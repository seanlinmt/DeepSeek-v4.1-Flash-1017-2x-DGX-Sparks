#!/usr/bin/env python3
"""Open-loop & saturation A/B benchmark for Chronos (Patch 0070 vs 0071)."""
import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import httpx

ALL_SUITES = ("cached_sat", "kv_cap", "pool_starve", "prod_replay")

# Calibrated prose vocabulary (matches tools/checklib.py: ~1.15 tokens/word on DeepSeek-V4.1-Flash).
WORDS = (
    "time year people way day man thing woman life child world school state family student group country "
    "problem hand part place case week company system program question work government number night point "
    "home water room mother area money story fact month lot right study book eye job word business issue "
    "side kind head house service friend father power hour game line end member law car city community name "
    "president team minute idea kid body information back parent face others level office door health person "
    "art war history party result change morning reason research girl guy moment air teacher force education "
    "river mountain signal engine garden theory market winter method bridge letter window voice paper field"
).split()
TOKENS_PER_WORD = 1.15


@dataclass
class Result:
    suite: str
    rep: int
    req_id: str
    cls: str
    t_submit: float
    in_flight: int
    ttft: float
    e2e: float
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    tpot_ms: Optional[float]
    token_sha: str
    error: str = ""


def default_url() -> str:
    return os.environ.get("API_URL", "http://127.0.0.1:" + os.environ.get("PORT", "8888"))


def resolve_api_key(api_key_file: Optional[str]) -> str:
    key = os.environ.get("API_KEY", "").strip()
    if key and key not in ("0", "1"):
        return key
    if api_key_file:
        with open(api_key_file, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    return s
    return ""


def make_text(salt: str, tag: str, approx_tokens: int) -> str:
    # Deterministic prose per (salt, tag) for cache control (~1.15 tokens/word).
    rng = random.Random(f"{salt}:{tag}")
    n_words = max(1, int(approx_tokens / TOKENS_PER_WORD))
    units: list[str] = []
    while len(units) < n_words:
        k = min(rng.randint(6, 16), n_words - len(units))
        seg = [rng.choice(WORDS) for _ in range(k)]
        seg[-1] += "."
        units.extend(seg)
    return f"[salt={salt} tag={tag}] " + " ".join(units)


async def send_req(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    suite: str,
    rep: int,
    req_id: str,
    cls: str,
    prompt: str,
    max_tokens: int,
    t0: float,
    in_flight_counter: list[int],
    ignore_eos: bool = False,
) -> Result:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if ignore_eos:
        payload["ignore_eos"] = True

    in_flight_at_submit = in_flight_counter[0]
    in_flight_counter[0] += 1
    t_start = time.perf_counter()
    t_first: Optional[float] = None
    t_last: Optional[float] = None
    chunks: list[str] = []
    usage: dict = {}
    err = ""
    try:
        async with client.stream("POST", f"{url}/v1/chat/completions", json=payload) as r:
            if r.status_code >= 400:
                body = (await r.aread()).decode("utf-8", errors="replace").strip()[:500]
                err = f"HTTP {r.status_code}: {body}"
            else:
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        if data == "[DONE]":
                            break
                        continue
                    obj = json.loads(data)
                    if obj.get("usage"):
                        usage = obj["usage"]
                    choices = obj.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta", {}).get("content")
                        if delta:
                            now = time.perf_counter()
                            if t_first is None:
                                t_first = now
                            t_last = now
                            chunks.append(delta)
    except Exception as e:
        err = repr(e)
    finally:
        in_flight_counter[0] -= 1

    t_end = time.perf_counter()
    ttft = (t_first - t_start) if t_first is not None else -1.0
    e2e = t_end - t_start
    comp_toks = int(usage.get("completion_tokens", 0))
    prompt_toks = int(usage.get("prompt_tokens", 0))
    cached_toks = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
    if t_first is not None and t_last is not None and comp_toks > 1:
        decode_s = max(1e-6, t_last - t_first)
        tpot_ms: Optional[float] = decode_s * 1000.0 / (comp_toks - 1)
    else:
        tpot_ms = None
    sha = hashlib.sha256("".join(chunks).encode()).hexdigest()[:12]
    return Result(
        suite=suite,
        rep=rep,
        req_id=req_id,
        cls=cls,
        t_submit=t_start - t0,
        in_flight=in_flight_at_submit,
        ttft=ttft,
        e2e=e2e,
        prompt_tokens=prompt_toks,
        cached_tokens=cached_toks,
        completion_tokens=comp_toks,
        tpot_ms=tpot_ms,
        token_sha=sha,
        error=err,
    )


async def warm_prefixes(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    salt: str,
    tag: str,
    count: int,
    tokens: int,
) -> list[str]:
    prefixes = [make_text(salt, f"{tag}_p{i}", tokens) for i in range(count)]
    t0 = time.perf_counter()
    counter = [0]
    actual_toks: list[int] = []
    # Warm sequentially to avoid pre-test contention
    for i, p in enumerate(prefixes):
        res = await send_req(client, url, model, "warm", 0, f"{tag}_p{i}", "warm", p, 4, t0, counter)
        if res.error:
            raise RuntimeError(f"Warm failed on {tag}_p{i}: {res.error}")
        actual_toks.append(res.prompt_tokens)
    if actual_toks:
        avg_toks = sum(actual_toks) / len(actual_toks)
        total_toks = sum(actual_toks)
        print(
            f"[warm] {tag}: {count} prefixes, target={tokens}, actual_avg={avg_toks:.0f}, "
            f"total={total_toks} tok in {time.perf_counter() - t0:.1f}s",
            file=sys.stderr,
            flush=True,
        )
        if tokens > 0 and abs(avg_toks - tokens) / tokens > 0.25:
            print(
                f"[warn] {tag}: actual prompt_tokens ({avg_toks:.0f}) deviates >25% from target ({tokens})",
                file=sys.stderr,
                flush=True,
            )
    return prefixes


async def run_cached_sat(
    client: httpx.AsyncClient, url: str, model: str, salt: str, rep: int, ignore_eos: bool
) -> tuple[list[Result], float]:
    prefixes = await warm_prefixes(client, url, model, salt, f"csat_r{rep}", 8, 140_000)
    # Pre-generate all prompts before t0 to avoid blocking the event loop mid-suite
    agent_prompts = [
        pref + "\n" + make_text(salt, f"csat_r{rep}_u{i}", 2500) + "\nWrite a detailed 1000-word technical architecture analysis."
        for i, pref in enumerate(prefixes)
    ]
    short_prompts = [
        make_text(salt, f"csat_r{rep}_s{j}", 500) + "\nSummarize in 3 short sentences."
        for j in range(24)
    ]

    t0 = time.perf_counter()
    counter = [0]
    tasks = []
    for i, turn in enumerate(agent_prompts):
        tasks.append(
            asyncio.create_task(
                send_req(
                    client, url, model, "cached_sat", rep, f"csat_r{rep}_a{i}",
                    "agent_cached", turn, 1024, t0, counter, ignore_eos=ignore_eos,
                )
            )
        )
    for j, short_p in enumerate(short_prompts):
        await asyncio.sleep(2.5)
        tasks.append(
            asyncio.create_task(
                send_req(
                    client, url, model, "cached_sat", rep, f"csat_r{rep}_s{j}",
                    "short_util", short_p, 128, t0, counter, ignore_eos=ignore_eos,
                )
            )
        )
    res = list(await asyncio.gather(*tasks))
    return res, time.perf_counter() - t0


async def run_kv_cap(
    client: httpx.AsyncClient, url: str, model: str, salt: str, rep: int, ignore_eos: bool
) -> tuple[list[Result], float]:
    # 6 x 300k = 1.8M total (exceeds 786k TF_ACTIVE_KV_CAP while leaving headroom in a 2.1M pool)
    prefixes = await warm_prefixes(client, url, model, salt, f"kvcap_r{rep}", 6, 300_000)
    prompts = [
        pref + "\n" + make_text(salt, f"kvcap_r{rep}_u{i}", 2000) + "\nProvide a thorough code review of all modules."
        for i, pref in enumerate(prefixes)
    ]

    t0 = time.perf_counter()
    counter = [0]
    tasks = [
        asyncio.create_task(
            send_req(
                client, url, model, "kv_cap", rep, f"kvcap_r{rep}_h{i}",
                "heavy_cached", turn, 512, t0, counter, ignore_eos=ignore_eos,
            )
        )
        for i, turn in enumerate(prompts)
    ]
    res = list(await asyncio.gather(*tasks))
    return res, time.perf_counter() - t0


async def run_pool_starve(
    client: httpx.AsyncClient, url: str, model: str, salt: str, rep: int, ignore_eos: bool
) -> tuple[list[Result], float]:
    prefixes = await warm_prefixes(client, url, model, salt, f"starve_r{rep}", 5, 350_000)
    bg_prompts = [
        pref + "\n" + make_text(salt, f"starve_r{rep}_u{i}", 1500) + "\nExplain step by step in detail."
        for i, pref in enumerate(prefixes)
    ]
    blocked_p = make_text(salt, f"starve_r{rep}_big", 450_000) + "\nAnalyze."
    short_prompts = [
        make_text(salt, f"starve_r{rep}_s{j}", 400) + "\nAnswer briefly."
        for j in range(12)
    ]

    t0 = time.perf_counter()
    counter = [0]
    tasks = [
        asyncio.create_task(
            send_req(
                client, url, model, "pool_starve", rep, f"starve_r{rep}_bg{i}",
                "bg_heavy", turn, 1024, t0, counter, ignore_eos=ignore_eos,
            )
        )
        for i, turn in enumerate(bg_prompts)
    ]
    await asyncio.sleep(3.0)
    # Large cold request intended to trigger ContextFull / full_wait
    tasks.append(
        asyncio.create_task(
            send_req(
                client, url, model, "pool_starve", rep, f"starve_r{rep}_big",
                "blocked_head", blocked_p, 256, t0, counter, ignore_eos=ignore_eos,
            )
        )
    )
    for j, short_p in enumerate(short_prompts):
        await asyncio.sleep(1.5)
        tasks.append(
            asyncio.create_task(
                send_req(
                    client, url, model, "pool_starve", rep, f"starve_r{rep}_s{j}",
                    "short_bypass", short_p, 128, t0, counter, ignore_eos=ignore_eos,
                )
            )
        )
    res = list(await asyncio.gather(*tasks))
    return res, time.perf_counter() - t0


async def run_prod_replay(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    salt: str,
    n_reqs: int = 120,
    rate_rps: float = 0.18,
    ignore_eos: bool = False,
) -> tuple[list[Result], float]:
    rng = random.Random(f"{salt}:replay")
    oc_prefs = await warm_prefixes(client, url, model, salt, "rep_oc", 8, 140_000)
    non_oc_prefs = await warm_prefixes(client, url, model, salt, "rep_noc", 6, 80_000)

    # Pre-generate workload and Poisson inter-arrivals before t0
    schedule: list[tuple[str, str, str, int, float]] = []
    for i in range(n_reqs):
        u = rng.random()
        if u < 0.60:
            cls, max_t = "opencode_cached", 1024
            p = rng.choice(oc_prefs) + "\n" + make_text(salt, f"r_oc_{i}", 2400) + "\nImplement patch."
        elif u < 0.85:
            cls, max_t = "non_opencode_cached", 512
            p = rng.choice(non_oc_prefs) + "\n" + make_text(salt, f"r_noc_{i}", 2000) + "\nDraft response."
        elif u < 0.95:
            cls, max_t = "short_util", 128
            p = make_text(salt, f"r_su_{i}", 500) + "\nExtract JSON."
        else:
            cls, max_t = "cold_large", 256
            p = make_text(salt, f"r_cl_{i}", 32_000) + "\nSummarize."
        delay = -math.log(max(1e-9, rng.random())) / rate_rps if i + 1 < n_reqs else 0.0
        schedule.append((f"replay_{i:04d}", cls, p, max_t, delay))

    t0 = time.perf_counter()
    counter = [0]
    tasks = []
    for req_id, cls, p, max_t, delay in schedule:
        tasks.append(
            asyncio.create_task(
                send_req(
                    client, url, model, "prod_replay", 0, req_id,
                    cls, p, max_t, t0, counter, ignore_eos=ignore_eos,
                )
            )
        )
        if delay > 0.0:
            await asyncio.sleep(delay)
    res = list(await asyncio.gather(*tasks))
    return res, time.perf_counter() - t0


def pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(len(s) - 1, int(math.ceil(p / 100.0 * len(s))) - 1)
    return s[max(0, idx)]


def summarize(results: list[Result], measured_wall_s: float, total_wall_s: float) -> None:
    by_key: dict[tuple[str, str], list[Result]] = {}
    for r in results:
        by_key.setdefault((r.suite, r.cls), []).append(r)
    print(
        f"\n{'Suite':<12} {'Class':<20} {'N':>4} {'Err':>3} {'InFl':>4} "
        f"{'Prompt':>8} {'Out':>5} {'Hit%':>6} {'TTFT p50':>9} {'TTFT p95':>9} {'TPOT p50':>9} {'E2E p95':>9}"
    )
    cached_classes = {"agent_cached", "heavy_cached", "bg_heavy", "opencode_cached", "non_opencode_cached"}
    for (suite, cls), rows in sorted(by_key.items()):
        ok = [r for r in rows if not r.error and r.ttft >= 0]
        tpots = [r.tpot_ms for r in ok if r.tpot_ms is not None]
        hit = (sum(r.cached_tokens for r in ok) / max(1, sum(r.prompt_tokens for r in ok))) * 100.0
        p_med = pct([float(r.prompt_tokens) for r in ok], 50)
        o_med = pct([float(r.completion_tokens) for r in ok], 50)
        infl_med = pct([float(r.in_flight) for r in rows], 50)
        tpot_str = f"{pct(tpots, 50):>7.1f}ms" if tpots else f"{'n/a':>9}"
        print(
            f"{suite:<12} {cls:<20} {len(rows):>4} {len(rows)-len(ok):>3} {infl_med:>4.0f} "
            f"{p_med:>8.0f} {o_med:>5.0f} {hit:>5.1f}% "
            f"{pct([r.ttft for r in ok], 50):>8.2f}s {pct([r.ttft for r in ok], 95):>8.2f}s "
            f"{tpot_str} {pct([r.e2e for r in ok], 95):>8.2f}s"
        )
        if cls in cached_classes and ok and hit < 80.0:
            print(
                f"[warn] {suite}/{cls}: low cache hit rate ({hit:.1f}%) — prefixes may have been evicted during warm",
                file=sys.stderr,
            )
    tot_out = sum(r.completion_tokens for r in results if not r.error)
    print(
        f"\nAggregate decode throughput: {tot_out / max(1e-6, measured_wall_s):.2f} tok/s "
        f"({tot_out} tok in {measured_wall_s:.1f}s measured; {total_wall_s:.1f}s total incl. warm)"
    )


async def main() -> None:
    ap = argparse.ArgumentParser(description="Open-loop & saturation A/B benchmark for Chronos.")
    ap.add_argument("--url", default=default_url(), help="Server base URL (default: API_URL or http://127.0.0.1:PORT/8888)")
    ap.add_argument("--model", default="DeepSeek-V4.1-Flash-1017")
    ap.add_argument("--api-key-file", default=None, help="Path to API key file (falls back to API_KEY env var)")
    ap.add_argument("--suites", nargs="+", choices=ALL_SUITES, default=list(ALL_SUITES))
    ap.add_argument("--reps", type=int, default=3, help="Repetitions for cached_sat, kv_cap, and pool_starve (ignored by prod_replay)")
    ap.add_argument("--replay-reqs", type=int, default=120, help="Number of requests for prod_replay")
    ap.add_argument("--salt", required=True)
    ap.add_argument("--rate", type=float, default=0.18, help="Poisson RPS for prod_replay")
    ap.add_argument("--ignore-eos", action="store_true", help="Pass ignore_eos=true to force decode length to max_tokens")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Fail fast if output path is not writable before running a long benchmark
    with open(out_path, "a", encoding="utf-8"):
        pass

    headers = {}
    api_key = resolve_api_key(args.api_key_file)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    limits = httpx.Limits(max_connections=None, max_keepalive_connections=128)
    timeout = httpx.Timeout(connect=10.0, read=600.0, write=120.0, pool=60.0)
    all_res: list[Result] = []
    suite_wall_s: dict[str, float] = {}
    measured_wall_s = 0.0
    t_wall0 = time.perf_counter()

    suite_fns = {
        "cached_sat": run_cached_sat,
        "kv_cap": run_kv_cap,
        "pool_starve": run_pool_starve,
    }

    async with httpx.AsyncClient(limits=limits, timeout=timeout, headers=headers) as client:
        for s in args.suites:
            if s == "prod_replay":
                res, elapsed = await run_prod_replay(
                    client, args.url, args.model, args.salt,
                    n_reqs=args.replay_reqs, rate_rps=args.rate, ignore_eos=args.ignore_eos,
                )
                all_res.extend(res)
                suite_wall_s[s] = elapsed
                measured_wall_s += elapsed
            else:
                s_elapsed = 0.0
                for rep in range(args.reps):
                    res, elapsed = await suite_fns[s](
                        client, args.url, args.model, args.salt, rep, args.ignore_eos
                    )
                    all_res.extend(res)
                    s_elapsed += elapsed
                suite_wall_s[s] = s_elapsed
                measured_wall_s += s_elapsed

    total_wall_s = time.perf_counter() - t_wall0
    summarize(all_res, measured_wall_s, total_wall_s)
    payload = {
        "config": {
            "url": args.url,
            "model": args.model,
            "suites": args.suites,
            "reps": args.reps,
            "replay_reqs": args.replay_reqs,
            "rate": args.rate,
            "salt": args.salt,
            "ignore_eos": args.ignore_eos,
        },
        "salt": args.salt,
        "wall_s": measured_wall_s,
        "measured_wall_s": measured_wall_s,
        "total_wall_s": total_wall_s,
        "suite_wall_s": suite_wall_s,
        "results": [asdict(r) for r in all_res],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
