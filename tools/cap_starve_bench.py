#!/usr/bin/env python3
"""Targeted >TF_ACTIVE_KV_CAP (~800k) starvation test under continuous overlapping short traffic."""
import argparse
import asyncio
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import httpx

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
class ReqResult:
    req_id: str
    cls: str
    t_submit: float
    t_first: float
    t_end: float
    ttft: float
    e2e: float
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    tpot_ms: Optional[float]
    error: str = ""


def make_text(salt: str, tag: str, approx_tokens: int) -> str:
    rng = random.Random(f"{salt}:{tag}")
    n_words = max(1, int(approx_tokens / TOKENS_PER_WORD))
    units: list[str] = []
    while len(units) < n_words:
        k = min(rng.randint(6, 16), n_words - len(units))
        seg = [rng.choice(WORDS) for _ in range(k)]
        seg[-1] += "."
        units.extend(seg)
    return f"[salt={salt} tag={tag}] " + " ".join(units)


def pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(len(s) - 1, int(math.ceil(p / 100.0 * len(s))) - 1)
    return s[max(0, idx)]


async def send_req(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    req_id: str,
    cls: str,
    prompt: str,
    max_tokens: int,
    t0: float,
) -> ReqResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
        "ignore_eos": True,
    }
    t_start = time.perf_counter()
    t_first: Optional[float] = None
    t_last: Optional[float] = None
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
    except Exception as e:
        err = repr(e)

    t_end = time.perf_counter()
    comp_toks = int(usage.get("completion_tokens", 0))
    prompt_toks = int(usage.get("prompt_tokens", 0))
    cached_toks = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
    ttft = (t_first - t_start) if t_first is not None else -1.0
    e2e = t_end - t_start
    if t_first is not None and t_last is not None and comp_toks > 1:
        decode_s = max(1e-6, t_last - t_first)
        tpot_ms: Optional[float] = decode_s * 1000.0 / (comp_toks - 1)
    else:
        tpot_ms = None
    return ReqResult(
        req_id=req_id,
        cls=cls,
        t_submit=t_start - t0,
        t_first=(t_first - t0) if t_first is not None else -1.0,
        t_end=t_end - t0,
        ttft=ttft,
        e2e=e2e,
        prompt_tokens=prompt_toks,
        cached_tokens=cached_toks,
        completion_tokens=comp_toks,
        tpot_ms=tpot_ms,
        error=err,
    )


async def run_bench(args: argparse.Namespace) -> None:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8"):
        pass

    limits = httpx.Limits(max_connections=None, max_keepalive_connections=64)
    timeout = httpx.Timeout(connect=10.0, read=1200.0, write=120.0, pool=60.0)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        prefix = make_text(args.salt, "mega_prefix", args.mega_tokens)
        tw0 = time.perf_counter()
        warm_res = await send_req(client, args.url, args.model, "warm_mega", "warm", prefix, 4, tw0)
        warm_s = time.perf_counter() - tw0
        if warm_res.error:
            raise RuntimeError(f"Warm failed: {warm_res.error}")
        if warm_res.prompt_tokens <= 786432:
            raise RuntimeError(f"Warm prompt_tokens={warm_res.prompt_tokens} <= 786432; increase --mega-tokens")
        print(
            f"[warm] mega_prefix: prompt_tokens={warm_res.prompt_tokens} cached={warm_res.cached_tokens} "
            f"in {warm_s:.1f}s",
            flush=True,
        )

        # Pre-generate all measured prompts before t0
        short_0_prompt = make_text(args.salt, "s0", 500) + "\nSummarize briefly."
        mega_prompt = prefix + "\n" + make_text(args.salt, "mega_u", 500) + "\nReply briefly."
        short_prompts = [
            make_text(args.salt, f"s{k}", 500) + "\nSummarize briefly."
            for k in range(1, args.n_short + 1)
        ]

        t0 = time.perf_counter()
        tasks = [
            asyncio.create_task(
                send_req(client, args.url, args.model, "short_0", "short_init", short_0_prompt, args.short_max_tokens, t0)
            )
        ]
        await asyncio.sleep(1.0)
        tasks.append(
            asyncio.create_task(
                send_req(client, args.url, args.model, "mega_cached", "mega_cached", mega_prompt, 32, t0)
            )
        )
        for k, sp in enumerate(short_prompts, start=1):
            await asyncio.sleep(1.0 if k == 1 else args.short_interval)
            tasks.append(
                asyncio.create_task(
                    send_req(client, args.url, args.model, f"short_{k:02d}", "short_stream", sp, args.short_max_tokens, t0)
                )
            )
        results = list(await asyncio.gather(*tasks))
        measured_s = time.perf_counter() - t0

    mega = next(r for r in results if r.cls == "mega_cached")
    shorts = [r for r in results if r.cls == "short_stream"]
    ok_shorts = [r for r in shorts if not r.error and r.ttft >= 0]
    started_before = sum(1 for r in ok_shorts if mega.t_first >= 0 and r.t_first < mega.t_first)
    finished_before = sum(1 for r in ok_shorts if mega.t_first >= 0 and r.t_end < mega.t_first)
    summary = {
        "warm_prompt_tokens": warm_res.prompt_tokens,
        "warm_wall_s": warm_s,
        "measured_wall_s": measured_s,
        "errors": sum(1 for r in results if r.error),
        "mega_prompt_tokens": mega.prompt_tokens,
        "mega_cached_tokens": mega.cached_tokens,
        "mega_t_submit_s": mega.t_submit,
        "mega_t_first_s": mega.t_first,
        "mega_ttft_s": mega.ttft,
        "mega_e2e_s": mega.e2e,
        "n_short_stream": len(shorts),
        "short_stream_started_before_mega": started_before,
        "short_stream_finished_before_mega": finished_before,
        "short_stream_ttft_p50": pct([r.ttft for r in ok_shorts], 50),
        "short_stream_ttft_p95": pct([r.ttft for r in ok_shorts], 95),
        "short_stream_ttft_max": max([r.ttft for r in ok_shorts], default=0.0),
    }

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\n{'ReqID':<12} {'Class':<13} {'t_sub':>7} {'t_1st':>7} {'t_end':>7} {'TTFT':>7} {'E2E':>7} {'Prompt':>8} {'Cached':>8} {'Out':>5} {'Err'}")
    for r in sorted(results, key=lambda x: x.t_submit):
        print(
            f"{r.req_id:<12} {r.cls:<13} {r.t_submit:7.2f} {r.t_first:7.2f} {r.t_end:7.2f} "
            f"{r.ttft:7.2f} {r.e2e:7.2f} {r.prompt_tokens:8d} {r.cached_tokens:8d} {r.completion_tokens:5d} {r.error}"
        )

    payload = {
        "config": vars(args),
        "warm": asdict(warm_res),
        "summary": summary,
        "results": [asdict(r) for r in results],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18888")
    ap.add_argument("--model", default="DeepSeek-V4.1-Flash-1017")
    ap.add_argument("--salt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mega-tokens", type=int, default=845000)
    ap.add_argument("--n-short", type=int, default=20)
    ap.add_argument("--short-interval", type=float, default=2.0)
    ap.add_argument("--short-max-tokens", type=int, default=256)
    args = ap.parse_args()
    asyncio.run(run_bench(args))


if __name__ == "__main__":
    main()
