#!/usr/bin/env python3
"""Before/after traffic & concurrency benchmark tool for DeepSeek / Chronos.

Measures traffic concurrency patterns:
  1. burst: seeds a shared 32k prefix, launches 4 concurrent cached agent turns,
     and 0.5s later launches 4 concurrent short utility requests.
  2. cut_in: seeds a 32k prefix, starts 1 cold 64k request, and --delay seconds later
     launches 2 concurrent short probes (short_utility and cached_turn).
  3. lane_reserve: launches 8 concurrent long requests (10k tokens, max_tokens=1024) to saturate standard lanes,
     and --lane-delay seconds later launches 4 staggered short utility requests.

Output:
  Writes structured JSON (--out) with schema, label, url, model, salt, reps, timestamp, and suites.
  Compares runs (--compare BEFORE.json AFTER.json) with a clean Markdown table comparing TTFT,
  total time, decode rates, aggregate throughput, and cut_in token bit-exactness.

Stdlib only.
"""
import argparse
import datetime
import hashlib
import json
import os
import random
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

WORDS = (
    "time year people way day man thing woman life child world school state family student group country "
    "problem hand part place case week company system program question work government number night point "
    "home water room mother area money story fact month lot right study book eye job word business issue "
    "side kind head house service friend father power hour game line end member law car city community name "
    "president team minute idea kid body information back parent face others level office door health person "
    "art war history party result change morning reason research girl guy moment air teacher force education "
    "river mountain signal engine garden theory market winter method bridge letter window voice paper field"
).split()
TOK_PER_SIZE = 0.82
DEFAULT_MODEL = "DeepSeek-V4.1-Flash-1017"

LONG_ESSAY_PROMPT = (
    "Write a comprehensive, multi-section technical essay analyzing distributed systems "
    "consensus, replication, fault tolerance, and recovery protocols in detail. "
    "Provide concrete architecture examples and do not stop early."
)

SHORT_UTILITY_TOPICS = (
    "how virtual memory page tables and TLB translation work in modern operating systems",
    "the trade-offs between optimistic concurrency control and two-phase locking in databases",
    "how TCP congestion control algorithms like BBR and CUBIC manage window sizing and latency",
    "the mechanisms of cache coherence protocols like MESI in multi-core CPU architectures",
    "how log-structured merge-trees (LSM trees) optimize write throughput compared to B-trees",
    "the design principles of Raft consensus including leader election and log compaction",
    "how epoll and kqueue implement event-driven I/O multiplexing in kernel architectures",
    "the differences between NVMe over Fabrics RDMA and TCP transport implementations",
)


def prose(size: int, seed) -> str:
    rng = random.Random(seed)
    out = []
    while len(out) < size * 0.72:
        sentence = [rng.choice(WORDS) for _ in range(rng.randint(6, 16))]
        out += sentence[:-1] + [sentence[-1] + "."]
    return " ".join(out)


def prose_tokens(tokens: int, seed) -> str:
    return prose(max(1, int(tokens / TOK_PER_SIZE)), seed)


def rep_salt(base_salt: str, rep_idx: int, total_reps: int) -> str:
    """Returns salt for given repetition index. Keeps base_salt verbatim when reps==1."""
    if total_reps <= 1:
        return base_salt
    return f"{base_salt}-r{rep_idx}"


def port_from_local_sh(local_sh_path: str | None = None) -> str | None:
    """Read kit PORT from scripts/local.sh beside this repo."""
    if local_sh_path is None:
        local_sh_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "local.sh")
    try:
        port = None
        with open(local_sh_path, encoding="utf-8") as f:
            for line in f:
                m = re.fullmatch(r"""\s*(?:export\s+)?PORT\s*=\s*(?:"(\d{1,5})"|'(\d{1,5})'|(\d{1,5}))\s*(?:#.*)?""", line.rstrip("\n"))
                if m:
                    port = next(value for value in m.groups() if value is not None)
        return port
    except OSError:
        pass
    return None


def default_base(local_sh_path: str | None = None) -> str:
    """Endpoint resolution: API_URL, else PORT from env or scripts/local.sh, else 8888."""
    if os.environ.get("API_URL"):
        return os.environ["API_URL"].rstrip("/")
    port = os.environ.get("PORT") or port_from_local_sh(local_sh_path) or "8888"
    return f"http://127.0.0.1:{port}"


def api_key() -> str:
    """The server's API key: API_KEY, else first key in API_KEY_FILE, else none."""
    key = os.environ.get("API_KEY", "").strip()
    if key:
        return key
    path = os.environ.get("API_KEY_FILE", os.path.expanduser("~/.local/state/dsv41-tensorfold/api-key"))
    try:
        with open(path, encoding="utf-8") as f:
            line = next((x.strip() for x in f if x.strip() and not x.startswith("#")), "")
    except OSError:
        return ""
    return line.split(":", 1)[1].strip() if ":" in line else line


def get_opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def stream_chat(
    url: str,
    model: str,
    messages: list,
    max_tokens: int,
    seed: int = 7,
    timeout: float = 900.0,
    start_gate: threading.Event | None = None,
    suite_start: float | None = None,
) -> dict:
    """Sends a streamed chat completion request via SSE and measures timing and usage."""
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    endpoint = url.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    k = api_key()
    if k:
        headers["Authorization"] = f"Bearer {k}"

    req = urllib.request.Request(endpoint, json.dumps(body).encode("utf-8"), headers)

    if start_gate is not None:
        start_gate.wait()

    t0 = time.monotonic()
    base_t = suite_start if suite_start is not None else t0
    started_at_rel_s = round(t0 - base_t, 4)

    first = None
    last = None
    usage = None
    server = None
    text = []
    events = 0
    finish = None
    opener = get_opener()

    try:
        with opener.open(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                ev = json.loads(data)
                if ev.get("usage"):
                    usage = ev["usage"]
                if ev.get("tensorfold"):
                    server = ev["tensorfold"]
                for ch in ev.get("choices") or []:
                    delta = ch.get("delta") or {}
                    piece = delta.get("content") or ""
                    if not piece:
                        piece = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if piece:
                        now = time.monotonic()
                        if first is None:
                            first = now
                        last = now
                        events += 1
                        text.append(piece)
                    if ch.get("finish_reason"):
                        finish = ch.get("finish_reason")
    except urllib.error.HTTPError as exc:
        end = time.monotonic()
        return {
            "error": f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:300]}",
            "total_s": round(end - t0, 4),
            "started_at_rel_s": started_at_rel_s,
            "first_token_at_rel_s": None,
            "finished_at_rel_s": round(end - base_t, 4),
            "ttft_s": None,
            "decode_tok_s": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "token_sha": "",
        }
    except Exception as exc:
        end = time.monotonic()
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "total_s": round(end - t0, 4),
            "started_at_rel_s": started_at_rel_s,
            "first_token_at_rel_s": None,
            "finished_at_rel_s": round(end - base_t, 4),
            "ttft_s": None,
            "decode_tok_s": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "token_sha": "",
        }

    end = time.monotonic()
    finished_at_rel_s = round(end - base_t, 4)
    total_s = round(end - t0, 4)
    first_token_at_rel_s = round(first - base_t, 4) if first is not None else None
    ttft_s = round(first - t0, 4) if first is not None else None

    pt = usage.get("prompt_tokens", 0) if usage else 0
    ct = usage.get("completion_tokens", 0) if usage else len(text)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) if usage else 0
    if cached == 0 and isinstance(server, dict):
        cached = next((v for k, v in server.items() if "cached" in k.lower() and isinstance(v, int)), 0)

    if ct > 1 and ttft_s is not None and total_s > ttft_s:
        decode_tok_s = round((ct - 1) / (total_s - ttft_s), 2)
    else:
        decode_tok_s = None

    if isinstance(server, dict) and server.get("token_sha"):
        token_sha = str(server["token_sha"])
    else:
        token_sha = hashlib.sha256("".join(text).encode("utf-8")).hexdigest()

    return {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "cached_tokens": cached,
        "ttft_s": ttft_s,
        "total_s": total_s,
        "decode_tok_s": decode_tok_s,
        "started_at_rel_s": started_at_rel_s,
        "first_token_at_rel_s": first_token_at_rel_s,
        "finished_at_rel_s": finished_at_rel_s,
        "token_sha": token_sha,
        "finish_reason": finish,
        "sse_events": events,
        "text": "".join(text),
    }


def calc_median(xs: list) -> float | None:
    valid = [x for x in xs if x is not None]
    return round(statistics.median(valid), 4) if valid else None


def calc_mean(xs: list) -> float | None:
    valid = [x for x in xs if x is not None]
    return round(statistics.mean(valid), 4) if valid else None


def calc_p95(xs: list) -> float | None:
    valid = sorted([x for x in xs if x is not None])
    if not valid:
        return None
    if len(valid) == 1:
        return round(valid[0], 4)
    k = (len(valid) - 1) * 0.95
    f = int(k)
    c = min(f + 1, len(valid) - 1)
    d = k - f
    return round(valid[f] + d * (valid[c] - valid[f]), 4)


def summarize_group(results: list[dict]) -> dict:
    ttfts = [r.get("ttft_s") for r in results if "error" not in r]
    totals = [r.get("total_s") for r in results if "error" not in r]
    decodes = [r.get("decode_tok_s") for r in results if "error" not in r]

    compl_tokens = sum(r.get("completion_tokens", 0) for r in results if "error" not in r)
    valid_firsts = [r["first_token_at_rel_s"] for r in results if "error" not in r and r.get("first_token_at_rel_s") is not None]
    valid_finished = [r["finished_at_rel_s"] for r in results if "error" not in r and r.get("finished_at_rel_s") is not None]

    agg_decode = None
    if valid_firsts and valid_finished and compl_tokens > 0:
        span = max(valid_finished) - min(valid_firsts)
        if span > 0:
            agg_decode = round(compl_tokens / span, 2)

    med_ttft = calc_median(ttfts)
    med_total = calc_median(totals)
    med_decode = calc_median(decodes)

    return {
        "count": len(results),
        "errors": sum(1 for r in results if "error" in r),
        "ttft_s": {"median": med_ttft, "p50": med_ttft, "mean": calc_mean(ttfts), "p95": calc_p95(ttfts)},
        "total_s": {"median": med_total, "p50": med_total, "mean": calc_mean(totals), "p95": calc_p95(totals)},
        "decode_tok_s": {"median": med_decode, "p50": med_decode, "mean": calc_mean(decodes), "p95": calc_p95(decodes)},
        "aggregate_decode_tok_s": agg_decode,
    }


def summarize_cutin_probe(results: list[dict]) -> dict:
    ttfts = [r.get("ttft_s") for r in results if "error" not in r]
    totals = [r.get("total_s") for r in results if "error" not in r]
    decodes = [r.get("decode_tok_s") for r in results if "error" not in r]
    pts = [r.get("prompt_tokens") for r in results if "error" not in r and r.get("prompt_tokens") is not None]
    cts = [r.get("cached_tokens") for r in results if "error" not in r and r.get("cached_tokens") is not None]
    shas = [r.get("token_sha") for r in results if "error" not in r and r.get("token_sha")]

    med_ttft = calc_median(ttfts)
    med_total = calc_median(totals)
    med_decode = calc_median(decodes)

    return {
        "count": len(results),
        "errors": sum(1 for r in results if "error" in r),
        "ttft_s": med_ttft,
        "total_s": med_total,
        "decode_tok_s": med_decode,
        "prompt_tokens": pts[0] if pts else 0,
        "cached_tokens": cts[0] if cts else 0,
        "token_sha": shas[0] if shas else "",
        "ttft_s_stats": {"median": med_ttft, "p50": med_ttft, "mean": calc_mean(ttfts), "p95": calc_p95(ttfts)},
        "total_s_stats": {"median": med_total, "p50": med_total, "mean": calc_mean(totals), "p95": calc_p95(totals)},
        "decode_tok_s_stats": {"median": med_decode, "p50": med_decode, "mean": calc_mean(decodes), "p95": calc_p95(decodes)},
        "p50": {"ttft_s": med_ttft, "total_s": med_total, "decode_tok_s": med_decode},
        "mean": {"ttft_s": calc_mean(ttfts), "total_s": calc_mean(totals), "decode_tok_s": calc_mean(decodes)},
        "p95": {"ttft_s": calc_p95(ttfts), "total_s": calc_p95(totals), "decode_tok_s": calc_p95(decodes)},
    }


def calc_overall_aggregate_decode(results: list[dict]) -> float | None:
    compl_tokens = sum(r.get("completion_tokens", 0) for r in results if "error" not in r)
    valid_firsts = [r["first_token_at_rel_s"] for r in results if "error" not in r and r.get("first_token_at_rel_s") is not None]
    valid_finished = [r["finished_at_rel_s"] for r in results if "error" not in r and r.get("finished_at_rel_s") is not None]
    if valid_firsts and valid_finished and compl_tokens > 0:
        span = max(valid_finished) - min(valid_firsts)
        if span > 0:
            return round(compl_tokens / span, 2)
    return None


def run_burst(
    url: str,
    model: str,
    salt: str,
    seed: int = 7,
    timeout: float = 900.0,
    burst_delay: float = 0.5,
) -> dict:
    suite_start = time.monotonic()
    # 1. Seed shared ~32,000-token prefix once
    shared_prefix = prose_tokens(32000, f"seed-burst-{salt}")
    seed_res = stream_chat(
        url,
        model,
        [{"role": "user", "content": shared_prefix + "\n\nReply with OK."}],
        max_tokens=1,
        seed=seed,
        timeout=timeout,
        suite_start=suite_start,
    )

    # 2. 4 concurrent cached agent turns with natural long essay prompt
    agent_results = [None] * 4
    agent_threads = []

    def _agent_worker(i):
        prompt = (
            shared_prefix
            + f"\n\n[agent turn {i}]\n"
            + prose_tokens(1500, f"burst-agent-{i}-{salt}")
            + f"\n\n{LONG_ESSAY_PROMPT} Variation tag: {salt}-agent-{i}"
        )
        agent_results[i] = stream_chat(
            url,
            model,
            [{"role": "user", "content": prompt}],
            max_tokens=256,
            seed=seed,
            timeout=timeout,
            suite_start=suite_start,
        )

    for i in range(4):
        t = threading.Thread(target=_agent_worker, args=(i,))
        agent_threads.append(t)
        t.start()

    # 3. burst_delay seconds later, launch 4 concurrent short utility requests with natural concise prompts
    time.sleep(burst_delay)
    short_results = [None] * 4
    short_threads = []

    def _short_worker(i):
        topic = SHORT_UTILITY_TOPICS[i % len(SHORT_UTILITY_TOPICS)]
        prompt = (
            f"[ref burst-short-{i}-{salt}]\n"
            + prose_tokens(512, f"burst-short-{i}-{salt}")
            + f"\n\nExplain {topic}. Provide a concise structured explanation. Variation tag: {salt}-short-{i}"
        )
        short_results[i] = stream_chat(
            url,
            model,
            [{"role": "user", "content": prompt}],
            max_tokens=64,
            seed=seed,
            timeout=timeout,
            suite_start=suite_start,
        )

    for i in range(4):
        t = threading.Thread(target=_short_worker, args=(i,))
        short_threads.append(t)
        t.start()

    for t in agent_threads + short_threads:
        t.join()

    agent_sum = summarize_group(agent_results)
    short_sum = summarize_group(short_results)
    agg_decode = calc_overall_aggregate_decode(agent_results + short_results)

    return {
        "seed_prefix": seed_res,
        "agent_turns": agent_results,
        "short_utility": short_results,
        "summary": {
            "agent_turns": agent_sum,
            "short_utility": short_sum,
            "aggregate_decode_tok_s": agg_decode,
        },
    }


def run_cut_in(
    url: str,
    model: str,
    salt: str,
    seed: int = 7,
    delay: float = 6.0,
    cutin_long_tokens: int = 64000,
    timeout: float = 900.0,
) -> dict:
    suite_start = time.monotonic()
    # 1. Seed 32,000-token prefix for cached_turn
    cached_prefix = prose_tokens(32000, f"seed-cutin-{salt}")
    seed_res = stream_chat(
        url,
        model,
        [{"role": "user", "content": cached_prefix + "\n\nReply with OK."}],
        max_tokens=1,
        seed=seed,
        timeout=timeout,
        suite_start=suite_start,
    )

    # 2. Start 1 cold ~64,000-token request with unique prefix salt (max_tokens=16)
    long_res = [None]
    long_prompt = (
        f"[ref cold-long-{salt}]\n"
        + prose_tokens(cutin_long_tokens, f"cutin-cold-long-{salt}")
        + "\n\nSummarize the text above in one sentence."
    )

    def _long_worker():
        long_res[0] = stream_chat(
            url,
            model,
            [{"role": "user", "content": long_prompt}],
            max_tokens=16,
            seed=seed,
            timeout=timeout,
            suite_start=suite_start,
        )

    t_long = threading.Thread(target=_long_worker)
    t_long.start()

    # 3. delay seconds later, fire 2 concurrent short probes with natural prompts (max_tokens=64)
    time.sleep(delay)
    short_res = [None]
    cached_res = [None]

    def _short_worker():
        topic = SHORT_UTILITY_TOPICS[0]
        prompt = (
            f"[ref cold-short-{salt}]\n"
            + prose_tokens(512, f"cutin-cold-short-{salt}")
            + f"\n\nExplain {topic}. Provide a concise structured explanation. Variation tag: {salt}-short"
        )
        short_res[0] = stream_chat(
            url,
            model,
            [{"role": "user", "content": prompt}],
            max_tokens=64,
            seed=seed,
            timeout=timeout,
            suite_start=suite_start,
        )

    def _cached_worker():
        topic = SHORT_UTILITY_TOPICS[1]
        prompt = (
            cached_prefix
            + "\n\n[cached turn]\n"
            + prose_tokens(1024, f"cutin-cached-delta-{salt}")
            + f"\n\nExplain {topic}. Provide a concise structured explanation. Variation tag: {salt}-cached"
        )
        cached_res[0] = stream_chat(
            url,
            model,
            [{"role": "user", "content": prompt}],
            max_tokens=64,
            seed=seed,
            timeout=timeout,
            suite_start=suite_start,
        )

    t_short = threading.Thread(target=_short_worker)
    t_cached = threading.Thread(target=_cached_worker)
    t_short.start()
    t_cached.start()

    t_long.join()
    t_short.join()
    t_cached.join()

    lr = long_res[0]
    su = short_res[0]
    ct = cached_res[0]

    return {
        "seed_prefix": seed_res,
        "long_request": lr,
        "short_utility": su,
        "cached_turn": ct,
        "summary": {
            "long_request": summarize_cutin_probe([lr]),
            "short_utility": summarize_cutin_probe([su]),
            "cached_turn": summarize_cutin_probe([ct]),
        },
    }


def run_lane_reserve(
    url: str,
    model: str,
    salt: str,
    seed: int = 7,
    long_tokens: int = 10000,
    lane_delay: float = 32.0,
    timeout: float = 900.0,
    lane_stagger: float = 0.5,
) -> dict:
    suite_start = time.monotonic()
    # 1. Launch 8 concurrent long requests (10k tokens, max_tokens=1024, natural essay prompt)
    long_results = [None] * 8
    long_threads = []

    def _long_worker(i):
        prompt = (
            f"[ref lane-long-{i}-{salt}]\n"
            + prose_tokens(long_tokens, f"lane-long-{i}-{salt}")
            + f"\n\n{LONG_ESSAY_PROMPT} Variation tag: {salt}-long-{i}"
        )
        long_results[i] = stream_chat(
            url,
            model,
            [{"role": "user", "content": prompt}],
            max_tokens=1024,
            seed=seed,
            timeout=timeout,
            suite_start=suite_start,
        )

    for i in range(8):
        t = threading.Thread(target=_long_worker, args=(i,))
        long_threads.append(t)
        t.start()

    # 2. lane_delay seconds later, launch 4 short utility requests staggered by lane_stagger (default 0.5s)
    time.sleep(lane_delay)
    short_results = [None] * 4
    short_threads = []

    def _short_worker(i):
        topic = SHORT_UTILITY_TOPICS[i % len(SHORT_UTILITY_TOPICS)]
        prompt = (
            f"[ref lane-short-{i}-{salt}]\n"
            + prose_tokens(500, f"lane-short-{i}-{salt}")
            + f"\n\nExplain {topic}. Provide a concise structured explanation. Variation tag: {salt}-short-{i}"
        )
        short_results[i] = stream_chat(
            url,
            model,
            [{"role": "user", "content": prompt}],
            max_tokens=64,
            seed=seed,
            timeout=timeout,
            suite_start=suite_start,
        )

    for i in range(4):
        if i > 0 and lane_stagger > 0:
            time.sleep(lane_stagger)
        t = threading.Thread(target=_short_worker, args=(i,))
        short_threads.append(t)
        t.start()

    for t in long_threads + short_threads:
        t.join()

    long_sum = summarize_group(long_results)
    short_sum = summarize_group(short_results)
    agg_decode = calc_overall_aggregate_decode(long_results + short_results)

    return {
        "long_requests": long_results,
        "short_utility": short_results,
        "summary": {
            "long_requests": long_sum,
            "short_utility": short_sum,
            "aggregate_decode_tok_s": agg_decode,
        },
    }


def _format_val(v, unit=""):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}{unit}" if unit else f"{v:.3f}"
    return f"{v}{unit}" if unit else str(v)


def _compute_delta_and_speedup(val_a, val_b, is_latency=True):
    if val_a is None or val_b is None:
        return "-", "-"
    if not isinstance(val_a, (int, float)) or not isinstance(val_b, (int, float)):
        return "-", "-"
    if val_a == 0:
        return "-", "-"

    delta_pct = ((val_b - val_a) / val_a) * 100.0
    delta_str = f"{delta_pct:+.1f}%"

    if is_latency:
        if val_b > 0:
            speedup = val_a / val_b
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "-"
    else:
        if val_a > 0:
            speedup = val_b / val_a
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "-"

    return delta_str, speedup_str


def compare(a_path: str, b_path: str) -> int:
    try:
        with open(a_path, encoding="utf-8") as f:
            a = json.load(f)
    except Exception as e:
        print(f"Error loading {a_path}: {e}", file=sys.stderr)
        return 1
    try:
        with open(b_path, encoding="utf-8") as f:
            b = json.load(f)
    except Exception as e:
        print(f"Error loading {b_path}: {e}", file=sys.stderr)
        return 1

    label_a = a.get("label") or "before"
    label_b = b.get("label") or "after"
    url_a = a.get("url")
    url_b = b.get("url")
    salt_a = a.get("salt")
    salt_b = b.get("salt")
    reps_a = a.get("reps", 1)
    reps_b = b.get("reps", 1)

    METRICS = [
        ("burst", "agent_turns ttft_s (median)", ("suites", "burst", "summary", "agent_turns", "ttft_s", "median"), True, "s"),
        ("burst", "agent_turns ttft_s (p95)", ("suites", "burst", "summary", "agent_turns", "ttft_s", "p95"), True, "s"),
        ("burst", "agent_turns total_s (median)", ("suites", "burst", "summary", "agent_turns", "total_s", "median"), True, "s"),
        ("burst", "agent_turns decode_tok_s (median)", ("suites", "burst", "summary", "agent_turns", "decode_tok_s", "median"), False, " tok/s"),
        ("burst", "short_utility ttft_s (median)", ("suites", "burst", "summary", "short_utility", "ttft_s", "median"), True, "s"),
        ("burst", "short_utility ttft_s (p95)", ("suites", "burst", "summary", "short_utility", "ttft_s", "p95"), True, "s"),
        ("burst", "short_utility total_s (median)", ("suites", "burst", "summary", "short_utility", "total_s", "median"), True, "s"),
        ("burst", "short_utility decode_tok_s (median)", ("suites", "burst", "summary", "short_utility", "decode_tok_s", "median"), False, " tok/s"),
        ("burst", "aggregate_decode_tok_s", ("suites", "burst", "summary", "aggregate_decode_tok_s"), False, " tok/s"),

        ("cut_in", "long_request ttft_s", ("suites", "cut_in", "summary", "long_request", "ttft_s"), True, "s"),
        ("cut_in", "long_request total_s", ("suites", "cut_in", "summary", "long_request", "total_s"), True, "s"),
        ("cut_in", "short_utility ttft_s", ("suites", "cut_in", "summary", "short_utility", "ttft_s"), True, "s"),
        ("cut_in", "short_utility total_s", ("suites", "cut_in", "summary", "short_utility", "total_s"), True, "s"),
        ("cut_in", "cached_turn ttft_s", ("suites", "cut_in", "summary", "cached_turn", "ttft_s"), True, "s"),
        ("cut_in", "cached_turn total_s", ("suites", "cut_in", "summary", "cached_turn", "total_s"), True, "s"),

        ("lane_reserve", "short_utility ttft_s (median)", ("suites", "lane_reserve", "summary", "short_utility", "ttft_s", "median"), True, "s"),
        ("lane_reserve", "short_utility ttft_s (p95)", ("suites", "lane_reserve", "summary", "short_utility", "ttft_s", "p95"), True, "s"),
        ("lane_reserve", "short_utility total_s (median)", ("suites", "lane_reserve", "summary", "short_utility", "total_s", "median"), True, "s"),
        ("lane_reserve", "long_requests ttft_s (median)", ("suites", "lane_reserve", "summary", "long_requests", "ttft_s", "median"), True, "s"),
        ("lane_reserve", "long_requests total_s (median)", ("suites", "lane_reserve", "summary", "long_requests", "total_s", "median"), True, "s"),
        ("lane_reserve", "aggregate_decode_tok_s", ("suites", "lane_reserve", "summary", "aggregate_decode_tok_s"), False, " tok/s"),
    ]

    def _get_in(d, keys):
        curr = d
        for k in keys:
            if not isinstance(curr, dict):
                return None
            curr = curr.get(k)
        if isinstance(curr, dict) and "median" in curr:
            return curr.get("median")
        return curr

    current_suite = None
    table_lines = []
    table_lines.append("| Suite / Metric | Before (A) | After (B) | Delta (%) | Speedup |")
    table_lines.append("| :--- | :---: | :---: | :---: | :---: |")

    for suite, name, key_path, is_lat, unit in METRICS:
        has_a = suite in a.get("suites", {})
        has_b = suite in b.get("suites", {})
        if not (has_a or has_b):
            continue

        if suite != current_suite:
            current_suite = suite
            table_lines.append(f"| **{suite}** | | | | |")

        val_a = _get_in(a, key_path)
        val_b = _get_in(b, key_path)

        delta_str, speedup_str = _compute_delta_and_speedup(val_a, val_b, is_latency=is_lat)
        a_str = _format_val(val_a, unit)
        b_str = _format_val(val_b, unit)

        table_lines.append(f"| {name} | {a_str} | {b_str} | {delta_str} | {speedup_str} |")

    print("# Chronos Traffic Benchmark Comparison\n")
    print(f"- **Before (A)**: `{label_a}` ({a_path})")
    print(f"- **After (B)**: `{label_b}` ({b_path})")
    print(f"- **Model**: `{a.get('model')}` vs `{b.get('model')}`")
    print(f"- **Endpoint**: `{url_a}` vs `{url_b}`")
    print(f"- **Salt**: `{salt_a}` vs `{salt_b}`")
    if reps_a > 1 or reps_b > 1:
        print(f"- **Reps**: `{reps_a}` vs `{reps_b}`")
    print()
    print("\n".join(table_lines))
    print()

    sha_a = _get_in(a, ("suites", "cut_in", "summary", "long_request", "token_sha"))
    sha_b = _get_in(b, ("suites", "cut_in", "summary", "long_request", "token_sha"))

    print("### Token Bit-Exactness (cut_in)")
    if sha_a or sha_b:
        if salt_a == salt_b:
            if sha_a and sha_b and sha_a == sha_b:
                print(f"- **Status**: MATCH ({sha_a})")
                print(f"- **Note**: Long-prompt token SHA matches across both runs with identical salt `{salt_a}`.")
            else:
                print(f"- **Status**: MISMATCH")
                print(f"- **Before SHA**: `{sha_a}`")
                print(f"- **After SHA**:  `{sha_b}`")
                print(f"- **Note**: Salts match (`{salt_a}`), but token SHA differs!")
        else:
            print(f"- **Status**: SALTS DIFFER")
            print(f"- **Before SHA**: `{sha_a}` (salt: `{salt_a}`)")
            print(f"- **After SHA**:  `{sha_b}` (salt: `{salt_b}`)")
            print(f"- **Note**: Different salts produce different prompts; exact bit comparison requires identical salts.")
    else:
        print("- **Status**: No cut_in token SHA recorded in runs.")
    print()

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--url", default=None, help="Endpoint URL (default from API_URL, PORT, or scripts/local.sh)")
    ap.add_argument("--model", default=os.environ.get("MODEL", DEFAULT_MODEL))
    ap.add_argument("--suite", default="burst,cut_in,lane_reserve", help="Comma-separated suites (default: burst,cut_in,lane_reserve)")
    ap.add_argument("--label", default="", help="Label for this benchmark run")
    ap.add_argument("--out", help="Write structured JSON to this path (creating parent dirs as needed)")
    ap.add_argument("--salt", default=None, help="Prefix salt (default: random 8-char hex)")
    ap.add_argument("--seed", type=int, default=7, help="Random seed for generation (default: 7)")
    ap.add_argument("--reps", type=int, default=1, help="Number of repetitions per suite (default: 1)")
    ap.add_argument("--delay", type=float, default=6.0, help="Delay (seconds) for cut_in short probes (default: 6.0)")
    ap.add_argument("--burst-delay", type=float, default=0.5, help="Delay (seconds) before short utility in burst (default: 0.5)")
    ap.add_argument("--lane-delay", type=float, default=32.0, help="Delay (seconds) before short utility in lane_reserve (default: 32.0)")
    ap.add_argument("--lane-stagger", type=float, default=0.5, help="Stagger delay (seconds) between short utility requests in lane_reserve (default: 0.5)")
    ap.add_argument("--long-tokens", type=int, default=10000, help="Long prompt tokens for lane_reserve (default: 10000)")
    ap.add_argument("--cutin-long-tokens", type=int, default=64000, help="Long prompt tokens for cut_in (default: 64000)")
    ap.add_argument("--timeout", type=float, default=900.0, help="HTTP request timeout in seconds (default: 900.0)")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE.json", "AFTER.json"), help="Compare two benchmark runs and print Markdown table")

    args = ap.parse_args()

    if args.compare:
        return compare(args.compare[0], args.compare[1])

    if args.reps < 1:
        ap.error("--reps must be at least 1")

    url = (args.url or default_base()).rstrip("/")
    salt = args.salt if args.salt is not None else f"{random.SystemRandom().randrange(16**8):08x}"
    suites_to_run = [s.strip() for s in args.suite.split(",") if s.strip()]

    print(f"=== Chronos Traffic Benchmark ===")
    print(f"URL:    {url}")
    print(f"Model:  {args.model}")
    print(f"Salt:   {salt}")
    print(f"Reps:   {args.reps}")
    print(f"Suites: {', '.join(suites_to_run)}")
    if args.label:
        print(f"Label:  {args.label}")
    print()

    run_data = {
        "schema": "chronos_traffic_bench.v1",
        "label": args.label,
        "url": url,
        "model": args.model,
        "salt": salt,
        "seed": args.seed,
        "reps": args.reps,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "suites": {},
        "summary": {},
    }

    for s in suites_to_run:
        print(f"--- Running suite: {s} ---")
        rep_results = []
        for rep in range(args.reps):
            rep_salt_val = rep_salt(salt, rep, args.reps)
            if args.reps > 1:
                print(f"  [rep {rep + 1}/{args.reps} salt={rep_salt_val}]")
            if s == "burst":
                res = run_burst(
                    url=url,
                    model=args.model,
                    salt=rep_salt_val,
                    seed=args.seed,
                    timeout=args.timeout,
                    burst_delay=args.burst_delay,
                )
                rep_results.append(res)
            elif s == "cut_in":
                res = run_cut_in(
                    url=url,
                    model=args.model,
                    salt=rep_salt_val,
                    seed=args.seed,
                    delay=args.delay,
                    cutin_long_tokens=args.cutin_long_tokens,
                    timeout=args.timeout,
                )
                rep_results.append(res)
            elif s == "lane_reserve":
                res = run_lane_reserve(
                    url=url,
                    model=args.model,
                    salt=rep_salt_val,
                    seed=args.seed,
                    long_tokens=args.long_tokens,
                    lane_delay=args.lane_delay,
                    timeout=args.timeout,
                    lane_stagger=args.lane_stagger,
                )
                rep_results.append(res)
            else:
                print(f"Unknown suite '{s}', skipping.", file=sys.stderr)
                break

        if not rep_results:
            continue

        if s == "burst":
            all_agents = []
            all_shorts = []
            for r in rep_results:
                all_agents.extend(r.get("agent_turns", []))
                all_shorts.extend(r.get("short_utility", []))
            agg_decodes = [
                r.get("summary", {}).get("aggregate_decode_tok_s")
                for r in rep_results
                if r.get("summary", {}).get("aggregate_decode_tok_s") is not None
            ]
            suite_summary = {
                "agent_turns": summarize_group(all_agents),
                "short_utility": summarize_group(all_shorts),
                "aggregate_decode_tok_s": calc_median(agg_decodes) if agg_decodes else None,
            }
            suite_data = {
                "reps": rep_results,
                "seed_prefix": rep_results[0].get("seed_prefix"),
                "agent_turns": all_agents,
                "short_utility": all_shorts,
                "summary": suite_summary,
            }
            run_data["suites"]["burst"] = suite_data
            sm = suite_summary
            print(f"  agent_turns:   TTFT med={sm['agent_turns']['ttft_s']['median']}s, decode med={sm['agent_turns']['decode_tok_s']['median']} tok/s")
            print(f"  short_utility: TTFT med={sm['short_utility']['ttft_s']['median']}s, decode med={sm['short_utility']['decode_tok_s']['median']} tok/s")
            print(f"  aggregate decode: {sm['aggregate_decode_tok_s']} tok/s")

        elif s == "cut_in":
            all_longs = [r["long_request"] for r in rep_results if "long_request" in r]
            all_shorts = [r["short_utility"] for r in rep_results if "short_utility" in r]
            all_cacheds = [r["cached_turn"] for r in rep_results if "cached_turn" in r]
            suite_summary = {
                "long_request": summarize_cutin_probe(all_longs),
                "short_utility": summarize_cutin_probe(all_shorts),
                "cached_turn": summarize_cutin_probe(all_cacheds),
            }
            suite_data = {
                "reps": rep_results,
                "seed_prefix": rep_results[0].get("seed_prefix"),
                "long_request": rep_results[0].get("long_request"),
                "short_utility": rep_results[0].get("short_utility"),
                "cached_turn": rep_results[0].get("cached_turn"),
                "long_requests": all_longs,
                "short_utility_probes": all_shorts,
                "cached_turns": all_cacheds,
                "summary": suite_summary,
            }
            run_data["suites"]["cut_in"] = suite_data
            sm = suite_summary
            sha_disp = (sm['long_request']['token_sha'] or '')[:16]
            print(f"  long_request:  TTFT={sm['long_request']['ttft_s']}s, total={sm['long_request']['total_s']}s, token_sha={sha_disp}...")
            print(f"  short_utility: TTFT={sm['short_utility']['ttft_s']}s, total={sm['short_utility']['total_s']}s")
            print(f"  cached_turn:   TTFT={sm['cached_turn']['ttft_s']}s, total={sm['cached_turn']['total_s']}s (cached_tokens={sm['cached_turn']['cached_tokens']})")

        elif s == "lane_reserve":
            all_longs = []
            all_shorts = []
            for r in rep_results:
                all_longs.extend(r.get("long_requests", []))
                all_shorts.extend(r.get("short_utility", []))
            agg_decodes = [
                r.get("summary", {}).get("aggregate_decode_tok_s")
                for r in rep_results
                if r.get("summary", {}).get("aggregate_decode_tok_s") is not None
            ]
            suite_summary = {
                "long_requests": summarize_group(all_longs),
                "short_utility": summarize_group(all_shorts),
                "aggregate_decode_tok_s": calc_median(agg_decodes) if agg_decodes else None,
            }
            suite_data = {
                "reps": rep_results,
                "long_requests": all_longs,
                "short_utility": all_shorts,
                "summary": suite_summary,
            }
            run_data["suites"]["lane_reserve"] = suite_data
            sm = suite_summary
            print(f"  short_utility: TTFT med={sm['short_utility']['ttft_s']['median']}s, total med={sm['short_utility']['total_s']['median']}s")
            print(f"  long_requests: TTFT med={sm['long_requests']['ttft_s']['median']}s, total med={sm['long_requests']['total_s']['median']}s")
            print(f"  aggregate decode: {sm['aggregate_decode_tok_s']} tok/s")

        print()

    summary = {}
    for s_name, s_data in run_data["suites"].items():
        if "summary" in s_data:
            summary[s_name] = s_data["summary"]
    run_data["summary"] = summary

    if args.out:
        out_path = os.path.abspath(args.out)
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(run_data, f, indent=2)
        print(f"Results written to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
