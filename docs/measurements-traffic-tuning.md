# Chronos Traffic Tuning & Benchmark Measurements

This document records the traffic profile, tuning configuration, native engine modifications (Patch `0070` locally / `0069` upstream), and empirical benchmark measurements for DeepSeek-V4.1-Flash-1017 served on Chronos (2x DGX Spark nodes: `ai` and `ai2`).

---

## 1. Traffic Profile & Problem Analysis

Based on historical serving distribution (`.archify/explainer-chronos-context-20261116-210959/chronos-context.html`):
1. **Bimodal Request Distribution**:
   - **Agent Sessions (OpenCode / Copilot)**: ~69% of traffic. Large prompt lengths (median ~143k tokens, up to 742k tokens) with a 97% KV cache prefix hit rate.
   - **Interactive Utility / Tool Queries**: Short prompts (<1k uncached tokens) requiring low Time-to-First-Token (TTFT) and fast completion.
2. **Short Output Demands**:
   - P99 completion length across all requests is <8,192 tokens. Capping default `MAX_TOKENS` at 16,384 preserves margin while conserving KV pool reservations when clients omit `max_tokens`.
3. **Burst Concurrency**:
   - 6.9% of active periods see >4 concurrent streams.
4. **Baseline Bottlenecks**:
   - **Monolithic Prefill**: Large cold prompts (e.g. 64k+ tokens) hold the GPU for ~29–34s unless `TF_FILL_CHUNKS` / `TF_FILL_CUT_IN` are enabled.
   - **8-Lane Long Saturation**: Even with `PARALLEL=8` and `TF_FILL_CUT_IN=8192` enabled, once 8 long requests occupy all 8 execution slots, newly arriving short requests stall for the full duration of a long decode (~58s in our test workload). Under pool pressure (`error.ContextFull`), strict FIFO `full_wait` likewise blocks short requests behind the blocked head entry.

---

## 2. Tuning Configuration & Native Engine Changes

### 2.1 Configuration Changes (`scripts/local.sh`)
- `PARALLEL=8`: Increased parallel active slots from 4 to 8.
- `MAX_TOKENS=16384`: Lowered default output reservation from 32,768 to 16,384 to conserve pool room per stream.
- `TF_FILL_CHUNKS=1`, `TF_FILL_FIRST=1`, `TF_FILL_CUT_IN=8192`: Upstream chunked prefill and cut-in settings. Prefills run in 2,048-token chunks and requests computing <=8,192 uncached tokens cut in between chunks.
- `TF_DS_SHORT_RESERVE_LANES=2`: Reserves 2 of the 8 slots for qualifying short requests when total concurrency exceeds 6 long requests (off by default: `0`).
- `TF_DS_SHORT_MAX_TOKENS=40960`: Maximum uncached prompt + `max_tokens` budget to qualify as short (default `40960` = upstream `32768` default `MAX_TOKENS` + `8192` `TF_FILL_CUT_IN`).
- `TENSORFOLD_PREPARE_FLOOR_GIB=3`: Image preparation headroom floor.

### 2.2 Native Engine Implementation: Patch `0070-short-request-lanes.patch`
Applied to `zig/src/core/lane_host.zig`:
1. **Short Qualification & Admission Latching (`Job.is_short`)**:
   - A request qualifies as short when `TF_DS_SHORT_RESERVE_LANES > 0`, uncached tokens to compute fit `cut_in` (when set), and `uncached_compute + max_tokens <= TF_DS_SHORT_MAX_TOKENS` (`40960` default).
   - `Job.is_short` is evaluated and stored on `Job` once at admission (`longAdmittedCount()` counts admitted jobs with `!job.is_short`), so prefix caching during execution does not reclassify an active long job as short.
2. **Short Lane Reservation**:
   - When `longAdmittedCount() >= lanes -| short_reserve`, `admitOne` withholds non-short queued requests so up to `short_reserve` lanes remain available for short arrivals.
   - Woken restores in `h.ready` preserve their existing priority whenever `h.full_wait == null`.
3. **Pool-Blocked Queue Bypass**:
   - When a request cannot be admitted due to `error.ContextFull` (or a restore hits `waitRoom`), it is marked `job.pool_blocked = true` and `requeue` sets `full_wait`.
   - Instead of blocking all admissions behind `full_wait`, `admitOne` and `requeue` scan past `pool_blocked` / non-short entries and immediately admit eligible short requests whose admission succeeds.

---

## 3. Benchmark Suite (`chronos_traffic_bench.py`)

Implemented at [chronos_traffic_bench.py](file:///Users/sean/projects/Eternity/networks/chronos/benchmarks/chronos_traffic_bench.py) (deployed to `tools/chronos_traffic_bench.py` on `ai`) with `--reps N` multi-repetition support and natural essay/utility generation prompts:

1. **`burst` Suite (per rep: 1 warm seed prefix + 4 agent turns + 4 short utility)** *(uses random-word filler prompts matching `exact.py`)*:
   - Fires 4 cached agent turns (~32k prompt, 28,672 cached) concurrently, followed 0.5s later by 4 uncached short utility prompts (~540 tokens).
2. **`cut_in` Suite (per rep: 1 cold 64k long + 1 short utility + 1 cached agent turn)**:
   - Sends a cold 64k prompt (`max_tokens=16`), then launches 1 short utility request and 1 cached turn 6.0s later while prefill runs.
3. **`lane_reserve` Suite (per rep: 8 long essay requests + 4 staggered short utility requests)**:
   - Sends 8 concurrent cold 10k-token prompt / `max_tokens=1024` technical essay requests (>8,192 cut-in threshold so each prefills in ~4.5s and decodes ~85–100s total).
   - At `t = 32.0s` (with `0.5s` stagger across `t = 32.0s..33.5s`), sends 4 cold short utility requests (~540 tokens, `max_tokens=64`).

> **Note on NVMe Spill v2 Cache Persistence**: With `SPILL_GIB=64` and `SPILL_MIN_TOKENS=8192`, `>=8192`-token prompt states persist on `/spill` across container restarts. Timing comparisons must use a fresh cold `--salt` per server restart (verification shows `cached_tokens == 0` on all 16 cold long requests in both runs below). Bit-exact `token_sha` parity was verified separately with matching salt `chronos_ab1` (`6ff8657fd5e5...`).

---

## 4. Isolated A/B Comparison (`TF_DS_SHORT_RESERVE_LANES=0` vs `2`, 2 Reps)

- **Held Constant Across Both Runs**: Image `tensorfold-dsv41:exl3-tp2-50ce64cbbae4`, `PARALLEL=8`, `MAX_TOKENS=16384`, `TF_FILL_CHUNKS=1`, `TF_FILL_FIRST=1`, `TF_FILL_CUT_IN=8192`, `TF_DS_SHORT_MAX_TOKENS=40960`, `SPILL_GIB=64`, `KV_SHARD=1`, `POOL=2101248`, `enable_thinking=False`.
- **Variables**: Only `TF_DS_SHORT_RESERVE_LANES=0` (`chronos_ab1`) vs `TF_DS_SHORT_RESERVE_LANES=2` (`chronos_ab2`), 2 cold reps each (`n=16` long requests, `n=8` short utility requests in `lane_reserve`).
- **Raw Artifacts**:
  - Baseline (`reserve=0`): [ab_reserve0.json](file:///Users/sean/projects/Eternity/networks/chronos/benchmarks/runs/ab_reserve0.json)
  - Tuned (`reserve=2`): [ab_reserve2.json](file:///Users/sean/projects/Eternity/networks/chronos/benchmarks/runs/ab_reserve2.json)

### 4.1 `lane_reserve` Results (All 16 Cold Long Requests + 8 Short Utility Requests)

| Metric | `reserve=0` (`chronos_ab1`) | `reserve=2` (`chronos_ab2`) | Delta / Trade-off |
| :--- | :---: | :---: | :--- |
| **`short_utility` TTFT (median, n=8)** | 4.486s | **3.059s** | **-31.8%** (1.47x faster) |
| **`short_utility` TTFT (mean, n=8)** | 17.020s | **3.246s** | **-80.9%** (5.24x faster) |
| **`short_utility` TTFT (P95, n=8)** | **58.382s** | **5.529s** | **-90.5%** (10.56x faster) |
| **`short_utility` Total Time (median)** | 10.912s | **7.438s** | **-31.8%** (1.47x faster) |
| **`short_utility` Total Time (P95)** | 61.309s | **8.454s** | **-86.2%** (7.25x faster) |
| **`long_requests` TTFT (median, n=16)** | 24.953s | 24.874s | -0.3% (unchanged for first 6) |
| **`long_requests` TTFT (mean, n=16)** | 26.669s | **35.617s** | **+33.6% slower** |
| **`long_requests` TTFT (P95, n=16)** | 52.239s | **87.121s** | **+66.8% slower** (7th & 8th queued) |
| **`long_requests` Total Time (median)** | 96.543s | **87.792s** | **-9.1% faster** (6-way vs 8-way decode) |
| **`long_requests` Total Time (P95)** | 100.441s | **111.825s** | **+11.3% slower** (7th & 8th finish later) |
| **Suite aggregate decode throughput** | 88.34 tok/s | **79.24 tok/s** | **-10.3% lower** |

### 4.2 Mechanism Observed in Request Timelines

- **Under `reserve=0`**: Because `TF_FILL_CUT_IN=8192` is active, the first two short requests (`short[0..1]`, arriving at `t = 32.0s..32.5s`) took the open slots before long requests #7 and #8 finished queueing, and `short[2]` took a freed slot at `t = 37s`. Once all 8 slots filled with 1,024-token long generations (`end ~ 90.4s..100.8s`), `short[3]` (arriving at `t = 33.5s`) stalled in queue for **58.90s** (rep 0) and **57.42s** (rep 1) until the first long stream exited at `90–92s`.
- **Under `reserve=2`**: Long requests were capped at 6 concurrent slots. Long requests #7 and #8 waited in queue until two of the first 6 finished (`ft = 80.9s..87.6s`, driving the +66.8% P95 TTFT and -10.3% aggregate decode trade-off). All 4 short utility requests executed cleanly across the 2 reserved slots with max TTFT **5.75s** and finished by `t = 42s`.

### 4.3 `burst` and `cut_in` Stability (Isolated A/B)

| Suite / Metric | `reserve=0` | `reserve=2` | Delta |
| :--- | :---: | :---: | :---: |
| **`burst` agent TTFT (median)** | 7.746s | 7.686s | -0.8% |
| **`burst` short TTFT (median)** | 16.963s | 16.908s | -0.3% |
| **`burst` aggregate decode** | 67.39 tok/s | 67.47 tok/s | +0.1% |
| **`cut_in` 64k prefill TTFT** | 33.875s | 33.998s | +0.4% |
| **`cut_in` short TTFT** | 0.669s | 0.698s | +4.3% |
| **`cut_in` cached turn TTFT** | 2.358s | 2.417s | +2.5% |

Both suites are flat within run-to-run noise because `cut_in` only saturates 1 long prefill at a time (governed by upstream `TF_FILL_CUT_IN=8192`) and `burst` agent turns resume from prefix cache (`~3,400` uncached compute `<= cut_in`).

---

## 5. Historical Combined Tuning Comparison (`before.json` vs `after.json`, Single Run v1)

For cluster tuning history only: an earlier single-run comparison moved four variables together from stock defaults (`PARALLEL=4`, `MAX_TOKENS=32768`, no `TF_FILL_*`, no Patch `0070`) to (`PARALLEL=8`, `MAX_TOKENS=16384`, `TF_FILL_CHUNKS=1`, `TF_FILL_FIRST=1`, `TF_FILL_CUT_IN=8192`, `TF_DS_SHORT_RESERVE_LANES=2`, `chronos_bench1`, pre-update 34k/random-word v1 script):
- Raw artifacts: [before.json](file:///Users/sean/projects/Eternity/networks/chronos/benchmarks/runs/before.json) vs [after.json](file:///Users/sean/projects/Eternity/networks/chronos/benchmarks/runs/after.json).
- `cut_in` short TTFT dropped **23.87s $\rightarrow$ 0.55s** (attributable to upstream `TF_FILL_CUT_IN=8192`, with bit-exact 64k SHA `e5dad5dd84e5...`).
- `lane_reserve` (v1 34k short-completion workload) short TTFT dropped **128.58s $\rightarrow$ 0.86s** (combined effect of `TF_FILL_CUT_IN=8192` + `PARALLEL=8` + Patch `0070`, alongside longer long-request TTFT 71.23s $\rightarrow$ 77.60s and -35.7% `burst` per-stream agent decode at 8-way vs 4-way concurrency).

Use **Section 4** (not Section 5) when attributing performance specifically to Patch `0070`/`0069` (`TF_DS_SHORT_RESERVE_LANES=2`).
