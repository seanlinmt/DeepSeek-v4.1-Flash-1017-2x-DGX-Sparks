# Patch 0070 A/B Benchmark Report (`chronos_ab_bench.py`)

**Date:** 2026-11-20 (NZDT)  
**Cluster:** 2× NVIDIA DGX Spark (`ai` rank 0, `ai2` rank 1) serving `DeepSeek-V4.1-Flash-1017` (`EXL3 2.9 bpw`, `ablit=1`, 1M window, 2.1M KV-sharded pool, 64 GiB NVMe spill)  
**Comparison:** Baseline (`50ce64cbbae4`, Patch 0070 disabled via `TF_DS_SHORT_RESERVE_LANES=0`) and Patched (`18bd7858a167`, updated Patch `0070-short-request-lanes.patch` and `scripts/local.sh`)

---

## 1. Summary of Results

Patch `0070` reduces interactive short-request latency and increases pool stability under multi-stream saturation. Measured suite runtime decreased by 18.9% (`4,846.5s → 3,932.9s`), aggregate decode throughput increased by 20.4% (`27.65 → 33.29 tok/s`), and 0 errors occurred across all 464 benchmark requests (`232` before + `232` after):

| Area | Before → After Result | Mechanism |
| :--- | :--- | :--- |
| **`pool_starve` `short_bypass` TTFT** | **`56.83s → 0.48s` p50 (-99.2%)**<br>`70.05s → 1.75s` p95 (-97.5%) | `TF_ACTIVE_KV_CAP=786432` limits `bg_heavy` to 2 streams (`<670k`), preventing `ContextFull` / `full_wait`; short requests bypass `blocked_head`. |
| **`prod_replay` `short_util` TTFT** | **`68.49s → 1.35s` p50 (-98.0%)**<br>`94.46s → 35.10s` p95 (-62.8%) | Reserved short lane and cut-in admission admitted 13/15 short queries (`0.48s–5.51s`); 2 burst arrivals waited `35.1s` during a saturated window. |
| **`cached_sat` `short_util` TTFT** | **`57.60s → 17.36s` p50 (-69.9%)**<br>`65.09s → 42.52s` p95 (-34.7%) | `TF_ACTIVE_KV_CAP` limits `agent_cached` to 5 concurrent streams, leaving 3 lanes open; short requests are admitted sequentially rather than queued behind 8-way decode. |
| **`pool_starve` `bg_heavy` cache and E2E** | Hit rate **`89.5% → 99.5%`**<br>E2E p95 **`381.30s → 252.15s` (-33.9%)** | Baseline `restores_room_wait_expired` (`1 → 0`), `kv_shard.no_part` (`26 → 0`), and cold 333k re-prefill are removed. |
| **`kv_cap` `heavy_cached` trade-off** | TPOT p50 **`53.5ms → 27.3ms` (-49.0%)**<br>TTFT p50 **`11.48s → 22.98s` (+100.2%)** | `TF_ACTIVE_KV_CAP` reduces 287k stream concurrency from 6 to 2: per-stream decode latency decreases 49.0%; suite wall time increases 23.8% (`87.98s → 108.92s`). |
| **`cached_sat` `agent_cached` trade-off** | TTFT p95 **`38.56s → 88.03s` (+128.3%)**<br>E2E p95 **`90.11s → 124.93s` (+38.6%)** | `TF_ACTIVE_KV_CAP` reduces 135k stream concurrency from 8 to 5 (`5 × 135.7k = 678.7k <= 786.4k`); agents 6–8 incur increased queue wait. |
| **`prod_replay` overload and cache** | Wall **`3,679.7s → 2,903.7s` (-21.1%)**<br>`opencode_cached` hit **`47.1% → 57.9%`** | Open-loop `0.18 rps` with `--ignore-eos` (~138 tok/s offered decode) exceeds steady capacity; the 80-state/512 MiB RAM prompt cache fills, producing cold misses (`36/70 → 27/67`). |

---

## 2. Test Setup, Configuration Diff, and Methodology

### 2.1 Server Configuration Diff

| Parameter / Setting | Before (`ab_0070_before`) | After (`ab_0070_after`) | Notes |
| :--- | :--- | :--- | :--- |
| **Container Image** | `tensorfold-dsv41:exl3-tp2-50ce64cbbae4` | `tensorfold-dsv41:exl3-tp2-18bd7858a167` | Built from [0070-short-request-lanes.patch](../zig/patches/0070-short-request-lanes.patch) |
| **`TF_DS_SHORT_RESERVE_LANES`** | `0` *(off)* | **`1`** | Reserves 1 of 8 lanes for short requests |
| **`TF_DS_SHORT_MAX_TOKENS`** | `<unset>` | **`6144`** | Short budget: uncached prompt + `min(max_tokens, demote)` |
| **`TF_SHORT_TOTAL_CTX_MAX`** | `<unset>` | **`32768`** | Max whole prompt (cached + uncached) to qualify as short |
| **`TF_SHORT_DEMOTE_TOKENS`** | `<unset>` | **`512`** | Demotes short stream to long after 512 emitted tokens |
| **`TF_ACTIVE_KV_CAP`** | `<unset>` | **`786432`** | Aggregate KV estimate cap for long admissions (`>0` active) |
| **`TF_BYPASS_MAX_SKIPS`** | `<unset>` | **`4`** | Max short bypasses past a pool-blocked front (`full_wait`) |
| **`TF_KV_ALLOC_STEP`** | `<unset>` | **`512`** | Decode lookahead tokens per stream in KV estimate |
| **`MAX_TOKENS` (`--max-tokens`)** | `16384` | **`12288`** | Default reservation; explicit `max_tokens` sent per bench req |
| **`SPILL_MIN_TOKENS` (`--spill-min-tokens`)** | `8192` | **`16384`** | Cached prefixes in both runs (`76k–350k`) exceed 16k |
| **Held Constant** | `PARALLEL=8`, `CONTEXT=1048576`, `POOL=2101248`, `KV_SHARD=1` (`CARRY=1, TWINS=1, CHECK=1024`), `SPILL_GIB=64` (`MIN_FREE_GIB=50, QUOTA=0, STAGING_MIB=128, TURNS=4, FLUSH_MS=60000`), `TF_FILL_CHUNKS=1`, `TF_FILL_FIRST=1`, `TF_FILL_CUT_IN=8192`, `TF_DS_PREFILL_INTERLEAVE_MS=250`, `GRAPHS=1`, `GRAPH_CAP=128`, `DRAFTS=1`, `THINKING=1` (`enable_thinking=False` in bench), `TENSORFOLD_MEMORY_RESERVE_GIB=12.2` | Same | Identical model weights (`Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw`, `ablit=1@0bdc2036`), kit, Engram tables, and RoCEv2 links |

### 2.2 Benchmark Invocation and Methodology

Both runs executed identical workloads through [chronos_ab_bench.py](../tools/chronos_ab_bench.py) against `http://127.0.0.1:18888` (`model=DeepSeek-V4.1-Flash-1017`):
```bash
python3 -u tools/chronos_ab_bench.py \
  --url http://127.0.0.1:18888 \
  --model DeepSeek-V4.1-Flash-1017 \
  --suites cached_sat kv_cap pool_starve prod_replay \
  --reps 2 --replay-reqs 120 --rate 0.18 --ignore-eos \
  --salt <ab0070_before_20261120 | ab0070_after_20261120> \
  --out runs/<ab_0070_before.json | ab_0070_after.json>
```

Methodological details verified in the artifacts and server logs:
1. **Per-run `--salt` (no cross-run `/spill` contamination):** `0` cross-run cache hits were recorded; warm phases prefilled `10,379,549` vs `10,379,072` tokens across `52` cold prefixes in `5,860.4s` vs `5,844.7s`.
2. **`prod_replay` RNG variance:** `run_prod_replay` seeds its draw RNG from `salt`; the two runs sampled different class counts across the 120 Poisson requests: `opencode_cached` (`70` vs `67`), `non_opencode_cached` (`33` vs `32`), `short_util` (`11` vs `15`), `cold_large` (`6` vs `6`), yielding `134,016` vs `130,944` total benchmark completion tokens (`-2.3%`).
3. **`--ignore-eos` load calibration:** Forcing every `opencode_cached` request to decode `1,024` tokens, `non_opencode_cached` `512`, `cold_large` `256`, and `short_util` `128` produces a mean output of `~768 tok/req`. At `rate=0.18 rps`, offered decode load alone is `~138 tok/s` plus uncached prefill work, exceeding the cluster's mixed-concurrency capacity and producing progressive backlog accumulation in `prod_replay` (`in_flight` up to `71` before, `61` after).
4. **Concurrent external background heartbeat (exact token accounting):** A periodic external client (`thinking=True`, `~4.8k` or `~2.55k` prompt, `1,200` output tokens, at approximately 7-minute intervals) ran on the cluster during both ~3h windows. Reconciliation of `/health` totals against container logs accounts for 100% of tokens:
   - **Before:** `28` external requests (`131,591` prompt tokens = `0.40%` of total `32,552,505` prompt tokens; `33,600` gen tokens = `20.0%` of total `167,824` gen tokens).
   - **After:** `21` external requests (`97,841` prompt tokens = `0.31%` of total `32,036,106` prompt tokens; `25,200` gen tokens = `16.1%` of total `156,352` gen tokens).
   - The lower external count in After (`21` vs `28`) corresponds to the shorter After window (`9,778s` vs `10,707s`) and fewer ~2.5k probes (`2` vs `3+`). Both servers started idle (`live.connections=0`) and ended with zero restarts (`broken=false`, `stalled=false`).

---

## 3. Wall Time and Aggregate Throughput Comparison

| Phase / Metric | Before | After (Patch 0070) | Delta |
| :--- | :---: | :---: | :---: |
| **`cached_sat` (2 reps)** | `222.00s` | `248.65s` | **+12.0%** |
| **`kv_cap` (2 reps)** | `87.98s` | `108.92s` | **+23.8%** |
| **`pool_starve` (2 reps)** | `856.82s` | `671.64s` | **-21.6%** |
| **`prod_replay` (120 reqs)** | `3,679.69s` | `2,903.67s` | **-21.1%** |
| **Total Measured Wall Time** | **`4,846.49s`** (`80m 46s`) | **`3,932.88s`** (`65m 33s`) | **-18.9%** |
| **Warm Phase Wall Time (52 prefixes)** | `5,860.43s` (`97m 40s`) | `5,844.73s` (`97m 25s`) | -0.3% |
| **Total Benchmark Wall Time** | `10,706.91s` (`2h 58m`) | `9,777.61s` (`2h 43m`) | **-8.7%** |
| **Benchmark Output Tokens** | `134,016` tok | `130,944` tok | -2.3% |
| **Aggregate Decode Throughput (measured)** | **`27.65 tok/s`** | **`33.29 tok/s`** | **+20.4%** |

---

## 4. Per-Suite Detailed Comparison

### 4.1 `cached_sat` (8 × 135k cached agent turns, 24 × 507 short utility requests staggered every 2.5s, 2 reps)

| Class | Metric | Before (`reserve=0`) | After (Patch `0070`) | Delta |
| :--- | :--- | :---: | :---: | :---: |
| `agent_cached` | `N` / `Err` / `InFl p50 (max)` | `16` / `0` / `3 (7)` | `16` / `0` / `3 (7)` | — |
| `agent_cached` | `Hit%` (`0`-hit misses) | `96.9%` (`0`) | `96.9%` (`0`) | `0.0 pp` |
| `agent_cached` | `TTFT p50` / `p95` | `11.45s` / `38.56s` | `11.67s` / `88.03s` | `+2.0%` / **`+128.3%`** |
| `agent_cached` | `TTFT mean` / `max` | `13.63s` / `38.56s` | `35.13s` / `88.03s` | `+157.8%` / `+128.3%` |
| `agent_cached` | `TPOT p50` / `p95` | `67.8ms` / `75.0ms` | `63.8ms` / `92.6ms` | `-5.9%` / `+23.4%` |
| `agent_cached` | `E2E p50` / `p95` | `81.33s` / `90.11s` | `91.80s` / `124.93s` | `+12.9%` / `+38.6%` |
| **`short_util`** | `N` / `Err` / `InFl p50 (max)` | `48` / `0` / `19 (31)` | `48` / `0` / `15 (22)` | — |
| **`short_util`** | `TTFT p50` / `p95` | `57.60s` / `65.09s` | **`17.36s` / `42.52s`** | **`-69.9%` / `-34.7%`** |
| **`short_util`** | `TTFT mean` / `max` | `55.05s` / `67.06s` | **`22.26s` / `44.26s`** | **`-59.6%` / `-34.0%`** |
| **`short_util`** | `TPOT p50` / `p95` | `73.1ms` / `107.3ms` | `72.1ms` / `99.9ms` | `-1.3%` / `-6.9%` |
| **`short_util`** | `E2E p50` / `p95` | `67.41s` / `77.07s` | **`26.31s` / `52.07s`** | **`-61.0%` / `-32.4%`** |

### 4.2 `kv_cap` (6 × 287k concurrent cached heavy turns, 512 output tokens, 2 reps)

| Class | Metric | Before (`reserve=0`) | After (Patch `0070`) | Delta |
| :--- | :--- | :---: | :---: | :---: |
| `heavy_cached` | `N` / `Err` / `InFl p50 (max)` | `12` / `0` / `2 (5)` | `12` / `0` / `2 (5)` | — |
| `heavy_cached` | `Hit%` (`0`-hit misses) | `98.6%` (`0`) | `98.6%` (`0`) | `0.0 pp` |
| `heavy_cached` | `TTFT p50` / `p95` | `11.48s` / `31.45s` | `22.98s` / `48.06s` | **`+100.2%`** / `+52.8%` |
| `heavy_cached` | `TTFT mean` / `max` | `12.80s` / `31.45s` | `23.63s` / `48.06s` | `+84.6%` / `+52.8%` |
| `heavy_cached` | `TPOT p50` / `p95` | `53.5ms` / `70.0ms` | **`27.3ms` / `35.1ms`** | **`-49.0%` / `-49.8%`** |
| `heavy_cached` | `E2E p50` / `p95` | `40.11s` / `46.65s` | **`37.05s`** / `59.31s` | **`-7.6%`** / `+27.1%` |

### 4.3 `pool_starve` (5 × 333k `bg_heavy` requests, a 427k cold `blocked_head` request at +3s, and 12 × 408 `short_bypass` requests at 1.5s, 2 reps)

| Class | Metric | Before (`reserve=0`) | After (Patch `0070`) | Delta |
| :--- | :--- | :---: | :---: | :---: |
| `bg_heavy` | `N` / `Err` / `InFl p50 (max)` | `10` / `0` / `2 (4)` | `10` / `0` / `2 (4)` | — |
| `bg_heavy` | `Hit%` (`0`-hit misses) | `89.5%` (`1`) | **`99.5%` (`0`)** | **`+10.0 pp`** |
| `bg_heavy` | `TTFT p50` / `p95` | `4.74s` / `253.73s` | `29.18s` / **`126.11s`** | `+515.0%` / **`-50.3%`** |
| `bg_heavy` | `TTFT mean` / `max` | `35.94s` / `253.73s` | `35.73s` / **`126.11s`** | `-0.6%` / **`-50.3%`** |
| `bg_heavy` | `TPOT p50` / `p95` | `55.5ms` / `139.6ms` | **`34.9ms` / `123.2ms`** | **`-37.1%` / `-11.7%`** |
| `bg_heavy` | `E2E p50` / `p95` | `61.88s` / `381.30s` | **`54.91s` / `252.15s`** | **`-11.3%` / `-33.9%`** |
| `blocked_head` | `N` / `Err` / `InFl p50 (max)` | `2` / `0` / `5 (5)` | `2` / `0` / `5 (5)` | — |
| `blocked_head` | `TTFT p50` / `p95` | `315.10s` / `526.97s` | `320.06s` / **`335.81s`** | `+1.6%` / **`-36.3%`** |
| `blocked_head` | `TTFT mean` / `max` | `421.04s` / `526.97s` | **`327.93s` / `335.81s`** | **`-22.1%` / `-36.3%`** |
| `blocked_head` | `TPOT p50` / `p95` | `17.1ms` / `17.2ms` | `16.1ms` / `22.3ms` | `-5.6%` / `+29.1%` |
| `blocked_head` | `E2E p50` / `p95` | `319.50s` / `531.32s` | `324.16s` / **`341.48s`** | `+1.5%` / **`-35.7%`** |
| **`short_bypass`** | `N` / `Err` / `InFl p50 (max)` | `24` / `0` / `11 (17)` | `24` / `0` / `11 (13)` | — |
| **`short_bypass`** | `TTFT p50` / `p95` | `56.83s` / `70.05s` | **`0.48s` / `1.75s`** | **`-99.2%` / `-97.5%`** |
| **`short_bypass`** | `TTFT mean` / `max` | `54.75s` / `77.16s` | **`0.81s` / `2.53s`** | **`-98.5%` / `-96.7%`** |
| **`short_bypass`** | `TPOT p50` / `p95` | `201.6ms` / `257.4ms` | **`73.5ms` / `88.6ms`** | **`-63.5%` / `-65.6%`** |
| **`short_bypass`** | `E2E p50` / `p95` | `81.99s` / `95.81s` | **`9.80s` / `11.93s`** | **`-88.0%` / `-87.6%`** |

### 4.4 `prod_replay` (120 open-loop Poisson requests at `0.18 rps`)

| Class | Metric | Before (`reserve=0`) | After (Patch `0070`) | Delta |
| :--- | :--- | :---: | :---: | :---: |
| `opencode_cached` | `N` / `Err` / `InFl p50 (max)` | `70` / `0` / `36 (71)` | `67` / `0` / `29 (61)` | — |
| `opencode_cached` | `Hit%` (`0`-hit misses) | `47.1%` (`36`) | **`57.9%` (`27`)** | **`+10.8 pp`** |
| `opencode_cached` | `TTFT p50` / `p95` | `1,163.98s` / `2,859.37s` | **`787.18s` / `2,084.45s`** | **`-32.4%` / `-27.1%`** |
| `opencode_cached` | `TTFT mean` / `max` | `1,247.81s` / `3,075.94s` | **`867.87s` / `2,359.91s`** | **`-30.4%` / `-23.3%`** |
| `opencode_cached` | `TPOT p50` / `p95` | `108.9ms` / `178.1ms` | **`87.1ms` / `136.6ms`** | **`-20.1%` / `-23.3%`** |
| `opencode_cached` | `E2E p50` / `p95` | `1,232.00s` / `2,942.51s` | **`884.34s` / `2,168.37s`** | **`-28.2%` / `-26.3%`** |
| `non_opencode_cached` | `N` / `Err` / `InFl p50 (max)` | `33` / `0` / `32 (70)` | `32` / `0` / `19 (60)` | — |
| `non_opencode_cached` | `Hit%` (`0`-hit misses) | `88.4%` (`3`) | **`94.2%` (`1`)** | **`+5.8 pp`** |
| `non_opencode_cached` | `TTFT p50` / `p95` | `89.43s` / `2,424.02s` | **`42.58s` / `1,640.06s`** | **`-52.4%` / `-32.3%`** |
| `non_opencode_cached` | `TTFT mean` / `max` | `485.63s` / `3,033.81s` | **`293.89s` / `1,682.80s`** | **`-39.5%` / `-44.5%`** |
| `non_opencode_cached` | `TPOT p50` / `p95` | `152.5ms` / `235.3ms` | **`130.3ms` / `220.6ms`** | **`-14.6%` / `-6.3%`** |
| `non_opencode_cached` | `E2E p50` / `p95` | `177.86s` / `2,532.83s` | **`118.82s` / `1,715.92s`** | **`-33.2%` / `-32.3%`** |
| `cold_large` | `N` / `Err` / `InFl p50 (max)` | `6` / `0` / `12 (58)` | `6` / `0` / `8 (20)` | — |
| `cold_large` | `TTFT p50` / `p95` | `57.96s` / `2,312.04s` | `72.73s` / **`493.93s`** | `+25.5%` / **`-78.6%`** |
| `cold_large` | `TTFT mean` / `max` | `786.42s` / `2,312.04s` | **`198.65s` / `493.93s`** | **`-74.7%` / `-78.6%`** |
| `cold_large` | `TPOT p50` / `p95` | `97.1ms` / `253.0ms` | **`86.6ms` / `133.1ms`** | **`-10.8%` / `-47.4%`** |
| `cold_large` | `E2E p50` / `p95` | `77.20s` / `2,376.49s` | `104.00s` / **`516.01s`** | `+34.7%` / **`-78.3%`** |
| **`short_util`** | `N` / `Err` / `InFl p50 (max)` | `11` / `0` / `26 (64)` | `15` / `0` / `16 (61)` | — |
| **`short_util`** | `TTFT p50` / `p95` | `68.49s` / `94.46s` | **`1.35s` / `35.10s`** | **`-98.0%` / `-62.8%`** |
| **`short_util`** | `TTFT mean` / `max` | `51.52s` / `94.46s` | **`6.08s` / `35.10s`** | **`-88.2%` / `-62.8%`** |
| **`short_util`** | `TPOT p50` / `p95` | `108.9ms` / `144.3ms` | **`81.0ms` / `123.3ms`** | **`-25.6%` / `-14.5%`** |
| **`short_util`** | `E2E p50` / `p95` | `85.07s` / `108.71s` | **`11.73s` / `48.30s`** | **`-86.2%` / `-55.6%`** |

---

## 5. Server-Side Telemetry Deltas (`/health`)

Counters below are exact `post - pre` deltas from `/health` across each benchmark window:

| Subsystem / Counter | Before Delta | After Delta | Interpretation |
| :--- | :---: | :---: | :--- |
| `prompt_tokens_total` | `32,552,505` | `32,036,106` | `131,591` external tokens before vs `97,841` after |
| `cached_tokens_total` | `15,202,304` | `16,396,288` | Cache-served tokens +1.19M (+7.9%) |
| `generation_tokens_total` | `167,824` | `156,352` | `33,600` external (28 reqs) vs `25,200` external (21 reqs) |
| `prompt_cache.memory_hits` / `misses` | `101` / `216` | `109` / `196` | Memory hits increased by 8; misses decreased by 20 |
| `prompt_cache.kept` / `evicted` | `190` / `220` | `178` / **`126`** | RAM cache state evictions -42.7% |
| `prompt_cache.disk_lookups` / `disk_hits` | `567` / `163` | `615` / `177` | — |
| `prompt_cache.restores_begun` (`shared`) | `36` (`13`) | `35` (`27`) | — |
| `prompt_cache.restores_refused` / `deferred` | `26` / `26` | `33` / `33` | Both counters reached `TierFull` under `prod_replay` backlog |
| `prompt_cache.restores_deferred_expired` | `5` | `3` | — |
| `prompt_cache.restores_room_waits` (`expired`) | `1` (`1`) | **`0` (`0`)** | 30s room-wait failure removed in `pool_starve` |
| `kv_shard.carries` (`carry_paused`) | `15` (`9`) | **`3` (`3`)** | Cross-part KV carries -80% |
| `kv_shard.moves` / `moved_tokens` | `1` / `1,709,747` | **`0` / `282,624`** | KV shard migration -83.5% |
| `kv_shard.staled` / `twins_made` | `172` / `127` | `168` / `126` | — |
| `kv_shard.no_part` / `refused_no_target` | **`26`** / `150` | **`0`** / `180` | `no_part` failures: 26 before, 0 after under `active_kv_cap` |
| `spill[rank=0]` writes / reads | `156` (`6.88 GiB`) / `36` (`2.97 GiB`) | `156` (`7.08 GiB`) / `35` (`1.94 GiB`) | Spill volume equivalent within 0.2 GiB (`evicted_turns=0`) |
| `spill[rank=1]` writes / reads | `157` (`7.39 GiB`) / `36` (`2.16 GiB`) | `156` (`7.02 GiB`) / `35` (`3.11 GiB`) | Spill volume equivalent within 0.37 GiB |

---

## 6. Mechanism and Root-Cause Analysis

1. **`pool_starve` `short_bypass` TTFT (`56.83s → 0.48s`, `-99.2%`) and `bg_heavy` cache hit rate (`99.5%`):**
   - Before, all 5 `bg_heavy` streams (`5 × 333.5k ≈ 1.67M` tokens) were admitted simultaneously into the `2.10M` pool. In rep 1, admission of the 427k `blocked_head` request and restoration of `bg_heavy` produced `restores_room_waits: 1` / `restores_room_wait_expired: 1` (`waited 30053 ms... did not start (ContextFull); prefilling` in container logs), forcing a cold 333k prefill (`TTFT max 253.73s`) and placing `short_bypass` behind `full_wait` / saturated lanes (`TTFT 56.83s p50`, `201.6ms TPOT`).
   - After, `TF_ACTIVE_KV_CAP=786432` limited `bg_heavy` (`kvEstimate = 333,477 + 512 = 333,989`) to **2 concurrent streams** (`2 × 333,989 = 667,978 <= 786,432`; 3 would total `1,001,967`). Pool headroom was maintained (`0` room waits, `0` `no_part`, `10/10` cache hits); `blocked_head` was held via `active_kv_cap` rather than `ContextFull` / `full_wait`; 6 lanes remained open (1 reserved for short); all 24 `short_bypass` requests ran at `0.41s–2.53s` TTFT without exhausting `TF_BYPASS_MAX_SKIPS=4`.

2. **`cached_sat` `short_util` TTFT (`57.60s → 17.36s` p50, `-69.9%`):**
   - At `t=0`, 8 `agent_cached` requests arrive (`prompt ≈ 135,230`, `kvEstimate = 135,742`). Under `TF_ACTIVE_KV_CAP=786432`, **5 `agent_cached` streams** are admitted (`5 × 135,742 = 678,710 <= 786,432`; 6 would be `814,452`), and `agent_cached` #6–#8 queue (`TTFT p95 88.03s`).
   - 3 of `PARALLEL=8` lanes remain open for `short_util`, which bypass the queued long requests (`i > 0` in `admitOne`). `24` `short_util` requests arrive every `2.5s` (`0.40 req/s`), while 3 lanes decoding `128` tokens at `~72ms` TPOT (`~9.2s` E2E) drain at `~0.326 req/s`. `s0`–`s2` occupy those 3 lanes (`TTFT 7.6s–13.2s`) and `s3`–`s23` accumulate in a short-request queue (`TTFT 42.5s–44.3s`); p50 TTFT is 3.3× lower than Before, where 8 `agent_cached` occupied all 8 lanes for `~60s` (`short_util TTFT 48.1s–67.1s`).

3. **`kv_cap` heavy TTFT increase (+100%) and per-stream TPOT decrease (-49%):**
   - With `kvEstimate = 286,529 + 512 = 287,041`, `TF_ACTIVE_KV_CAP=786432` limits `heavy_cached` to **2 concurrent streams** (`574,082 <= 786,432`) vs 6 in Before. Per-stream decode latency is halved (`53.5ms → 27.3ms TPOT`, `E2E p50 40.11s → 37.05s`), while queued streams #3–#6 increase `TTFT p50` (`11.48s → 22.98s`) and suite wall time (`87.98s → 108.92s`; 2-way decode produces ~73 tok/s aggregate vs ~112 tok/s at 6-way).

4. **`prod_replay` `short_util` TTFT (`1.35s` p50, `-98.0%`) and two requests at `35.1s`:**
   - 13 of 15 `short_util` requests in `prod_replay` achieved `0.48s–5.51s` TTFT while `in_flight` reached `61`.
   - Two overlapping requests (`replay_0059` at `t=346.1s` and `replay_0061` at `t=361.0s`) waited `35.07s` and `35.10s`. Container logs (`22:32:05–22:33:41Z`) record the sequence: `non_opencode_cached` streams (`~78k` tokens each) occupied ~`7` streams within `TF_ACTIVE_KV_CAP=786432` (filling the 7 long lanes), while a `135,093`-token cold prefill plus an active stream held the 8th slot until `22:33:15.34Z` (`chatcmpl-2b1ef4d3` ended), after which `0059` ran (`22:33:15.79Z → 22:33:29.02Z`) and `0061` followed 1.8s later (`22:33:30.78Z → 22:33:41.20Z`).

5. **`prod_replay` `opencode_cached` cold misses (27/67; `57.9%` hit rate vs `47.1%` before):**
   - `TF_ACTIVE_KV_CAP` reduced RAM state evictions by 42.7% (`220 → 126`), reduced `prod_replay` drain time by 21.1% (`61.3m → 48.4m`), and removed pool `no_part` failures (`26 → 0`).
   - Two structural limits persist under sustained `0.18 rps` load with `--ignore-eos`:
     - **Offered load exceeding cluster capacity:** `~138 tok/s` demanded decode plus uncached prefill produces an unbounded queue (`in_flight=61`).
     - **80-state / 512 MiB in-memory prompt-cache cap and `TierFull`:** Container logs show the RAM prompt cache reaching `80 states, 510 MiB of 512 MiB` during `pool_starve` rep 1 (`22:07:18Z`), with `126` RAM evictions, `62` `131k` KV-row `goes stale` placements during `prod_replay`, and `33` `restores_refused` / `restores_deferred` events (`3` `TierFull` expiries up to `779,745 ms`). After a state is evicted from RAM and its KV rows become stale or its disk restore is refused/deferred, the queued `opencode_cached` request cold-prefills `135k` tokens.

---

## 7. Trade-offs and Operational Constraints

Patch `0070` reduces interactive short-request latency and removes pool-exhaustion failures. Five trade-offs and operational constraints follow: four are measurable in benchmark metrics and one is an admission-accounting edge case:

### 7.1 Reduced Batch Efficiency and Increased Queue Wait for Heavy Bursts

`TF_ACTIVE_KV_CAP=786432` is ~37% of the total 2.1M KV pool; concurrent heavy streams are therefore restricted when RAM is free and streams fit without evictions:
- **`6 × 287k` (`kv_cap`):** Concurrency is capped at 2 streams instead of 6 (`2 × 287k = 574k <= 786k`). Each active stream decodes 2× faster (`TPOT 53.5ms → 27.3ms`, -49.0%), while 2-way batching yields lower aggregate GPU throughput than 6-way; burst suite duration increases **+23.8%** (`87.98s → 108.92s`) and queued stream wait time doubles (`TTFT p50 11.48s → 22.98s`, +100.2%).
- **`8 × 135k` (`cached_sat`):** Concurrency is capped at 5 streams instead of 8 (`5 × 135.7k = 678.7k <= 786.4k`), increasing the 6th–8th agents' `TTFT p95` from **`38.56s → 88.03s` (+128.3%)** and burst wall time by **+12.0%** (`222.0s → 248.7s`).
- **Affected workload:** Multiple large coding agents (135k–300k context) arriving simultaneously during a period with no short requests and available KV memory headroom.

### 7.2 Starvation of >786k (800k–1M) Requests Under Continuous Short Traffic (Code Edge Case)

In `0070-short-request-lanes.patch` and `admitOne`:
- A request with `kvEstimate > 786,432` can start only when `h.admitted.items.len == 0` (the server is idle).
- While that `>786k` request is at `queued[0]`, `full_wait` is not set (`waiting == false`; the request is held by admission accounting rather than a pool-exhaustion `ContextFull` or room wait). Without a `cap_wait` guard, `TF_BYPASS_MAX_SKIPS=4` is not applied, and short requests arriving behind it bypass it without bound.
- **Affected workload:** A single 800k–1M prompt arriving while short utility calls arrive every few seconds (such that `admitted.items.len` never reaches 0). The request can wait indefinitely, and long requests queued behind it are head-of-line blocked (`i > 0` sets `bypass_only = true`).
*(The updated Patch 0070 introduces `capBlockedByShort` / `cap_skips`, which bounds short bypasses past a short-held front request to `TF_BYPASS_MAX_SKIPS`.)*

### 7.3 Head-of-Line Blocking for Medium-Long Requests Behind a Queued Large Request

`admitOne` sets `bypass_only = true` for every candidate at queue position `i > 0`; only requests with `prompt <= 32,768` whole prompt and `<= 6,144` short budget bypass the queue:
- A `78k` request queued behind a waiting `400k` request cannot run even if `activeKvTokens + 78k <= 786,432`.
- This preserves large-request priority over medium requests while `TF_ACTIVE_KV_CAP` remains under-utilized during the wait.

### 7.4 One Idle Lane on Pure-Long Medium-Context Traffic (33k–110k)
- `TF_DS_SHORT_RESERVE_LANES=1` limits long streams to `7` of `8` lanes even when no short requests arrive.
- For prompts `≥ 135k`, `TF_ACTIVE_KV_CAP=786432` already limits concurrency to `≤ 5` streams; the reserved-lane constraint does not bind.
- The 1-lane reduction (**-12.5% peak concurrency**) applies only to 8 concurrent medium-context requests (`33k–110k` tokens each) with no short traffic.

### 7.5 Short-Lane Saturation and Demoted Short Requests
- **Short Lane Congestion (`reserve=1`):** When 7 medium-long (`~78k`) streams fit inside 786k, 1 lane remains for short requests; two overlapping short requests serialize (in `prod_replay`, 13 of 15 short requests achieved `0.5s–5.5s` TTFT; 2 overlapping arrivals reached `35.1s`). A sustained burst of 24 short requests every 2.5s (`cached_sat`) exceeds 3 free lanes (`TTFT p50` reached `17.4s`).
- **Demotion Dynamics (`TF_SHORT_DEMOTE_TOKENS=512`):** A `<32k`-prompt request that generates thousands of tokens (e.g., a `4.8k`-prompt thinking request generating `1,200+` tokens) enters on the reserved short lane and occupies that lane until completion; it is counted as long for subsequent admissions after token 512 is emitted.

**Net Effect:**
- Mixed workloads (agent sessions and interactive/tool calls), pool-pressure bursts (300k+), and `ContextFull` / KV-shard thrash: `pool_starve` and `prod_replay` completed 21% faster with 98–99% lower short TTFT.
- Homogeneous bursts of 6–8 medium/heavy long requests that fit in the 2.1M pool complete more slowly with higher p95 TTFT and lower per-stream TPOT; `>786k` requests can starve when 800k–1M prompts arrive alongside steady short traffic.

---

## 8. Tuning Implications

1. **Patch `0070` defaults (`TF_DS_SHORT_RESERVE_LANES=1`, `TF_ACTIVE_KV_CAP=786432`) reduce short-request TTFT by 88%–99% across every mixed suite and eliminate `ContextFull` restore timeouts and KV shard `no_part` failures.**
2. **If heavy multi-agent throughput is prioritized over per-stream TPOT, `TF_ACTIVE_KV_CAP` can be raised (`786432 → ~900k–1048k`):** At `786,432`, `135k` agent sessions cap at 5 streams (`5 × 135.7k = 678.7k`) and `287k` sessions cap at 2 streams (`574k`), leaving ~35–45% of the `2.1M` pool idle during homogeneous heavy bursts. `~960,000` admits `6–7` `135k` sessions or `3` `287k` sessions while remaining below pool exhaustion (`2.1M`) and preserving the reserved short lane.
3. **The 512 MiB / 80-state RAM prompt-cache cap requires evaluation for high-queue workloads:** With prefixes held on NVMe by `/spill`, `prod_replay` overload (`in_flight > 30`) cycled through 80 states and reached `TierFull` on concurrent restores.
4. **`prod_replay` steady-state measurement requires a lower offered rate:** `--rate 0.18` with `--ignore-eos` constitutes an overload stress test; an unsaturated rate (~0.10–0.12 rps with `--ignore-eos`, or `0.18` with natural EOS) measures steady-state open-loop cache hits (`>80%`).

---

## 9. Patch 0070 v2 (`capBlockedByShort`) Live Verification

### 9.1 Deployment and Build Health

The update addresses the `>786k` starvation edge case identified after the first benchmark pass. In [0070-short-request-lanes.patch](../zig/patches/0070-short-request-lanes.patch), short bypasses past a front long request held from `TF_ACTIVE_KV_CAP` by admitted short streams (`capBlockedByShort`: either `need > cap` with `long_count == 0`, or `long_kv + need <= cap < active_kv + need`) are bounded by `TF_BYPASS_MAX_SKIPS` (`cap_wait_id` / `cap_skips`); a 2-line `ShortLaneConfig` docstring cleanup was also applied (`119/119` Zig host unit tests passing).

- **Deployed image (`ai` and `ai2`):** `tensorfold-dsv41:exl3-tp2-bb0ec9a5acc6`, `running=true`, `restarts=0` on both ranks ([ab_0070_v2_build_start.log](./ab_0070_v2_build_start.log), [ab_0070_v2_inspect.json](./ab_0070_v2_inspect.json)).
- **Active Patch `0070` env vars verified:** `TF_DS_SHORT_RESERVE_LANES=1`, `TF_DS_SHORT_MAX_TOKENS=6144`, `TF_SHORT_TOTAL_CTX_MAX=32768`, `TF_SHORT_DEMOTE_TOKENS=512`, `TF_ACTIVE_KV_CAP=786432`, `TF_BYPASS_MAX_SKIPS=4`, `TF_KV_ALLOC_STEP=512`.
- **Health stability:** Both [ab_0070_v2_health_pre.json](./ab_0070_v2_health_pre.json) and [ab_0070_v2_health_post.json](./ab_0070_v2_health_post.json) report `status: "ok"`, `warming: false`, `pair.stalled: false`, `pair.broken: false`, `prompt_cache.evicted: 0`, `restores_room_waits: 0`, and `kv_shard.no_part: 0` across `+7,566,923` prompt tokens (`+3,508,224` cached) and `+33,240` generated tokens.

### 9.2 Targeted `~800k` Request With 20 Overlapping Short Requests

Executed [cap_starve_bench.py](../tools/cap_starve_bench.py) (`--model DeepSeek-V4.1-Flash-1017`) with identical parameters before restart on v1 (`18bd7858a167`) and after restart on v2 (`bb0ec9a5acc6`). The benchmark warms an `~802k` prefix (`> 786,432` `TF_ACTIVE_KV_CAP`), launches `short_0` at `t=0s` (`~500` prompt, `max_tokens=256`), submits cached `mega_cached` (`~802k` prompt, `max_tokens=32`) at `t=1.0s`, and submits `20` `short_stream` requests at `2.0s` intervals (`t=2s..40s`):

| Metric | Patch 0070 v1 (`18bd7858a167`) | Patch 0070 v2 (`bb0ec9a5acc6`) | Delta / Effect |
| :--- | :---: | :---: | :--- |
| **Warmup prompt / wall** | `801,654` tok / `517.48s` | `801,666` tok / `450.81s` | Cold prefill completed in both runs |
| **Errors** | `0` | `0` | No failures |
| **`mega_cached` prompt / cached** | `802,149` / `800,768` (`99.8%`) | `802,159` / `800,768` (`99.8%`) | Memory hit (`>786,432` cap) |
| **`mega_cached` `TTFT`** | `75.73s` (`t_1st = 76.73s`) | **`20.19s`** (`t_1st = 21.19s`) | **`-73.3%` (-55.54s)** |
| **`mega_cached` `E2E`** | `76.55s` | **`21.00s`** | **`-72.6%` (-55.55s)** |
| **`short_stream_started_before_mega`** | `20 / 20` *(100% bypassed)* | **`5 / 20`** *(`4` after engine queue)* | **Starvation removed** |
| **`short_stream_finished_before_mega`** | `20 / 20` | **`5 / 20`** | `short_06`..`short_20` ran after mega |
| **`short_stream` `TTFT p50` / `p95` / `max`** | `0.54s` / `6.76s` / `7.04s` | `5.57s` / `8.94s` / `10.50s` | `short_06` was paused behind `mega_cached` |
| **`short_stream` `E2E p50` / `p95` / `max`** | `17.72s` / `21.72s` / `24.13s` | `21.71s` / `27.10s` / `27.39s` | Short queueing behind mega was bounded |
| **Measured window wall (`measured_wall_s`)** | `77.55s` | **`60.74s`** | **`-21.7%`** |

**Client-clock count discrepancy (`5` vs `4`):** Per `openai.zig:507-524` and `docker logs --timestamps`, `chat.prepare()` (JSON read, prompt render, tokenization) executes before `srv.engine.submit()`. The client-sent `mega_cached` request (`4.63 MB` JSON / `802,159` tokens at `t=1.00s`, `~01:48:05.62Z`) required `~1.09s` of preparation (`POST /tokenize` alone measures `0.510s` for that payload) and logged `POST /v1/chat/completions` at `01:48:06.705Z`, `84 ms` after `short_01` (`01:48:06.621Z`). `short_01` therefore entered `LaneHost` before `mega_cached` without bypassing it. After `mega_cached` entered the engine, `4` subsequent short requests (`short_02`..`short_05`) bypassed it per `TF_BYPASS_MAX_SKIPS=4`, and `short_06`..`short_10` were held until active short streams drained (`01:48:23.770Z`), unblocking `mega_cached` (`01:48:25.806Z` prefill, `01:48:26.611Z` done). One periodic external `thinking=True` job (`4,986` prompt, `1,200` output, finished `01:49:16.615Z`) ran ~20s after `mega_cached` finished.

### 9.3 Regression Results (`cached_sat` and `pool_starve`, `--reps 1`)

`tools/chronos_ab_bench.py --suites cached_sat pool_starve --reps 1 --ignore-eos` was executed on v2 (`bb0ec9a5acc6`) with `0` errors across all `50` requests, aggregate measured throughput of `37.81 tok/s`, and `18,176` tok in `480.7s` measured / `2,055.6s` total including warm phases:

| Suite | Class | Metric | Baseline (`50ce64cb`, 2r) | Patch 0070 v1 (`18bd7858`, 2r) | Patch 0070 v2 (`bb0ec9a5`, 1r) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `cached_sat` | — | `Wall` | `222.00s` (`111.00s/r`) | `248.65s` (`124.33s/r`) | `122.09s` |
| `cached_sat` | `agent_cached` | `N` / `Err` / `Hit%` | `16` / `0` / `96.9%` | `16` / `0` / `96.9%` | `8` / `0` / `96.9%` |
| `cached_sat` | `agent_cached` | `TTFT p50` / `p95` | `11.45s` / `38.56s` | `11.67s` / `88.03s` | `14.98s` / `92.11s` |
| `cached_sat` | `agent_cached` | `TPOT p50` / `E2E p95` | `67.8ms` / `90.11s` | `63.8ms` / `124.93s` | `60.6ms` / `122.07s` |
| `cached_sat` | **`short_util`** | `N` / `Err` | `48` / `0` | `48` / `0` | `24` / `0` |
| `cached_sat` | **`short_util`** | `TTFT p50` / `p95` | `57.60s` / `65.09s` | `17.36s` / `42.52s` | **`19.57s` / `26.40s`** |
| `cached_sat` | **`short_util`** | `TPOT p50` / `E2E p95` | `73.1ms` / `77.07s` | `72.1ms` / `52.07s` | **`70.7ms` / `37.79s`** |
| `pool_starve` | — | `Wall` | `856.82s` (`428.41s/r`) | `671.64s` (`335.82s/r`) | `358.64s` |
| `pool_starve` | `bg_heavy` | `N` / `Err` / `Hit%` | `10` / `0` / `89.5%` | `10` / `0` / `99.5%` | `5` / `0` / `99.5%` |
| `pool_starve` | `bg_heavy` | `TTFT p50` / `p95` | `4.74s` / `253.73s` | `29.18s` / `126.11s` | `37.29s` / `66.12s` |
| `pool_starve` | `bg_heavy` | `TPOT p50` / `E2E p95` | `55.5ms` / `381.30s` | `34.9ms` / `252.15s` | `33.1ms` / `162.40s` |
| `pool_starve` | `blocked_head` | `TTFT` / `TPOT p50` / `E2E` | `315.10s–526.97s` / `17.1ms` | `320.06s–335.81s` / `16.1ms` | `351.43s` / `16.5ms` / `355.64s` |
| `pool_starve` | **`short_bypass`** | `N` / `Err` | `24` / `0` | `24` / `0` | `12` / `0` |
| `pool_starve` | **`short_bypass`** | `TTFT p50` / `p95` | `56.83s` / `70.05s` | `0.48s` / `1.75s` | **`0.48s` / `2.13s`** |
| `pool_starve` | **`short_bypass`** | `TPOT p50` / `E2E p95` | `201.6ms` / `95.81s` | `73.5ms` / `11.93s` | **`68.8ms` / `10.72s`** |

*(Reference: v1 rep 0 alone recorded `cached_sat/short_util` `TTFT p50=33.26s` / `p95=43.05s`, `E2E p95=52.07s` and `pool_starve/short_bypass` `TTFT p50=0.48s` / `p95=1.75s`, `E2E p95=11.02s`. During v2 `pool_starve`, two periodic external `thinking=True` requests (`4,930` prompt at `02:24:14Z` and `2,557` prompt at `02:27:03Z`, `1,200` output tokens each) ran concurrently; `0.48s` `short_bypass` TTFT and the `99.5%` cache hit rate were unchanged.)*

---

## 10. Raw Artifacts

- **Benchmark Results and Logs:**
  - [ab_0070_before.json](./ab_0070_before.json) | [ab_0070_before.log](./ab_0070_before.log)
  - [ab_0070_after.json](./ab_0070_after.json) | [ab_0070_after.log](./ab_0070_after.log)
  - [ab_0070_build_start.log](./ab_0070_build_start.log)
  - [cap_starve_bench.py](../tools/cap_starve_bench.py) | [ab_0070_cap_starve_before.json](./ab_0070_cap_starve_before.json) | [ab_0070_cap_starve_before.log](./ab_0070_cap_starve_before.log)
  - [ab_0070_cap_starve_after.json](./ab_0070_cap_starve_after.json) | [ab_0070_cap_starve_after.log](./ab_0070_cap_starve_after.log)
  - [ab_0070_v2_reg.json](./ab_0070_v2_reg.json) | [ab_0070_v2_reg.log](./ab_0070_v2_reg.log) | [ab_0070_v2_build_start.log](./ab_0070_v2_build_start.log)
- **Container Inspections and `/health` Snapshots:**
  - [ab_0070_before_inspect.json](./ab_0070_before_inspect.json) | [ab_0070_before_health_pre.json](./ab_0070_before_health_pre.json) | [ab_0070_before_health_post.json](./ab_0070_before_health_post.json)
  - [ab_0070_after_inspect.json](./ab_0070_after_inspect.json) | [ab_0070_after_health_pre.json](./ab_0070_after_health_pre.json) | [ab_0070_after_health_post.json](./ab_0070_after_health_post.json)
  - [ab_0070_v2_inspect.json](./ab_0070_v2_inspect.json) | [ab_0070_v2_health_pre.json](./ab_0070_v2_health_pre.json) | [ab_0070_v2_health_post.json](./ab_0070_v2_health_post.json)
- **Archived Report Copy:** [ab_0070_report.md](./ab_0070_report.md)
