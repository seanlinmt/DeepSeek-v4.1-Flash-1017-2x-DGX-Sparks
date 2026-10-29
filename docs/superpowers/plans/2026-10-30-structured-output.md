# Structured Output (patch 0034) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce `response_format`/grammar-constrained decoding in the dsv41 TensorFold engine on the two-Spark pair.

**Architecture:** xgrammar v0.2.2 (pinned `4d145cc…`) built with the image's g++/libstdc++ behind a POD-only C shim in `core/structure/`; a mutexed compile cache in `core/structure.zig`; compile at admission on worker threads with the handle carried on `Request.structure`; per-stream matcher state in the dsv41 family; the mask applied by a `.cu` kernel after the graph launch and before `sampleDev`; rank 0 owns the matcher and ships rank 1 its vocabulary slice over the RDMA ring.

**Tech Stack:** Zig 0.17.0, CUDA 13 (sm_121a), C++17 (g++/libstdc++), xgrammar v0.2.2, Docker image built by `scripts/prepare.sh`.

**Spec:** `docs/STRUCTURED-OUTPUT.md` rev 3 — read it with this plan.

## Global Constraints

- xgrammar pinned at commit `4d145cc13d878c751ebeed36af1c013074be76bc`; CMake `-DXGRAMMAR_BUILD_PYTHON_BINDINGS=OFF`, `-DCMAKE_BUILD_TYPE=Release`, clone `--recursive` (dlpack).
- Both C++ components build with the image's **g++/libstdc++**; the engine links libstdc++ and libgcc, never libc++ (`linkLibCpp` off). The shim is compiled by a **system `g++ -c` + `ar` step**, not `addCSourceFile`.
- `vocab_type = RAW`; only the real vocab is passed, padded size as `vocab_size`; every special token is exported with empty bytes.
- `any_whitespace=false` for JSON/JSON-Schema; a bounded pre-grammar whitespace phase (≤ 4 characters) handles leading whitespace.
- `strict_mode=true` always; `strict:false` accepted and ignored.
- Cache key **(kind, full text)** with text compared on a hash hit; mutexed; in-flight compile dedupe; byte bound from `tfso_grammar_bytes`, `TF_DS_SO_CACHE_BYTES` default 268435456; library cache off.
- Compiles run on worker threads only (`TF_DS_SO_COMPILE_THREADS` default 2), at admission; `TF_DS_SO_TIMEOUT_S` default 20 → `cx.refuse` before headers.
- Constrained streams never draft and never share a round with drafted rows.
- Matcher on rank 0 only; rank 1 receives a bit-shifted vocabulary slice over the RDMA ring, only in constrained rounds; rank 1 never initializes xgrammar.
- `TF_DS_SO=0` → `Info.structures` false, old 400, old bits. `TF_DS_SO_FAULT=accept` injects a matcher failure.
- Patch exported by `scripts/port-patches.sh` (squash) after a base commit of `TF_REF` + `0001`–`0033`; it must pass its own apply/equality checks.
- Image built only by `scripts/prepare.sh`; host tests need `-Dxgrammar=<prefix>` and skip the shim tests without it.
- Spec line references are for a clean pre-patch `TF_REF`; locate by symbol.

## Review Focus

1. `\n\n` after the think-close token before `{` — accepted, not newline-until-length. Task 6 tests, Task 11 probe.
2. `strict: false` — accepted and ignored, reply still valid. Task 3 test, Task 11 probe.
3. An unsupported Lark grammar or regex construct — 400 with the compiler's message, never a crash or hung admission. Task 2 test, Task 11 probe.
4. An unsatisfiable schema within `max_tokens` — that request ends `length`, server and other streams healthy; `TF_DS_SO_FAULT` covers the `failed` frame. Task 9 test, Task 11 probe.
5. Cancel mid-decode on a constrained stream — matcher freed, no leak, others unaffected. Task 6 test, Task 11 probe.

---

## File Structure

Engine repo (`ai:~/repos/tensorfold-dsv41-dev`, branch `structured-output`):

- `zig/src/core/structure/tfso.h`, `tfso.cc` — the shim (new).
- `zig/src/core/structure.zig` — cache, compile pool, kind mapping (new).
- `zig/src/families/dsv41/structure.zig` — per-stream matcher, whitespace phase, vocab-export wiring (new).
- `zig/src/families/dsv41/structure_mask.cu` — the mask kernel (new).
- `zig/src/families/dsv41/layer_main.zig` — mask + matcher at the two `sampleDev` sites (modify).
- `zig/src/families/dsv41/dsv41.zig` — `Info.structures` (modify).
- `zig/src/families/dsv41/lanes.zig` — drafting exclusion and mixed rounds (modify).
- `zig/src/core/engine_api.zig` — `Structure.handle` (modify).
- `zig/src/server/server.zig` — compile at admission (modify).
- `zig/src/server/chat.zig` — scoped error frame (modify).
- `build.zig` — `-Dxgrammar`, g++/ar step, link flags (modify).
- Tests: `zig/src/core/structure_test.zig`, `zig/src/families/dsv41/structure_test.zig`, `tools/so_gpu_test.cpp` (GPU binary).

Recipe repo (worktree `ai:~/repos/dsv41-structured-output`):

- `zig/Dockerfile`, `scripts/config.sh`, `scripts/prepare.sh`, `zig/patches/0034-structured-output.patch` (generated), `zig/patches/README.md` (generated row), `tools/structure_probe.py`.

---

### Task 0: Base commit, toolchain, xgrammar build  [engine repo + host]

**Files:** engine checkout `~/repos/tensorfold-dsv41-dev`; `~/opt/zig-0.17.0`; `~/opt/xgrammar-0.2.2`.

**Interfaces:**
- Produces: branch `structured-output` whose base commit tree equals `TF_REF` + `0001`–`0033`; a working `zig`; `~/opt/xgrammar-0.2.2/{include,lib/libxgrammar.a}`.

- [ ] **Step 1: Apply the hand-written patches and commit the base**

Run: `git checkout -B structured-output 629c524 && for p in ~/repos/dsv41-structured-output/zig/patches/*.patch; do git apply "$p"; done && git add -A && git commit -m "base: TF_REF + 0001-0033" && git tag so-base`
Expected: one commit; `git status` clean.

- [ ] **Step 2: Install Zig 0.17.0 from the pinned tarball**

Run: download `zig-aarch64-linux-0.17.0.tar.xz`, check sha256 against `scripts/config.sh`'s `ZIG_SHA256`, extract to `~/opt/zig-0.17.0`.
Expected: `~/opt/zig-0.17.0/zig version` prints `0.17.0`.

- [ ] **Step 3: Build xgrammar standalone (the fallback decision point)**

Run: `git clone --recursive https://github.com/mlc-ai/xgrammar && git checkout 4d145cc13d878c751ebeed36af1c013074be76bc && cmake -B build -DXGRAMMAR_BUILD_PYTHON_BINDINGS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/opt/xgrammar-0.2.2 && cmake --build build -j && cmake --install build`
Expected: `~/opt/xgrammar-0.2.2/lib/libxgrammar.a` exists.

- [ ] **Step 4: Smoke-compile against it**

Run: `g++ -std=c++17 -I$HOME/opt/xgrammar-0.2.2/include smoke.cc -L$HOME/opt/xgrammar-0.2.2/lib -lxgrammar -o smoke && ./smoke`
Expected: exit 0 after building a `TokenizerInfo` from 3 tokens and compiling `{"type":"object"}`.

- [ ] **Step 5: Record the fallback if Step 3 failed**

If it did not build, add one line to spec §12 and switch every later task's shim body to llguidance's C API. Commit: `git commit -m "docs: xgrammar build verified | llguidance fallback"` (recipe repo).

### Task 1: Shim and build wiring  [engine repo]

**Files:**
- Create: `zig/src/core/structure/tfso.h`, `zig/src/core/structure/tfso.cc`
- Modify: `build.zig`
- Test: `zig/src/core/structure_test.zig`

**Interfaces:**
- Produces: the `tfso_*` surface of spec §4, every call taking `char* err, int cap`; `tfso_grammar_bytes`; `tfso_fill_bitmask` returning the need-apply flag.

- [ ] **Step 1: Write the first failing test — the skip behaviour**

Test `shim tests are skipped without -Dxgrammar`: build without the option and assert the test binary prints the skip note and exits 0.

- [ ] **Step 2: Run it** — `~/opt/zig-0.17.0/zig build test`; expected PASS after the wiring exists (this test is what pins the wiring).

- [ ] **Step 3: Wire `build.zig`** — a `-Dxgrammar` option (default empty); when set, a system step runs `g++ -std=c++17 -fPIC -I<xg>/include -c zig/src/core/structure/tfso.cc -o tfso.o` and `ar rcs libtfso.a tfso.o`; the test and native targets link `libtfso.a`, `libxgrammar.a`, `stdc++`, `gcc`; `linkLibCpp` stays off; when unset, the shim tests are skipped with the printed note.

- [ ] **Step 4: Write the shim tests** — fake 20-token vocab; `tfso_init`; `tfso_compile_json_schema` with `{"type":"object","properties":{"a":{"type":"string"}}}`; `tfso_fill_bitmask` sets `{` and clears a plain word; an invalid schema returns non-zero with a message in the caller's buffer (two concurrent failures keep distinct messages).

- [ ] **Step 5: Run** — `~/opt/zig-0.17.0/zig build test -Dxgrammar=$HOME/opt/xgrammar-0.2.2`; expected PASS.

- [ ] **Step 6: Implement** `tfso.cc` — `TokenizerInfo` (RAW, padded size, stop ids), `GrammarCompiler` with its cache disabled, `GrammarMatcher`; every call wrapped in `try/catch (const std::exception&)` copying `what()` into the caller's buffer; `tfso_grammar_bytes` returns `MemorySizeBytes()`.

- [ ] **Step 7: Commit** — `git add zig/src/core/structure zig/src/core/structure_test.zig build.zig && git commit -m "feat(core): xgrammar shim and its g++ build step"`

### Task 2: Compile cache and kind mapping  [engine repo]

**Files:**
- Create: `zig/src/core/structure.zig`; extend `zig/src/core/structure_test.zig`

**Interfaces:**
- Consumes: Task 1's shim.
- Produces: `pub const Kind = enum { json, json_schema, regex, choice, grammar };` `pub const Cache = struct { pub fn init(a: Allocator, bytes: usize) Cache; pub fn getOrCompile(self: *Cache, kind: Kind, text: []const u8, timeout_ms: u64) !*Grammar; pub fn deinit(self: *Cache) void; };` `pub fn choiceGrammar(a: Allocator, choices_json: []const u8) ![]const u8;` `GrammarsEntry = struct { g: *Grammar, bytes: usize, text: []const u8 };`

- [ ] **Step 1: Write the failing tests** — (a) two puts with the same hash and different text yield different grammars; (b) the byte bound evicts LRU; (c) cache 0 compiles per call; (d) two threads asking for the same uncached schema trigger **one** compile (dedupe); (e) `choiceGrammar("[\"a\",\"b\"]")` → `root ::= "\"a\"" | "\"b\""`; (f) a timeout returns `error.Timeout` while the worker finishes; (g) an invalid schema returns the compiler's message in the error buffer.

- [ ] **Step 2: Run** — expect failures for the missing declarations.

- [ ] **Step 3: Implement** — mutexed map keyed `(kind, text)` with the text stored and compared; in-flight map for dedupe; worker pool capped by `TF_DS_SO_COMPILE_THREADS`; eviction by `tfso_grammar_bytes`; the kind→call mapping of spec §4; `choice` alternation with JSON-quote escaping.

- [ ] **Step 4: Run** — expect PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(core): structured-output compile cache with dedupe and byte bound"`

### Task 3: Compile at admission  [engine repo]

**Files:**
- Modify: `zig/src/core/engine_api.zig` (`Structure.handle`), `zig/src/server/server.zig` (`checkFeatures`, boot init)
- Test: `zig/src/core/structure_test.zig` (server-level test file if one exists)

**Interfaces:**
- Produces: `Structure.handle: ?*anyopaque = null`; at boot, when `info.structures`, the server initializes `core/structure` from `srv.text`; `checkFeatures` calls `getOrCompile` and `cx.refuse`s on error/timeout, then sets `request.structure.handle`.

- [ ] **Step 1: Write the failing test** — a request with a valid schema through `checkFeatures` yields a non-null `handle`; an invalid schema is refused with the compiler message; a simulated timeout is refused (no header sent); `strict:false` is accepted and the handle is valid.

- [ ] **Step 2: Run** — expect failure.

- [ ] **Step 3: Implement** — the field, the boot init (rank 0 only; rank 1 skips), the compile call, the refusals; no compile ever runs on the lane thread.

- [ ] **Step 4: Run** — expect PASS.

- [ ] **Step 5: Update the spec's §2** to record that `server.zig` and `engine_api.zig` are modified, and commit — `git commit -m "feat(server): compile structured output at admission"`

### Task 4: Vocabulary export  [engine repo]

**Files:**
- Create/modify: `zig/src/families/dsv41/structure.zig`; extend `zig/src/families/dsv41/structure_test.zig`; wire into rank 0's boot in `dsv41.zig`/`native.zig`

**Interfaces:**
- Produces: `pub fn exportVocab(a: Allocator, tok: *const Tokenizer, padded: usize) !void`; `pub fn vocabReady() bool`.

- [ ] **Step 1: Write the failing test** — a 10-token fixture including a special token: export succeeds; the special token's bytes are empty; `vocab_size` is the padded size; a second call is a no-op.

- [ ] **Step 2: Run** — expect failure.

- [ ] **Step 3: Implement** — iterate the real vocab only; empty bytes for specials; stop ids; RAW; call `tfso_init` once; wire the call into rank 0's boot when `info.structures` and `TF_DS_SO` is on.

- [ ] **Step 4: Run** — expect PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(dsv41): export the vocabulary (specials blocked) at rank 0 boot"`

### Task 5: Mask kernel and application points  [engine repo]

**Files:**
- Create: `zig/src/families/dsv41/structure_mask.cu`, `tools/so_gpu_test.cpp`
- Modify: `zig/src/families/dsv41/layer_main.zig` (both `sampleDev` sites), `build.zig` (nvcc step for the `.cu`)

**Interfaces:**
- Produces: `pub fn maskRow(stream: *cuda.Stream, logits: [*]f32, vocab: usize, bits: []const i32, rows: []const u32) void;` a pinned host staging buffer + device buffer allocated once at boot (max rows × words).

- [ ] **Step 1: Write the failing host test** — a CPU reference of the bit math: a 64-wide row, bits allowing 5 and 9, flag 1 → others `-inf`; the same with a row not constrained → unchanged (bit-identical copy).

- [ ] **Step 2: Run** — expect failure.

- [ ] **Step 3: Implement the kernel and the wiring** — `.cu` compiled by the image's nvcc; the host fills the bitmask, H2D-copies into the staging buffer, launches the kernel **after `round.forward` returns and before `sampleDev`** at both sites (:1185, :1316), scoped to the round's constrained rows; confirm in code that both sites are outside the captured stretch (`round.zig` `Graphs`) and record it in the patch header.

- [ ] **Step 4: Run the GPU test binary on the Spark** — `tools/so_gpu_test.cpp` builds and runs outside `zig build test`: a 128-wide row, mask, expect the `-inf` set and the untouched rows byte-equal; run with `-inf` values through the sampler (`sampleRow`) and assert no NaN with a single allowed token.

- [ ] **Step 5: Commit** — `git commit -m "feat(dsv41): device mask kernel applied before sampling"`

### Task 6: Matcher state machine and streaming hook  [engine repo]

**Files:**
- Modify: `zig/src/families/dsv41/structure.zig`, `zig/src/families/dsv41/layer_main.zig`
- Test: extend `zig/src/families/dsv41/structure_test.zig`

**Interfaces:**
- Produces: `pub const Stream = struct { matcher: ?*anyopaque, phase: enum { idle, ws, engaged, done }, ws_used: u8, after: ?u32 };` `pub fn prepare(cache: *core_structure.Cache, s: *Stream, kind: Kind, text: []const u8, after: ?u32) !void`; `pub fn onToken(s: *Stream, id: u32) !void`; `pub fn fill(s: *Stream, buf: []i32) !void`; `pub fn release(s: *Stream) void`.

- [ ] **Step 1: Write the failing tests** — (a) idle until `after`; (b) the whitespace phase accepts `\n\n` then engages; (c) a 5th whitespace character is refused; (d) after completion `fill` is all-ones and `onToken` never fails on special ids; (e) `release` frees and is idempotent; (f) cancel mid-decode leaves no matcher.

- [ ] **Step 2: Run** — expect failures.

- [ ] **Step 3: Implement** — the state machine with a boot-computed whitespace-token set; the round hook advances with the last accepted token and fills the mask before `sampleDev`; `release` on finish and on cancel; the matcher touched only by the lane thread.

- [ ] **Step 4: Run** — expect PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(dsv41): matcher state machine with bounded whitespace phase"`

### Task 7: TP slicing over the ring  [engine repo]

**Files:**
- Modify: `zig/src/families/dsv41/structure.zig`, the ring message in `zig/src/families/dsv41/rdma.zig` (or the decode-size exchange path), `layer_main.zig` (rank 1 receives before its launch)
- Test: extend `zig/src/families/dsv41/structure_test.zig`

**Interfaces:**
- Produces: `pub fn sliceFor(rank: u32, world: u32, full: []const i32, vocab_per_rank: []const usize) Slice` where `Slice = struct { bits: []const i32, bit_offset: u6 };`; a ring message carrying `bits` + `bit_offset` + the round id; rank 1 shifts its slice in.

- [ ] **Step 1: Write the failing tests** — a boundary at 1000 (not 32-aligned): rank 1's slice has `bit_offset = 1000 % 32` and masks the right ids; a message round-trip of the slice + offset; no send when the round has no constrained row.

- [ ] **Step 2: Run** — expect failure.

- [ ] **Step 3: Implement** — rank 0 fills and sends only in constrained rounds; rank 1 receives, shifts, uploads; check the ring slot capacity and grow it in the ring's own sizing terms if needed; the existing candidate exchange and agreement guard untouched.

- [ ] **Step 4: Run** — expect PASS; GPU binary: rank 1's masked row produces the same token as rank 0's.

- [ ] **Step 5: Commit** — `git commit -m "feat(dsv41): rank 1 receives a bit-shifted mask slice"`

### Task 8: Capability flag, kill switch, drafting exclusion  [engine repo]

**Files:**
- Modify: `zig/src/families/dsv41/dsv41.zig`, `zig/src/families/dsv41/lanes.zig`

**Interfaces:**
- Produces: `Info.structures = soEnabled()` from `TF_DS_SO`; every admission path excludes a structured stream from drafting; constrained rows never share a round with drafted rows.

- [ ] **Step 1: Write the failing test** — `TF_DS_SO=0` → `Info.structures` false; a structured stream never qualifies for a drafted round; a mixed round is never formed.

- [ ] **Step 2: Run** — expect failure.

- [ ] **Step 3: Implement** — the flag and the admission conditions (mirror `loneFits`'s `r.structure != null` in the family's own paths).

- [ ] **Step 4: Run** — expect PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(dsv41): advertise structures; constrained streams never draft"`

### Task 9: Metrics, fault injection, scoped failure frame  [engine repo]

**Files:**
- Modify: `zig/src/families/dsv41/structure.zig`, `zig/src/server/chat.zig`, the metrics registry
- Test: a server-level test file next to `chat.zig`

**Interfaces:**
- Produces: `so_requests_total`, `so_aborted_total`, `so_cache_bytes`, `so_compile_ms`, `so_compile_threads_busy`; `TF_DS_SO_FAULT=accept` makes the matcher reject the next token; a structured request's mid-stream failure emits one `data: {"error": …}` frame and closes; other failure classes unchanged.

- [ ] **Step 1: Write the failing test** — counters increment; with `TF_DS_SO_FAULT=accept` a streamed structured request produces exactly one error frame and closes; a non-structured failure keeps today's behaviour.

- [ ] **Step 2: Run** — expect failure.

- [ ] **Step 3: Implement** — counters; the fault switch; the error frame scoped by `request.structure != null`.

- [ ] **Step 4: Run** — expect PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat: structured-output metrics, fault switch, scoped error frame"`

### Task 10: Image build and the feature-off checkpoint  [recipe repo]

**Files:**
- Modify: `zig/Dockerfile`, `scripts/config.sh`, `scripts/prepare.sh`

**Interfaces:**
- Produces: `XGRAMMAR_REF` pinned in `config.sh`, passed by `prepare.sh`, echoed by `image_hash`; the Dockerfile clones xgrammar `--recursive`, builds it with g++ (Python bindings off, Release, install) **before** `zig build`; the engine links `libtfso.a` + `libxgrammar.a` + libstdc++ + libgcc; the build-context comment says `zig/patches` and `zig/patches-inputs`.

- [ ] **Step 1: Record the baseline** — on the current `main` image, `tools/exact.py --drafts on` → save the record.

- [ ] **Step 2: Wire the build** — `config.sh` pin; `prepare.sh --build-arg XGRAMMAR_REF`; `image_hash` echoes it; Dockerfile steps in order; link flags.

- [ ] **Step 3: Build and run the checkpoint** — `scripts/prepare.sh`; `TF_DS_SO=0 ./stop.sh && TF_DS_SO=0 ./start.sh`; `tools/exact.py --like <record>`.
Expected: the image builds, the server serves, the bits match the baseline.

- [ ] **Step 4: Commit** — `git add zig/Dockerfile scripts/config.sh scripts/prepare.sh && git commit -m "feat(zig): build xgrammar and the shim into the engine image"`

### Task 11: Probe and full live gates  [recipe repo]

**Files:**
- Create: `tools/structure_probe.py`

- [ ] **Step 1: Write the probe** — the matrix of spec §10.4: schemas × thinking × streaming, the formerly-400 case, malformed schema, unsupported Lark/regex, spec beside required `tool_choice`, spec + `tools` + "auto", `\n\n` after think-close, `strict:false`, unsatisfiable schema (`length`), `TF_DS_SO_FAULT=accept` (error frame), cancel mid-decode, mixed 4-stream run.

- [ ] **Step 2: Run the gates** — feature on (`./stop.sh` + `./start.sh`), then `exact.py` (byte-identity), the probe, the mixed run, the sampler-sanity run (`-inf` + penalties/top-p/min-p + single allowed token), `TF_DS_SO=0` (old 400), one rehearsed rollback, `/metrics`.

- [ ] **Step 3: Commit** — `git add tools/structure_probe.py && git commit -m "feat: structured-output probe and live gates"`

### Task 12: Export patch 0034  [recipe repo]

**Files:**
- Generate: `zig/patches/0034-structured-output.patch`; `zig/patches/README.md` row.

- [ ] **Step 1: Export** — `scripts/port-patches.sh --engine ~/repos/tensorfold-dsv41-dev --branch structured-output --base so-base --mode squash --name structured-output --title "…"`.
Expected: the script verifies the base, applies the patch on a clean `TF_REF` + `0001`–`0033`, and confirms the result equals the branch; it writes `0034-structured-output.patch` and the README row.

- [ ] **Step 2: Fill the receipts** — add the measured decode cost (drafts off), the cache bound, and the gate results to the patch header.

- [ ] **Step 3: Commit** — `git add zig/patches && git commit -m "feat(zig): export structured-output patch 0034 with gate receipts"`
