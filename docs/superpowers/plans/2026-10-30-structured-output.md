# Structured Output (patch 0034) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce `response_format`/grammar-constrained decoding in the dsv41 TensorFold engine on the two-Spark pair, so a request's reply cannot leave its schema.

**Architecture:** xgrammar 0.2.2, pinned by git SHA and built with the image's g++/libstdc++, behind a ~400-LOC `extern "C"` shim; a Zig-side compile cache and matcher lifecycle in `core/structure.zig`; per-stream state in the dsv41 family; the mask applied on-device to the full logits row inside the captured decode stretch (per-lane bitmask buffer + per-row enable flag); rank 0 owns the matcher and ships rank 1 its vocabulary slice over the RDMA ring.

**Tech Stack:** Zig 0.17.0, CUDA 13 (sm_121a), C++ (g++/libstdc++), xgrammar 0.2.2, Docker image built by `scripts/prepare.sh`.

**Spec:** `docs/STRUCTURED-OUTPUT.md` (rev 2, commit `4994774`) — read it with this plan; every value here is copied from it.

## Global Constraints

- Zig 0.17.0 only; no new runtime dependencies beyond xgrammar 0.2.2 pinned by full SHA.
- Both C++ components build with the image's g++ against libstdc++; the engine's final link adds libstdc++ and **does not** call `linkLibCpp`.
- `vocab_type = RAW` (bytes are already decoded; a byte-level type would decode twice).
- JSON/JSON-Schema compile with compact whitespace (`any_whitespace=false`, verified against the 0.2.2 headers) and a root leading `\s*`.
- `strict_mode=true` always; a client's `strict:false` is ignored and documented.
- Cache key is **(kind, full text)**, text compared on hash hit; byte-bounded (`TF_DS_SO_CACHE_BYTES`, default 268435456, 0 = off); the library's own cache stays off.
- Compiles run on worker threads, never on the lane thread; `TF_DS_SO_COMPILE_THREADS` default 2; `TF_DS_SO_TIMEOUT_S` default 20 → 400 at admission.
- Constrained streams never draft; every other stream keeps DSpark drafts.
- Matcher lives on rank 0 only; rank 1 receives its vocabulary slice of the bitmask over the RDMA ring and never initializes xgrammar.
- `TF_DS_SO=0` at boot leaves `Info.structures` false and the old 400 behaviour.
- The patch lands as `zig/patches/0034-structured-output.patch`, applied after `0033` in filename order, and must `git apply --check` on a clean `TF_REF` after `0001`–`0033`.
- The image is built only by `scripts/prepare.sh`; host tests run with `zig build test` (shim tests need `-Dxgrammar=<prefix>`).
- Spec line references are for a clean pre-patch `TF_REF`; locate by symbol, not line.

## Review Focus

1. A model that emits `\n\n` after the think-close token before `{` — the grammar must accept it (not newline-until-length). Owned by Task 6 tests + Task 11 probe.
2. A client sending `strict: false` — must be accepted and ignored, reply still schema-valid. Owned by Task 3 test + Task 11 probe.
3. A Lark grammar or regex construct xgrammar cannot compile — must be a 400 with the compiler's message, never a crash or a hung admission. Owned by Task 3 test + Task 11 probe.
4. A schema that cannot complete inside `max_tokens` — that request ends `length`/`failed`, the server and other streams stay healthy. Owned by Task 9 test + Task 11 probe.
5. A cancel mid-decode on a constrained stream — matcher freed, no leak, no crash, other streams unaffected. Owned by Task 6 test + Task 11 probe.

---

## File Structure

Engine checkout (`ai:~/repos/tensorfold-dsv41-dev`, `TF_REF` + `0001`–`0033` applied):

- `zig/src/families/dsv41/structure/tfso.h` — the shim's C declarations (new).
- `zig/src/families/dsv41/structure/tfso.cc` — the shim implementation (new).
- `zig/src/core/structure.zig` — cache, compile pool, kind→shim mapping (new).
- `zig/src/families/dsv41/structure.zig` — vocab export, per-stream matcher registry, mask preparation (new).
- `zig/src/families/dsv41/structure_mask.zig` — device buffer, flag, kernel wrapper (new).
- `zig/src/families/dsv41/round.zig` — mask node in the captured stretch (modify).
- `zig/src/families/dsv41/lanes.zig` — drafting exclusion (modify).
- `zig/src/families/dsv41/dsv41.zig` — `Info.structures` (modify).
- `zig/src/server/chat.zig` — failure surfacing (modify).
- `build.zig` — link shim/xgrammar/libstdc++, `-Dxgrammar` option (modify).
- Tests: `zig/src/core/structure_test.zig`, `zig/src/families/dsv41/structure_test.zig` (new).

Recipe repo (worktree `ai:~/repos/dsv41-structured-output`):

- `zig/Dockerfile`, `scripts/config.sh`, `scripts/prepare.sh`, `zig/patches/README.md`, `zig/patches/0034-structured-output.patch`, `tools/structure_probe.py`.

---

### Task 1: Verify xgrammar 0.2.2 builds standalone

**Files:**
- Create: `ai:~/opt/xgrammar-0.2.2/` (build prefix), `docs/STRUCTURED-OUTPUT.md` decision note if the fallback is taken.

**Interfaces:**
- Produces: `~/opt/xgrammar-0.2.2/{include,lib/libxgrammar.a}` or the recorded decision to use llguidance.

- [ ] **Step 1: Clone and build with the toolchain the image will use**

Run: `git clone --recursive https://github.com/mlc-ai/xgrammar && git checkout 0.2.2 && cmake -B build -DXGRAMMAR_BUILD_PYTHON_LIB=OFF && cmake --build build -j`
Expected: `libxgrammar.a` and `include/xgrammar/*.h` exist; no Python required.

- [ ] **Step 2: Compile a smoke TU against the headers**

Run: `g++ -std=c++17 -Iinclude smoke.cc lib/libxgrammar.a -o smoke && ./smoke`
Expected: exit 0 after building a `TokenizerInfo` from a 3-token vocab and compiling `{"type":"object"}`.

- [ ] **Step 3: Record the decision**

If it builds, no spec change. If it does not, write one line into `docs/STRUCTURED-OUTPUT.md` §12 and switch every later task's shim body to llguidance's C API (same Zig-side design).

- [ ] **Step 4: Commit (only if the spec changed)**

```bash
git add docs/STRUCTURED-OUTPUT.md && git commit -m "docs: xgrammar build verified | fallback to llguidance"
```

### Task 2: The shim

**Files:**
- Create: `zig/src/families/dsv41/structure/tfso.h`, `zig/src/families/dsv41/structure/tfso.cc`
- Test: `zig/src/families/dsv41/structure_test.zig`

**Interfaces:**
- Produces: `tfso_init(const char* const* vocab_bytes, const int32_t* lens, int32_t n, int32_t vocab_type, int32_t padded_vocab, const int32_t* stops, int32_t nstops) -> void*`; `tfso_compile_json_schema(void*, const char*, int strict, int any_whitespace) -> void*`; `tfso_compile_regex(void*, const char*) -> void*`; `tfso_compile_grammar(void*, const char*) -> void*`; `tfso_matcher_create(void*, void*) -> void*`; `tfso_fill_bitmask(void*, int32_t*) -> int`; `tfso_accept_token(void*, int32_t) -> int`; `tfso_rollback(void*, int32_t) -> int`; `tfso_finished(void*) -> int`; `tfso_free_*`; every failing call writes a message into a caller buffer (`tfso_last_error(char*, int)`).

- [ ] **Step 1: Write the failing test** — a fake 20-token vocab (ASCII bytes), `tfso_init` with `vocab_type=RAW`, then `tfso_compile_json_schema("{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\"}}}", 1, 0)`, a matcher, `tfso_fill_bitmask`; assert bit for token `{` is set and bit for a plain word token is clear.

- [ ] **Step 2: Run it** — `zig build test -Dxgrammar=$HOME/opt/xgrammar-0.2.2`; expect a compile error naming the missing `tfso_*` symbols.

- [ ] **Step 3: Implement** `tfso.cc` — `TokenizerInfo` with the RAW type and padded size, `GrammarCompiler`, `GrammarMatcher`; `fill_bitmask` writes int32 words, bit 1 = allowed; errors are caught and copied into the error buffer, never thrown across the C boundary.

- [ ] **Step 4: Run the test** — same command; expect PASS.

- [ ] **Step 5: Commit** — `git add zig/src/families/dsv41/structure zig/src/families/dsv41/structure_test.zig && git commit -m "feat(dsv41): xgrammar shim (tfso)"`

### Task 3: Core compile cache and kind mapping

**Files:**
- Create: `zig/src/core/structure.zig`, `zig/src/core/structure_test.zig`
- Modify: `build.zig` (test module)

**Interfaces:**
- Consumes: the `tfso_*` surface from Task 2.
- Produces: `pub const Kind = enum { json, json_schema, regex, choice, grammar };` `pub const Cache = struct { pub fn init(a: Allocator, bytes: usize) Cache; pub fn get(self: *Cache, kind: Kind, text: []const u8) !?*Grammar; pub fn put(self: *Cache, kind: Kind, text: []const u8, g: *Grammar) !*Grammar; pub fn deinit(self: *Cache) void; };` `pub fn compileAsync(self: *Cache, kind: Kind, text: []const u8) !Job;` `pub fn wait(self: *Cache, job: Job, timeout_ms: u64) !*Grammar;` `pub fn choiceGrammar(a: Allocator, choices_json: []const u8) ![]const u8;`

- [ ] **Step 1: Write the failing tests** — (a) two puts with the same hash but different text return different grammars; (b) a put past the byte bound evicts the least recently used; (c) cache 0 compiles per call; (d) `choiceGrammar("[\"a\",\"b\"]")` yields `root ::= "\"a\"" | "\"b\""` with escaped quotes; (e) an invalid schema yields an error carrying the compiler's message; (f) `wait` past the timeout yields `error.Timeout` while the worker keeps running.

- [ ] **Step 2: Run** — `zig build test`; expect failures for missing declarations.

- [ ] **Step 3: Implement** — LRU map `(kind,text)` with stored text compared on hit; worker pool capped by `TF_DS_SO_COMPILE_THREADS` (default 2) reading the flag through the engine's config; compile calls mapped per kind (`json` → the object schema text with `any_whitespace=0`; `json_schema` → strict; `regex`; `grammar`; `choice` → the generated alternation).

- [ ] **Step 4: Run** — expect PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(core): structured-output compile cache and kind mapping"`

### Task 4: Vocabulary export

**Files:**
- Create: `zig/src/families/dsv41/structure.zig` (export part)
- Test: `zig/src/families/dsv41/structure_test.zig` (extend)

**Interfaces:**
- Consumes: the engine tokenizer (`core/tokenizer`), `tfso_init`.
- Produces: `pub fn exportVocab(a: Allocator, tok: *const Tokenizer, padded_vocab: usize) !void` — builds `(bytes, lens, stops)` arrays, calls `tfso_init` once; `pub fn vocabReady() bool`.

- [ ] **Step 1: Write the failing test** — a small hand-built tokenizer fixture (10 tokens incl. `<|end|>`) → `exportVocab` succeeds, `vocabReady()` true; calling it twice is a no-op.

- [ ] **Step 2: Run** — expect failure (function absent).

- [ ] **Step 3: Implement** — iterate ids in order, skip nothing (padded ids above the real vocab get empty bytes and are excluded from the mask by the shim's stop/vocab-size handling), pass `vocab_type=RAW`, all stop/eos ids.

- [ ] **Step 4: Run** — PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(dsv41): export the vocabulary to the grammar engine"`

### Task 5: Device mask and captured-stretch node

**Files:**
- Create: `zig/src/families/dsv41/structure_mask.zig`
- Modify: `zig/src/families/dsv41/round.zig` (the `Graphs` capture path)

**Interfaces:**
- Produces: `pub const MaskBuf = struct { dev: *anyopaque, host: []i32, words: usize, flag_dev: *anyopaque, pub fn upload(self: *MaskBuf, stream: *cuda.Stream) !void; };` `pub fn maskRowKernel(stream: *cuda.Stream, logits: [*]f32, vocab: usize, buf: *MaskBuf, row_flag: *anyopaque) void;`

- [ ] **Step 1: Write the failing test** — host-side: a 64-wide fake row, a mask with bit 5 and 9 allowed, flag 1 → the row's other entries become `-inf`; flag 0 → row unchanged (bit-identical copy).

- [ ] **Step 2: Run** — `zig build test`; expect failure.

- [ ] **Step 3: Implement** — a persistent per-lane device buffer allocated with the stream, the flag in device memory, the kernel launched from the same stream the round uses and captured inside the stretch (`round.Graphs.run`), the host upload called before every launch; `TF_DS_SO=0` keeps the flag 0 forever.

- [ ] **Step 4: Run** — PASS; then live: `GRAPHS=1` unconstrained traffic must stay byte-identical (Task 11 gate 2 exercises it).

- [ ] **Step 5: Commit** — `git commit -m "feat(dsv41): on-device grammar mask inside the captured stretch"`

### Task 6: Matcher state machine and round hook

**Files:**
- Modify: `zig/src/families/dsv41/structure.zig`, `zig/src/families/dsv41/layer_main.zig` (the `sampleDev` call sites)
- Test: `zig/src/families/dsv41/structure_test.zig`

**Interfaces:**
- Consumes: Task 2 shim, Task 3 cache, Task 4 export, Task 5 mask.
- Produces: `pub const Stream = struct { matcher: ?*anyopaque, engaged: bool, completed: bool, after: ?u32 };` `pub fn prepare(cache: *core_structure.Cache, stream: *Stream, kind: Kind, text: []const u8, after: ?u32) !void`; `pub fn onToken(stream: *Stream, id: u32) !void` (advances or engages at `after`; on completion marks `completed`); `pub fn fill(stream: *Stream, buf: *MaskBuf) !void` (all-ones once `completed`; before `engaged`, the full mask); `pub fn release(stream: *Stream) void`.

- [ ] **Step 1: Write the failing tests** — (a) idle until `after` is emitted, then masked; (b) after completion, `fill` yields all-ones and `onToken` never fails; (c) `onToken` with a special-token id after completion does not error; (d) `release` frees the matcher and a second `release` is safe; (e) `\n\n` before `{` is accepted (the grammar's leading `\s*`).

- [ ] **Step 2: Run** — expect failures.

- [ ] **Step 3: Implement** — the state machine; the round hook advances the matcher with the last accepted token and uploads the mask before `sampleDev`; cancel and finish call `release`.

- [ ] **Step 4: Run** — PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(dsv41): matcher lifecycle and decode-round hook"`

### Task 7: TP=2 mask slicing

**Files:**
- Modify: `zig/src/families/dsv41/structure.zig`, the ring exchange path in `zig/src/families/dsv41/round.zig` / `comm.zig`
- Test: `zig/src/families/dsv41/structure_test.zig`

**Interfaces:**
- Produces: `pub fn sliceFor(rank: u32, world: u32, full: []const i32, vocab: usize) []const i32`; rank 0 sends rank 1's slice over the existing RDMA ring message used for decode-size exchanges; rank 1 receives it into its mask buffer and never calls `tfso_*`.

- [ ] **Step 1: Write the failing test** — a 128-word mask split across 2 ranks: `sliceFor(1,2,...)` returns the second half word-aligned, and `sliceFor(0,2,...)` the first; a vocab not divisible by 32 rounds up and never reads past the end.

- [ ] **Step 2: Run** — expect failure.

- [ ] **Step 3: Implement** — rank 0 compiles/advances/fills; the slice is sent before the round's graph launch; rank 1 uploads its slice; the existing candidate exchange and agreement guard are untouched.

- [ ] **Step 4: Run** — PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(dsv41): rank 1 receives its vocabulary slice of the mask"`

### Task 8: Capability flag, kill switch, drafting exclusion

**Files:**
- Modify: `zig/src/families/dsv41/dsv41.zig` (`Info`), `zig/src/families/dsv41/lanes.zig`

**Interfaces:**
- Produces: `Info.structures = soEnabled()` where `soEnabled()` reads `TF_DS_SO` (default 1); every structured stream is excluded from draft admission.

- [ ] **Step 1: Write the failing test** — host test: with `TF_DS_SO=0` the advertised `Info.structures` is false; with it unset/1, true. A structured stream never qualifies for a drafted round.

- [ ] **Step 2: Run** — expect failure.

- [ ] **Step 3: Implement** — the flag and the admission condition (mirroring `loneFits`'s `r.structure != null` check in the family's own admission).

- [ ] **Step 4: Run** — PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(dsv41): advertise structures; constrained streams never draft"`

### Task 9: Metrics and failure surfacing

**Files:**
- Modify: `zig/src/families/dsv41/structure.zig` (counters), `zig/src/server/chat.zig` (streaming failure frame), the metrics registry
- Test: `zig/src/core/structure_test.zig`

**Interfaces:**
- Produces: `so_requests_total`, `so_aborted_total`, `so_cache_bytes`, `so_compile_ms`, `so_compile_threads_busy`; a matcher failure after headers ends a stream with `data: {"error": …}` then closes, and a non-streaming failure keeps the engine's existing `cx.other` reply.

- [ ] **Step 1: Write the failing test** — counters increment on compile/abort; a simulated failure produces exactly one error frame and a closed stream.

- [ ] **Step 2: Run** — expect failure.

- [ ] **Step 3: Implement** — counters on the existing `/metrics` path; the streaming branch reuses the `.failed` path at `chat.zig` (~:470) with an error frame instead of a finish reason.

- [ ] **Step 4: Run** — PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat: structured-output metrics and streamed failure frame"`

### Task 10: Image build wiring and patch export

**Files:**
- Modify: `zig/Dockerfile`, `scripts/config.sh`, `scripts/prepare.sh`, `zig/patches/README.md`
- Create: `zig/patches/0034-structured-output.patch`

**Interfaces:**
- Produces: `XGRAMMAR_REF` pinned in `config.sh`, passed by `prepare.sh` as `--build-arg`, echoed by `image_hash`; the Dockerfile clones xgrammar `--recursive`, builds it with g++ (Python off) **before** `zig build`, and the engine links `libtfso.a` + `libxgrammar.a` + libstdc++ (no `linkLibCpp`); the build context wording says `zig/patches` **and** `zig/patches-inputs`.

- [ ] **Step 1: Export the patch** — in the engine checkout: `git diff > 0034-structured-output.patch`; verify `git apply --check` on a clean `TF_REF` with `0001`–`0033` applied, then `git apply` it and confirm the tree matches.

- [ ] **Step 2: Run the build** — `scripts/prepare.sh`; expect a new image tag and a successful `zig build` with the shim linked.

- [ ] **Step 3: Add the README row** — the patch table entry: what it does, switches (`TF_DS_SO`, `TF_DS_SO_CACHE_BYTES`, `TF_DS_SO_TIMEOUT_S`, `TF_DS_SO_COMPILE_THREADS`), defaults, and the measurements to be filled from Task 11.

- [ ] **Step 4: Commit** — `git add zig/Dockerfile scripts/config.sh scripts/prepare.sh zig/patches/README.md zig/patches/0034-structured-output.patch && git commit -m "feat(zig): structured-output patch 0034 and its image build"`

### Task 11: Probe tool and live gates

**Files:**
- Create: `tools/structure_probe.py`
- Modify: `zig/patches/0034-structured-output.patch` (header receipts), `zig/patches/README.md`

- [ ] **Step 1: Write the probe** — schemas (flat, nested, enum, long-string, regex, choice) × thinking on/off × stream/non-stream; validate every reply against its schema; include the `\n\n`-after-think case, `strict:false`, an unsupported Lark grammar (expect 400), `response_format`+`tools`+"auto" (expect no calls), an unsatisfiable schema (expect length/failed, server healthy), and a cancel mid-decode.

- [ ] **Step 2: Run the gates** — `./stop.sh && ./start.sh` with `/mnt/nvmeof` healthy; then `tools/exact.py` (unconstrained byte-identical, `GRAPHS=1`, drafts on), the probe, a mixed 4-stream run, and `/metrics` checks.

- [ ] **Step 3: Record the receipts** — decode-speed cost with drafts off on a constrained request, the cache bound in use, and the gate results, into the patch header and README row (the repo's culture).

- [ ] **Step 4: Commit** — `git commit -m "feat: structured-output probe and live gate receipts (0034)"`
