# Structured output — the design of `zig/patches/0034`

Branch `engine-structured-output`. Revision 3, 2026-10-30: the plan review's
items are integrated (patch export, compile-at-admission, shim build toolchain,
mask outside the captured graph, TP slicing with bit shifts, bounded whitespace,
special tokens, threading and lifetime, fault injection, gates). The patch lands
as `zig/patches/0034-structured-output.patch` (exported by
`scripts/port-patches.sh`), a generated `zig/patches/README.md` row, a
`zig/Dockerfile` edit, a probe in `tools/`, and this document.

## 0. Status

Design approved 2026-10-30, including the surface superset. Revision 3 applies
the plan review; no open questions remain.

## 1. The decisions

| Decision | Choice |
| --- | --- |
| Grammar engine | **xgrammar v0.2.2**, pinned at commit `4d145cc13d878c751ebeed36af1c013074be76bc`. Both xgrammar and the shim build with the image's **g++/libstdc++**; the engine links libstdc++ and libgcc, **not** libc++. The shim's C surface is POD-only. **llguidance** (plain C) is the named fallback. |
| Request surface | The engine's whole parsed surface (`response_format` `json_object`/`json_schema`, the four `guided_*`, `structured_outputs`) — signed off 2026-10-30. xgrammar's schema/regex subset means some `guided_*` requests that vLLM accepts get a **compile-time 400** with the compiler's message. |
| Whitespace | JSON/JSON-Schema compile with `any_whitespace=false` (compact). Leading whitespace is **not** expressible through `CompileJSONSchema` (verified: no such parameter), so a **bounded pre-grammar phase** accepts at most 4 whitespace characters after the think-close token, then engages the matcher. `structural_tag` `sequence[regex, json_schema]` was considered and rejected: its `json_schema` element has no whitespace knob, so inner whitespace would return to unlimited. |
| Drafts | A constrained request never drafts. Constrained rows never share a round with drafted rows. |
| Validation | Full live gates, plus a `TF_DS_SO=0` checkpoint on the real image before the feature is turned on. |

## 2. What exists at `TF_REF` (629c524)

Line numbers are for a **clean pre-patch `TF_REF`**; implementation locates by
symbol. The patch **does** extend this server-side plumbing (rev 3):
`engine_api.zig` (a compiled-grammar handle on `Structure`) and `server.zig`
(the compile at admission). Everything else below is reused unchanged.

- `zig/src/server/grammar.zig` (101 lines): parses the whole surface; malformed
  spec = 400; `grammar.refusal` refuses a spec beside `tool_choice`
  "required"/named.
- `zig/src/server/server.zig` (`checkFeatures`, ~:158-179): the gate on
  `srv.info.structures`, then `request.structure = .{ .kind, .text, .after }`.
- `zig/src/core/engine_api.zig`: `Structure { kind, text, after }` (no strict
  field, no handle yet); `Info.structures = false`.
- `zig/src/core/lane_host.zig` (`loneFits`, ~:397): `r.structure != null` already
  keeps a request off the lone drafted fast path; the lane core runs on its own
  thread (:104).
- `zig/src/server/anthropic_translate.zig` (~:270): the Anthropic frontend maps
  into the chat body — one gate serves both frontends.
- `zig/src/server/tool_parse.zig` (~:228,:233): non-call JSON is content.
- `zig/src/server/chat.zig:469-470`: `.failed => cx.other(...)` — a failed reply
  is an error reply, not a finish reason.
- **Sampling is outside the captured graph**: `round.forward` captures the
  forward stretches; `sampleDev` is called by its callers in
  `layer_main.zig` (:1185 prompt/first token, :1316 round/verify) after
  `round.forward` returns (:1303). The mask therefore needs no graph changes.
- No grammar enforcement exists; `build.zig.zon` has zero dependencies.

## 3. What the patch adds

1. **`Info.structures = true`** in the dsv41 host's `Info`, gated by
   `TF_DS_SO` (default 1; `0` keeps the old 400 and the old bits).
2. **Compile at admission, never on the lane thread.** `Structure` gains
   `handle: ?*anyopaque`. `checkFeatures` calls `core/structure.compile(...)`,
   which runs on a worker thread (cap `TF_DS_SO_COMPILE_THREADS`, default 2) and
   returns the handle; a compile error or `TF_DS_SO_TIMEOUT_S` expiry is
   `cx.refuse(...)` **before any header goes out**. On timeout the worker's
   compile runs to completion in the background and its result is discarded
   (xgrammar cannot cancel); the pool is capped, so the burn is bounded.
3. **`zig/src/core/structure.zig`**: cache keyed on **(kind, full text)** with
   the text compared on a hash hit, guarded by a mutex and with in-flight
   dedupe (two requests for the same schema share one compile); byte bound from
   `tfso_grammar_bytes` (`CompiledGrammar::MemorySizeBytes`, which covers the
   grammar and its adaptive mask cache), `TF_DS_SO_CACHE_BYTES` default 256 MiB,
   0 = off; the library's own cache stays off (CVE-2025-32381).
   Eviction is safe under a live matcher: `GrammarMatcher` holds the compiled
   grammar by value (`CompiledGrammar` is a `shared_ptr` pimpl), so dropping our
   handle cannot free it.
4. **`zig/src/core/structure/tfso.h` + `tfso.cc`** — the shim, in `core/`
   because `core/structure.zig` uses it (layering). Built by a **system step**
   (`g++ -std=c++17 -fPIC -c` then `ar rcs libtfso.a`), not `addCSourceFile`
   (which uses Zig's clang/libc++ headers — the mix the spec rules out).
5. **The mask point.** `sampleDev` reads only the row's top `top_k + MARGIN`
   candidates from the GPU, so masking runs on the device, on the full row,
   before that read:
   - a `.cu` kernel (built by the image's nvcc, wired where the engine's CUDA
     sources are built) sets `logits[i] = -inf` where bit i is 0;
   - the bitmask is filled on the host, H2D-copied into a per-round staging
     buffer, and the kernel is launched **after the graph launch and before
     `sampleDev`** on the same stream — stream-ordered, no capture changes, no
     fixed-address buffers, no stale flags (rows are re-mapped per round, so the
     mask is uploaded per round for that round's constrained rows only);
   - the kernel is gated per row: rows that are not constrained are untouched,
     so drafted/plain rows in the same round keep their bits.
6. **Matcher state machine** (per stream, in `families/dsv41/structure.zig`):
   idle until the `after` token; then a **bounded whitespace phase** (at most 4
   whitespace characters, from a boot-computed whitespace-token set); then
   engaged, advancing with each accepted token; on completion only the stop
   token is allowed, then **all-ones** for whatever follows EOS. Freed on
   finish and on cancel; touched only by that stream's lane thread. It survives
   `0005`'s interleaved prefill as per-stream state.
7. **Vocab export** (rank 0 only, at boot, once): only the **real** vocab is
   passed, with the model's padded size as `vocab_size`; `vocab_type = RAW`;
   every special token is exported with **empty bytes** so the mask can never
   allow it (otherwise a JSON string could legitimately contain a think marker).
8. **TP=2 — one matcher, on rank 0.** Rank 0 compiles and steps the matcher,
   fills the full bitmask, and sends rank 1 its slice over the RDMA ring **only
   in rounds that contain a constrained row**. The slice is not word-aligned
   (`vocabOf` gives a rank's vocabulary part, not a multiple of 32), so the send
   carries a bit offset and rank 1 shifts in. Rank 1 never initializes xgrammar.
   The existing candidate exchange and per-step agreement guard are unchanged.
   The ring message is sized and tested; if the ring's slot cannot carry it, the
   patch grows the slot in the ring's own sizing terms.
9. **Failure surfacing**, scoped to structured requests: `Reason.failed` with
   the message; non-streaming keeps the engine's `cx.other` reply; streaming
   sends one `data: {"error": …}` frame and closes. The change is scoped so
   other failure classes keep today's behaviour. `TF_DS_SO_FAULT=accept` injects
   a matcher failure so the path is testable (an unsatisfiable schema ends
   `length`, not `failed`).
10. **`/metrics`**: `so_requests_total`, `so_aborted_total`, `so_cache_bytes`,
    `so_compile_ms`, `so_compile_threads_busy`.

## 4. The shim's C surface

Every call takes an error out-parameter (`char* buf, int cap`) written on
failure — no global error state (compile workers and the lane thread run
concurrently). All xgrammar exceptions are caught inside the shim.

`tfso_init(const char* const* bytes, const int32_t* lens, int32_t n,
int32_t padded_vocab, const int32_t* stops, int32_t nstops, char*, int)` ·
`tfso_compile_json_schema(ctx, const char* schema, int strict, char*, int)`
(`any_whitespace=false` fixed) · `tfso_compile_regex` ·
`tfso_compile_grammar` · `tfso_grammar_bytes(grammar) -> size_t` ·
`tfso_matcher_create(ctx, grammar)` · `tfso_fill_bitmask(matcher, int32_t* buf)
-> int` (the need-apply flag) · `tfso_accept_token(matcher, id) -> int` ·
`tfso_terminated(matcher) -> int` · `tfso_rollback(matcher, n) -> int`
(**v2 only**; nothing calls it with drafts off) · `tfso_free_*`.

Kind mapping: `json` → the schema `{"type":"object"}`; `json_schema` → strict
compile; `regex`; `grammar`; `choice` → an EBNF alternation generated from the
JSON array, quote-escaped. Bitmask: int32 words, bit = 1 allowed, `(vocab+31)/32`.

## 5. Build changes (`zig/Dockerfile`)

- `XGRAMMAR_REF` is pinned in `scripts/config.sh` to
  `4d145cc13d878c751ebeed36af1c013074be76bc` (`v0.2.2`), passed by
  `scripts/prepare.sh` as `--build-arg`, and echoed by `image_hash` — the
  `TF_REF` pattern; no Dockerfile default.
- The clone is `--recursive` (only `3rdparty/dlpack` is needed for the C++
  core); the build is CMake with **`-DXGRAMMAR_BUILD_PYTHON_BINDINGS=OFF`**
  (the 0.2.2 option name), `-DCMAKE_BUILD_TYPE=Release`, an install prefix, and
  `cmake --install`. It runs **before** the engine's `zig build`.
- `build.zig` compiles the shim with a **system g++ step** and `ar`, links
  `libtfso.a` + `libxgrammar.a` + **libstdc++** + **libgcc** (exception
  unwinding) against glibc; `linkLibCpp` stays off.
- `prepare.sh`'s build context carries `zig/patches` **and
  `zig/patches-inputs`**; the Dockerfile's header comment is corrected.
- The image hash changes by itself.

## 6. Switches and defaults

| Switch | Default | Meaning |
| --- | --- | --- |
| `TF_DS_SO` | `1` | `0`: `Info.structures` false, the old 400 |
| `TF_DS_SO_CACHE_BYTES` | `268435456` | compile cache bound; `0`: compile per request |
| `TF_DS_SO_TIMEOUT_S` | `20` | a compile past it is a 400 at admission |
| `TF_DS_SO_COMPILE_THREADS` | `2` | concurrent compiles; the rest wait in admission |
| `TF_DS_SO_FAULT` | unset | `accept`: inject a matcher failure (test only) |

## 7. Budget

Boot: the tokenizer-info build on rank 0, one-time (tens of MiB host). Per
schema: a ms-scale worker compile, cached. Per constrained round: a tens-of-µs
matcher advance + fill, a ≤17 KiB H2D, one kernel launch. Expected under 1% of
the decode step; the real cost is drafts-off on constrained requests, measured
and recorded in the patch header.

## 8. Host tests (`zig build test`)

The shim needs xgrammar on the host: `zig build test
-Dxgrammar=$HOME/opt/xgrammar-0.2.2`; **without the flag the shim tests are
skipped with a printed note** and the rest of the suite runs. With it: a fake
20-token vocab (init, each compile kind, fill, accept); bitmask word/bit math;
LRU eviction, byte bound, cache off, concurrent-compile dedupe; the whitespace
phase (`\n\n{` accepted, a 5th whitespace character refused); the state machine
(idle → whitespace → engaged → completed → all-ones); `choice` alternation and
escaping; the compile-error message out-parameter; special tokens never in the
mask.

## 9. Dev loop

Engine checkout `ai:~/repos/tensorfold-dsv41-dev`: `TF_REF`, then `0001`–`0033`
applied in filename order, **committed as a base commit**, then a
`structured-output` branch carries the work. Zig 0.17.0 installed at
`~/opt/zig-0.17.0` (the pinned tarball, no sudo). xgrammar built once at
`~/opt/xgrammar-0.2.2` (CMake flags above). Export is
`scripts/port-patches.sh --engine ~/repos/tensorfold-dsv41-dev --branch
structured-output --base <base> --mode squash --name structured-output` — it
verifies the base, applies every patch, and only then writes
`zig/patches/0034-*.patch` plus the README row. The serving build stays in the
image via `scripts/prepare.sh`.

## 10. Live gates (the pair)

0. Record `tools/exact.py` on the **current `main` image** before deploying
   (it only covers greedy, thinking-off, non-streaming requests; record with
   `--drafts on`, compare with `--like`).
1. Build the image with the feature present and **`TF_DS_SO=0`**; run
   `exact.py` against the recorded baseline. This is the checkpoint: the shim,
   the extra kernel launch path and the link changes are exercised with the
   feature off, and the bits must match.
2. Turn the feature on: `./stop.sh` + `./start.sh` with `/mnt/nvmeof` healthy.
3. `exact.py` again — unconstrained traffic still byte-identical with
   `GRAPHS=1` and drafts on.
4. `tools/structure_probe.py`: flat, nested, enum, long-string, regex, choice
   specs × thinking on/off × stream/non-stream, every reply validated; the
   formerly-400 request now 200; malformed schema → 400; an unsupported
   Lark/regex construct → 400 with the message; spec beside `tool_choice`
   "required" → 400; spec + `tools` + "auto" → constrained content, no calls;
   `\n\n` after think-close completes; `strict:false` accepted and ignored;
   unsatisfiable schema → `length`; `TF_DS_SO_FAULT=accept` → the error frame;
   a cancel mid-decode frees the matcher; mixed 4-stream run with plain replies
   byte-identical.
5. Sampler sanity with `-inf` entries: penalties + top-p + min-p, and a mask
   that leaves a single allowed token (no NaN, no crash).
6. `TF_DS_SO=0` restores the old 400; a rehearsed rollback (§11) is performed
   once.
7. `/metrics` counters present.

## 11. Deploy and rollback

Deploy: branch worktree `~/repos/dsv41-structured-output`, `prepare.sh`,
`stop.sh`, `start.sh`, the gates above. Rollback (rehearsed once): `main` back
on the live checkout — the image tag resolves to the old hash, still cached —
`./stop.sh` + `./start.sh`. No re-download.

## 12. Risks and fallbacks

- **xgrammar's standalone build is verified first**; if it fights us, llguidance
  drops in (plain C, no C++ ABI question), same Zig-side design.
- **The TP vocabulary boundary is not 32-aligned** — the slice carries a bit
  offset; the message is sized and tested, and sent only in constrained rounds.
- **The ring slot's capacity** is checked; the patch grows it in the ring's own
  terms if needed.
- **The whitespace phase** must accept the model's `\n\n` and refuse unbounded
  whitespace; both are host tests and probes.
- **Drafting exclusion** is verified across every admission path (`loneFits` is
  only one), and constrained rows never share a drafted round.
- **Some `guided_*` grammars now 400 at compile time** — intended, documented.

## 13. Out of scope (follow-ups)

bifrost-proxy passing `response_format` through to `chronos-local`; per-draft
masks (v2, `tfso_rollback` reserved); honouring a client's `strict: false`;
offering the patch upstream.

## 14. Credits

The enforcement contract follows DeepInfer's `structure_output` design
(deepseek-ai/DeepInfer, Apache-2.0). The grammar engine is xgrammar (mlc-ai,
Apache-2.0) at v0.2.2.
