# Structured output — the design of `zig/patches/0034`

Branch `engine-structured-output`. Revision 2, 2026-10-30: every item of the
spec review is integrated below (toolchain, graphs, TP placement, compile
threading, vocab type, whitespace, strict, cache key, failure surfacing,
matcher lifetime, host-test wiring, rollback, XGRAMMAR_REF wiring, build
context, line-number scope, tools interplay). The patch lands as
`zig/patches/0034-structured-output.patch` (one file, applied after `0033` in
filename order), a row in `zig/patches/README.md`, a `zig/Dockerfile` edit, a
probe in `tools/`, and this document.

## 0. Status

The design decisions below were approved on 2026-10-30. One item needs explicit
sign-off (§1, "surface"): enabling the engine's capability flag accepts more
fields than the originally approved `response_format` minimum.

## 1. The decisions

| Decision | Choice |
| --- | --- |
| Grammar engine | **xgrammar 0.2.2**, pinned by git SHA. **Both** xgrammar and the shim are built with the image's **g++ against libstdc++**; the engine's final link adds libstdc++ and does **not** link libc++ (`linkLibCpp` off). One C++ runtime, one `std::string`/`std::vector` layout — no objects cross between two runtimes. The shim's `extern "C"` surface carries only POD (byte buffers, int32 masks, opaque handles), so nothing C++ crosses into Zig. **llguidance** (plain C API, no C++ objects at all) is the named fallback if xgrammar's standalone build fails. Runtime cost is a wash at this serving scale (<1% of a 30–60 ms step either way); xgrammar wins on build footprint and on contract parity with DeepInfer, whose integration recipe this follows. |
| Request surface | `response_format` `json_object` / `json_schema` (the approved minimum). **Awaiting explicit sign-off:** the server parses the whole structured-output surface as one unit behind `Info.structures` (§2), so enabling it also accepts `guided_json` / `guided_regex` / `guided_choice` / `guided_grammar` and `structured_outputs`. A superset — refusing fields the engine already parses adds code for no benefit. **Known subset limit:** xgrammar accepts JSON Schema, its own regex dialect (Rust-style) and EBNF/GBNF-style grammars; it does **not** take Lark grammars that vLLM's `guided_grammar` accepts, and its regex support is a subset. Such requests will now get a **400 at compile time** with the compiler's message instead of today's blanket 400 — a behaviour change for `guided_*` clients only. |
| Drafts | A constrained request never drafts: plain decode for its lifetime. All other streams keep DSpark drafts. |
| Validation | Full live gates on the pair (§10). |

## 2. What exists at `TF_REF` (629c524) — the patch changes none of it

Line numbers below refer to a **clean `TF_REF` checkout, before any recipe
patch**; after `0001`–`0033` they may shift by a few lines, so implementation
locates by symbol, not by line.

- `zig/src/server/grammar.zig` (101 lines) parses the whole surface:
  `response_format` (`text`, `json_object`, `json_schema`), the four `guided_*`
  fields, and `structured_outputs` (`json`, `json_object`, `regex`, `choice`,
  `grammar`); a malformed spec is a 400 at parse; `grammar.refusal` refuses a
  spec beside `tool_choice` "required"/named.
- `zig/src/server/server.zig` (`checkFeatures`, around :177-178): the only gate
  — `if (!srv.info.structures) return cx.refuse(...)` — then it builds
  `request.structure = .{ .kind = s.kind, .text = s.text, .after = if (thinking)
  srv.text.tokenId(srv.markers.close) else null }`. The think-boundary token is
  already wired.
- `zig/src/core/engine_api.zig`: `Structure { kind, text, after: ?u32 }`;
  `Request.structure`; `Info.structures = false`. `Structure` has **no strict
  field**.
- `zig/src/core/lane_host.zig` (`loneFits`, around :397): `r.structure != null`
  already keeps a request off the lone drafted fast path. The lane core runs on
  its own thread (`lane_host.zig:104`).
- `zig/src/server/anthropic_translate.zig` (~:270): the Anthropic frontend maps
  its response format into the chat body — one gate serves both frontends.
- `zig/src/server/tool_parse.zig` (~:228,:233): JSON that is not a tool call is
  already the reply's content.
- `zig/src/server/chat.zig:469-470`: `.cancelled => "cancelled"`,
  `.failed => cx.other(...)` — a failed reply is an **error reply**, not a
  finish reason.
- Nothing enforces a grammar: no family sets `Info.structures`, and
  `build.zig.zon` has zero dependencies.

## 3. What the patch adds — the enforcement side

1. **`Info.structures = true`** in the dsv41 host's `Info` construction, the
   way `0004-call-gate` made the host advertise `call_gates`. Kill switch:
   `TF_DS_SO=0` at boot leaves the flag false and the old 400 in place.
2. **`zig/src/core/structure.zig`** (new, family-agnostic): compile cache +
   matcher lifecycle.
   - Cache keyed on **(kind, full text)** — the text is stored beside the
     grammar and compared on a hash hit, so a collision can never return the
     wrong grammar. Byte-bounded (`TF_DS_SO_CACHE_BYTES`, default 256 MiB, 0 =
     no cache). The library's own cache stays off (the unbounded-cache DoS,
     CVE-2025-32381).
   - **Compiles run on a worker thread, before the response headers go out**,
     never on the lane thread that runs decode. Concurrent compiles are capped
     (`TF_DS_SO_COMPILE_THREADS`, default 2); a request whose compile is still
     queued waits in admission, bounded. On `TF_DS_SO_TIMEOUT_S` the request is
     a 400 and its worker's compile runs to completion in the background
     (xgrammar cannot cancel) — acceptable because real schemas compile in
     milliseconds and the bound only catches pathological input; the worker
     pool is capped, so the burn is bounded too.
   - Matcher state machine: idle until the `after` token is emitted (the think
     block is unconstrained), then advance/fill; on grammar completion only
     the stop token is allowed, then an all-ones mask for whatever follows EOS.
   - **Failure surfacing, defined:** a matcher failure after headers is
     `Reason.failed` with the message — non-streaming gets the engine's
     existing error reply (`chat.zig:470`), streaming terminates the SSE stream
     with an error frame (`data: {"error": …}`) and closes, as vLLM does.
     Counted as `so_aborted_total`. In practice this is near-unreachable in v1:
     with drafts off and special tokens excluded from the mask, `accept_token`
     has nothing to reject.
   - **Matcher lifetime:** owned by the stream; freed on finish and on cancel;
     it survives a pause because it is per-stream state (patch `0005`'s
     prefill interleave runs other streams' rounds in between — the structured
     stream's matcher simply idles). It is touched only by the lane thread that
     runs that stream's rounds, so no lock of its own.
   - **`strict`:** v1 always compiles `strict_mode=true`. A client's
     `strict: false` is ignored, documented in the patch header and the probe.
     `Structure` has no strict field; extending it upstream is deferred.
   - **Whitespace:** JSON/JSON-Schema compile with **compact whitespace**
     (`any_whitespace=false`; the exact option name is verified against the
     pinned 0.2.2 headers at implementation), and the grammar's root allows
     leading whitespace (`\s*`) so `\n\n` after the think-close token before
     `{` is accepted. This is the fix for the known "newlines until length"
     failure; the probe reproduces that case.
   - **`response_format` + `tools` + `tool_choice: "auto"`:** intended
     behaviour — the reply's content is constrained, so the model emits no
     tool calls for that request. Documented in the patch header and the probe.
3. **`zig/src/families/dsv41/structure.zig`** (new): per-stream matcher
   registry, the boot-time vocab export, the round hook.
4. **The mask point, and CUDA graphs.** The served sampler
   (`zig/src/families/dsv41/sampling.zig`, `sampleRow`/`sample_rows` of one
   row's fp32 logits on the device) reads only the row's top `top_k + MARGIN`
   candidates from the GPU, so masking **must run on the device, on the full
   logits row, before that read**. Decode stretches are captured as CUDA graphs
   (`round.zig` `Graphs`: one capture per rows/context bucket/stretch), so the
   mask kernel is **captured inside the stretch**:
   - a persistent per-lane device bitmask buffer (allocated when a stream is
     constrained) and a per-row enable flag;
   - the host fills the mask and H2D-copies it **before the graph launch** each
     round (≤ ~17 KiB at a ~131k vocab — microseconds, on the same stream ahead
     of the launch);
   - the captured kernel is gated by the flag: flag 0 → it returns immediately,
     so unconstrained rows keep their bits; flag 1 → `logits[i] = -inf` where
     the bit is 0;
   - graph keys are unchanged (same rows/bucket/stretch/shard), so no new
     captures and no cache growth. **Cost:** one gated kernel node per captured
     stretch (a no-op when the flag is 0, ~1–2 µs) plus one H2D per constrained
     lane per round. §10's "graphs untouched" claim is therefore corrected:
     graphs gain that one gated node; unconstrained replies stay byte-identical.
5. **Vocab export at boot**, once, **on rank 0 only**: the engine tokenizer
   (`zig/src/core/tokenizer/`) supplies every token's exact decoded bytes, the
   special/stop ids, the model's padded vocab size, and `vocab_type = RAW` —
   the bytes are already decoded, so a byte-level type would decode them a
   second time. This is the whole tokenizer coupling.
6. **Drafts.** `loneFits` covers the lone fast path only; the family's round and
   draft admission must skip structured streams everywhere. Verified during
   implementation (`lanes.zig` / `round.zig`); the condition is added where it
   is missing.
7. **TP=2 — one matcher, on rank 0.** The vocabulary is split across ranks
   (`layer_main.zig` `vocabOf`: "the ranks' vocabulary parts"), so:
   - rank 0 owns the compile and the matcher, fills the **full** bitmask, and
     sends rank 1 its slice over the existing RDMA ring (≤ ~8.5 KiB, the same
     path the decode-size exchanges use);
   - each rank masks its own vocabulary part before its `sampleDev`; the
     existing candidate exchange and per-step agreement guard are unchanged, and
     the two slices are identical by construction, so the ranks cannot diverge
     on a masked row;
   - **rank 1 never initializes xgrammar**: it receives masks, nothing more.
     That removes the lockstep-matcher risk and halves the boot cost.
8. **`/metrics`**: `so_requests_total`, `so_aborted_total`, `so_cache_bytes`,
   `so_compile_ms`, `so_compile_threads_busy` (naming matched to the engine's
   existing style).

## 4. The shim's C surface

`tfso_init(encoded_vocab, vocab_type=RAW, padded_vocab_size, stop_token_ids,
count) → ctx` · `tfso_compile_json_schema(ctx, text, strict, any_whitespace)` ·
`tfso_compile_regex` · `tfso_compile_grammar` (EBNF) · `tfso_matcher_create` ·
`tfso_fill_bitmask(matcher, int32* buf)` · `tfso_accept_token(matcher, id)` ·
`tfso_rollback(matcher, n)` · `tfso_finished` · `tfso_free_*`. Errors return a
code + message buffer; Zig turns them into that request's failure. Kind
mapping: `json` → the schema `{"type":"object"}` (compact whitespace);
`json_schema` → strict compile (as DeepInfer); `regex` → regex; `grammar` →
EBNF; `choice` → an EBNF alternation generated from the JSON array,
quote-escaped. **`tfso_rollback` is kept but is v2-only**: with drafts off
nothing calls it in v1; it exists for the per-draft-step masks that a later
patch will need.

The shim's `.cc`/`.h` ship **inside the patch** as new files in the engine tree
(`zig/src/families/dsv41/structure/`), so the build context is unchanged.

## 5. Build changes (`zig/Dockerfile`)

- `ARG XGRAMMAR_REF` is **wired through all three files** the way `TF_REF` is:
  `scripts/config.sh` pins it, `scripts/prepare.sh` passes it as `--build-arg`,
  and `config.sh`'s `image_hash` echoes it — no Dockerfile default, so the hash
  covers it exactly like the engine pin.
- A `RUN` clones `mlc-ai/xgrammar` at that ref **`--recursive`** (its kernels
  live in submodules), builds its C++ core with the image's **g++** (CMake,
  **Python bindings off**), producing `libxgrammar.a`. This step runs **before**
  the engine's `zig build`.
- The patched engine's `build.zig` compiles the shim against those headers with
  g++ and links `libtfso.a` + `libxgrammar.a` + **libstdc++** into the native
  target (no `linkLibCpp`).
- `prepare.sh`'s build context carries `zig/patches` **and `zig/patches-inputs`**;
  the Dockerfile's header comment is corrected to say so.
- The image hash changes by itself.

## 6. Switches and defaults

| Switch | Default | Meaning |
| --- | --- | --- |
| `TF_DS_SO` | `1` | `0`: `Info.structures` stays false; the old 400, the old bits |
| `TF_DS_SO_CACHE_BYTES` | `268435456` | the compile cache's byte bound; `0`: compile per request |
| `TF_DS_SO_TIMEOUT_S` | `20` | a compile that outlives it is a 400 at admission |
| `TF_DS_SO_COMPILE_THREADS` | `2` | concurrent compiles; the rest wait in admission |

No recipe-level serving setting: the feature is request-driven. The README
table row says so.

## 7. Budget

Boot: the tokenizer-info build on rank 0, one-time, tens of MiB of host memory.
Per schema: a ms-scale compile on a worker thread, cached. Per constrained row
per round: a tens-of-µs matcher advance + fill, a ≤17 KiB H2D ahead of the
graph launch, a gated elementwise kernel inside the captured stretch. Expected
under 1% of the decode step; the real cost is drafts-off on the constrained
requests themselves, measured and recorded in the patch header (the repo's
culture).

## 8. Host tests (in the patch; `zig build test`)

The shim is C++ and needs xgrammar on the host: `zig build test
-Dxgrammar=<prefix>` points at a local build (on `ai`: `~/opt/xgrammar-0.2.2`,
built once with g++/CMake, Python off, `--recursive`). Without the flag the
shim tests are **skipped with a printed note** so the rest of the suite still
runs. With it: a fake ~20-token vocab through the shim (init, each compile
kind, fill, accept); the bitmask word/bit math against a reference; LRU
eviction and the byte bound; cache off; the `after` state machine (idle →
active → completed → post-EOS all-ones); `choice` → alternation generation and
escaping; compact-whitespace acceptance of `\n\n{`; the compile-failure error
path.

## 9. Dev loop

The engine checkout is `ai:~/repos/tensorfold-dsv41-dev` at `TF_REF` (already
created, clean). Apply `0001`–`0033` in filename order, develop on top, export
`git diff` as `0034` (it must `git apply --check` on a clean `TF_REF` after
`0033`). Host tests need Zig 0.17.0 on `ai` — the pinned tarball `config.sh`
already names, installed at `~/opt/zig-0.17.0` (no sudo). The serving build
stays in the image via `scripts/prepare.sh`, never by hand.

## 10. Live gates (the pair)

1. `scripts/prepare.sh` (the image rebuilds; first boot ~10 min) →
   `./stop.sh` + `./start.sh`, with `/mnt/nvmeof` healthy.
2. `tools/exact.py`: unconstrained traffic byte-identical with `GRAPHS=1` and
   drafts on — the gated kernel node must not perturb a single token.
3. New `tools/structure_probe.py`: flat, nested, enum, long-string, regex and
   choice specs × thinking on/off × stream/non-stream, every reply validated
   against its schema; the formerly-400 request now 200; a malformed schema is
   still a 400; an unsupported Lark grammar or regex construct is a 400 with
   the compiler's message; a spec beside `tool_choice` "required" is still a
   400; a spec with `tools` + "auto" yields constrained content and no calls;
   the newline-until-length case (a schema where the model wants `\n\n` after
   the think block) completes; a mixed run (structured + plain, 4 streams) with
   the plain replies still byte-identical; a mid-stream matcher failure (forced
   by an unsatisfiable schema) ends with the error frame, not a hang.
4. `/metrics` counters present; an unsatisfiable schema ends its own request
   with the server and the other streams healthy.

## 11. Deploy and rollback

Deploy is the recipe's normal path: the branch worktree
`~/repos/dsv41-structured-output`, `prepare.sh`, `stop.sh`, `start.sh`, the
gates above. Rollback: `main` back on the live checkout — the image tag
resolves to the old hash, still cached locally — `./stop.sh` + `./start.sh`.
No re-download, no kit rebuild.

## 12. Risks and fallbacks

- **xgrammar's standalone C++ build is the first thing to verify.** If its
  CMake core fights us, llguidance drops in: plain C API (no C++ ABI question
  at all), same Zig-side design, identical runtime cost.
- **The TP=2 head layout** is read, not assumed: slice-masking per rank as
  §3.7 says; the candidate-exchange and agreement path is walked and its
  unchanged behaviour confirmed.
- **Drafting exclusion scope**: `loneFits` is one path; the family's admission
  is walked and the condition added where it is missing.
- **xgrammar's grammar/regex subset** means some `guided_*` requests that
  vLLM would accept now get a compile-time 400 — intended, documented, probed.
- **Compact whitespace** must still accept the model's `\n\n` before `{`; the
  root's leading `\s*` covers it, probed.

## 13. Out of scope (follow-ups)

- bifrost-proxy passing `response_format` through to `chronos-local` instead of
  the forced-tool rewrite (the Eternity repo, once this serves).
- Per-draft-step masks (a v2 that keeps drafts on constrained requests; the
  rollback surface is already in the shim for it).
- Honouring a client's `strict: false` (needs a `Structure` field upstream).
- Offering the patch upstream (MiaAI-Lab / bertholomus) once the gates pass.

## 14. Credits

The enforcement contract follows DeepInfer's `structure_output` design
(deepseek-ai/DeepInfer, Apache-2.0): the int32 `1 = allowed` bitmask applied to
logits before sampling, the post-EOS all-ones mask, rollback bookkeeping, and
the byte-bounded compile cache. The grammar engine is xgrammar (mlc-ai,
Apache-2.0), pinned at 0.2.2 — the same engine DeepInfer and vLLM's default
backend use.
