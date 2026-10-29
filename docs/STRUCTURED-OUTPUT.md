# Structured output — the design of `zig/patches/0034`

Branch `engine-structured-output`. This is the design the patch implements; the
decisions below were approved on 2026-10-30. The patch lands as
`zig/patches/0034-structured-output.patch` (one file, applied after `0033` in
filename order), a row in `zig/patches/README.md`, a `zig/Dockerfile` edit, a
probe in `tools/`, and this document.

## 1. The decisions

| Decision | Choice |
| --- | --- |
| Grammar engine | **xgrammar 0.2.2**, pinned by git SHA, behind a ~400-LOC C++ shim (`extern "C"`). **llguidance** (C API) is the named fallback if xgrammar's standalone C++ build fails. Runtime cost is a wash at this serving scale (≤4 streams, 20–60 tok/s: under ~200 mask calls/s against a 30–60 ms decode step, <1% either way); xgrammar wins on build footprint (no Rust toolchain, the image's existing g++, the repo's git-pin pattern) and on contract parity with DeepInfer, whose integration recipe this follows. |
| Request surface | `response_format` `json_object` / `json_schema` (the approved minimum). **Accepted deviation:** the server parses the whole structured-output surface as one unit behind `Info.structures` (§2); enabling it also accepts `guided_json` / `guided_regex` / `guided_choice` / `guided_grammar` and `structured_outputs`. A superset of the approval — writing refusal code for fields the engine already parses correctly adds risk for no benefit. |
| Drafts | A constrained request never drafts: plain decode for its lifetime. All other streams keep DSpark drafts. |
| Validation | Full live gates on the pair (§10). |

## 2. What exists at `TF_REF` (629c524) — the patch changes none of it

- `zig/src/server/grammar.zig` (101 lines) parses the whole surface:
  `response_format` (`text`, `json_object`, `json_schema`), the four `guided_*`
  fields, and `structured_outputs` (`json`, `json_object`, `regex`, `choice`,
  `grammar`); a malformed spec is a 400 at parse; `grammar.refusal` refuses a
  spec beside `tool_choice` "required"/named.
- `zig/src/server/server.zig:177-178` (`checkFeatures`): the only gate —
  `if (!srv.info.structures) return cx.refuse("structured output … is not
  supported by this engine yet")` — then it builds
  `request.structure = .{ .kind = s.kind, .text = s.text, .after = if (thinking)
  srv.text.tokenId(srv.markers.close) else null }`. The think-boundary token is
  already wired.
- `zig/src/core/engine_api.zig`: `Structure { kind: {json, json_schema, regex,
  choice, grammar}, text, after: ?u32 }`; `Request.structure`; `Info.structures
  = false` (line 129).
- `zig/src/core/lane_host.zig:397` (`loneFits`): `r.structure != null` already
  keeps a request off the lone drafted fast path.
- `zig/src/server/anthropic_translate.zig:270`: the Anthropic frontend maps its
  response format into the chat body — one gate serves both frontends.
- `zig/src/server/tool_parse.zig:228,233`: JSON that is not a tool call is
  already the reply's content.
- Nothing anywhere enforces a grammar: no family sets `Info.structures`, and
  `build.zig.zon` has zero dependencies. The 400 the proxy sees today is the
  dsv41 family simply not opting in.

## 3. What the patch adds — the enforcement side

1. **`Info.structures = true`** in the dsv41 host's `Info` construction, the
   way `0004-call-gate` made the host advertise `call_gates`. Kill switch:
   `TF_DS_SO=0` at boot leaves the flag false and the old 400 in place.
2. **`zig/src/core/structure.zig`** (new, family-agnostic): compile cache +
   matcher lifecycle. An LRU keyed on (kind, text) hash, byte-bounded
   (`TF_DS_SO_CACHE_BYTES`, default 256 MiB, 0 = no cache); per-request matcher
   state machine — idle until the `after` token is emitted (the think block is
   unconstrained), then advance/fill/rollback; on grammar completion only the
   stop token is allowed, then an all-ones mask for whatever follows EOS (the
   engine's stop handling is untouched); a matcher error finishes only that
   request (`Reason.failed`), never another stream.
3. **`zig/src/families/dsv41/structure.zig`** (new): per-stream matcher
   registry; the boot-time vocab export; the round hook.
4. **The mask point.** The served sampler (`zig/src/families/dsv41/sampling.zig`,
   `sample_rows` of one row of fp32 logits on the device — `layer_main.zig:651`
   is the documented call site) reads only the row's top `top_k + MARGIN`
   candidates from the GPU, so masking **must run on the device, on the full
   logits row, before that read**: advance the matcher with the last accepted
   token → fill the int32 bitmask (bit = 1 means allowed — DeepInfer's format)
   → H2D (≤ ~17 KiB at a ~131k vocab) → a small elementwise kernel sets
   `logits[i] = -inf` where the bit is 0 → the sampler runs exactly as before.
5. **Vocab export at boot**, once: the engine tokenizer
   (`zig/src/core/tokenizer/`) supplies every token's exact decoded bytes, the
   special/stop ids, the vocab type, and the model's *padded* vocab size to the
   shim's `TokenizerInfo`. This is the whole tokenizer coupling.
6. **Drafts.** `loneFits` covers the lone fast path only; the family's round and
   draft admission must skip structured streams everywhere. Verified during
   implementation (`lanes.zig` / `round.zig`); the condition is added where it
   is missing.
7. **TP=2.** The mask applies wherever the full row exists. If the lm-head is
   vocab-split across ranks, each rank masks its slice of the same bitmask
   (deterministic: both ranks compile the same grammar). The exact layout is
   read from the head path and written into the patch header.
8. **`/metrics`**: `so_requests_total`, `so_aborted_total`, `so_cache_bytes`,
   `so_compile_ms` (naming matched to the engine's existing style).

## 4. The shim's C surface

`tfso_init(encoded_vocab, vocab_type, padded_vocab_size, stop_token_ids,
count) → ctx` · `tfso_compile_json_schema(ctx, text, strict) → grammar` ·
`tfso_compile_regex` · `tfso_compile_grammar` (EBNF) · `tfso_matcher_create` ·
`tfso_fill_bitmask(matcher, int32* buf)` · `tfso_accept_token(matcher, id)` ·
`tfso_rollback(matcher, n)` · `tfso_finished` · `tfso_free_*`. Errors return a
code + message buffer; Zig turns them into that request's failure. Kind
mapping: `json` → the schema `{"type":"object"}`; `json_schema` → strict
compile (as DeepInfer); `regex` → regex; `grammar` → EBNF; `choice` → an EBNF
alternation generated from the JSON array, quote-escaped. The compile cache is
owned by the Zig side (the bound and the metrics are ours); the library's own
cache stays off — the unbounded-cache DoS (CVE-2025-32381) is the lesson.

The shim's `.cc`/`.h` ship **inside the patch** as new files in the engine tree
(`zig/src/families/dsv41/structure/`), so `scripts/prepare.sh`'s build context
(Dockerfile + `zig/patches` only) is unchanged.

## 5. Build changes (`zig/Dockerfile`)

- `ARG XGRAMMAR_REF` (a full SHA, like `TF_REF`); a `RUN` clones
  `mlc-ai/xgrammar` at that ref and builds its C++ core with the image's
  existing g++ (no Python bindings) into `libxgrammar.a`.
- The patched engine's `build.zig` compiles the shim against it and links
  `libtfso.a` + xgrammar + libc++ into the native target.
- The image hash changes by itself (`config.sh`'s `image_hash` cats the patches
  and the Dockerfile).

## 6. Switches and defaults

| Switch | Default | Meaning |
| --- | --- | --- |
| `TF_DS_SO` | `1` | `0`: `Info.structures` stays false; the old 400, the old bits |
| `TF_DS_SO_CACHE_BYTES` | `268435456` | the compile cache's byte bound; `0`: compile per request |
| `TF_DS_SO_TIMEOUT_S` | `20` | a compile that outlives it is a 400 at admission |

No recipe-level serving setting: the feature is request-driven. The README
table row says so.

## 7. Budget

Boot: the tokenizer-info build, one-time, tens of MiB of host memory. Per
schema: a ms-scale compile, cached. Per constrained row per round: a tens-of-µs
matcher advance + fill, a ≤17 KiB H2D, a one-pass elementwise kernel. Expected
under 1% of the decode step; the real cost is drafts-off on the constrained
requests themselves, measured and recorded in the patch header (the repo's
culture) — plain decode is what a constrained request would need anyway to be
exactly right.

## 8. Host tests (in the patch; `zig build test`, no nvcc)

A fake ~20-token vocab through the shim (init, each compile kind, fill,
accept, rollback); the bitmask word/bit math against a reference; LRU eviction
and the byte bound; cache off; the `after` state machine (idle → active →
completed → post-EOS all-ones); `choice` → alternation generation and escaping;
the compile-failure error path.

## 9. Dev loop

The engine checkout is `ai:~/repos/tensorfold-dsv41-dev` at `TF_REF` (already
created, clean). Apply `0001`–`0033` in filename order, develop on top, export
`git diff` as `0034` (it must `git apply --check` on a clean `TF_REF` after
`0033`). Host tests need Zig 0.17.0 on `ai` — the pinned tarball `config.sh`
already names, installed at `~/opt/zig-0.17.0` (no sudo, no host pollution
beyond that dir). The serving build stays in the image via
`scripts/prepare.sh`, never by hand.

## 10. Live gates (the pair)

1. `scripts/prepare.sh` (the image rebuilds; first boot ~10 min) →
   `./stop.sh` + `./start.sh`, with `/mnt/nvmeof` healthy.
2. `tools/exact.py`: unconstrained traffic byte-identical — the patch is a
   no-op without the fields, drafts and graphs untouched.
3. New `tools/structure_probe.py`: flat, nested, enum, long-string, regex and
   choice specs × thinking on/off × stream/non-stream, every reply validated
   against its schema; the formerly-400 request now 200; a malformed schema is
   still a 400; a spec beside `tool_choice` "required" is still a 400; a mixed
   run (structured + plain, 4 streams) with the plain replies still
   byte-identical.
4. `/metrics` counters present; an unsatisfiable schema ends its own request
   (`length`/`failed`) with the server and the other streams healthy.

## 11. Deploy and rollback

Deploy is the recipe's normal path: the branch checked out (or its worktree
`~/repos/dsv41-structured-output`), `prepare.sh`, `stop.sh`, `start.sh`, the
gates above. Rollback: `main` back on the live checkout — the image tag
resolves to the old hash, which is still cached locally — `./stop.sh` +
`./start.sh`. No re-download, no kit rebuild.

## 12. Risks and fallbacks

- **xgrammar's standalone C++ build is the first thing to verify.** If its
  CMake core fights us, llguidance drops in: same Zig-side design, the shim
  swapped for its C API, identical runtime cost.
- **The TP=2 head layout** is read, not assumed: slice-masking per rank if the
  vocab is split.
- **Drafting exclusion scope**: `loneFits` is one path; the family's admission
  is walked and the condition added where it is missing.
- **`json_object` with thinking off** must still emit valid JSON from token 0 —
  the probe covers it.

## 13. Out of scope (follow-ups)

- bifrost-proxy passing `response_format` through to `chronos-local` instead of
  the forced-tool rewrite (the Eternity repo, once this serves).
- Per-draft-step masks (a v2 that keeps drafts on constrained requests).
- Offering the patch upstream (MiaAI-Lab / bertholomus) once the gates pass.

## 14. Credits

The enforcement contract follows DeepInfer's `structure_output` design
(deepseek-ai/DeepInfer, Apache-2.0): the int32 `1 = allowed` bitmask applied to
logits before sampling, the post-EOS all-ones mask, rollback bookkeeping, and
the byte-bounded compile cache. The grammar engine is xgrammar (mlc-ai,
Apache-2.0), pinned at 0.2.2 — the same engine DeepInfer and vLLM's default
backend use.
