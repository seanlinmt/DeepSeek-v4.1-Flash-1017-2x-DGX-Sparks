# Resume: structured output (patch 0034)

**Read these first, in order**

1. `docs/STRUCTURED-OUTPUT.md` — the design (rev 4: the shim is a runtime-loaded shared library).
2. `docs/superpowers/plans/2026-10-30-structured-output.md` — the 13-task plan.
3. `.superpowers/sdd/2026-10-30-structured-output/progress.md` — the ledger: every task's state and every ruling.

**Branches and checkouts**

| What | Where |
|---|---|
| Recipe repo (this one) | `ai:~/repos/dsv41-structured-output`, branch `engine-structured-output` (worktree of the main checkout; `main` stays untouched on `ai:~/repos/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks`) |
| Engine repo | `ai:~/repos/tensorfold-dsv41-dev`, branch `structured-output`; `so-base` tags `TF_REF 629c524` + every `zig/patches/*.patch` (30 files) |
| Engine commits | `6b3a0c3` shim + build step · `b1829fa` compile cache · `9a567ea` vocab export · `2f867ea` admission compile · `db0ec1e` mask arithmetic |
| Toolchain on `ai` | `~/opt/zig-0.17.0` (pinned tarball), `~/opt/xgrammar-0.2.2` (v0.2.2 = `4d145cc13d878c751ebeed36af1c013074be76bc`, CMake Release, `-DXGRAMMAR_BUILD_PYTHON_BINDINGS=OFF`) |

**Verification command (both directions)**

```bash
cd ~/repos/tensorfold-dsv41-dev
~/opt/zig-0.17.0/zig build test -Dxgrammar=$HOME/opt/xgrammar-0.2.2   # 337/339, 2 pre-existing spill failures
~/opt/zig-0.17.0/zig build test                                        # shim tests skip (4 skipped)
```

The two failures (`spill.store_test … write faults`, `spill.fsio.test … partial direct write`) are in
untouched modules and existed before this work.

**Done**

- xgrammar 0.2.2 built standalone; a POD-only C shim (`zig/src/core/structure/tfso.{h,cc}`) built by g++
  into `libtfso.so` (static libstdc++/libgcc inside), loaded with `std.DynLib` (`core/structure/api.zig`).
- Compile cache (`core/structure.zig`): `(kind, full text)` keys compared as strings, byte-bounded LRU,
  in-flight dedupe, worker-thread compiles, `TF_DS_SO_TIMEOUT_S`, choice→EBNF generation.
- Vocabulary export (`server/structure.zig`): per-id decode, specials blocked with empty bytes, EOS stops,
  `TF_DS_SO` / `TF_DS_SO_CACHE_BYTES` readers.
- Admission (`server.zig` + `engine_api.zig`): `Structure.handle`, compile in `checkFeatures` before any
  header, `cx.refuse` on timeout or compile failure, `So` freed in `deinit`.
- Mask arithmetic (`families/dsv41/structure_mask.zig`): `maskRow` (bit 0 = disallowed; **ids past the
  mask words are masked too**, so a padded vocabulary cannot leak), `fillAll`.

**Not done — the next block (in this order)**

1. `core/lanes/stream.zig`: carry `structure` (kind, text, after, compiled handle, whitespace ids) beside
   the existing `call` field. `api.Structure` needs `ws_ids: []const u32` for the whitespace phase.
2. `core/lane_host.zig`: pass `r.structure` into the lane message (the `loneFits` exclusion already exists).
3. `families/dsv41/lanes.zig` / `lanes_main.zig`: hold the per-stream matcher; free on finish and cancel.
4. `families/dsv41/layer_main.zig`: at both `sampleDev` sites (~:1185 prompt/first token, ~:1316 round rows)
   advance the matcher with the last accepted token, fill the bitmask, `cuMemcpyDtoH` → `maskRow` →
   `cuMemcpyHtoD`, then sample. A constrained stream never drafts.
5. `native/cuda.zig`: `.structures = soEnabled()` — **only after 1–4 land** (today a flipped flag would
   answer constrained requests unconstrained; the change was made and deliberately reverted).

Then: metrics + `TF_DS_SO_FAULT`, `zig/Dockerfile` + `scripts/config.sh`/`prepare.sh`/`image_hash`
(`XGRAMMAR_REF` wired the `TF_REF` way), the `TF_DS_SO=0` checkpoint with `tools/exact.py`,
`tools/structure_probe.py`, the live gates, and `scripts/port-patches.sh --engine ~/repos/tensorfold-dsv41-dev
--branch structured-output --base so-base --mode squash --name structured-output`.

**Invariants not to break**

- `Info.structures` stays false until the mask actually applies.
- The shim is a shared library, never statically linked into the Zig build (GCC-LTO libstdc++ is unreadable
  by lld; every archive arrangement left xgrammar members unresolved).
- The mask must cover ids past the tokenizer's vocabulary.
- Zig 0.17: `std.Io.Mutex`/`std.Io.sleep` (no `std.Thread.Mutex`, no `milliTimestamp`), no `dupeZ`, and the
  `**` array-repeat literal fails to parse — use explicit arrays.
- The engine is built only by `scripts/prepare.sh`; host tests only by `zig build test`.

## Session log (append-only)

- 2026-10-31: engine commits 6b3a0c3 (shim+build), b1829fa (cache), 9a567ea (vocab export), 2f867ea (admission), db0ec1e (mask arithmetic), 8a63267 (grammar carried through the lane stream). Host tests 337/339 - the 2 failures are the pre-existing spill cases. Feature stays OFF: Info.structures is false.
- Next (family enforcement): the live[] window state near layer_main.zig:1300 holds slot/row/n/len/smp/req; add the matcher handle + phase; advance at sampleDev ~1185 (first token) and ~1316 (round rows); cuMemcpyDtoH -> structure_mask.maskRow -> cuMemcpyHtoD before the sample.
- Then: native/cuda.zig .structures = soEnabled(); drafting exclusion in the family admission; TP mask delivery only in constrained rounds; metrics + TF_DS_SO_FAULT; zig/Dockerfile + scripts/config.sh/prepare.sh/image_hash (XGRAMMAR_REF pinned like TF_REF); TF_DS_SO=0 checkpoint with tools/exact.py; tools/structure_probe.py; live gates; scripts/port-patches.sh --engine ~/repos/tensorfold-dsv41-dev --branch structured-output --base so-base --mode squash --name structured-output.

- 2026-10-31 (fixer): enforcement lands in dsv41/lanes.zig, NOT layer_main.zig — layer_main is the recordings gate (runDrafted over fixtures, DStream from a recording), so enforcing there would be dead code. soMask at lanes.zig:778, applied at :1044 (firstFn) and :1122 (verifyFn). StructureOps lives in core/lanes/stream.zig and is aliased from engine_api (Zig rejects importing core/structure/api.zig from the lanes module). The served path already copies logits to the host, so the mask is applied there and no extra DtoH/HtoD is needed. Whitespace cap = 2 tokens. Commits 6fe48ef, c980405; tests 337/339, only the two pre-existing spill failures. Info.structures still false.

## Merged

Patch 0034 is on `main` (merge `dc452e3`, image `tensorfold-dsv41:exl3-tp2-93c750801527`, the image the pair runs).
Rollback is no longer "check out main": revert the merge (`git revert -m 1 dc452e3`) or start the previous
tag with `IMAGE=tensorfold-dsv41:exl3-tp2-8e32557ada53 ./start.sh restart` (still cached).
The feature stays switchable without a rebuild: `TF_DS_SO=0`.
