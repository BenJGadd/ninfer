# Qwen3.5-9B Port and Reproduction Guide

This is the maintainer record for adding `qwen3.5-9b/groupwise-int` to NInfer as a
**rebase-survivable patch series**. Its purpose is that a future NInfer revision can carry the
9B target again with a mechanical re-apply, not a re-derivation. Model mathematics are in
[`qwen3.5-9b-model.md`](qwen3.5-9b-model.md); the artifact contract is in
[`qwen3.5-9b-artifact.md`](qwen3.5-9b-artifact.md).

Base revision of this port: NInfer `a16b6442` (2026-09-07), plus the local
`qwen3_8_27b` nvfp4 commit `2d568cf6` beneath it. Branch: `qwen3_5_9b`.

## 1. Why the port has two halves

`src/targets/` is organised **per shape**, `tools/convert/` **per checkpoint**. Qwen3.8-27B
needed only a converter because it is shape-identical to Qwen3.6-27B. Qwen3.5-9B is a new shape
(32 layers, hidden 4096, 16 query heads, 32 GDN value heads), so it needs both a converter target
and an engine target. Both reuse the Qwen3.6 family runtime (`src/targets/qwen3_6/`) unchanged:
the family code is parameterised on a `Variant` type whose `TextConfig` supplies every dimension.

## 2. Single source of truth for dimensions

Eight primitive numbers define the shape. They live in exactly two places and a test keeps
them equal:

| Primitive | Value | Python | C++ |
|---|---:|---|---|
| hidden | 4096 | `geometry.py: Geometry.hidden` | `config.h: TextConfig::hidden` |
| layers | 32 | `.layers` | `::layers` |
| intermediate | 12288 | `.intermediate` | `::intermediate` |
| vocab rows | 248320 | `.vocab_rows` | `::output_rows` |
| query heads / KV heads / head dim | 16 / 4 / 256 | `.query_heads .kv_heads .head_dim` | same names |
| GDN key heads / value heads / head dim | 16 / 32 / 128 | `.gdn_key_heads .gdn_value_heads .gdn_head_dim` | `::gdn_key_heads ::gdn_value_heads ::gdn_{key,value}_head_dim` |

Everything else is a formula of those (`query_size = heads × head_dim`,
`convolution_dim = 2·key_dim + value_dim`, `gate_up_rows = 2·intermediate`, …).
`tools/convert/qwen3_5_9b/geometry.py` derives them for Python;
`src/targets/qwen3_5_9b/impl/config.h` derives them for C++;
`tests/convert/qwen3_5_9b/test_config_h.py` parses `config.h` and asserts the primitives match.
`convert.py` additionally rebuilds a `Geometry` from the checkpoint's own `config.json` and
refuses a checkpoint whose geometry differs.

The Qwen3.6-27B target, by contrast, spells 210 literal dimensions in its `bindings.cpp`. Do not
copy that style into this target: a mistyped literal there mis-shapes a tensor view and can load
silently and produce garbage rather than fail.

## 3. Files

### 3.1 New files (no merge conflicts on rebase)

```text
tools/convert/qwen3_5_9b/
  __init__.py  geometry.py  inventory.py  recipe.py  draft_head.py
  convert.py   verify.py    generation_config.json
src/targets/qwen3_5_9b/
  CMakeLists.txt
  impl/config.h  impl/package.cpp  impl/variant.h  impl/variant.cpp
  impl/load/bindings.h  impl/load/bindings.cpp
  export/ninfer/targets/qwen3_5_9b/package.h
tests/convert/qwen3_5_9b/
  __init__.py  test_inventory.py  test_recipe.py  test_convert.py  test_config_h.py
  config.json  model.safetensors.index.json          (fixtures from the pinned checkpoint)
docs/maintainer/qwen3.5-9b-{port,model,artifact}.md
docs/performance/qwen3.5-9b.md
model-cards/Qwen3.5-9B-NInfer/
```

### 3.2 Conflict surface (the re-apply checklist)

Exactly these shared files are edited. On a rebase, every conflict is in one of them and the
edit is mechanical: add the 9B alongside the existing targets.

| File | Edit |
|---|---|
| `src/CMakeLists.txt` | `add_subdirectory(targets/qwen3_5_9b)` after the 35B line |
| `src/targets/registry.h` | include `qwen3_5_9b/package.h`; `using Qwen3_5_9B`; `LoadedQwen3_5_9B` + `Qwen3_5_9BInstance` structs (copies of the 35B pair); add `std::unique_ptr<Qwen3_5_9BInstance>` to `ActiveTarget` |
| `src/targets/registry.cpp` | the two constructor/destructor definitions; one `if (identity.model_id == Qwen3_5_9B::model_id)` arm in `construct_target` |
| `src/runtime/engine/engine.cpp` | `Core9`/`ScoreCore9` aliases in the `Core` variant; one `else if constexpr` arm in the core factory; add `ScoreCore9` to the two `is_same_v` scoring checks |
| `tools/bench/run_serve_corpus.py` | `"qwen3_5_9b": "qwen3.5-9b"` in `TARGET_MODEL_IDS` |
| `AGENTS.md` | add `qwen3.5-9b/groupwise-int` to the registered identity list |
| `docs/performance.md` | one coverage row |
| `tools/convert/qwen3_6/common/recipe.py` | `Cast` accepts a BF16 target (the 9B stores `linear_attn.norm.weight` as F32) |

Ops-layer hunks. The kernels are shape-generic; only their dispatch tables and validators
named the two registered geometries:

| File | Edit |
|---|---|
| `src/ops/softmax_attention/dense/causal_cache/geometry.cuh` | `CausalD256H16Kv4 = CausalAttentionGeometry<16, 4, 2>` |
| same dir: `prompt*.cu`, `small_t*.cu` (8 files, 13 sites) | before every `<CausalD256H16Kv2>` launch: `if (cache.num_kv_heads == CausalD256H16Kv4::KVHeads) { …<CausalD256H16Kv4>…; return; }` |
| same dir: `small_t.cu` INT8 route | `if constexpr (Geometry::GroupSize == 4 && TokenTile >= 5)` picks 16/8 warps (power-of-two consumer warps per row tile) |
| `…/causal_softmax_attention.cpp` | `require_causal_geometry` admits (16, 4) |
| `src/ops/launcher/rope.cu` | `launch_fixed<…, 16, 4>` arms for 1-D and MRoPE |
| `src/ops/linear_attention/gated_delta_net/recurrent.cuh`, `recurrent.cu`, `replay.cpp` | `FoldGeometry24x32 = FoldGeometry<24, 16, 32, 8192>` + dispatch arm + registered-geometry predicate |
| `src/ops/linear/{q4,q5,q6,w8}/*_dispatch.cpp` | one fallback block before the final throw: unregistered `(n, k)` with `k % 128 == 0` take the generic simt/mma routes |
| `src/ops/kernel/gdn_gating.cuh`, `launcher/gdn_gating.cu`, `wrapper/gdn_gating.cpp` | head count becomes a runtime parameter (`A_log.ne[0]`) instead of the literal 48 |
| `src/targets/qwen3_6/impl/frontend/chat_template.cpp` | the Qwen3.5 template digest (`a4aee8af…`) resolves to the native ThinkingToggle renderer; the 9B template is the 3.6 template minus its `preserve_thinking` branch, with scalar tool arguments rendered by `string` instead of `tojson`. Upstream master executes jinja directly, so this hunk is dropped on a rebase past it |

To find these again after a rebase: `git log --oneline qwen3_5_9b ^<upstream> -- src/ops` lists the
commits; each hunk is small and carries a "Qwen3.5-9B" comment.

`src/runtime/engine/context_cost_defaults.cpp` is **not** touched: an unregistered model falls
back to the generic (27B-derived, conservative) prefill cost, which is the same path the
35B-A3B takes.

## 4. What the engine target does differently from the 27B template

- **Every leaf is a composition of shape-generic public Ops.** The fused Ops the 27B/35B leaves
  call (`attn_input_proj`, `gdn_input_proj*`, `linear_swiglu`, `linear_add`, `linear_pair`,
  `gdn_norm_gating_proj`) are closed catalogs of exactly the 27B and 35B-A3B shapes and throw on
  anything else. The 9B leaves instead use `ops::linear` (with the fallback routes above),
  `ops::rmsnorm`, `ops::silu_mul`, `ops::residual_add`, `ops::gdn_gating`, and the family's
  projected causal-convolution launcher (`ops::detail::gdn_projected_conv_{snapshot,record}_launch`,
  which already registers the 8192-channel 2048/2048/4096 geometry the 35B-A3B uses). This is
  exactly how the 27B's MTP leaves are already written. Cost: more launches and one BF16
  intermediate per fused group; that is the untuned baseline the performance page measures.
- Because `ops::linear` outputs must be contiguous, projection groups are stored so each output
  is one parent or one row view: attention query/key/gate/value are row views of the two
  27B-style parents; GDN uses `query_key_value`, `z` and a fused W8 `a_b_projection`
  (artifact reference, Section 1).
- One weights profile, `WeightsProfile::Qwen35GroupwiseInt`. No NVFP4/FP8 leaves.
- `DFlashConfig::supported = false`, `backend = SpeculativeBackend::None`. No masked-draft
  companion model exists for Qwen3.5-9B. `--spec dflash`/`dflash2` fail at startup with
  "selected masked draft backend is not supported by this target". MTP (`--spec mtp`) and the
  shortlisted proposal head (`--lm-head-draft`) work.
- Graph frontier ranges and MTP window boundaries are copied from the 27B: they are properties
  of the KV attention kernels, not of the model width.
- `kAttentionScale = 1/√256`, `kGdnScale = 1/√128` are unchanged because both head widths are
  unchanged; a `static_assert` guards that assumption.

## 5. What the converter does differently

- `Qwen/Qwen3.5-9B` ships **no `generation_config.json`**. The engine reads it for default stop
  ids, so the converter embeds `tools/convert/qwen3_5_9b/generation_config.json` (structure of
  the Qwen3.5-27B file; ids `248046` `<|im_end|>` and `248044` `<|endoftext|>` from the 9B
  `tokenizer_config.json`; sampling values from the model card's thinking-mode row). Its SHA-256
  is pinned like the other five resources. `--generation-config` overrides the path.
- Two GDN sources are F32 in the checkpoint (`linear_attn.A_log`, `linear_attn.norm.weight`)
  where the 27B stores BF16. `A_log` is FP32 in the artifact regardless; the norm weight is
  rounded to the BF16 the family GDN kernel consumes (`Cast(..., BF16)` in the recipe).
- The other five frontend files come from the checkpoint directory and are hash-pinned in
  `convert.OFFICIAL_RESOURCE_SHA256`. `tokenizer.json`, `preprocessor_config.json` and
  `video_preprocessor_config.json` are byte-identical to the Qwen3.6 official set;
  `tokenizer_config.json` and `chat_template.jinja` differ (different hashes, same token ids).
- Inventory counts are never typed: `convert.expected_object_counts()` derives them from the
  layer schedule and `inventory.expected_text_core_count()` re-counts the built tuple.
- Quantization map (Q6 endpoints, Q4 query-side/gate_up/draft head, Q5 value-side/down,
  W8 MTP/merger) is copied from the 27B. It was perplexity-tuned for the 27B; a 9B perplexity
  run (`docs/perplexity.md`) is owed before calling it final.

## 6. Reproduction on this machine (nixos-desktop, RTX 5090)

Environment facts that bit during the first run and are now handled:

- `nix-ld` is no longer in the system profile, so the old uv/torch venv cannot execute.
  The converter environment is a nix `python3.11` venv with pip wheels
  (`~/Projects/ninfer-repro/py-convert-venv.sh`, exports in `py-convert-env.sh`). Wheels need
  `LD_LIBRARY_PATH=<gcc-lib>/lib:/run/opengl-driver/lib`.
- **Do not** build a nixpkgs `torch-bin` environment: it pulls nvshmem and NCCL from source
  (784 CUDA objects, unbounded nvcc parallelism). With 30 GiB RAM and no swap that crashed the
  machine on 2026-09-21. Run every heavy job under
  `systemd-run --user --scope -p MemoryMax=<N>G` and cap `ninja -j`.
- The CUDA 13.1 build shell is `~/Projects/ninfer-repro/shell.nix`; the incremental rebuild
  script is `~/Projects/ninfer-repro/build-9b.sh` (`ninja -j6`).

Step list:

```bash
# 0. checkpoint (pinned revision), aggressive resumable download, sha256 from the HF LFS metadata
cd ~/Projects/ninfer-repro/models-src
./hfdl.sh Qwen/Qwen3.5-9B c202236235762e1c871ad0ccb60c8ee5ba337b9a Qwen3.5-9B.manifest "$PWD/Qwen3.5-9B" 4

# 1. python env (once)
cd ~/Projects/ninfer-repro && ./py-convert-venv.sh && . ./py-convert-env.sh

# 2. converter unit tests (fixtures are the pinned config.json + weight map)
cd ninfer && "$PY" -m pytest tests/convert/qwen3_5_9b

# 3. engine build (fresh build-9b/ tree: the old build/ cache names the pre-move source path)
cd ~/Projects/ninfer-repro
systemd-run --user --scope -p MemoryMax=22G -- nix-shell --impure shell.nix --run ./build-9b.sh

# 4. convert (GPU quantization; ~6.05 GiB output)
cd ninfer && "$PY" -m tools.convert.qwen3_5_9b.convert \
  --model ../models-src/Qwen3.5-9B --out ../out/qwen3_5_9b.ninfer

# 5. verify structure + representative payloads against the source shards
"$PY" -m tools.convert.qwen3_5_9b.verify ../out/qwen3_5_9b.ninfer --model ../models-src/Qwen3.5-9B

# 6. smoke serve (never port 8001: that is the desktop 35B server). Stop it with
#    `pkill -x ninfer-serve`; a `pkill -f` pattern that appears in your own command line kills
#    your shell first.
cd ~/Projects/ninfer-repro && ./serve-9b.sh          # build-9b/apps/ninfer-serve on 127.0.0.1:8090
curl -s http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-9b","messages":[{"role":"user","content":"Capital of France?"}],"enable_thinking":false}'

# 7. benchmark (methodology: docs/performance/methodology.md); its server uses port 8091
./bench-9b.sh
```

## 7. Re-applying on a newer NInfer

### 7.0 Upstream master after 2026-09-14 is a different architecture

Surveyed 2026-09-21 (`origin/master` 50 commits past `a16b6442`): upstream deleted
`src/targets/*`, `src/targets/registry.h` and the per-target `Variant`/bindings, and replaced them
with one generic `src/models/qwen3_5/` whose geometry is read from a **v3 artifact** at load time
(`src/models/qwen3_5/config.cpp` parses `hidden_size`, attention and GDN head counts;
`src/models/registry.cpp` selects by architecture string only). The converter became generic too
(`tools/convert/qwen3_5.py`, `official_recipes.py::_dense_groupwise`), and
`tools/upgrade_ninfer_v2_to_v3.py` upgrades official v2 files.

Consequences for this branch:

- **Do not rebase it onto master.** Sections 3.1–3.2's C++ half has nothing to re-apply against;
  only the converter ideas and the ops admissions carry over.
- **Port again on master as a smaller job** (`qwen3_5_9b-v3`):
  1. converter: an official recipe entry for `Qwen/Qwen3.5-9B` reusing `_dense_groupwise`, plus
     the pinned `generation_config.json` and the F32→BF16 norm cast;
  2. the ops-layer admissions of Section 3.2 (attention 16/4, RoPE, fold 24×32, gating heads):
     those gates are unchanged on master;
  3. shape entries in the fused-op catalogs (`attn_input_proj`, `gdn_input_proj`, `linear_swiglu`,
     `linear_add`, and the rewritten per-shape `linear` tables): master's execution layer calls the
     fused Ops directly, so composition leaves are not an option there. Master's
     `docs/maintainer/linear-tuning.md` documents adding and tuning entries; that path also yields
     tuned speeds instead of this branch's composition baseline.
- The chat-template digest registration (Section 3.2) is replaced on master by
  `src/models/qwen3_5/frontend/digest.cpp`; check whether the Qwen3.5 template digest is admitted
  there.

The rest of this section describes re-applying within the pre-2026-09-14 architecture.

1. `git rebase <new-upstream>` the `qwen3_5_9b` branch. Conflicts can only appear in the
   Section 3.2 files; resolve each by re-adding the 9B line next to whatever the upstream now
   has for the 27B/35B.
2. If the family `Variant` contract changed (new leaf, renamed member), diff
   `src/targets/qwen3_6_27b/impl/variant.h` against the previous base and mirror the change in
   `src/targets/qwen3_5_9b/impl/variant.h`; the 9B leaves are the 27B groupwise leaves with
   `TextConfig` substituted.
3. If the shared converter helpers changed (`tools/convert/qwen3_6/common/*`), the 9B
   `inventory.py`/`recipe.py` only call `tensor_spec`, `build_vision_specs`,
   `build_vision_recipes`, `attention_qproj_part` and the expression classes; update call sites.
4. If the artifact container version changed, reconvert; the recipe is the contract, the
   file is derived.
5. Rerun steps 2–7 above. Byte totals and hashes in the artifact doc and model card are
   **computed** (`tools.convert.qwen3_6.common.conversion.tensor_payload_bytes`) and pasted;
   never edit them by hand.
