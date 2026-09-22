# Qwen3.5-9B on the v3 engine

Maintainer record for running `Qwen/Qwen3.5-9B` on NInfer's v3 (artifact-configured) engine.
Branch `qwen3_5_9b-v3` on the BenJGadd/ninfer mirror, based on upstream master `9e163eee`
(2026-09-21). The earlier port of the same model onto the pre-v3 engine lives on branch
`qwen3_5_9b` (its port guide describes the per-target architecture that master replaced).

## 1. What the v3 engine already gives you

Master reads model geometry from the artifact (`src/models/qwen3_5/config.cpp`), selects the
implementation by architecture string only (`src/models/registry.cpp`), executes chat templates
as jinja, and converts any Qwen3.5 Dense checkpoint with the generic recipe helpers. So the 9B
needs **no engine target, no bindings, no template registration**. What it still needs is
below, and all of it is small.

## 2. Files

### 2.1 New files

```text
docs/maintainer/qwen3.5-9b-port.md                  (this file)
docs/performance/qwen3.5-9b.md                      (bench of the composed baseline)
model-cards/Qwen3.5-9B-NInfer/
src/models/qwen3_5/execution/composed.{h,cpp}       (composed projection routes, Section 4)
tools/frontend_resources/qwen3_5_9b/generation_config.json
```

### 2.2 Edited files (the re-apply checklist)

| File | Edit |
|---|---|
| `tools/convert/official_recipes.py` | `qwen3_5_9b` recipe: `_dense_groupwise(Q6)`, GDN `query|key|value` grouped as one Q5 parent, `a_projection`/`b_projection` Q8 (Section 3) |
| `src/models/qwen3_5/execution/parameters.h` | `ComposedAttentionProjection`, `ComposedGdnProjection`, `ComposedGdnControl`; `composed`/`output_composed` fields on the block parameters |
| `src/models/qwen3_5/execution/parameters.cpp` | `block()` and `dense()` decide fused vs composed from the plan catalogs (`registered_*`) and prepare the logical projections for the composed route |
| `src/models/qwen3_5/execution/{attention,gdn,ffn,text}.cpp` | one `composed` branch per leaf; FFN reuses its existing MTP composition |
| `src/models/qwen3_5/program/planning/startup.cpp` | `add_scratch` and the GDN control scratch consult the composed byte formulas |
| `src/models/qwen3_5/execution_sources.cmake` | `composed.cpp` |
| `src/ops/linear/{q4,q5,q6,q8}/*_dispatch.cpp` | one fallback block before the throw: unregistered `(n, k)` with `k % 128 == 0` take the generic simt/mma routes |
| `src/ops/softmax_attention/dense/causal_cache/geometry.cuh` + 8 `.cu` + `causal_softmax_attention.cpp` | `CausalD256H16Kv4` (16 query / 4 KV heads); dispatch keyed on `cache.num_kv_heads`; INT8 small-T route for group size 4 |
| `src/ops/launcher/rope.cu` | `launch_fixed<…, 16, 4>` arms |
| `src/ops/linear_attention/gated_delta_net/{recurrent.cuh,recurrent.cu,replay.cpp}` | `FoldGeometry24x32` (24 GDN layers × 32 value heads) |
| `src/ops/kernel/gdn_gating.cuh`, `launcher/gdn_gating.cu`, `wrapper/gdn_gating.cpp` | head count from `A_log.ne[0]` instead of the literal 48 |

The ops-layer hunks are byte-identical to the pre-v3 branch's (those files did not change
between `a16b6442` and master); `git diff a16b6442 qwen3_5_9b -- <paths> | git apply` transplanted
them. Everything else was written against master.

## 3. Converting

```bash
python -m tools.convert --model /path/to/Qwen3.5-9B --recipe qwen3_5_9b \
  --components text,vision,mtp --proposal --name qwen3.5-9b \
  --resource generation_config.json=tools/frontend_resources/qwen3_5_9b/generation_config.json \
  --out out/qwen3_5_9b.ninfer
```

- The upstream repository ships no `generation_config.json`; the engine needs one for its
  default stop ids. The pinned file mirrors Qwen3.5-27B's (`eos_token_id [248046, 248044]`,
  `pad 248044`, model-card thinking-mode sampling).
- Two GDN sources are F32 in the checkpoint (`linear_attn.A_log`, `linear_attn.norm.weight`);
  the generic converter casts direct parameters, so no recipe change is needed.
- The recipe differs from `qwen3_6_27b` only in storage: `query|key|value` as one Q5 parent
  (the composed route wants one contiguous `linear` into the convolution input; query/key move
  from Q4 to Q5 because a group has one format) and Q8 for the two 32-row control projections
  (the BF16 linear catalog registers three exact 27B shapes and nothing else).
- Output: 716 objects (710 tensors, 6 resources), formats BF16 358 · FP32 48 · I32 1 · Q4 95 ·
  Q5 174 · Q6 3 · Q8 31; file 6,558,595,328 bytes.

## 4. The composed route

The fused Ops the execution layer calls (`attn_input_proj`, `gdn_input_proj*`, `linear_swiglu`,
`linear_add`, `gdn_norm_gating_proj`) are closed catalogs of the two tuned geometries and throw
for any other shape; the per-shape `linear` tables likewise. `composed.h` runs the same
mathematics through shape-generic Ops instead:

| Leaf | Fused (registered) | Composed (unregistered) |
|---|---|---|
| attention input | `attn_input_proj(pair)` | four `linear` calls on the logical query/key/gate/value |
| GDN input (prefill) | `gdn_input_proj(pair)` | `linear(query\|key\|value)` + `linear(z)` |
| GDN snapshot / record | `gdn_input_proj_conv_{snapshot,record}(pair)` | the two linears + `ops::detail::gdn_projected_conv_{snapshot,record}_launch` (already registers the 8192-channel 2048/2048/4096 geometry) |
| GDN control | `gdn_norm_gating_proj` | `rmsnorm` + `linear(a)` + `linear(b)` + `gdn_gating` |
| output projections | `linear_add` | `linear` + `residual_add` |
| dense FFN | `linear_swiglu` + `linear_add` | the leaf's existing MTP composition |

The choice is made once per block in `parameters.cpp` by asking the plans' `*_admits`
predicates with the config geometry, so a future catalog entry for a 9B shape flips that block
to the fused route with no further change. Composed leaves report their own workspace bytes and
startup planning consults those. Cost: extra launches and a BF16 intermediate per fused group;
on the pre-v3 branch the same composition measured ~80 % of the tuned engine's bandwidth
efficiency.

## 5. Reproduction on this machine

Scripts in `~/Projects/ninfer-repro/`: `convert-9b-v3.sh`, `build-v3.sh` (nix-shell shell.nix,
`ninja -j6`, build tree `ninfer-v3/build-v3/`), `serve-9b.sh`/`bench-9b.sh` variants for the
v3 binary. Run heavy jobs under `systemd-run --user --scope -p MemoryMax=…`: 30 GiB, no swap.

## 6. Re-applying on a newer master

Rebase `qwen3_5_9b-v3`. Conflicts can only land in the Section 2.2 files; each hunk is a few
lines with a "Qwen3.5-9B" comment. If upstream registers 9B shapes in a fused catalog, the
`registered_*` predicates start returning true and the composed code becomes dead for that leaf
without edits. If upstream generalises the linear tables, drop the fallback blocks.
