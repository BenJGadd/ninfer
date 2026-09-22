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

## 7. The other local artifacts on v3

Upstream's `tools/upgrade_ninfer_v2_to_v3.py` converted the four other deployed v2 files on
2026-09-22 (weights preserved, maintained templates installed), each verified by loading on this
branch's engine with its deployed flags and answering one request:

| v2 file | v3 name | identity | objects |
|---|---|---|---:|
| `qwen3_6_35b_a3b.ninfer` | `qwen3_6_35b_a3b.v3.ninfer` | `qwen3.6-35b-a3b` + dflash | 940 |
| `qwen3_6_35b_a3b_uncensored.ninfer` | `qwen3_6_35b_a3b_uncensored.v3.ninfer` | `qwen3.6-35b-a3b` + dflash | 940 |
| `qwen3_8_27b_nvfp4.ninfer` | `qwen3_8_27b_nvfp4.v3.ninfer` | `qwen3.8-27b` + dflash2 | 1190 |
| `qwen3_8_27b_nvfp4_uncensored.ninfer` | `qwen3_8_27b_nvfp4_uncensored.v3.ninfer` | `qwen3.8-27b` + dflash2 | 1190 |

The 9B was reconverted from source with the `qwen3_5_9b` recipe rather than upgraded. The
`qwen3_8_27b_thinkingcap_nvfp4_w8g32` variant (`weights_id nvfp4_w8g32`, 1307 objects) is outside
the tool's table and was retired instead of ported.

## 8. Weight-only NVFP4 (`qwen3_5_9b_nvfp4`)

Qwen publishes no NVFP4 checkpoint for this size, and upstream's `import_encoded` only copies
pre-encoded words, so the artifact is quantized in-repo from the BF16 checkpoint.

**Method** `nvfp4_blockwise` (`tools/convert/methods.py`), the inverse of the decode contract
in `docs/maintainer/tensor-formats.md` §3.3, `W = e2m1(c) · e4m3fn(s) / d_w`:

- pass one: matrix max-abs `A` over every input row of the parent;
- `d_w = 448 · 6 / A`, so the largest block scale lands exactly on the E4M3FN maximum;
- per 16-wide block: `s = e4m3fn_rne(clamp(amax_block · d_w / 6, 0, 448))`, then
  `c = e2m1_rne(clamp(w · d_w / s, −6, 6))` with ties to the even code; an all-zero block
  stores scale word 0 and zero codes;
- weight-only: no activation divisor is produced, every site keeps `A16Only`.

A synthetic round trip (`quantize_nvfp4_matrix` → `encode_nvfp4` → `decode_nvfp4_words`)
returns the identical words, and the E2M1 tie cases 0.25/0.75/1.25/1.75/2.5/3.5/5.0 round to
codes 0/2/2/4/4/6/6.

**Recipe** (`tools/convert/official_recipes.py`): every `text/layers/*` projection except
`gdn/{a,b}_projection` is NVFP4; a/b stay Q8, endpoints Q6, vision and MTP as in
`qwen3_5_9b`. A non-standard method is never auto-packed (recipe.py `standard`), which is what
the composed route wants for attention — query/key/gate/value are four whole NVFP4 parents,
because NVFP4 row views are rejected (`weight_view.cpp` "requires a complete FP8/NVFP4
parent") — while GDN query|key|value and MLP gate|up are grouped explicitly so `single()` sees
one contiguous parent.

**Engine** — the NVFP4 linear catalog is per-shape (`Nvfp4Geometry<N,K>` at compile time), so
five shape files were added under `src/ops/linear/nvfp4/shapes/`, registered in
`nvfp4_shapes.h`, `nvfp4_dispatch.cpp` (`kShapes`) and `sources.cmake`:

| shape | sites |
|---|---|
| `n4096_k4096` | attention query, gate, output; GDN z, output |
| `n1024_k4096` | attention key, value |
| `n8192_k4096` | GDN query\|key\|value |
| `n24576_k4096` | MLP gate\|up |
| `n4096_k12288` | MLP down |

Each carries the A16 routes of `n5120_k6144.cu` unchanged (the schedules need only
`K % 512 == 0`); the A4 slot is `reject_a4` and `uses_a4` is false. Re-apply: five new files plus
three one-line-per-shape registrations.

**Reproduction**: `~/Projects/ninfer-repro/convert-9b-nvfp4.sh` (env `MODEL_SRC`, `OUT`,
`NAME` for a same-shape fine-tune), `serve-nvfp4.sh`, `perplexity-nvfp4.sh`.

**Result**: see `docs/performance/qwen3.5-9b.md` for the NVFP4 numbers against the Q6/Q5
artifact.
