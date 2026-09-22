# Qwen3.5-9B Model Reference

Qwen3.5-9B is the same architecture as Qwen3.6-27B (`qwen3_5` / `qwen3_5_text` in Hugging
Face): a hybrid GDN + full-attention text decoder, one MTP draft layer, and the 27-layer Vision
tower. Everything in [`qwen3.6-27b-model.md`](qwen3.6-27b-model.md) Sections 3–13 applies
verbatim with the dimensions below substituted. This page records only what differs.

## 1. Global dimensions

| Field | 27B | **9B** |
|---|---:|---:|
| hidden size | 5120 | **4096** |
| decoder layers | 64 | **32** |
| intermediate size | 17408 | **12288** |
| output/embedding rows | 248320 | 248320 |
| tokenizer-addressable IDs | 248077 | 248077 |
| full-attention interval | 4 | 4 |
| full-attention layers | 16 | **8** (`3, 7, 11, 15, 19, 23, 27, 31`) |
| GDN layers | 48 | **24** |
| RMSNorm epsilon / RoPE theta | `1e-6` / `1e7` | same |
| position capacity | 262144 | 262144 |
| MTP layers | 1 | 1 |

## 2. Full attention

| Field | 27B | **9B** |
|---|---:|---:|
| query heads / KV heads / head dim | 24 / 4 / 256 | **16** / 4 / 256 |
| Q width / K,V width | 6144 / 1024 | **4096** / 1024 |
| Q heads per KV head | 6 | **4** |
| rotated dims per head | 64 | 64 (`partial_rotary_factor 0.25`) |
| attention scale | `1/√256` | same |
| MRoPE sections | `[11,11,10]` | same |

`q_proj` is head-interleaved query|gate as in the 27B (`[8192,4096]` = 16 heads × 512 rows).
`attn_output_gate: true` in the checkpoint config.

## 3. Gated DeltaNet

| Field | 27B | **9B** |
|---|---:|---:|
| Q/K heads × dim | 16 × 128 | 16 × 128 |
| V heads × dim | 48 × 128 | **32** × 128 |
| Q, K width | 2048 | 2048 |
| V, Z width | 6144 | **4096** |
| A/B gate width | 48 | **32** |
| conv channels (2·QK + V) | 10240 | **8192** |
| V heads per Q/K head | 3 | **2** |
| delta-rule scale | `1/√128` | same |

The 2:1 value-to-key head ratio is the one Qwen3.6-35B-A3B already exercises in the family GDN
kernels.

## 4. MTP and Vision

MTP: one full-attention layer with the Section 2 layout; `fc` is `[4096, 8192]`.
Vision: identical tower (depth 27, hidden 1152, intermediate 4304, 16 heads, 2304 positions);
only the merger's output width follows the text hidden size (`fc2: [4096, 4608]`).

## 5. Speculation

MTP (`--spec mtp`, K ≤ 5) and the 131072-row shortlisted proposal head (`--lm-head-draft`) are
supported. No DFlash/DFlash2 draft model exists for this checkpoint; the target reports
`SpeculativeBackend::None` for masked drafting and rejects `--spec dflash*` at startup.

## 6. Implementation map

| Concern | Source |
|---|---|
| primitive dimensions and derived extents | `src/targets/qwen3_5_9b/impl/config.h` ↔ `tools/convert/qwen3_5_9b/geometry.py` |
| artifact bindings (groupwise-int only) | `src/targets/qwen3_5_9b/impl/load/` |
| execution leaves (plain-Op compositions), workspace, graph ranges | `src/targets/qwen3_5_9b/impl/variant.{h,cpp}` |
| ops-layer geometry admissions (attention, RoPE, fold, linear fallbacks, gating) | see port guide Section 3.2 |
| everything else | shared `src/targets/qwen3_6/` family, see the 27B map |
| converter / verifier | `tools/convert/qwen3_5_9b/` |
| port record and re-apply checklist | [`qwen3.5-9b-port.md`](qwen3.5-9b-port.md) |
