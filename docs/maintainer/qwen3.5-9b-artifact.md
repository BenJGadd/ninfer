# Qwen3.5-9B Artifact Reference

Identity `qwen3.5-9b / groupwise-int`, target key `qwen3_5_9b`, container version 2. The
object namespace, writer order, numeric-format assignment, fused row order, aliases, frontend
resources and source transforms are those of the Qwen3.6-27B groupwise-int artifact
([`qwen3.6-27b-artifact.md`](qwen3.6-27b-artifact.md) Sections 3–12) with the shapes below.
Every shape here is produced by `tools/convert/qwen3_5_9b/geometry.py`; this page is a
rendering of that module, not a second source.

Three GDN storage roles differ from the 27B, because the 9B engine leaves are compositions of
plain `linear` calls whose outputs must be contiguous (Section 4 of the port guide):

- `gdn/query_key_value` is the checkpoint's `in_proj_qkv` unchanged (Q5; the 27B splits it into a
  Q4 query|key parent and a Q5 value|z parent);
- `gdn/z` is `in_proj_z` unchanged (Q5);
- `gdn/a_b_projection` fuses `in_proj_a` | `in_proj_b` as one W8 parent. The 27B keeps them BF16,
  but the BF16 linear catalog only registers three exact 27B shapes, while W8 has a generic route.

## 1. Fixed target facts

| Fact | Value |
|---|---:|
| vocabulary rows / tokenizer-addressable IDs | 248320 / 248077 |
| Text hidden / layers / intermediate | 4096 / 32 / 12288 |
| full-attention layers | 8 (`3, 7, 11, 15, 19, 23, 27, 31`) |
| GDN layers | 24 |
| query / KV heads, head width | 16 / 4, 256 |
| query / KV widths | 4096 / 1024 |
| GDN key heads × width | 16 × 128 = 2048 |
| GDN value heads × width | 32 × 128 = 4096 |
| GDN convolution channels / taps | 8192 / 4 |
| optimized draft-head rows | 131072 |
| Vision merger input / output | 4608 / 4096 |
| native context capacity | 262144 |

## 2. Inventory

| Group | Objects |
|---|---:|
| frontend resources | 6 |
| text core (`1 + 32·4 + 8·5 + 24·8 + 2`) | 363 |
| optimized MTP draft head | 2 |
| MTP | 12 |
| Vision | 333 |
| **tensors** | **710** |
| **objects** | **716** |

Numeric-format counts: BF16 358 · FP32 48 · I32 1 · Q4G64_F16S 95 · Q5G64_F16S 174 ·
Q6G64_F16S 3 · W8G32_F16S 31. Layouts: contiguous-le-v1 407 · row-split-k128-v1 303.

Computed byte totals (`tools.convert.qwen3_6.common.conversion`):

| Quantity | Bytes |
|---|---:|
| `tensor_payload_bytes` | 6,545,522,592 (6.10 GiB) |
| `device_arena_bytes` (256-aligned) | 6,545,536,512 |
| text core / draft head / MTP / Vision | 5,710,573,568 / 285,736,960 / 258,515,968 / 290,696,096 |

## 3. Per-layer shapes `[N,K]`

| Object | Format | Shape |
|---|---|---:|
| `text/token_embedding`, `text/output_head` | Q6 | `[248320, 4096]` |
| `text/layers/{l}/input_norm`, `post_attention_norm` | BF16 | `[4096]` |
| `attention/query_key` (query 0..4096 · key 4096..5120) | Q4 | `[5120, 4096]` |
| `attention/gate_value` (gate 0..4096 · value 4096..5120) | Q5 | `[5120, 4096]` |
| `attention/query_norm`, `key_norm` | BF16 | `[256]` |
| `attention/output` | Q5 | `[4096, 4096]` |
| `gdn/a_log`, `gdn/dt_bias` | FP32 | `[32]` |
| `gdn/convolution` (alias `channel_major_convolution` = `[8192, 4]`) | BF16 | `[4, 8192]` |
| `gdn/a_b_projection` (a 0..32 · b 32..64) | W8 | `[64, 4096]` |
| `gdn/query_key_value` (query 0..2048 · key 2048..4096 · value 4096..8192) | Q5 | `[8192, 4096]` |
| `gdn/z` | Q5 | `[4096, 4096]` |
| `gdn/norm` | BF16 | `[128]` |
| `gdn/output` | Q5 | `[4096, 4096]` |
| `mlp/gate_up` (gate 0..12288 · up 12288..24576) | Q4 | `[24576, 4096]` |
| `mlp/down` | Q5 | `[4096, 12288]` |
| `text/draft_head` / `text/draft_head_token_ids` | Q4 / I32 | `[131072, 4096]` / `[131072]` |
| `mtp/input_projection` | W8 | `[4096, 8192]` |
| `mtp/layer/attention/query_key_gate_value` (q 0..4096 · k ..5120 · gate ..9216 · v ..10240) | W8 | `[10240, 4096]` |
| `mtp/layer/attention/output` | W8 | `[4096, 4096]` |
| `mtp/layer/mlp/gate_up` / `down` | W8 | `[24576, 4096]` / `[4096, 12288]` |
| `vision/merger/fc2` / `fc2_bias` | W8 / BF16 | `[4096, 4608]` / `[4096]` |

## 4. Source

`Qwen/Qwen3.5-9B` revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, four BF16 shards,
775 source tensors, all consumed exactly once (`tests/convert/qwen3_5_9b/test_recipe.py`).
Source role names and transforms are identical to the 27B (`model.language_model.*`,
`mtp.*`, `model.visual.*`, `lm_head.weight`).

Frontend resources and pinned SHA-256 (`convert.OFFICIAL_RESOURCE_SHA256`):

| Resource | Origin | SHA-256 |
|---|---|---|
| `tokenizer.json` | checkpoint (= Qwen3.6 official) | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |
| `tokenizer_config.json` | checkpoint | `316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8` |
| `chat_template.jinja` | checkpoint | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` |
| `generation_config.json` | **repository file** `tools/convert/qwen3_5_9b/generation_config.json` (upstream ships none) | `98cc62c0ad60faa4067de78f12b909e65e4dc093d099f406ac5bb163ad95a2f4` |
| `preprocessor_config.json` | checkpoint (= Qwen3.6 official) | `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516` |
| `video_preprocessor_config.json` | checkpoint (= Qwen3.6 official) | `7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13` |

## 5. Produced artifact

Recipe id `qwen3_5_9b-v1`. File facts are filled from the conversion report and `sha256sum`
after each conversion; see the model card `model-cards/Qwen3.5-9B-NInfer/` for the published
values.
