---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
base_model_relation: quantized
tags: [ninfer, qwen3.5, multimodal, conversational, cuda, rtx-5090]
---

# Qwen3.5-9B for NInfer (v3 artifact)

[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) converted to the NInfer v3 `.ninfer`
format by the `qwen3_5_9b` official recipe on branch `qwen3_5_9b-v3` of the BenJGadd/ninfer
mirror (upstream master `9e163eee` plus the port in `docs/maintainer/qwen3.5-9b-port.md`).

## Artifact

| Field | Value |
|---|---|
| Filename | `qwen3_5_9b_v3.ninfer` |
| Size | 6,558,595,328 bytes (6.11 GiB) |
| SHA-256 | `3d87bfe1735dc048426da408fae68f4d6d553d4fd4e692e64395d2dbdbf9d12c` |
| Container version | 3 |
| Name | `qwen3.5-9b` |
| Recipe | `qwen3_5_9b` |
| Converter revision | `3e45a90a` |

716 objects (710 tensors, 6 resources): BF16 358 · FP32 48 · INT32 1 · Q4 95 · Q5 174 · Q6 3 ·
Q8 31. `generation_config.json` is repository-pinned (`tools/frontend_resources/qwen3_5_9b/`)
because the upstream checkpoint ships none.

```bash
printf '%s  %s\n' '3d87bfe1735dc048426da408fae68f4d6d553d4fd4e692e64395d2dbdbf9d12c' 'qwen3_5_9b_v3.ninfer' | sha256sum --check
```

## Run

```bash
./build/apps/ninfer-serve models/qwen3_5_9b_v3.ninfer --host 127.0.0.1 --port 8090 \
  --max-context 262144 --kv-capacity auto --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

No DFlash/DFlash2 draft model exists for this checkpoint.

## Evaluation

Causal perplexity (`ninfer-perplexity --quick --kv-dtype int8`, context 4096 / stride 2048,
corpus `ninfer-ppl-1m-v1`, 2026-09-22): overall **5.07** (chinese_reference 5.48,
english_long_form 7.61, english_reference 7.84, ninfer_code 2.01; 261,167 scored tokens) —
identical to six decimals with the pre-v3 engine's result for the same weights.

## Performance

See `docs/performance/qwen3.5-9b.md`.
