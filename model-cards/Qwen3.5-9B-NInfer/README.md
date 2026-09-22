---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.5
  - multimodal
  - conversational
  - cuda
  - rtx-5090
---

# Qwen3.5-9B for NInfer

[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) converted to the native
[NInfer](https://github.com/Neroued/ninfer) `.ninfer` artifact format by the local
`qwen3_5_9b` port (branch `qwen3_5_9b`, see `docs/maintainer/qwen3.5-9b-port.md`). The
artifact is intended only for NInfer; it is not a Transformers checkpoint, Safetensors
distribution, or GGUF file.

## Artifact

| Field | Value |
|---|---|
| Filename | `qwen3_5_9b.ninfer` |
| Size | 6,558,480,640 bytes (6.11 GiB) |
| SHA-256 | `3a471aa18d59d0760db30e9ad7fcf339580deb55665cb47b8be362627133139c` |
| Container version | 2 |
| NInfer model ID | `qwen3.5-9b` |
| NInfer weights ID | `groupwise-int` |
| NInfer target key | `qwen3_5_9b` |

The file contains the registered Text, Vision, MTP, proposal-head, tokenizer, chat-template,
generation, and media-processor objects required by NInfer (716 objects: 710 tensors, 6
resources). `generation_config.json` is a repository-pinned file because the upstream
checkpoint ships none.

```bash
printf '%s  %s\n' '3a471aa18d59d0760db30e9ad7fcf339580deb55665cb47b8be362627133139c' 'qwen3_5_9b.ninfer' | sha256sum --check
```

## Requirements

- NInfer with the `qwen3_5_9b` target (this port), built from source;
- 64-bit Linux; NVIDIA GeForce RTX 5090 (`sm_120a`); CUDA Toolkit 13.1 or newer.

## Run

```bash
./build/apps/ninfer-serve models/qwen3_5_9b.ninfer \
  --host 127.0.0.1 --port 8090 \
  --max-context 262144 --kv-capacity auto --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

Masked-draft backends (`--spec dflash`, `--spec dflash2`) are not available for this model:
no companion draft model exists.

## Source

| Field | Value |
|---|---|
| Base repository | `Qwen/Qwen3.5-9B` |
| Revision | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Recipe id | `qwen3_5_9b-v1` |
| Converter revision | `9badf959` (branch `qwen3_5_9b`) |

## Evaluation

Causal perplexity (`ninfer-perplexity --quick --kv-dtype int8`, context 4096 / stride 2048,
corpus `ninfer-ppl-1m-v1`, 261,167 scored tokens, 2026-09-21):

| Domain | Tokens | Perplexity |
|---|---:|---:|
| chinese_reference | 65,510 | 5.48 |
| english_long_form | 65,455 | 7.61 |
| english_reference | 65,304 | 7.84 |
| ninfer_code | 64,898 | 2.01 |
| **overall** | 261,167 | **5.07** |

## Performance

See `docs/performance/qwen3.5-9b.md`.
