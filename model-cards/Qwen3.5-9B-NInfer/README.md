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
| Size | @ARTIFACT_BYTES@ bytes (@ARTIFACT_GIB@ GiB) |
| SHA-256 | `@ARTIFACT_SHA256@` |
| Container version | 2 |
| NInfer model ID | `qwen3.5-9b` |
| NInfer weights ID | `groupwise-int` |
| NInfer target key | `qwen3_5_9b` |

The file contains the registered Text, Vision, MTP, proposal-head, tokenizer, chat-template,
generation, and media-processor objects required by NInfer (740 objects: 734 tensors, 6
resources). `generation_config.json` is a repository-pinned file because the upstream
checkpoint ships none.

```bash
printf '%s  %s\n' '@ARTIFACT_SHA256@' 'qwen3_5_9b.ninfer' | sha256sum --check
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
| Converter revision | @CONVERTER_REVISION@ |

## Performance

See `docs/performance/qwen3.5-9b.md`.
