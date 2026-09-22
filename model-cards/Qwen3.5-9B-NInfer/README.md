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
| Size | 6,558,595,840 bytes (6.11 GiB) |
| SHA-256 | `9046ad3ad6ba7813e6ca69c6b1a6d26abc8543b2c673effd29e43556e5c940b9` |
| Container version | 3 |
| Name | `qwen3.5-9b` |
| Recipe | `qwen3_5_9b` |
| Converter revision | `3e45a90a` |

716 objects (710 tensors, 6 resources): BF16 358 · FP32 48 · INT32 1 · Q4 95 · Q5 174 · Q6 3 ·
Q8 31. `generation_config.json` is repository-pinned (`tools/frontend_resources/qwen3_5_9b/`)
because the upstream checkpoint ships none. `chat_template.jinja` is the maintained
`tools/chat_templates/qwen3_5.jinja`: the checkpoint template plus the `developer` role alias
(the checkpoint template raises "Unexpected message role" on it, which breaks OpenAI-style
clients such as pi that send their system prompt as `developer`).

```bash
printf '%s  %s\n' '9046ad3ad6ba7813e6ca69c6b1a6d26abc8543b2c673effd29e43556e5c940b9' 'qwen3_5_9b_v3.ninfer' | sha256sum --check
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

## NVFP4 variant

`qwen3_5_9b_nvfp4_v3.ninfer` — recipe `qwen3_5_9b_nvfp4` + override
`qwen3_5_9b_nvfp4_a4.py`: NVFP4 quantized in-repo from the BF16 checkpoint (`nvfp4_blockwise`)
with calibrated W4A4 activation divisors (port guide §8-8.1). 6,334,364,164 bytes, SHA-256
`3d5d3f938166ebacec463c6004a2d2c79e4a4ca37aa082f1e64eb903ee7e04d0`, 988 objects (176 NVFP4
projections + 256 divisors; a/b Q8, endpoints Q6, vision and MTP unchanged). Served with the
same flags and `--model-id qwen3.5-9b-nvfp4`. Quick perplexity 5.30 against 5.07 for the
artifact above; prefill 33k tok/s against ~10.7k, decode ~610 tok/s against ~450 on the same
smoke prompts.

## Defiant Fable variant (uncensored fine-tune)

`qwen3_5_9b_defiant_nvfp4_v3.ninfer` — the same recipe, override and engine route applied to
`DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP` (16-bit
safetensors, revision 7af0a9c4) with its own calibration; official tokenizer files (port guide
§8.3). 6,334,364,164 bytes, SHA-256
`ad55284cc744904551ed030e95e1fa36bd67c71adbe96a8f544de31f673fd84a`. `--model-id
qwen3.5-9b-defiant-nvfp4`. Quick perplexity 5.01. Not scored for capability or refusal
behaviour: treat it as unmeasured beyond loading, answering and perplexity.
