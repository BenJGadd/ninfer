# Qwen3.5-9B serving performance (v3 engine)

[Performance index](../performance.md) · [Measurement and publication rules](methodology.md) ·
[Port guide](../maintainer/qwen3.5-9b-port.md)

**No corpus campaign was run for this branch** (skipped deliberately on 2026-09-22). The
figures below are single greedy smoke requests read from the server log, not methodology-grade
statistics; the pre-v3 branch `qwen3_5_9b` carries the full MTP0/MTP3 corpus measurements of the
same weights on the same machine, and its composed leaves are the same composition as this
branch's, so those numbers are the reference until this branch is benchmarked.

Revision `3e45a90a6e2784adcaeb16b83171291da9ff911a`, artifact SHA-256 `3d87bfe1735dc048…`, RTX 5090, `--kv-dtype int8`, MTP3 with the
shortlisted proposal head, 32,768-token context.

| Request | Prompt tokens | Output tokens | Prefill (tok/s) | Decode (tok/s) | MTP acceptance |
|---|---:|---:|---:|---:|---:|
| factual, thinking off | 29 | 16 | 695.5 | 425.0 | 86.7% |
| 704-token prompt, thinking off | 704 | 25 | 7,010 | 495.3 | 90.5% |
| arithmetic, thinking on | 42 | 587 | 876.4 | 522.5 | 90.7% |

Greedy outputs of the first two requests are byte-identical to the pre-v3 engine's. Perplexity
(quick, int8 KV) is 5.073971 overall on both engines.

## Weight-only NVFP4 artifact (`qwen3_5_9b_nvfp4`, 2026-09-22)

Same engine revision, artifact SHA-256 `c6ec1107dec5605484d4393872315e8fd3a6964098d13beab0de10e482a6871f`
(6,334,245,120 bytes, 732 objects, 176 NVFP4 projections), `--kv-dtype int8`, MTP3, 32,768-token
context. Smoke requests only, no corpus campaign yet:

| Request | Prompt tokens | Output tokens | Prefill (tok/s) | Decode (tok/s) |
|---|---:|---:|---:|---:|
| "Reply with exactly: nvfp4 ok" | 20 | 5 | 924.3 | 691.5 |
| capital of Australia | 24 | 8 | 1,060 | 605.3 |
| iterative Fibonacci function | 26 | 160 | 947.5 | 631.9 |

Quick perplexity (int8 KV, corpus `ninfer-ppl-1m-v1`, 261,167 scored tokens): overall **5.201481**
(chinese_reference 5.464, english_long_form 7.627, english_reference 8.622,
ninfer_code 2.023) against 5.073971 for the Q4/Q5 artifact — a 2.5% perplexity cost for
weight-only NVFP4 with in-repo block quantization and no calibration.

### Corpus campaign, A16 route (2026-09-22)

`tools/bench/run_serve_corpus.py`, modes mtp0 and mtp3, int8 KV, stochastic sampling, RTX 5090
at its 575 W power cap (79-82 °C, no thermal throttling). Full tables:
`campaigns/qwen3.5-9b-nvfp4-a16/summary.md`.

| Profile | Fixture / category | NVFP4 (A16) | Q4/Q5 (pre-v3 branch) |
|---|---|---:|---:|
| MTP0 prefill, 7,680 tokens | long_niah_8k | 1,834.9 ± 50.5 tok/s | ~10,700 tok/s |
| MTP0 prefill, 260,096 tokens | long_niah_256k | 1,544.7 ± 1.8 tok/s | — |
| MTP0 decode, 7,680-token context | long_niah_8k | 243.5 ± 2.1 tok/s | ~200 tok/s |
| MTP3 decode | code | 476.9 ± 32.6 tok/s (65.1% acc.) | 391.9 ± 20.8 |
| MTP3 decode | story | 343.9 ± 29.6 tok/s (37.4%) | 269.8 ± 17.2 |
| MTP3 decode | translation | 483.7 ± 24.9 tok/s (65.3%) | 391.8 ± 23.5 |
| MTP3 decode | structured | 580.8 ± 41.3 tok/s (86.3%) | 456.7 ± 41.5 |
| MTP3 long decode | aime26_01 | 560.0 ± 8.8 tok/s (82.6%) | — |

Decode is 22-27% faster than the Q4/Q5 artifact in every category, with the device wait at
6.1-6.6 ms per MTP3 round. Prefill is 5-6x *slower*: every NVFP4 site is `A16Only`, and the A16
route runs prompts through 32-token SIMT chunks. The W4A4 route (port guide §8.1) is the fix;
this campaign is the A16 baseline it is measured against.
