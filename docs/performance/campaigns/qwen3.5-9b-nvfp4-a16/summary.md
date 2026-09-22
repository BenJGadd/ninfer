# Serving corpus performance summary

All values are arithmetic mean ± sample standard deviation. Sampling: stochastic.

## MTP0 context-length profile

| Target | Weights | Fixture | n | Prompt tokens | Prefill tok/s | Server TTFT ms | Decode tok/s | Host exposure ms | Decode Host us/round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3_5_9b_nvfp4 | cd986128587168d5d5a00752d5180bf27e4a611e24eb5f04bfc4bee55f9dbf5f | long_niah_8k | 5 | 7680.0 ± 0.0 | 1834.9 ± 50.5 | 4191.7 ± 119.5 | 243.5 ± 2.1 | 3456.4 ± 98.8 | 7.4 ± 2.3 |
| qwen3_5_9b_nvfp4 | cd986128587168d5d5a00752d5180bf27e4a611e24eb5f04bfc4bee55f9dbf5f | long_niah_64k | 5 | 64512.0 ± 0.0 | 1759.9 ± 14.8 | 36690.0 ± 312.4 | 212.4 ± 0.9 | 30505.8 ± 263.3 | 7.5 ± 2.1 |
| qwen3_5_9b_nvfp4 | cd986128587168d5d5a00752d5180bf27e4a611e24eb5f04bfc4bee55f9dbf5f | long_niah_128k | 5 | 130048.0 ± 0.0 | 1684.1 ± 3.7 | 77293.3 ± 168.7 | 184.0 ± 0.5 | 63968.5 ± 136.6 | 7.5 ± 1.1 |
| qwen3_5_9b_nvfp4 | cd986128587168d5d5a00752d5180bf27e4a611e24eb5f04bfc4bee55f9dbf5f | long_niah_256k | 5 | 260096.0 ± 0.0 | 1544.7 ± 1.8 | 168527.5 ± 196.2 | 147.0 ± 0.4 | 138345.8 ± 162.8 | 7.9 ± 1.4 |

## MTP3 long-decode reasoning

| Target | Weights | Fixture | n | Completion tokens | Decode tok/s | Decode Host us/round | Device wait us/round | Spec acceptance | Spec tokens/round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3_5_9b_nvfp4 | cd986128587168d5d5a00752d5180bf27e4a611e24eb5f04bfc4bee55f9dbf5f | long_decode_aime26_01 | 5 | 7238.6 ± 1174.8 | 560.0 ± 8.8 | 13.1 ± 0.2 | 6196.3 ± 29.8 | 82.6% ± 1.5% | 3.48 ± 0.05 |
| qwen3_5_9b_nvfp4 | cd986128587168d5d5a00752d5180bf27e4a611e24eb5f04bfc4bee55f9dbf5f | long_decode_aime26_15 | 5 | 64011.2 ± 3409.6 | 483.1 ± 6.6 | 12.9 ± 0.1 | 6658.2 ± 27.3 | 74.1% ± 1.3% | 3.22 ± 0.04 |
| qwen3_5_9b_nvfp4 | cd986128587168d5d5a00752d5180bf27e4a611e24eb5f04bfc4bee55f9dbf5f | long_decode_aime26_30 | 5 | 56544.4 ± 10411.4 | 500.7 ± 8.5 | 13.2 ± 0.3 | 6589.9 ± 81.3 | 76.8% ± 1.9% | 3.31 ± 0.06 |

## MTP3 cross-scenario decode

| Target | Weights | Category | n | Decode tok/s | Decode Host us/round | Device wait us/round | Spec acceptance | Spec tokens/round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3_5_9b_nvfp4 | cd986128587168d5d5a00752d5180bf27e4a611e24eb5f04bfc4bee55f9dbf5f | code | 15 | 476.9 ± 32.6 | 13.6 ± 0.7 | 6181.9 ± 74.5 | 65.1% ± 6.7% | 2.95 ± 0.20 |
| qwen3_5_9b_nvfp4 | cd986128587168d5d5a00752d5180bf27e4a611e24eb5f04bfc4bee55f9dbf5f | story | 15 | 343.9 ± 29.6 | 13.3 ± 0.3 | 6159.8 ± 5.4 | 37.4% ± 6.1% | 2.12 ± 0.18 |
| qwen3_5_9b_nvfp4 | cd986128587168d5d5a00752d5180bf27e4a611e24eb5f04bfc4bee55f9dbf5f | translation | 15 | 483.7 ± 24.9 | 14.3 ± 0.9 | 6098.6 ± 5.9 | 65.3% ± 5.1% | 2.96 ± 0.15 |
| qwen3_5_9b_nvfp4 | cd986128587168d5d5a00752d5180bf27e4a611e24eb5f04bfc4bee55f9dbf5f | structured | 15 | 580.8 ± 41.3 | 13.2 ± 0.2 | 6165.8 ± 3.8 | 86.3% ± 8.6% | 3.59 ± 0.26 |
