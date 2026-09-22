# Qwen3.5-9B serving performance

[Performance index](../performance.md) · [Measurement and publication rules](methodology.md) ·
[Port guide](../maintainer/qwen3.5-9b-port.md)

On this page: [run records](#scope-and-run-records), [context profile](#no-speculation-context-profile),
[single-request decode](#single-request-speculative-decode), [corpus makespan](#corpus-makespan),
[limitations](#comparisons-and-limitations), [reproduction](#reproduction-and-reports).

## Scope and run records

All runs use the [common RTX 5090 serving profile](methodology.md#common-serving-profile) with
stochastic sampling. This is the **untuned baseline** of the local `qwen3_5_9b` port: every leaf
is a composition of generic `linear` routes with no shape-specific tuning (port guide, Section 4).
The machine is the same nixos-desktop RTX 5090 that reproduced the 35B-A3B numbers in
`~/Projects/ninfer-repro/LOG.md`; driver 595.71.05, CUDA compile 13.1.

| Run | Campaign date | Weights ID | Measurement | Context ceiling | C | KV capacity (tokens) |
|---|---|---|---|---:|---|---|
| [B0](#b0) | 2026-09-21 | `groupwise-int` | MTP0 context profile | 262,144 | 1 | auto |
| [B3](#b3) | 2026-09-21 | `groupwise-int` | MTP3 serial corpus | 262,144 | 1 | auto |

| Run | Tested Git revision (branch `qwen3_5_9b`) |
|---|---|
| B0, B3 | `d2a0baca182b5c7b2f47ccf40ebf1d2dd0da60fd` |

Artifact: `qwen3_5_9b.ninfer`, SHA-256 `3a471aa18d59d0760db30e9ad7fcf339580deb55665cb47b8be362627133139c`, 6.11 GiB.

## No-speculation context profile

MTP0, [single-request method](methodology.md#single-request-phases), five samples per context.
Run [B0](#b0).

| Prompt tokens | Samples | Prefill phase (tok/s) | Server TTFT (ms) | Decode phase (tok/s) |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 10561.8 ± 35.8 | 730.9 ± 2.5 | 194.4 ± 0.7 |
| 64,512 | 5 | 8434.0 ± 25.9 | 7672.2 ± 23.7 | 172.6 ± 0.6 |
| 130,048 | 5 | 6661.6 ± 91.8 | 19570.2 ± 270.4 | 152.7 ± 1.4 |
| 260,096 | 5 | 4900.5 ± 20.5 | 53163.7 ± 221.7 | 127.0 ± 0.1 |

## Single-request speculative decode

Category rows pool 3 fixtures × 5 seeds ([method](methodology.md#single-request-phases)).

Run [B3](#b3), MTP3.

| Fixture | Samples | Completion tokens | Decode phase (tok/s) | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 8113.4 ± 3586.0 | 462.3 ± 11.2 | 81.7% ± 1.5% | 3.45 ± 0.04 |
| `long_decode_aime26_15` | 5 | 65536.0 ± 0.0 | 404.5 ± 9.2 | 73.6% ± 2.2% | 3.21 ± 0.06 |
| `long_decode_aime26_30` | 5 | 58658.4 ± 8503.3 | 423.9 ± 14.8 | 80.0% ± 4.3% | 3.40 ± 0.13 |

| Category | Samples | Decode phase (tok/s) | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 391.9 ± 20.8 | 63.9% ± 5.1% | 2.92 ± 0.15 |
| Story | 15 | 269.8 ± 17.2 | 34.1% ± 4.1% | 2.02 ± 0.12 |
| Translation | 15 | 391.8 ± 23.5 | 64.2% ± 3.8% | 2.93 ± 0.11 |
| Structured | 15 | 456.7 ± 41.5 | 81.3% ± 10.5% | 3.44 ± 0.32 |

## Corpus makespan

Not published: the serial runner reports per-request phases only (the same scope as the
27B's G3 run).

## Comparisons and limitations

- No published Qwen3.5-9B numbers exist upstream; the only comparison is against the 27B/35B
  pages on the same GPU. The 9B is an untuned composition baseline; the 27B/35B numbers come
  from shape-tuned fused kernels.
- MTP acceptance and decode rates are stochastic-sampling values as the methodology requires;
  the smoke tests in the port guide used greedy decoding.
- Output checks: termination reasons from `run.jsonl` are tabulated below; token accounting is
  the runner's; no content audit was performed. Every Code and Structured request ran to its
  4,096-token budget, as did 7 of 15 reasoning requests at 65,536 tokens: this model thinks and
  writes long. Throughput alone does not establish task completion.

| Category | Requests | Stop token | Output limit |
|---|---:|---:|---:|
| Context profile (128-token budget) | 20 | 0 | 20 |
| Reasoning (65,536-token budget) | 15 | 8 | 7 |
| Code (4,096) | 15 | 0 | 15 |
| Story (4,096) | 15 | 10 | 5 |
| Translation (4,096) | 15 | 15 | 0 |
| Structured (4,096) | 15 | 0 | 15 |
- Vision and DFlash: Vision was not benchmarked; DFlash is not available for this model.

## Reproduction and reports

Campaign reports: `~/Projects/ninfer-repro/ninfer-run-logs/bench-9b/` (`run.jsonl`,
`summary.csv`, `summary.md`, `server/`).

```bash
cd ~/Projects/ninfer-repro && ./bench-9b.sh      # wraps the command below
"$PY" tools/bench/run_serve_corpus.py --serve build-9b/apps/ninfer-serve \
  --artifact qwen3_5_9b=../out/qwen3_5_9b.ninfer --mode mtp0 --mode mtp3 \
  --port 8091 --output ../ninfer-run-logs/bench-9b
```

### B0

MTP0: no `--spec`; the four `long_niah_*` fixtures, five seeds each.

### B3

MTP3: `--spec mtp --draft-tokens 3 --lm-head-draft`; the 75-request corpus.
