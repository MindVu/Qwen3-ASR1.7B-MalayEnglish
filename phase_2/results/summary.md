# Phase 3 — qwen-asr vLLM wrapper (offline batch) Benchmark Summary

## Batch-size sweep

| Batch size (C) | Requests | Failed | Avg RTF | P50 RTF | P95 RTF | Agg RTF | Audio sec/s | Req/s | Batch time (s) | GPU % | Steady Audio sec/s | Steady GPU % | CPU Sys % | CPU Client % | CPU Engine % | VRAM (MB) | WER | CER | Status (<=0.5) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 3072 | 0 | 0.0786 | 0.0454 | 0.1968 | 0.0007 | 1421.95 | 133.92 | 0.48 | 89.8% | 1422.48 | 90.4% | 1.5% | 25.2% | 97.0% | 37969.9 | 16.05% | 4.79% | PASS |
| 128 | 3072 | 0 | 0.1001 | 0.0577 | 0.2507 | 0.0004 | 2232.13 | 210.22 | 0.61 | 86.5% | 2231.54 | 86.5% | 1.6% | 34.0% | 95.2% | 37969.9 | 16.06% | 4.80% | PASS |
| 256 | 3072 | 0 | 0.1484 | 0.0859 | 0.3739 | 0.0003 | 3009.86 | 283.47 | 0.90 | 80.9% | 3012.95 | 81.9% | 1.7% | 45.0% | 90.2% | 37969.9 | 16.03% | 4.79% | PASS |
| 512 | 3072 | 0 | 0.2509 | 0.1440 | 0.6299 | 0.0003 | 3559.46 | 335.23 | 1.53 | 81.0% | 3629.50 | 81.8% | 1.6% | 53.6% | 88.8% | 37969.9 | 16.04% | 4.79% | FAIL |
| 1024 | 3072 | 0 | 0.4452 | 0.2576 | 1.1162 | 0.0002 | 4011.63 | 377.82 | 2.71 | 74.6% | 4013.62 | 76.0% | 1.7% | 60.3% | 84.6% | 37969.9 | 16.01% | 4.79% | FAIL |

## Success criterion (per-request RTF, informational)

- **P95 RTF ≤ 0.5:** maximum sustainable batch size = **256**
- **P95 RTF ≤ 0.3:** maximum sustainable batch size = **128**

## How to read the metrics

- One `transcribe(list)` call per batch; all requests of a batch finish together.
- **Avg/P50/P95 RTF** = batch wall time / the request's OWN audio duration. Pessimistic for short clips batched with long ones (they wait for the longest clip in the batch).
- **Agg RTF** = batch wall time / total audio of the batch (inverse of batch throughput).
- **Steady** columns exclude the first and last batch.
- Batch time includes everything inside the wrapper call: audio normalization, prompt building, vLLM generate and output parsing.

