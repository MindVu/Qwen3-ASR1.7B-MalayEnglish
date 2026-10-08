# Phase 2 — vLLM Concurrency Benchmark Summary (continuous batching)

## Concurrency Sweep Results

| Concurrency | Requests | Failed | Avg RTF | P50 RTF | P95 RTF | Client Avg RTF | Client P95 RTF | Audio sec/s | Req/s | GPU % | Steady Audio sec/s | Steady GPU % | CPU Sys % | CPU Client % | CPU Engine % | VRAM (MB) | WER | CER | Status (<=0.5) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 192 | 0 | 0.0183 | 0.0175 | 0.0263 | 0.0183 | 0.0263 | 61.27 | 5.78 | 97.8% | 61.21 | 97.8% | 1.5% | 17.8% | 101.1% | 34359.9 | 17.24% | 5.10% | PASS |
| 2 | 192 | 0 | 0.0188 | 0.0181 | 0.0272 | 0.0188 | 0.0272 | 119.13 | 11.23 | 99.8% | 120.02 | 99.8% | 1.4% | 10.4% | 101.0% | 34361.9 | 16.50% | 4.94% | PASS |
| 4 | 192 | 0 | 0.0189 | 0.0182 | 0.0274 | 0.0189 | 0.0274 | 235.08 | 22.16 | 99.5% | 237.11 | 99.8% | 1.4% | 13.0% | 100.8% | 34361.9 | 16.53% | 4.93% | PASS |
| 8 | 192 | 0 | 0.0194 | 0.0186 | 0.0281 | 0.0194 | 0.0281 | 454.24 | 42.83 | 98.6% | 463.33 | 99.3% | 2.0% | 16.9% | 100.5% | 34361.9 | 16.45% | 4.92% | PASS |
| 16 | 192 | 0 | 0.0208 | 0.0199 | 0.0303 | 0.0208 | 0.0303 | 807.31 | 76.11 | 95.7% | 877.71 | 97.5% | 2.4% | 22.6% | 99.6% | 34361.9 | 16.48% | 4.92% | PASS |
| 32 | 192 | 0 | 0.0240 | 0.0230 | 0.0360 | 0.0240 | 0.0360 | 1308.86 | 123.40 | 91.1% | 1545.58 | 90.9% | 2.8% | 32.2% | 98.1% | 34361.9 | 16.53% | 4.94% | PASS |
| 64 | 192 | 0 | 0.0291 | 0.0268 | 0.0487 | 0.0291 | 0.0487 | 2018.43 | 190.29 | 87.8% | 2909.26 | 82.0% | 2.4% | 41.1% | 96.1% | 34361.9 | 16.50% | 4.94% | PASS |

## Comparison with Frozen Phase 1 HF Baseline

| Metric | HF Baseline (Phase 1) | vLLM (C=1) | vLLM Best Throughput |
|---|---:|---:|---:|
| **P95 RTF** | 0.387 | 0.0263 | 0.0487 |
| **Avg RTF** | 0.251 | 0.0183 | 0.0291 |
| **Client P95 RTF** | see Phase 1 `baseline_summary` | 0.0263 | 0.0487 |
| **Audio Throughput** | 4.71 sec/s | 61.27 sec/s | 2018.43 sec/s (C=64) |
| **GPU Utilization** | 33.5% | 97.8% | 87.8% |
| **Peak VRAM** | 5.9 GB | 33.55 GB | 33.56 GB |
| **WER** | 28.3% | 17.24% | 16.50% |
| **CER** | see Phase 1 `baseline_summary` | 5.10% | 4.94% |

## Success Criterion Assessment

- **Primary Target (P95 RTF ≤ 0.5):** Maximum Sustainable Concurrency = **64**
- **Strong Target (P95 RTF ≤ 0.3):** Maximum Sustainable Concurrency = **64**

