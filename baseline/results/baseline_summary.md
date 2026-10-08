# Phase 1: Baseline Inference Benchmark Summary

Hardware: NVIDIA A100-SXM4-40GB | Precision: FP32 / Default | Architecture: Single GPU Worker (Batch Size = 1) + Bounded Async Queue

| Concurrency | Avg RTF | P50 RTF | P95 RTF | Throughput | GPU Util. | VRAM |
| ----------: | ------: | ------: | ------: | ---------: | --------: | ---: |
|           1 |  0.1148 |  0.1038 |  0.1797 | 10.44 audio-s/s | 40.2% | 5883.9 MB |
|           2 |  0.1148 |  0.1038 |  0.1797 | 10.46 audio-s/s | 40.7% | 5883.9 MB |
|           4 |  0.1148 |  0.1038 |  0.1797 | 10.47 audio-s/s | 41.0% | 5883.9 MB |
|           8 |  0.1148 |  0.1038 |  0.1797 | 10.47 audio-s/s | 41.0% | 5883.9 MB |
|          16 |  0.1148 |  0.1038 |  0.1797 | 10.47 audio-s/s | 41.0% | 5883.9 MB |
|          32 |  0.1148 |  0.1038 |  0.1797 | 10.47 audio-s/s | 41.0% | 5883.9 MB |
|          64 |  0.1148 |  0.1038 |  0.1797 | 10.47 audio-s/s | 41.0% | 5883.9 MB |

### Duration Category Breakdown (Empirical Baseline)
| Bucket | Count | Avg Audio Dur. | Avg RTF | P50 RTF | P95 RTF |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **2–5s** | 13 | 3.65s | 0.1260 | 0.1218 | 0.1554 |
| **5–15s** | 19 | 10.42s | 0.0917 | 0.0938 | 0.1192 |
| **15–30s** | 4 | 18.25s | 0.0838 | 0.0899 | 0.1021 |
| **Overall** | 43 | 8.18s | 0.1148 | 0.1038 | 0.1797 |

### Latency Breakdown
- Preprocessing (Audio -> Tensor): 2.9 ms (0.4%)
- Model Generation (`model.generate`): 780.5 ms (99.6%)
- Decoding (`tokenizer.batch_decode`): 0.2 ms (<0.1%)
- Total Average Latency per Request: ~783.6 ms
