# Inference Optimization Report: Qwen3-ASR 1.7B

## 1. Environment Specifications
* **GPU Model:** [e.g., NVIDIA L4]
* **GPU Memory:** [e.g., 24 GB]
* **CPU & RAM:** [e.g., 16 vCPUs, 64 GB RAM]
* **CUDA / PyTorch:** [e.g., CUDA 12.1 / PyTorch 2.2.0]
* **Precision:** BF16

## 2. Baseline Measurements (Single Stream HF)
| Bucket | Avg RTF | P50 RTF | P95 RTF | GPU Util | VRAM | Throughput |
|---|---|---|---|---|---|---|
| 2-5s | | | | | | |
| 5-15s | | | | | | |
| 15-30s | | | | | | |

## 3. Optimization Journey

| Configuration | Max Concurrent Streams | P95 RTF | VRAM Usage | WER |
|---|---|---|---|---|
| Baseline (HF pipeline, sequential) | 1 | | | |
| Opt 1 (BF16 + FlashAttention) | | | | |
| Opt 2 (Dynamic Batching) | | | | |
| Opt 3 (vLLM Continuous Batching) | | | | |
| Final (Optimized vLLM config) | | | | |

### Detailed Experiments
**Hypothesis 1:** GPU utilization is low because inference requests are processed sequentially without leveraging Tensor Cores effectively.
* **Change:** Enabled BF16 and FlashAttention-2 in the HuggingFace pipeline.
* **Measurement:** [Insert results]
* **Result:** [Insert analysis]

**Hypothesis 2:** [Insert hypothesis about batching]
* **Change:** [Insert change]
* **Measurement:** [Insert results]
* **Result:** [Insert analysis]

*Note on failed experiments:* [Document any configuration like `torch.compile` that degraded performance or OOM'd, and explain why].

## 4. Concurrent Benchmark Results (Final Config)
| Concurrent Streams | Avg RTF | P50 RTF | P95 RTF | GPU Util | VRAM Usage | Throughput |
|---|---|---|---|---|---|---|
| 1 | | | | | | |
| 2 | | | | | | |
| 4 | | | | | | |
| 8 | | | | | | |
| 16 | | | | | | |
| 32 | | | | | | |
| 64 | | | | | | |
| 128 | | | | | | |

**Maximum sustainable concurrent streams at P95 RTF ≤ 0.5:** `[X] streams`

## 5. Bottleneck Analysis
Based on profiling with `nvidia-smi dmon` and PyTorch profiler at [X] streams:
* **The primary bottleneck is [e.g., KV Cache Memory / Memory Bandwidth / Compute].**
* *Evidence:* [e.g., GPU utilization was only 65% but VRAM was 100% full, indicating we hit the `max_num_seqs` limit before saturating compute.]

## 6. Recommendations for Further Optimization
1. [e.g., Implement INT8 W8A16 quantization to reduce memory bandwidth requirements]
2. [e.g., Offload audio feature extraction (whisper/qwen processor) to a dedicated CPU worker pool to prevent CPU-bound stalls]
