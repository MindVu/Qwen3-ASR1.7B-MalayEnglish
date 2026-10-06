# Phase 2 — vLLM Migration & Concurrent Inference

This directory contains the complete implementation for **Phase 2: vLLM Migration & Concurrent Inference** for **Qwen3-ASR 1.7B**.

---

## 1. Goal & Hypothesis

* **Goal:** Replace the HuggingFace `generate()` inference backend with the official Qwen-ASR vLLM backend (`Qwen3ASRModel.LLM`), then measure how well a single shared vLLM engine handles simultaneous ASR requests across concurrency levels $C \in [1, 2, 4, 8, 16, 32, 64]$.
* **Hypothesis:** The HF batch-1 generation path from Phase 1 underutilizes the A100 GPU (GPU utilization $\approx 33.5\%$, `generate()` accounted for $99.8\%$ of latency). vLLM's internal continuous batching and PagedAttention request scheduler will turn this idle GPU headroom into useful parallel execution, reducing P95 RTF and increasing throughput under concurrency.

---

## 2. Frozen Phase 1 Baseline

To ensure scientific rigor, all Phase 1 variables remain strictly frozen:
* **Model:** `Qwen/Qwen3-ASR-1.7B`
* **Fine-Tuned Checkpoint:** LoRA adapter (`runs/lora_r16/checkpoint-42`)
* **Audio Specs:** 16 kHz mono, duration buckets ($<2s$, $2-5s$, $5-15s$, $15-30s$, $>30s$)
* **Generation Parameters:** `max_new_tokens=512`, `temperature=0.0`
* **Precision:** `bfloat16`
* **Dataset:** 43 benchmark utterances (`data/test.jsonl`)
* **Warmup:** 3 requests
* **Evaluation:** Standardized text normalization (`prepare_data.normalize_text`), language tag stripping (`<asr_text>`), and dynamic programming / evaluate WER computation.

### Baseline Reference (Phase 1 HF Evaluation):
| Metric | HF Baseline (Phase 1) |
|---|---:|
| **P95 RTF** | **0.387** |
| **Avg RTF** | **0.251** |
| **Audio Throughput** | **4.71 sec/s** |
| **GPU Utilization** | **33.5%** |
| **Peak VRAM** | **5.9 GB** |
| **WER** | **28.3%** |

---

## 3. Architecture Overview

```text
                                  ┌── Request 1 ──┐
                                  ├── Request 2 ──┤
Async Load Tester (Semaphore = C) ├── Request 3 ──┼──> Shared vLLM Engine (max_batch_size = 1) ──> GPU
                                  ├── Request 4 ──┤
                                  └── Request N ──┘
```

1. **Shared Model Instance:** A single instance of `Qwen3ASRModel.LLM` is loaded into GPU VRAM once at startup. No per-request model loading.
2. **`max_inference_batch_size = 1`:** Intentionally configured to 1 per `transcribe()` invocation. This allows individual requests to enter vLLM independently so vLLM's internal continuous batching scheduler manages concurrent streams rather than manual client-side batch collation.
3. **Async Dispatch:** Because `Qwen3ASRModel.transcribe()` is synchronous, the async load tester dispatches requests concurrently via `asyncio.to_thread` bounded by `asyncio.Semaphore(concurrency)`.

---

## 4. File Structure

```text
phase2/
├── common.py          # Normalized WER, dataset loading, duration buckets, percentiles
├── vllm_backend.py    # Official Qwen3ASRModel.LLM wrapper, single-model singleton
├── monitoring.py      # Background GPU sampler (NVML / nvidia-smi fallback)
├── load_test.py       # Async concurrent load tester for shared vLLM engine
├── benchmark_vllm.py  # Concurrency sweep (C=1→64), metrics collector, Markdown/CSV report
├── config.yaml        # Centralized Phase 2 parameters
└── results/           # Per-concurrency JSON results & summary tables
    ├── concurrency_1.json
    ├── concurrency_2.json
    ├── concurrency_4.json
    ├── concurrency_8.json
    ├── concurrency_16.json
    ├── concurrency_32.json
    ├── concurrency_64.json
    ├── summary.csv
    └── summary.md
```

Also exposed through the modular `inference/` package:
```text
inference/
├── common.py
└── vllm_backend.py
```

---

## 5. Usage & Execution Guide

### Step 1: Offline LoRA Weight Merging (Matching Phase 1 Wrapper)
Since vLLM loads weights directly from model directory checkpoints, merge the fine-tuned LoRA adapter (`runs/lora_r16/checkpoint-42`) into the base model using the official `qwen_asr` wrapper (matching Phase 1):
```bash
python phase2/merge_lora.py \
    --base_model Qwen/Qwen3-ASR-1.7B \
    --adapter runs/lora_r16/checkpoint-42 \
    --output_dir models/lora_r16_merged
```

### Step 2: Run Concurrency Sweep with vLLM
### Option 1: Full Concurrency Sweep ($C = 1 \to 64$)
Sweep concurrency levels $1, 2, 4, 8, 16, 32, 64$:
```bash
python phase2/benchmark_vllm.py --model models/lora_r16_merged --concurrencies 1,2,4,8,16,32,64
```

### Option 2: Single Concurrency Baseline ($C = 1$)
Establish direct comparison between HF ($C=1$) and vLLM ($C=1$):
```bash
python phase2/benchmark_vllm.py --model models/lora_r16_merged --concurrencies 1
```

### Option 3: Target Concurrency Load Test
Run a specific concurrency level using the standalone load tester:
```bash
python phase2/load_test.py --model models/lora_r16_merged --concurrency 8
```

---

## 6. Success Criteria

The primary requirement of the assessment:
* **Primary Target:** Maximum sustainable concurrency where $\text{P95 RTF} \le 0.5$.
* **Strong Target:** Maximum sustainable concurrency where $\text{P95 RTF} \le 0.3$.

The benchmark suite automatically evaluates both targets and annotates `PASS` / `FAIL` per concurrency level in `phase2/results/summary.md`.
