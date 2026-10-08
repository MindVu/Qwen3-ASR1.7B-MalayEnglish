# Qwen3-ASR 1.7B Assessment

This repository contains the codebase for evaluating, fine-tuning, and optimizing the **Qwen/Qwen3-ASR-1.7B** model on a code-switched (Malay-English) dataset. The project progresses through data preparation, LoRA fine-tuning, and a staged inference optimization ladder to scale throughput using vLLM.

## Project Structure

- **`data/` & `process/`**: Scripts to download raw audio, classify language using Qwen3-8B, and form the 60-minute code-switched evaluation and training dataset.
- **`finetune/`**: Contains scripts to prepare the dataset, train a LoRA adapter on the decoder, and evaluate Word Error Rate (WER).
- **`baseline/`** (Phase 1): Baseline inference benchmark using a sequential HuggingFace server over HTTP to establish the initial performance (throughput and RTF).
- **`phase_2/`**: Offline batch load tester using the official `Qwen3ASRModel.LLM` vLLM wrapper to test static batch sizes.
- **`phase_3/`**: Concurrent load tester for a shared `AsyncLLMEngine` (vLLM) with continuous batching to maximize sustained throughput and hardware saturation.
- **`plot_graph.py`**: Utility to visualize and plot training loss and GPU memory usage from fine-tuning logs.

---

## Quick Start

### 0. Install Dependencies
Install the required packages for fine-tuning, inference, and benchmarking:
```bash
pip install -r requirements.txt
```

### 1. Data Preprocessing
*(Note: If the dataset is already prepared in `data/`, you can skip to step 2.)*
To download and filter the raw audio into the 60-minute code-switched benchmark, run the processing scripts:
```bash
cd data
# Download the dataset from Hugging Face
python ../process/download_data.py

# Classify languages (Malay, English, Code-switched)
python ../process/language_classify.py

# Form the final 60-minute subset
python ../process/form_final_dataset.py
cd ..
```

### 2. Fine-Tuning (LoRA)
**Prepare the data:**
```bash
python finetune/prepare_data.py
```
*This filters the dataset, resamples to 16kHz mono, normalizes text, and splits into a 90/10 train/validation split.*

**Run Training:**
```bash
python finetune/train_lora.py   --train_file data/train.jsonl   --eval_file data/valid.jsonl   --output_dir ./runs/lora_r16_1   --use_lora
```
*This freezes the encoder, applies LoRA to the decoder (`q_proj`, `v_proj`), and saves the fine-tuned checkpoint.*

**Visualize Training Logs:**
```bash
python plot_graph.py
```
*Extracts training loss and GPU memory usage to generate visual plots.*

### 3. Benchmarking Inference (The Optimization Ladder)

We measure Real-Time Factor (RTF) and Throughput (audio-s/s) across different concurrency/batch levels ($C=1, 2, 4, 8, 16, 32, 64$).

**Baseline (Sequential HF HTTP Server):**
```bash
cd baseline
# Start the server (in one terminal)
python run_phase1.py
```

**Phase 2 (Static Batching with vLLM):**
```bash
cd phase_2
python benchmark_vllm.py --model ../models/lora_r16_merged
```

**Phase 3 (Continuous Batching with Async vLLM):**
```bash
cd phase_3
python benchmark_vllm.py --model ../models/lora_r16_merged
```

*(Results for each phase are automatically saved to their respective `results/` folder in Markdown, CSV, and JSON formats).*
