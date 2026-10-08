# Phase 1: Baseline Inference Server & Benchmark Suite

This directory contains the complete implementation for **Phase 1: Baseline + Async Request Handling** for **Qwen3-ASR 1.7B**.

## Architecture Overview

```
Client 1 ─┐
Client 2 ─┤
Client 3 ─┼──> FastAPI (Async POST /transcribe) ──> Bounded Queue (max: 64) ──> Dedicated GPU Worker (Batch Size = 1)
Client 4 ─┘                                        (HTTP 429 if full)           (Sequential execution)
```

### Components

1. **`config.yaml`**: Centralized configuration for model path, precision, server host/port, bounded queue limit, benchmark parameters, and GPU monitoring intervals.
2. **`model.py`**: Reusable model loader (`Qwen3ASRModelWrapper`) that loads Qwen3-ASR 1.7B and its processor **strictly once** at startup, configures precision (`float32`, `float16`, `bfloat16`), and sets `eval()` mode.
3. **`inference.py`**: Pure inference module with a clean `transcribe(...) -> dict` interface. Fully decoupled from HTTP logic. Performs audio decoding/resampling, chat templating, `torch.inference_mode()` execution, and records sub-millisecond GPU-synchronized timing breakdown (`t_preprocess`, `t_generate`, `t_decode`, `rtf`).
4. **`server.py`**: High-performance FastAPI asynchronous server. Implements:
   - Lifespan context manager to load model once at startup.
   - Bounded inference queue (`asyncio.Queue(maxsize=64)`). Rejects excess requests with `HTTP 429 Too Many Requests`.
   - Dedicated single GPU worker task that drains requests one by one without dynamic batching.
   - Non-blocking client response futures that accurately measure `queue_wait_time` and `total_latency`.
5. **`monitoring.py`**: Background GPU monitoring daemon (`GPUMonitor`) that samples GPU utilization (%) and VRAM (MB) via `pynvml` (direct C-API) or fallback `nvidia-smi`.
6. **`benchmark.py` & `load_test.py`**: Async client benchmark suite that evaluates 2–5s, 5–15s, and 15–30s audio across concurrency levels 1→64.
7. **`run_phase1.py`**: **Single-terminal automated runner** designed specifically for cluster environments (e.g., NSCC `qsub -I`). Automatically spawns the server, waits for model readiness, runs the full benchmark sweep, and shuts down the server.

---

## How to Run

### Setup Environment
```bash
module load miniforge3
conda activate /scratch/users/ntu/csducmin/env/qwenasr
```

---

### Option A: Single-Terminal Run (Recommended for NSCC `qsub -I`)

Since NSCC interactive jobs only provide a single terminal session, use `run_phase1.py`:

```bash
python run_phase1.py
```

This single command:
1. Starts `server.py` in the background (logging output to `server.log`).
2. Polls `http://127.0.0.1:8000/health` until the model is loaded in GPU VRAM.
3. Runs the benchmark across concurrency levels (1, 2, 4, 8, 16, 32, 64).
4. Prints the summary table and saves `results/baseline_c*.json`, `baseline_summary.csv`, and `baseline_summary.md`.
5. Automatically stops the server when complete.

---

### Option B: Backgrounding in a Single Terminal (`&`)

You can also run both manually within the same terminal:

```bash
# 1. Start server in background
python server.py > server.log 2>&1 &
SERVER_PID=$!

# 2. Wait until server is healthy
until curl -s http://127.0.0.1:8000/health | grep -q '"status":"healthy"'; do
    echo "Waiting for model to load..."
    sleep 5
done
echo "Server is ready!"

# 3. Run benchmark in the same terminal
python benchmark.py --concurrency 1,2,4,8,16,32,64 --test-file data/test.jsonl

# 4. Stop the server
kill $SERVER_PID
```

---

### Option C: Connecting Across Nodes or From Local Laptop

If you want to access the server from another machine or login node:
- In `config.yaml` or via CLI, `server.py` listens on `0.0.0.0` (all interfaces on the node).
- To access from your **local laptop**, SSH port-forward to the compute node hostname:
  ```bash
  # On your compute node, find hostname:
  hostname
  # e.g., asp2a-compute-0042

  # From your local laptop terminal:
  ssh -L 8000:asp2a-compute-0042:8000 <username>@asp2a.nscc.sg
  ```
  Now `http://localhost:8000` on your laptop connects directly to your NSCC compute node!
