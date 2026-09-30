# 7-Day Plan: Qwen3-ASR 1.7B Assessment

## Day 0 (a few hours): Setup and decisions
- Rent or confirm a GPU (A100/L4/4090 all work; note the model and VRAM, since results are GPU-dependent).
- Read the Qwen3-ASR model card and repo. Check the official inference path, vLLM support, and any fine-tuning recipe. Don't rely on memory for this.
- Define "stream" in the README. My suggestion is one concurrent offline request on a 2-30s clip, with RTF = processing time / audio duration.
- Create the repo and a `results/` folder where every script writes JSON, so tables are generated rather than hand-typed.

## Day 1: Data and baseline evaluation
- Add a `data/` folder and download raw audio data. Use an LLM to classify the language of the audio clips, and filter the results to extract a high-quality 60-minute dataset of Malay, Malaysian English, or code-switched audio.
- Resample the final dataset to 16 kHz mono, filter to under 30s, normalize text, and split 90/10 by speaker or source to avoid leakage.
- Write `wer_eval.py` with one fixed text normalizer used everywhere.
- Run **pre-fine-tune WER** on the eval set. Build the benchmark set with 2-5s, 5-15s and 15-30s buckets, about 30 clips each.
- Record the environment (GPU, CPU, RAM, CUDA, PyTorch, framework, precision).

## Day 2: Fine-tuning
- Train **LoRA on the decoder, encoder frozen, BF16**. This is cheap, fits 30-60 minutes of data, and is easy to justify. If time allows, run a second experiment (different rank or LR) as a comparison.
- Log everything the brief lists: trainable parameters, batch size, gradient accumulation, LR, steps/epochs, precision, peak VRAM, duration and loss curve.
- Prove the training worked:
  - The weights differ from the base model (LoRA delta norm).
  - The checkpoint reloads.
  - Inference with the fine-tuned model runs.
  - WER after fine-tuning, plus a few before/after transcript examples.
- Write `FINETUNING_REPORT.md` the same day, while the details are fresh.

## Day 3: Baseline inference benchmark (Part 2)
- Run single-stream HF transformers on the bucketed set. Record avg, P50 and P95 RTF, GPU utilization (NVML sampling at 100 ms), VRAM and throughput in audio-seconds per wall-second.
- Write `bench_single.py` and `loadtest.py` (async client with a semaphore, ramping 1→2→4→8→16→32→64→128).
- Run the load test on the baseline. It will saturate early, and that becomes your "before" row.
- Add per-stage timers (feature extraction, encoder, queue wait, decode) from the start.

## Day 4: Optimization ladder (Parts 3 and 5)
Do one change at a time and write the Hypothesis → Change → Measurement → Result entry as you go.

| Step | Change | Hypothesis |
|---|---|---|
| Opt 1 | BF16 + FlashAttention/SDPA | Cheaper compute and memory |
| Opt 2 | Dynamic batching in a serving loop | GPU idle between sequential requests |
| Opt 3 | vLLM continuous batching | Decode is bandwidth-bound, so it needs batching plus paged KV cache |
| Opt 4 | Tune `max_num_seqs`, `gpu_memory_utilization`, max model length | KV-cache headroom limits batch size |
| Opt 5 | Move feature extraction to a CPU worker pool, pipelined | CPU preprocessing stalls the GPU at high concurrency |
| Opt 6 (optional) | FP8/INT8 quantization | More KV space, higher batch. Measure the WER cost |

Also include at least one experiment that probably won't help (for example `torch.compile` on the HF path, or a length-sorted batching variant), and explain why it failed. The brief explicitly rewards this.

## Day 5: Concurrency results and WER under load (Part 4)
- Run the full load test on each configuration, producing the table for 1→128 streams.
- Run WER at each concurrency level on the final config to show there's no quality degradation from batching or quantization.
- Report the **max sustainable streams at P95 RTF ≤ 0.5**, and also at ≤ 0.3.

## Day 6: Bottleneck analysis (Part 6)
- Profile the final config at its saturation point with `nvidia-smi dmon`, the PyTorch Profiler and the stage timers. Optional: one Nsight Systems trace.
- Decide what actually limits you: decode bandwidth, KV-cache capacity, encoder compute, CPU preprocessing or scheduling. Back the claim with a plot or table, for example GPU utilization and queue time versus concurrency.

## Day 7: Write-up and polish
- README with exact repro steps (pinned versions, one `make` or shell entry point per stage).
- Inference Optimization Report, including the journey table, failed experiments and recommendations for further optimization.
- Answer the 5 Final Questions. For the 1,000-session design: replicated vLLM workers behind a load balancer, length-aware routing, autoscaling on queue depth, a separate CPU preprocessing tier, and capacity math from your measured streams per GPU with headroom.
- Do a clean-clone run to verify reproducibility.

## Repo layout
```
finetune/    prepare_data.py, train_lora.py, eval_wer.py
serving/     hf_baseline.py, batched_server.py, vllm_server.py
bench/       bench_single.py, loadtest.py, gpu_monitor.py
results/     *.json (raw), plots/
reports/     FINETUNING_REPORT.md, INFERENCE_REPORT.md
README.md
```

## Risks and mitigations
- **Qwen3-ASR not supported in your chosen engine.** Check on Day 0, and keep the HF batched path as the fallback.
- **Data leakage or a tiny eval set inflating WER changes.** Split by speaker and report the eval duration.
- **Noisy benchmarks.** Do warmup runs, several repetitions, and a fixed audio set.
- **Time overrun.** If Day 4 slips, drop the quantization step. A clean, well-explained ladder scores higher than many half-measured steps.
