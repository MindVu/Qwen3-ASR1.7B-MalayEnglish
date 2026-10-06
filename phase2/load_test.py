"""
Phase 2: Concurrent Load Tester for Shared vLLM Engine.

Architecture:
                  ┌── Request 1 ──┐
                  ├── Request 2 ──┤
Async load tester ├── Request 3 ──┼──> Shared vLLM engine (Batch Size 1) → GPU
                  ├── Request 4 ──┤
                  └── Request N ──┘

All requests dispatch against one shared Qwen3ASRModel.LLM instance using asyncio.to_thread
bounded by an asyncio.Semaphore(concurrency).
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import (
    BUCKETS,
    compute_percentiles,
    compute_wer,
    load_benchmark_dataset,
    strip_language_tag,
)
from monitoring import GPUMonitor
from vllm_backend import VLLMInferenceBackend, get_vllm_backend

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phase2.load_test")


async def run_single_request(
    sem: asyncio.Semaphore,
    backend: VLLMInferenceBackend,
    sample: Dict[str, Any],
    req_index: int,
) -> Dict[str, Any]:
    """Execute a single transcription request against the shared vLLM backend."""
    t_submit = time.perf_counter()

    async with sem:
        t_start = time.perf_counter()
        queue_wait = t_start - t_submit

        try:
            # Dispatch to worker thread since Qwen3ASRModel.transcribe() is synchronous
            res = await asyncio.to_thread(
                backend.transcribe_detailed,
                audio=sample["audio"],
                context="",
                language=None,
            )
            t_end = time.perf_counter()
            exec_time = t_end - t_start
            e2e_latency = t_end - t_submit

            dur = max(float(sample["duration"]), 1e-3)
            rtf_client = e2e_latency / dur
            rtf_exec = exec_time / dur

            pred_text = res.get("text", "")
            tokens_approx = len(pred_text.split())

            return {
                "success": True,
                "request_index": req_index,
                "sample_id": sample["id"],
                "bucket": sample["bucket"],
                "audio_duration_s": round(dur, 3),
                "queue_wait_s": round(queue_wait, 4),
                "exec_time_s": round(exec_time, 4),
                "latency_e2e_s": round(e2e_latency, 4),
                "rtf": round(rtf_client, 4),
                "rtf_exec": round(rtf_exec, 4),
                "prediction": pred_text,
                "raw_prediction": res.get("raw_text", ""),
                "reference": sample.get("reference", ""),
                "generated_tokens": tokens_approx,
            }
        except Exception as e:
            t_end = time.perf_counter()
            logger.error("Request %d (%s) failed: %s", req_index, sample.get("id"), e)
            return {
                "success": False,
                "request_index": req_index,
                "sample_id": sample["id"],
                "bucket": sample["bucket"],
                "audio_duration_s": sample.get("duration", 0.0),
                "error": str(e),
                "latency_e2e_s": time.perf_counter() - t_submit,
            }


async def run_concurrent_benchmark(
    backend: VLLMInferenceBackend,
    samples: List[Dict[str, Any]],
    concurrency: int = 1,
    warmup_requests: int = 3,
    gpu_monitor: Optional[GPUMonitor] = None,
) -> Dict[str, Any]:
    """
    Executes concurrent workload against the shared vLLM engine at fixed concurrency level.
    """
    logger.info("Starting benchmark run for Concurrency = %d on %d samples (warmup=%d)...",
                concurrency, len(samples), warmup_requests)

    sem = asyncio.Semaphore(concurrency)

    # 1. Warm-up
    if warmup_requests > 0 and samples:
        warmup_subset = samples[:min(warmup_requests, len(samples))]
        logger.info("Executing %d warmup request(s)...", len(warmup_subset))
        for i, s in enumerate(warmup_subset):
            await asyncio.to_thread(backend.transcribe, s["audio"])
        logger.info("Warmup complete.")

    # 2. Start GPU monitor
    if gpu_monitor is not None:
        gpu_monitor.start()

    wall_start = time.perf_counter()

    # 3. Launch concurrent requests
    tasks = [
        run_single_request(sem, backend, sample, idx + 1)
        for idx, sample in enumerate(samples)
    ]
    raw_results = await asyncio.gather(*tasks)

    wall_time_s = time.perf_counter() - wall_start

    # 4. Stop GPU monitor
    gpu_stats = gpu_monitor.stop() if gpu_monitor is not None else {}

    # 5. Aggregate metrics
    successful = [r for r in raw_results if r.get("success")]
    failed = [r for r in raw_results if not r.get("success")]

    total_audio_s = sum(r["audio_duration_s"] for r in successful)
    total_tokens = sum(r["generated_tokens"] for r in successful)

    client_rtfs = [r["rtf"] for r in successful]
    exec_rtfs = [r["rtf_exec"] for r in successful]
    latencies = [r["latency_e2e_s"] for r in successful]
    queue_waits = [r["queue_wait_s"] for r in successful]

    # WER calculation
    preds = [r["prediction"] for r in successful if r.get("reference")]
    refs = [r["reference"] for r in successful if r.get("reference")]
    overall_wer = compute_wer(preds, refs) if refs else 0.0

    # Bucket breakdown
    bucket_breakdown: Dict[str, Any] = {}
    for name, _, _ in BUCKETS:
        b_items = [r for r in successful if r["bucket"] == name]
        if b_items:
            b_preds = [r["prediction"] for r in b_items if r.get("reference")]
            b_refs = [r["reference"] for r in b_items if r.get("reference")]
            bucket_breakdown[name] = {
                "n_requests": len(b_items),
                "rtf": compute_percentiles([r["rtf"] for r in b_items]),
                "wer": round(compute_wer(b_preds, b_refs), 4) if b_refs else None,
            }

    throughput_audio_s_per_wall_s = total_audio_s / max(wall_time_s, 1e-4)
    requests_per_s = len(successful) / max(wall_time_s, 1e-4)
    tokens_per_s = total_tokens / max(wall_time_s, 1e-4)

    summary = {
        "concurrency": concurrency,
        "config": {
            "backend": "vllm",
            "model": backend.model_path,
            "max_inference_batch_size": backend.max_inference_batch_size,
            "max_new_tokens": backend.max_new_tokens,
            "gpu_memory_utilization": backend.gpu_memory_utilization,
            "total_requests": len(samples),
            "warmup_requests": warmup_requests,
        },
        "status": {
            "successful_requests": len(successful),
            "failed_requests": len(failed),
        },
        "throughput": {
            "wall_time_s": round(wall_time_s, 3),
            "total_audio_s": round(total_audio_s, 2),
            "audio_s_per_wall_s": round(throughput_audio_s_per_wall_s, 3),
            "requests_per_s": round(requests_per_s, 3),
            "generated_tokens_per_s": round(tokens_per_s, 2),
        },
        "rtf": compute_percentiles(client_rtfs),
        "rtf_exec_only": compute_percentiles(exec_rtfs),
        "latency_e2e_s": compute_percentiles(latencies),
        "queue_wait_time_s": compute_percentiles(queue_waits),
        "wer": {
            "overall": round(overall_wer, 4),
            "n_evaluated": len(refs),
        },
        "bucket_breakdown": bucket_breakdown,
        "gpu": gpu_stats,
        "per_sample": raw_results,
    }

    logger.info("Results for Concurrency = %d:", concurrency)
    logger.info("  Avg RTF: %.4f | P50 RTF: %.4f | P95 RTF: %.4f",
                summary["rtf"]["avg"], summary["rtf"]["p50"], summary["rtf"]["p95"])
    logger.info("  Throughput: %.2f audio-sec/s (%.2f req/s)",
                throughput_audio_s_per_wall_s, requests_per_s)
    logger.info("  WER: %.2f%% (%d samples)", overall_wer * 100, len(refs))
    logger.info("  GPU Util: %.1f%% (Max: %.1f%%) | Peak VRAM: %s MB",
                gpu_stats.get("gpu_util_avg_pct", 0.0),
                gpu_stats.get("gpu_util_max_pct", 0.0),
                gpu_stats.get("gpu_mem_used_peak_mb", 0.0))

    return summary


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Concurrent vLLM Load Tester")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B", help="Base model path or HF ID")
    parser.add_argument("--test_file", default="data/test.jsonl", help="Path to test.jsonl")
    parser.add_argument("--audio_dir", default="data/audio_clips/test", help="Path to test audio directory")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrency level")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup requests")
    parser.add_argument("--gpu_mem", type=float, default=0.7, help="GPU memory utilization for vLLM")
    args = parser.parse_args()

    backend = get_vllm_backend(
        model_path=args.model,
        gpu_memory_utilization=args.gpu_mem,
        max_inference_batch_size=1,
    )
    samples = load_benchmark_dataset(test_file=args.test_file, audio_dir=args.audio_dir)
    gpu_monitor = GPUMonitor(device_index=0, interval=0.05)

    asyncio.run(
        run_concurrent_benchmark(
            backend=backend,
            samples=samples,
            concurrency=args.concurrency,
            warmup_requests=args.warmup,
            gpu_monitor=gpu_monitor,
        )
    )


if __name__ == "__main__":
    main()
