"""
phase 3: Concurrent Load Tester for Shared vLLM Engine (continuous batching).

Same workload as Phase 1, but bypasses HTTP and submits requests directly
to one shared AsyncLLM engine.

Phase 1:
    async HTTP requests -> ASR server -> vLLM

phase 3:
    asyncio tasks -> AsyncVLLMBackend -> AsyncLLM (continuous batching)

RTF definition (identical to Phase 1):

    RTF = inference processing time / audio duration

    inference processing time = time spent in the backend call (backend-measured
    `latency_s`), i.e. from submitting the request to the engine until its
    result is complete. It excludes the client-side semaphore wait and audio
    decoding (audio is preloaded). Audio duration is the dataset `duration`.
    Under continuous batching this time naturally grows with concurrency because
    requests share GPU steps; that is part of the request's processing time.

Secondary metric (matches Phase 1's `rtf_client_e2e`):

    client RTF = (time from dispatch, i.e. semaphore acquired, to result) / audio duration

    Same boundaries as Phase 1's client RTF (excludes the client semaphore wait).
    Here it equals the wall time of the backend call, so it also covers coroutine
    scheduling and chunking overhead on top of the engine time.

Workload (default, configurable via --requests_per_bucket):
    2-5s   : 30 requests
    5-15s  : 30 requests
    15-30s : 30 requests
    Total  : 90 requests / concurrency level
"""

import argparse
import asyncio
import logging
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
    compute_cer,
    compute_wer,
    load_benchmark_dataset,
)
from monitoring import GPUMonitor, format_cpu_summary, steady_state_window
from vllm_backend import AsyncVLLMBackend


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("phase_3.load_test")


REQUESTS_PER_BUCKET = 30
DEFAULT_CONCURRENCIES = [1, 2, 4, 8, 16, 32, 64]
DEFAULT_WARMUP = 16
BUCKET_ORDER = ["2-5s", "5-15s", "15-30s"]   # used for logging / reporting
# Request order in the workload: longest -> shortest, interleaved round-robin
# (15-30s, 5-15s, 2-5s, 15-30s, 5-15s, 2-5s, ...). Keep identical in Phase 1.
POOL_ORDER = ["15-30s", "5-15s", "2-5s"]


def preload_audio(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Decode every clip once, before the sweep, so the benchmark measures the
    engine and not disk reads / resampling inside the event loop.
    Converts sample["audio"] from a path into a (np.ndarray, sample_rate) tuple,
    which normalize_audios() accepts without further decoding.
    """
    from qwen_asr.inference.utils import SAMPLE_RATE, normalize_audios

    n = 0
    for s in samples:
        if isinstance(s["audio"], str):
            wav = normalize_audios(s["audio"])[0]
            s["audio"] = (wav, SAMPLE_RATE)
            n += 1
    logger.info("Preloaded %d audio clips into memory.", n)
    return samples


async def run_single_request(
    sem: asyncio.Semaphore,
    backend: AsyncVLLMBackend,
    sample: Dict[str, Any],
    req_index: int,
) -> Dict[str, Any]:
    t_submit = time.perf_counter()

    async with sem:
        t_start = time.perf_counter()
        semaphore_wait = t_start - t_submit

        try:
            # Native coroutine: no thread hop. Many of these run inside the
            # engine at once and are continuously batched by vLLM.
            result = await backend.transcribe_detailed(
                audio=sample["audio"],
                context="",
                language=None,
            )

            t_end = time.perf_counter()

            call_time = t_end - t_start
            e2e_latency = t_end - t_submit

            # Inference processing time: measured inside the backend call.
            inference_time = float(result.get("latency_s", call_time))

            audio_duration = max(float(sample["duration"]), 1e-3)

            # RTF = inference processing time / audio duration
            rtf = inference_time / audio_duration

            # Client RTF = dispatch -> result (same boundaries as Phase 1's client RTF)
            client_rtf = call_time / audio_duration

            return {
                "success": True,
                "req_index": req_index,
                "bucket": sample["bucket"],
                "audio_duration_s": audio_duration,
                "semaphore_wait_s": round(semaphore_wait, 4),
                "inference_time_s": round(inference_time, 4),
                "call_time_s": round(call_time, 4),
                "e2e_latency_s": round(e2e_latency, 4),
                "rtf": round(rtf, 4),
                "client_rtf": round(client_rtf, 4),
                "t_start_perf": t_start,
                "t_done_perf": t_end,
                "prediction": result.get("text", ""),
                "reference": sample.get("reference", ""),
            }

        except Exception as e:
            logger.exception("Request %d failed", req_index)

            return {
                "success": False,
                "req_index": req_index,
                "bucket": sample.get("bucket"),
                "audio_duration_s": float(sample["duration"]),
                "semaphore_wait_s": round(time.perf_counter() - t_submit, 4),
                "inference_time_s": None,
                "e2e_latency_s": None,
                "rtf": None,
                "prediction": "",
                "reference": sample.get("reference", ""),
                "error": str(e),
            }


# def build_phase1_measurement_pool(
#     samples_by_bucket: Dict[str, List[Dict[str, Any]]],
#     requests_per_bucket: int = REQUESTS_PER_BUCKET,
# ) -> List[Dict[str, Any]]:
#     """
#     Exactly reproduce Phase 1's measurement pool construction:
#     for each bucket, append requests_per_bucket samples, cycling through
#     available samples if necessary.
#     """
#     measurement_pool: List[Dict[str, Any]] = []

#     for bucket_name in BUCKET_ORDER:
#         bucket_samples = samples_by_bucket.get(bucket_name, [])

#         if not bucket_samples:
#             logger.warning("Bucket %s has no samples and will be skipped.", bucket_name)
#             continue

#         for i in range(requests_per_bucket):
#             measurement_pool.append(bucket_samples[i % len(bucket_samples)])

#     return measurement_pool

def build_phase1_measurement_pool(
    samples_by_bucket: Dict[str, List[Dict[str, Any]]],
    requests_per_bucket: int = REQUESTS_PER_BUCKET,
) -> List[Dict[str, Any]]:
    """
    Build the measurement workload (same construction as Phase 1).

    Each bucket contributes `requests_per_bucket` requests, cycling through its
    clips if it has fewer clips than requests. Requests are INTERLEAVED
    round-robin in POOL_ORDER, so the order is:

        15-30s[0], 5-15s[0], 2-5s[0], 15-30s[1], 5-15s[1], 2-5s[1], ...

    This keeps the mix of clip lengths roughly constant throughout the run
    (instead of all short clips first and all long clips last), and means the
    warmup (taken from the front of the pool) touches every bucket.
    """
    active = []
    for bucket_name in POOL_ORDER:
        bucket_samples = samples_by_bucket.get(bucket_name, [])

        if not bucket_samples:
            logger.warning("Bucket %s has no samples and will be skipped.", bucket_name)
            continue

        active.append(bucket_samples)

    measurement_pool: List[Dict[str, Any]] = []
    for i in range(requests_per_bucket):
        for bucket_samples in active:
            measurement_pool.append(bucket_samples[i % len(bucket_samples)])

    return measurement_pool


async def run_concurrent_benchmark(
    backend: AsyncVLLMBackend,
    samples_by_bucket: Dict[str, List[Dict[str, Any]]],
    concurrency: int,
    requests_per_bucket: int = REQUESTS_PER_BUCKET,
    warmup_requests: int = DEFAULT_WARMUP,
    gpu_monitor: Optional[GPUMonitor] = None,
) -> Dict[str, Any]:

    logger.info("=" * 70)
    logger.info("RUNNING BENCHMARK - CONCURRENCY LEVEL: %d", concurrency)
    logger.info("=" * 70)

    if concurrency > backend.max_num_seqs:
        logger.warning(
            "concurrency=%d > engine max_num_seqs=%d: extra requests will queue inside vLLM.",
            concurrency,
            backend.max_num_seqs,
        )

    measurement_pool = build_phase1_measurement_pool(
        samples_by_bucket=samples_by_bucket,
        requests_per_bucket=requests_per_bucket,
    )

    logger.info("Measurement pool: %d requests", len(measurement_pool))

    for bucket_name in BUCKET_ORDER:
        count = sum(1 for s in measurement_pool if s["bucket"] == bucket_name)
        logger.info("  %s: %d requests", bucket_name, count)

    sem = asyncio.Semaphore(concurrency)

    # ------------------------------------------------------------
    # 1. Warm-up
    # ------------------------------------------------------------

    if warmup_requests > 0 and measurement_pool:
        logger.info("Warming up engine with %d requests...", warmup_requests)

        warmup_tasks = [
            run_single_request(
                sem=sem,
                backend=backend,
                sample=measurement_pool[i % len(measurement_pool)],
                req_index=-(i + 1),
            )
            for i in range(warmup_requests)
        ]

        warmup_results = await asyncio.gather(*warmup_tasks)

        warmup_failed = [r for r in warmup_results if not r.get("success")]
        if warmup_failed:
            logger.error(
                "Warmup: %d/%d requests FAILED. First error: %s",
                len(warmup_failed),
                len(warmup_results),
                warmup_failed[0].get("error"),
            )
            if len(warmup_failed) == len(warmup_results):
                raise RuntimeError(
                    "All warmup requests failed; the engine is not working. "
                    f"Error: {warmup_failed[0].get('error')}"
                )

        logger.info("Warmup complete.")

    # ------------------------------------------------------------
    # 2. Start GPU monitoring
    # ------------------------------------------------------------

    if gpu_monitor is not None:
        gpu_monitor.start()

    # ------------------------------------------------------------
    # 3. Measurement
    # ------------------------------------------------------------

    logger.info(
        "Dispatching %d benchmark requests at concurrency %d...",
        len(measurement_pool),
        concurrency,
    )

    wall_start = time.perf_counter()
    wall_start_epoch = time.time()  # same instant on the clock the GPU/CPU monitor uses

    tasks = [
        asyncio.create_task(
            run_single_request(
                sem=sem,
                backend=backend,
                sample=sample,
                req_index=i + 1,
            )
        )
        for i, sample in enumerate(measurement_pool)
    ]

    raw_results = await asyncio.gather(*tasks)

    wall_time_s = time.perf_counter() - wall_start

    # Convert per-request timestamps to seconds since the start of the measurement
    for r in raw_results:
        if r.get("success"):
            r["t_start_s"] = round(r.pop("t_start_perf") - wall_start, 4)
            r["t_done_s"] = round(r.pop("t_done_perf") - wall_start, 4)

    # ------------------------------------------------------------
    # 4. Stop GPU monitoring
    # ------------------------------------------------------------

    gpu_stats = gpu_monitor.stop() if gpu_monitor is not None else {}

    # ------------------------------------------------------------
    # 5. Metrics
    # ------------------------------------------------------------

    successful = [r for r in raw_results if r.get("success")]
    failed = [r for r in raw_results if not r.get("success")]

    if failed:
        logger.warning(
            "%d/%d requests failed at C=%d. First error: %s",
            len(failed),
            len(raw_results),
            concurrency,
            failed[0].get("error"),
        )

    total_audio_s = sum(r["audio_duration_s"] for r in successful)

    rtfs = [r["rtf"] for r in successful]
    client_rtfs = [r["client_rtf"] for r in successful]
    inference_times = [r["inference_time_s"] for r in successful]
    latencies = [r["e2e_latency_s"] for r in successful]
    semaphore_waits = [r["semaphore_wait_s"] for r in successful]

    # WER
    wer_items = [r for r in successful if r.get("reference")]
    predictions = [r["prediction"] for r in wer_items]
    references = [r["reference"] for r in wer_items]

    overall_wer = compute_wer(predictions, references) if references else 0.0
    overall_cer = compute_cer(predictions, references) if references else 0.0

    # Per-bucket metrics
    bucket_breakdown: Dict[str, Any] = {}

    for bucket_name, _, _ in BUCKETS:
        bucket_results = [r for r in successful if r["bucket"] == bucket_name]

        bucket_predictions = [r["prediction"] for r in bucket_results if r.get("reference")]
        bucket_references = [r["reference"] for r in bucket_results if r.get("reference")]

        bucket_breakdown[bucket_name] = {
            "n_requests": len(bucket_results),
            "total_audio_seconds": round(
                sum(r["audio_duration_s"] for r in bucket_results), 2
            ),
            "rtf": compute_percentiles([r["rtf"] for r in bucket_results]),
            "rtf_client_e2e": compute_percentiles([r["client_rtf"] for r in bucket_results]),
            "wer": (
                round(compute_wer(bucket_predictions, bucket_references), 4)
                if bucket_references
                else None
            ),
            "cer": (
                round(compute_cer(bucket_predictions, bucket_references), 4)
                if bucket_references
                else None
            ),
        }

    # Throughput
    throughput_audio_s_per_wall_s = total_audio_s / max(wall_time_s, 1e-4)
    requests_per_s = len(successful) / max(wall_time_s, 1e-4)

    # Steady-state throughput + GPU/CPU averages over the same trimmed window
    # (excludes ramp-up and tail; independent of how many requests were sent)
    win = steady_state_window(
        [(r["t_done_s"], r["audio_duration_s"]) for r in successful],
        concurrency,
    )
    if win is None:
        steady: Dict[str, Any] = {
            "available": False,
            "reason": (
                f"needs at least 3*C successful requests "
                f"(C={concurrency}, successful={len(successful)})"
            ),
        }
    else:
        steady = {
            "available": True,
            "t_start_s": round(win["t_start_s"], 4),
            "t_end_s": round(win["t_end_s"], 4),
            "window_s": round(win["window_s"], 3),
            "n_requests": win["n_requests"],
            "audio_s_per_wall_s": round(win["audio_s_per_s"], 3),
            "requests_per_s": round(win["requests_per_s"], 3),
        }
        if gpu_monitor is not None:
            steady["gpu"] = gpu_monitor.get_window_stats(
                wall_start_epoch + win["t_start_s"],
                wall_start_epoch + win["t_end_s"],
            )
        if win["n_requests"] < concurrency:
            logger.warning(
                "Steady-state window only holds %d requests (< C=%d); "
                "use more requests per bucket for a stable estimate.",
                win["n_requests"], concurrency,
            )

    summary = {
        "concurrency": concurrency,
        "config": {
            "backend": "vllm-async",
            "model": backend.model_path,
            "max_num_seqs": backend.max_num_seqs,
            "engine_kwargs": getattr(backend, "engine_kwargs", {}),
            "engine_load_time_s": round(getattr(backend, "load_time_s", 0.0), 2),
            "max_inference_batch_size": backend.max_inference_batch_size,
            "max_new_tokens": backend.max_new_tokens,
            "gpu_memory_utilization": backend.gpu_memory_utilization,
            "requests_per_bucket": requests_per_bucket,
            "total_requests": len(measurement_pool),
            "warmup_requests": warmup_requests,
            "buckets": {
                name: sum(1 for s in measurement_pool if s["bucket"] == name)
                for name in BUCKET_ORDER
            },
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
        },
        "rtf": compute_percentiles(rtfs),
        "rtf_client_e2e": compute_percentiles(client_rtfs),
        "inference_time_s": compute_percentiles(inference_times),
        "latency_e2e_s": compute_percentiles(latencies),
        "semaphore_wait_time_s": compute_percentiles(semaphore_waits),
        "wer": {
            "overall": round(overall_wer, 4),
            "n_evaluated": len(references),
        },
        "cer": {
            "overall": round(overall_cer, 4),
            "n_evaluated": len(references),
        },
        "bucket_breakdown": bucket_breakdown,
        "gpu": gpu_stats,
        "steady_state": steady,
        "per_sample": raw_results,
    }

    # Console output
    logger.info("Results for Concurrency = %d:", concurrency)
    logger.info(
        "  Avg RTF: %.4f | P50 RTF: %.4f | P95 RTF: %.4f",
        summary["rtf"]["avg"],
        summary["rtf"]["p50"],
        summary["rtf"]["p95"],
    )
    logger.info(
        "  Client RTF: avg %.4f | P50 %.4f | P95 %.4f",
        summary["rtf_client_e2e"]["avg"],
        summary["rtf_client_e2e"]["p50"],
        summary["rtf_client_e2e"]["p95"],
    )
    logger.info(
        "  Throughput: %.2f audio-s/s (%.2f req/s)",
        throughput_audio_s_per_wall_s,
        requests_per_s,
    )
    if steady.get("available"):
        sg = steady.get("gpu", {})
        logger.info(
            "  Steady state (%.2fs window, %d reqs): %.2f audio-s/s (%.2f req/s) | GPU %.1f%%",
            steady["window_s"],
            steady["n_requests"],
            steady["audio_s_per_wall_s"],
            steady["requests_per_s"],
            sg.get("gpu_util_avg_pct", 0.0),
        )
        if sg:
            logger.info("  Steady-state CPU: %s", format_cpu_summary(sg))
    else:
        logger.info("  Steady state: not available (%s)", steady.get("reason"))
    logger.info(
        "  Semaphore wait avg: %.3fs (P95: %.3fs)",
        summary["semaphore_wait_time_s"]["avg"],
        summary["semaphore_wait_time_s"]["p95"],
    )
    logger.info("  WER: %.2f%% | CER: %.2f%% (%d samples)",
                overall_wer * 100, overall_cer * 100, len(references))
    logger.info(
        "  GPU Util: %.1f%% (Max: %.1f%%) | Peak VRAM: %s MB",
        gpu_stats.get("gpu_util_avg_pct", 0.0),
        gpu_stats.get("gpu_util_max_pct", 0.0),
        gpu_stats.get("gpu_mem_used_peak_mb", 0.0),
    )
    logger.info("  CPU Util: %s", format_cpu_summary(gpu_stats))

    return summary


def main():

    parser = argparse.ArgumentParser(description="phase 3: Concurrent vLLM Load Tester")

    parser.add_argument("--model", default="models/lora_r16_merged",
                        help="Model path or Hugging Face model ID")
    parser.add_argument("--test_file", default="data/test.jsonl", help="Path to test.jsonl")
    parser.add_argument("--audio_dir", default="data/audio_clips/test",
                        help="Path to test audio directory")
    parser.add_argument("--concurrency", type=str, default=None,
                        help="Comma-separated concurrency levels")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help="Number of warmup requests")
    parser.add_argument("--requests_per_bucket", type=int, default=REQUESTS_PER_BUCKET,
                        help="Requests per duration bucket (default: matches Phase 1)")
    parser.add_argument("--gpu_mem", type=float, default=0.9,
                        help="GPU memory utilization for vLLM")
    parser.add_argument("--max_num_seqs", type=int, default=None,
                        help="vLLM max concurrent sequences (default: max(128, max concurrency))")

    args = parser.parse_args()

    if args.concurrency:
        concurrencies = [int(c.strip()) for c in args.concurrency.split(",") if c.strip()]
    else:
        concurrencies = DEFAULT_CONCURRENCIES

    max_num_seqs = args.max_num_seqs or max(128, max(concurrencies))

    logger.info("Initializing phase 3 benchmark with:")
    logger.info("  Model: %s", args.model)
    logger.info("  Test File: %s", args.test_file)
    logger.info("  Audio Directory: %s", args.audio_dir)
    logger.info("  Concurrencies: %s", concurrencies)
    logger.info("  Requests per bucket: %d", args.requests_per_bucket)
    logger.info("  Warmup requests: %d", args.warmup)
    logger.info("  max_num_seqs: %d", max_num_seqs)

    # Load and categorize samples
    samples = load_benchmark_dataset(
        test_file=args.test_file,
        audio_dir=args.audio_dir,
    )
    preload_audio(samples)

    samples_by_bucket: Dict[str, List[Dict[str, Any]]] = {name: [] for name in BUCKET_ORDER}

    for sample in samples:
        bucket = sample.get("bucket")
        if bucket in samples_by_bucket:
            samples_by_bucket[bucket].append(sample)

    logger.info("Loaded benchmark sample distribution:")
    for bucket_name in BUCKET_ORDER:
        logger.info("  Bucket %s: %d available clips", bucket_name, len(samples_by_bucket[bucket_name]))

    gpu_monitor = GPUMonitor(device_index=0, interval=0.05)

    async def run_all():
        # The AsyncLLM engine must be created inside the running event loop.
        backend = await AsyncVLLMBackend.create(
            model_path=args.model,
            gpu_memory_utilization=args.gpu_mem,
            max_num_seqs=max_num_seqs,
        )
        try:
            all_results = []
            for concurrency in concurrencies:
                result = await run_concurrent_benchmark(
                    backend=backend,
                    samples_by_bucket=samples_by_bucket,
                    concurrency=concurrency,
                    requests_per_bucket=args.requests_per_bucket,
                    warmup_requests=args.warmup,
                    gpu_monitor=gpu_monitor,
                )
                all_results.append(result)
            return all_results
        finally:
            backend.shutdown()

    all_results = asyncio.run(run_all())

    print()
    print("=" * 90)
    print("phase 3 SUMMARY")
    print("=" * 90)

    print(
        f"{'Concurrency':>12} "
        f"{'Requests':>10} "
        f"{'Wall(s)':>10} "
        f"{'Avg RTF':>10} "
        f"{'P95 RTF':>10} "
        f"{'Audio/s':>12} "
        f"{'Req/s':>10}"
    )
    print("-" * 90)

    for result in all_results:
        print(
            f"{result['concurrency']:>12} "
            f"{result['config']['total_requests']:>10} "
            f"{result['throughput']['wall_time_s']:>10.3f} "
            f"{result['rtf']['avg']:>10.4f} "
            f"{result['rtf']['p95']:>10.4f} "
            f"{result['throughput']['audio_s_per_wall_s']:>12.3f} "
            f"{result['throughput']['requests_per_s']:>10.3f}"
        )


if __name__ == "__main__":
    main()