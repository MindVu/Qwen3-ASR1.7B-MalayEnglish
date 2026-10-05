"""
Benchmark and Load Testing Suite for Qwen3-ASR 1.7B Baseline Inference.

Features:
- Tests concurrency levels: 1, 2, 4, 8, 16, 32, 64
- Evaluates audio categorized into duration buckets:
    * 2-5s
    * 5-15s
    * 15-30s
- Measures per request & aggregates:
    * Avg, P50, P95 RTF (Real-Time Factor)
    * Throughput (audio-seconds per wall-second, requests/s)
    * Queue wait times & total latency
    * Background GPU utilization & VRAM tracking
- Automatically outputs machine-readable JSON per concurrency level:
    results/baseline_c1.json, baseline_c2.json, ...
- Generates summary CSV and Markdown tables
"""

import aiohttp
import argparse
import asyncio
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

# Ensure current directory / phase1 is in pythonpath
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from monitoring import GPUMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("qwen_asr.benchmark")

BUCKETS = [
    ("2-5s", 2.0, 5.0),
    ("5-15s", 5.0, 15.0),
    ("15-30s", 15.0, 30.0),
]


def bucket_of_duration(dur: float) -> str:
    """Classify duration into designated buckets."""
    for name, lo, hi in BUCKETS:
        if lo <= dur < hi:
            return name
    if dur < 2.0:
        return "<2s"
    return ">30s"


def compute_percentiles(values: List[float]) -> Dict[str, float]:
    """Compute count, mean, min, max, p50, p90, p95 for a list of floats."""
    if not values:
        return {"n": 0, "avg": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(values)
    return {
        "n": len(values),
        "avg": round(float(np.mean(arr)), 4),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
    }


def load_dataset_samples(
    test_file: str,
    audio_dir: Optional[str] = None,
    requests_per_bucket: int = 20,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load test samples partitioned into duration categories: 2-5s, 5-15s, 15-30s.
    """
    categorized: Dict[str, List[Dict[str, Any]]] = {name: [] for name, _, _ in BUCKETS}
    categorized["<2s"] = []
    categorized[">30s"] = []

    if os.path.exists(test_file):
        with open(test_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                dur = float(item.get("duration", 0.0))
                audio_path = item.get("audio", "")
                
                # Check path existence or attempt audio_dir resolution
                if not os.path.exists(audio_path) and audio_dir:
                    fname = os.path.basename(audio_path)
                    candidate = os.path.join(audio_dir, fname)
                    if os.path.exists(candidate):
                        audio_path = candidate
                
                bucket = bucket_of_duration(dur)
                categorized[bucket].append({
                    "id": item.get("id", os.path.basename(audio_path)),
                    "audio_path": audio_path,
                    "duration": dur,
                    "prompt": item.get("prompt", "Transcribe the audio accurately."),
                    "reference": item.get("transcript") or item.get("text", ""),
                    "bucket": bucket,
                })

    # Fallback to audio_dir directory scan if test_file is missing or empty
    if not any(categorized.values()) and audio_dir and os.path.exists(audio_dir):
        import soundfile as sf
        for root, _, files in os.walk(audio_dir):
            for file in files:
                if file.endswith((".wav", ".mp3", ".flac")):
                    full_path = os.path.join(root, file)
                    try:
                        info = sf.info(full_path)
                        dur = float(info.duration)
                        bucket = bucket_of_duration(dur)
                        categorized[bucket].append({
                            "id": file,
                            "audio_path": full_path,
                            "duration": dur,
                            "prompt": "Transcribe the audio accurately.",
                            "reference": "",
                            "bucket": bucket,
                        })
                    except Exception:
                        pass

    # Log available samples
    logger.info("Loaded benchmark sample distribution:")
    for b_name in ["2-5s", "5-15s", "15-30s"]:
        samples = categorized.get(b_name, [])
        logger.info("  Bucket %s: %d available clips", b_name, len(samples))

    return categorized


async def send_transcribe_request(
    session: aiohttp.ClientSession,
    url: str,
    sample: Dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """Send a single transcription request under concurrency semaphore constraint."""
    audio_path = sample["audio_path"]
    if not os.path.exists(audio_path):
        return {
            "success": False,
            "error": f"Audio file not found: {audio_path}",
            "sample_id": sample["id"],
            "bucket": sample["bucket"],
            "duration": sample["duration"],
        }

    with open(audio_path, "rb") as f:
        audio_data = f.read()

    data = aiohttp.FormData()
    data.add_field(
        "file",
        audio_data,
        filename=os.path.basename(audio_path),
        content_type="audio/wav",
    )
    data.add_field("prompt", sample.get("prompt", "Transcribe the audio accurately."))

    client_send_time = time.perf_counter()
    async with semaphore:
        client_dispatched_time = time.perf_counter()
        try:
            async with session.post(url, data=data) as resp:
                client_recv_time = time.perf_counter()
                status_code = resp.status
                if status_code == 200:
                    payload = await resp.json()
                    e2e_client_latency = client_recv_time - client_dispatched_time
                    rtf_client = e2e_client_latency / max(sample["duration"], 1e-4)

                    return {
                        "success": True,
                        "status_code": status_code,
                        "sample_id": sample["id"],
                        "bucket": sample["bucket"],
                        "audio_duration_s": round(sample["duration"], 3),
                        "client_e2e_latency_s": round(e2e_client_latency, 4),
                        "client_rtf": round(rtf_client, 4),
                        "server_rtf": payload.get("rtf"),
                        "server_rtf_model": payload.get("rtf_model"),
                        "server_processing_time_s": payload.get("processing_time"),
                        "server_timing": payload.get("timing", {}),
                        "text": payload.get("text", ""),
                        "reference": sample.get("reference", ""),
                    }
                else:
                    err_text = await resp.text()
                    return {
                        "success": False,
                        "status_code": status_code,
                        "error": err_text,
                        "sample_id": sample["id"],
                        "bucket": sample["bucket"],
                        "duration": sample["duration"],
                    }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "sample_id": sample["id"],
                "bucket": sample["bucket"],
                "duration": sample["duration"],
            }


async def run_concurrency_benchmark(
    url: str,
    concurrency: int,
    samples_by_bucket: Dict[str, List[Dict[str, Any]]],
    requests_per_bucket: int = 20,
    warmup_requests: int = 5,
    gpu_monitor: Optional[GPUMonitor] = None,
) -> Dict[str, Any]:
    """
    Execute benchmark for a single concurrency level.
    """
    logger.info("=" * 70)
    logger.info("RUNNING BENCHMARK - CONCURRENCY LEVEL: %d", concurrency)
    logger.info("=" * 70)

    # Prepare measurement payload containing 2-5s, 5-15s, and 15-30s audio
    measurement_pool: List[Dict[str, Any]] = []
    for b_name in ["2-5s", "5-15s", "15-30s"]:
        b_samples = samples_by_bucket.get(b_name, [])
        if not b_samples:
            continue
        # Cycle through samples to fulfill requests_per_bucket count
        for i in range(requests_per_bucket):
            measurement_pool.append(b_samples[i % len(b_samples)])

    semaphore = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=180.0)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        # 1. Warm-up Phase (not measured)
        if warmup_requests > 0 and measurement_pool:
            logger.info("Warming up server with %d requests...", warmup_requests)
            warmup_tasks = [
                send_transcribe_request(session, url, measurement_pool[i % len(measurement_pool)], semaphore)
                for i in range(warmup_requests)
            ]
            await asyncio.gather(*warmup_tasks)
            logger.info("Warmup complete.")

        # 2. Start GPU Monitoring
        if gpu_monitor:
            gpu_monitor.start()

        # 3. Measurement Phase
        logger.info("Dispatching %d benchmark requests at concurrency %d...", len(measurement_pool), concurrency)
        t_bench_start = time.perf_counter()

        tasks = [
            asyncio.create_task(send_transcribe_request(session, url, s, semaphore))
            for s in measurement_pool
        ]
        results = await asyncio.gather(*tasks)

        t_bench_end = time.perf_counter()
        wall_time_s = t_bench_end - t_bench_start

        # 4. Stop GPU Monitoring
        gpu_stats = gpu_monitor.stop() if gpu_monitor else {}

    # 5. Process Metrics
    successful_results = [r for r in results if r.get("success")]
    failed_results = [r for r in results if not r.get("success")]

    total_audio_s = sum(r["audio_duration_s"] for r in successful_results)
    server_rtfs = [r["server_rtf"] for r in successful_results if r.get("server_rtf") is not None]
    client_rtfs = [r["client_rtf"] for r in successful_results if r.get("client_rtf") is not None]
    model_rtfs = [r["server_rtf_model"] for r in successful_results if r.get("server_rtf_model") is not None]
    queue_waits = [
        r["server_timing"].get("queue_wait_time", 0.0)
        for r in successful_results
        if "server_timing" in r
    ]
    latencies = [
        r["server_timing"].get("total_latency", r["client_e2e_latency_s"])
        for r in successful_results
    ]

    # Per-bucket metrics
    bucket_breakdown: Dict[str, Any] = {}
    for b_name in ["2-5s", "5-15s", "15-30s"]:
        b_res = [r for r in successful_results if r["bucket"] == b_name]
        b_rtfs = [r["server_rtf"] for r in b_res if r.get("server_rtf") is not None]
        b_audio_s = sum(r["audio_duration_s"] for r in b_res)
        bucket_breakdown[b_name] = {
            "n_requests": len(b_res),
            "total_audio_seconds": round(b_audio_s, 2),
            "rtf": compute_percentiles(b_rtfs),
        }

    throughput_audio_s_per_wall_s = total_audio_s / max(wall_time_s, 1e-4)
    requests_per_s = len(successful_results) / max(wall_time_s, 1e-4)

    summary = {
        "concurrency": concurrency,
        "config": {
            "warmup_requests": warmup_requests,
            "total_requests": len(measurement_pool),
            "requests_per_bucket": requests_per_bucket,
        },
        "status": {
            "successful_requests": len(successful_results),
            "failed_requests": len(failed_results),
        },
        "throughput": {
            "wall_time_s": round(wall_time_s, 3),
            "total_audio_s": round(total_audio_s, 2),
            "audio_s_per_wall_s": round(throughput_audio_s_per_wall_s, 3),
            "requests_per_s": round(requests_per_s, 3),
        },
        "rtf_server": compute_percentiles(server_rtfs),
        "rtf_client_e2e": compute_percentiles(client_rtfs),
        "rtf_model_only": compute_percentiles(model_rtfs),
        "queue_wait_time_s": compute_percentiles(queue_waits),
        "latency_e2e_s": compute_percentiles(latencies),
        "bucket_breakdown": bucket_breakdown,
        "gpu": gpu_stats,
    }

    # Console display
    logger.info("Results for Concurrency = %d:", concurrency)
    logger.info("  Avg RTF: %.4f | P50 RTF: %.4f | P95 RTF: %.4f",
                summary["rtf_server"]["avg"], summary["rtf_server"]["p50"], summary["rtf_server"]["p95"])
    logger.info("  Throughput: %.2f audio-s/s (%.2f req/s)",
                throughput_audio_s_per_wall_s, requests_per_s)
    logger.info("  Queue wait avg: %.3fs (P95: %.3fs)",
                summary["queue_wait_time_s"]["avg"], summary["queue_wait_time_s"]["p95"])
    logger.info("  GPU Util: %.1f%% (Max: %.1f%%) | Peak VRAM: %s MB",
                gpu_stats.get("gpu_util_avg_pct", 0.0),
                gpu_stats.get("gpu_util_max_pct", 0.0),
                gpu_stats.get("gpu_mem_used_peak_mb", 0.0))

    return summary


def save_concurrency_result(result: Dict[str, Any], output_dir: str):
    """Save machine-readable JSON for a single concurrency level."""
    os.makedirs(output_dir, exist_ok=True)
    c = result["concurrency"]
    filepath = os.path.join(output_dir, f"baseline_c{c}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved machine-readable result to %s", filepath)


def generate_summary_table(results_list: List[Dict[str, Any]], output_dir: str):
    """Generate Markdown and CSV summary tables across concurrency levels."""
    headers = ["Concurrency", "Avg RTF", "P50 RTF", "P95 RTF", "Throughput", "GPU Util.", "VRAM"]
    rows = []
    csv_lines = [",".join(headers)]

    for r in results_list:
        c = r["concurrency"]
        avg_rtf = f"{r['rtf_server']['avg']:.4f}"
        p50_rtf = f"{r['rtf_server']['p50']:.4f}"
        p95_rtf = f"{r['rtf_server']['p95']:.4f}"
        tp = f"{r['throughput']['audio_s_per_wall_s']:.2f} audio-s/s"
        gpu_u = f"{r['gpu'].get('gpu_util_avg_pct', 0.0):.1f}%"
        vram = f"{r['gpu'].get('gpu_mem_used_peak_mb', 0.0):.1f} MB"

        rows.append([str(c), avg_rtf, p50_rtf, p95_rtf, tp, gpu_u, vram])
        csv_lines.append(f"{c},{avg_rtf},{p50_rtf},{p95_rtf},{r['throughput']['audio_s_per_wall_s']:.2f},{r['gpu'].get('gpu_util_avg_pct', 0.0)},{r['gpu'].get('gpu_mem_used_peak_mb', 0.0)}")

    # Markdown format
    md_lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join([" ---: " for _ in headers]) + "|",
    ]
    for row in rows:
        md_lines.append("| " + " | ".join(row) + " |")
    md_table = "\n".join(md_lines)

    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "baseline_summary.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines) + "\n")

    with open(os.path.join(output_dir, "baseline_summary.md"), "w", encoding="utf-8") as f:
        f.write(md_table + "\n")

    print("\n" + "=" * 70)
    print("PHASE 1 BASELINE BENCHMARK SUMMARY TABLE")
    print("=" * 70)
    print(md_table)
    print("=" * 70 + "\n")


async def main_async():
    parser = argparse.ArgumentParser(description="Phase 1 Baseline Benchmark Suite")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--url", default=None, help="Server URL (defaults to config)")
    parser.add_argument("--concurrency", type=str, default=None,
                        help="Comma-separated concurrency levels (e.g. 1,2,4,8,16,32,64)")
    parser.add_argument("--audio-dir", default=None, help="Directory containing eval audio clips")
    parser.add_argument("--test-file", default=None, help="Path to test.jsonl")
    parser.add_argument("--requests-per-bucket", type=int, default=None,
                        help="Requests per duration bucket (2-5s, 5-15s, 15-30s)")
    parser.add_argument("--warmup", type=int, default=None, help="Number of warmup requests")
    parser.add_argument("--output-dir", default=None, help="Output results directory")
    args = parser.parse_args()

    # Resolve config
    config_path = args.config
    if not os.path.exists(config_path):
        config_path = os.path.join(CURRENT_DIR, args.config)
    cfg = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    bench_cfg = cfg.get("benchmark", {})
    url = args.url or bench_cfg.get("server_url", "http://127.0.0.1:8000/transcribe")
    test_file = args.test_file or bench_cfg.get("test_file", "data/test.jsonl")
    audio_dir = args.audio_dir or bench_cfg.get("audio_dir", "data/audio_clips/test")
    output_dir = args.output_dir or bench_cfg.get("output_dir", "results")
    warmup = args.warmup if args.warmup is not None else bench_cfg.get("warmup_requests", 5)
    reqs_per_bucket = (
        args.requests_per_bucket
        if args.requests_per_bucket is not None
        else bench_cfg.get("requests_per_bucket", 20)
    )

    if args.concurrency:
        concurrencies = [int(c.strip()) for c in args.concurrency.split(",")]
    else:
        concurrencies = bench_cfg.get("concurrencies", [1, 2, 4, 8, 16, 32, 64])

    logger.info("Initializing baseline benchmark with:")
    logger.info("  Server URL: %s", url)
    logger.info("  Test File: %s", test_file)
    logger.info("  Audio Directory: %s", audio_dir)
    logger.info("  Concurrencies: %s", concurrencies)
    logger.info("  Output Directory: %s", output_dir)

    samples = load_dataset_samples(test_file=test_file, audio_dir=audio_dir, requests_per_bucket=reqs_per_bucket)
    gpu_monitor = GPUMonitor(device_index=cfg.get("monitoring", {}).get("gpu_device_index", 0))

    all_results = []
    for c in concurrencies:
        result = await run_concurrency_benchmark(
            url=url,
            concurrency=c,
            samples_by_bucket=samples,
            requests_per_bucket=reqs_per_bucket,
            warmup_requests=warmup,
            gpu_monitor=gpu_monitor,
        )
        save_concurrency_result(result, output_dir=output_dir)
        all_results.append(result)

    generate_summary_table(all_results, output_dir=output_dir)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
