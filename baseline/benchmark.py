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
- Word Error Rate (WER) and Character Error Rate (CER), computed with the same
  normalizer / algorithm as Phase 2
- Automatically outputs machine-readable JSON per concurrency level:
    results/baseline_c1.json, baseline_c2.json, ...
- Generates summary CSV and Markdown tables

RTF definition (headline, identical in Phase 1 and Phase 2):

    RTF = inference processing time / audio duration

    Phase 1: inference processing time = server `processing_time`
             (preprocessing + model.generate + decoding). It excludes audio
             file decoding, HTTP transfer, and queue wait. Audio duration is the
             dataset `duration` field (same source as Phase 2).
             Reported under `rtf_server` / per-sample `rtf_processing`.

Secondary metric (NOT the headline):
    rtf_client_e2e  : (dispatch -> FULL response body received) / audio duration.
                      Includes HTTP + upload + server queue wait.
"""

import aiohttp
import argparse
import asyncio
import json
import logging
import mimetypes
import os
import re
import socket
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

# Ensure current directory / phase1 is in pythonpath
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from monitoring import GPUMonitor, fmt_num, fmt_pct, format_cpu_summary, steady_state_window

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
BUCKET_ORDER = [name for name, _, _ in BUCKETS]   # used for logging / reporting
# Request order in the workload: longest -> shortest, interleaved round-robin
# (15-30s, 5-15s, 2-5s, 15-30s, ...). Keep identical in Phase 2.
POOL_ORDER = ["15-30s", "5-15s", "2-5s"]
ASR_TEXT_TAG = "<asr_text>"


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


def normalize_text(text: Any) -> str:
    """
    Standard text normalization (identical to Phase 2): lowercasing, punctuation
    removal, whitespace collapse. Preserves apostrophes inside words.
    """
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_language_tag(text: Any) -> str:
    """Remove the 'language {Lang}<asr_text>' prefix if present (identical to Phase 2)."""
    if text is None:
        return ""
    text = str(text)
    if ASR_TEXT_TAG in text:
        return text.split(ASR_TEXT_TAG, 1)[1].strip()
    return text.strip()


def compute_wer_builtin(predictions: List[str], references: List[str]) -> float:
    """Dynamic-programming WER (fallback when evaluate / jiwer are unavailable)."""
    total_words = 0
    total_edits = 0
    for pred, ref in zip(predictions, references):
        ref_words = ref.strip().split()
        pred_words = pred.strip().split()
        r_len, p_len = len(ref_words), len(pred_words)
        total_words += r_len

        dp = [[0] * (p_len + 1) for _ in range(r_len + 1)]
        for i in range(r_len + 1):
            dp[i][0] = i
        for j in range(p_len + 1):
            dp[0][j] = j
        for i in range(1, r_len + 1):
            for j in range(1, p_len + 1):
                if ref_words[i - 1] == pred_words[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        total_edits += dp[r_len][p_len]

    if total_words == 0:
        return 0.0
    return total_edits / total_words


_WER_METRIC = None


def compute_wer(predictions: List[str], references: List[str]) -> float:
    """
    Corpus-level WER with normalization. Same chain as Phase 2:
    evaluate -> jiwer -> builtin DP.
    """
    global _WER_METRIC
    if len(predictions) == 0:
        return 0.0

    preds = [normalize_text(strip_language_tag(p)) for p in predictions]
    refs = [normalize_text(strip_language_tag(r)) for r in references]

    try:
        import evaluate
        if _WER_METRIC is None:
            _WER_METRIC = evaluate.load("wer")
        return float(_WER_METRIC.compute(predictions=preds, references=refs))
    except Exception:
        try:
            import jiwer
            return float(jiwer.wer(reference=refs, hypothesis=preds))
        except Exception:
            return float(compute_wer_builtin(predictions=preds, references=refs))


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings (two-row dynamic programming)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def compute_cer_builtin(predictions: List[str], references: List[str]) -> float:
    """Total character-level edits / total reference characters (spaces count)."""
    total_chars = 0
    total_edits = 0
    for pred, ref in zip(predictions, references):
        total_chars += len(ref)
        total_edits += _edit_distance(ref, pred)
    if total_chars == 0:
        return 0.0
    return total_edits / total_chars


_CER_METRIC = None


def compute_cer(predictions: List[str], references: List[str]) -> float:
    """
    Corpus-level CER with the same normalization as WER (identical to Phase 2).
    Chain: evaluate -> jiwer -> builtin DP. Spaces count as characters.
    """
    global _CER_METRIC
    if len(predictions) == 0:
        return 0.0

    preds = [normalize_text(strip_language_tag(p)) for p in predictions]
    refs = [normalize_text(strip_language_tag(r)) for r in references]

    try:
        import evaluate
        if _CER_METRIC is None:
            _CER_METRIC = evaluate.load("cer")
        return float(_CER_METRIC.compute(predictions=preds, references=refs))
    except Exception:
        try:
            import jiwer
            return float(jiwer.cer(reference=refs, hypothesis=preds))
        except Exception:
            return float(compute_cer_builtin(predictions=preds, references=refs))


def _extract_reference(item: Dict[str, Any]) -> str:
    """
    Pick the reference transcript (same key order as Phase 2's loader) and strip
    any 'language X<asr_text>' prefix.
    """
    for key in ["transcript", "reference", "text", "normalized_text", "raw_text"]:
        if key in item and item[key] is not None:
            val = str(item[key]).strip()
            if val:
                return strip_language_tag(val)
    return ""


def load_dataset_samples(
    test_file: str,
    audio_dir: Optional[str] = None,
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
                audio_path = item.get("audio", "")

                # Check path existence or attempt audio_dir resolution
                if not os.path.exists(audio_path) and audio_dir:
                    fname = os.path.basename(audio_path)
                    candidate = os.path.join(audio_dir, fname)
                    if os.path.exists(candidate):
                        audio_path = candidate

                # Duration: use the field if present, otherwise read it from the file.
                # (A missing duration would otherwise land in "<2s" and be dropped.)
                dur = item.get("duration")
                if dur is None and os.path.exists(audio_path):
                    try:
                        import soundfile as sf
                        dur = float(sf.info(audio_path).duration)
                    except Exception:
                        dur = 0.0
                dur = float(dur or 0.0)

                bucket = bucket_of_duration(dur)
                categorized[bucket].append({
                    "id": item.get("id", os.path.basename(audio_path)),
                    "audio_path": audio_path,
                    "duration": dur,
                    "prompt": item.get("prompt", "Transcribe the audio accurately."),
                    "reference": _extract_reference(item),
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
    for b_name in BUCKET_ORDER:
        logger.info("  Bucket %s: %d available clips", b_name, len(categorized.get(b_name, [])))

    return categorized


def preload_audio_bytes(samples_by_bucket: Dict[str, List[Dict[str, Any]]]) -> None:
    """
    Read every audio file from disk once, before the sweep, so requests don't
    do blocking file I/O inside the event loop.
    """
    n = 0
    for b_name in BUCKET_ORDER:
        for s in samples_by_bucket.get(b_name, []):
            path = s["audio_path"]
            if "audio_bytes" not in s and os.path.exists(path):
                with open(path, "rb") as f:
                    s["audio_bytes"] = f.read()
                n += 1
    logger.info("Preloaded %d audio files into memory.", n)


async def fetch_server_pid(transcribe_url: str) -> Optional[int]:
    """
    Ask the server's /health endpoint for its PID so its CPU usage can be tracked.
    Only used when the server runs on THIS machine (hostname must match).
    Returns None if unavailable (older server without pid/hostname, remote host, error).
    """
    health_url = transcribe_url.rsplit("/", 1)[0] + "/health"
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5.0)) as session:
            async with session.get(health_url) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
    except Exception as e:
        logger.info("Could not query %s for server PID (%s); server CPU will not be tracked.", health_url, e)
        return None

    pid, host = data.get("pid"), data.get("hostname")
    if pid is None:
        logger.info("Server /health has no 'pid' field; server CPU will not be tracked.")
        return None
    if host and host != socket.gethostname():
        logger.info("Server runs on another host (%s); server CPU will not be tracked.", host)
        return None
    logger.info("Tracking server CPU usage (PID %s).", pid)
    return int(pid)


async def send_transcribe_request(
    session: aiohttp.ClientSession,
    url: str,
    sample: Dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """Send a single transcription request under concurrency semaphore constraint."""
    audio_path = sample["audio_path"]
    audio_data = sample.get("audio_bytes")

    if audio_data is None:
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

    content_type = mimetypes.guess_type(audio_path)[0] or "audio/wav"

    # FormData is single-use, so build a fresh one per request.
    data = aiohttp.FormData()
    data.add_field(
        "file",
        audio_data,
        filename=os.path.basename(audio_path),
        content_type=content_type,
    )
    data.add_field("prompt", sample.get("prompt", "Transcribe the audio accurately."))

    t_submit = time.perf_counter()
    async with semaphore:
        t_dispatch = time.perf_counter()
        client_queue_wait = t_dispatch - t_submit
        try:
            async with session.post(url, data=data) as resp:
                status_code = resp.status
                # Read the FULL body before stopping the clock; the context
                # manager returns as soon as headers arrive.
                if status_code == 200:
                    payload = await resp.json()
                    err_text = ""
                else:
                    payload = {}
                    err_text = await resp.text()
                t_recv = time.perf_counter()

            if status_code != 200:
                return {
                    "success": False,
                    "status_code": status_code,
                    "error": err_text,
                    "sample_id": sample["id"],
                    "bucket": sample["bucket"],
                    "duration": sample["duration"],
                }

            e2e_client_latency = t_recv - t_dispatch
            rtf_client = e2e_client_latency / max(sample["duration"], 1e-4)

            # Headline RTF = inference processing time / audio duration
            proc_time = payload.get("processing_time")
            rtf_processing = (
                proc_time / max(sample["duration"], 1e-4) if proc_time is not None else None
            )

            return {
                "success": True,
                "status_code": status_code,
                "sample_id": sample["id"],
                "bucket": sample["bucket"],
                "audio_duration_s": round(sample["duration"], 3),
                "client_semaphore_wait_s": round(client_queue_wait, 4),
                "client_e2e_latency_s": round(e2e_client_latency, 4),
                "client_rtf": round(rtf_client, 4),
                "t_start_perf": t_dispatch,
                "t_done_perf": t_recv,
                "rtf_processing": round(rtf_processing, 4) if rtf_processing is not None else None,
                "server_rtf": payload.get("rtf"),  # server's own value, kept for reference only
                "server_rtf_model": payload.get("rtf_model"),
                "server_processing_time_s": payload.get("processing_time"),
                "server_timing": payload.get("timing") or {},
                "text": payload.get("text", ""),
                "reference": sample.get("reference", ""),
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"{type(e).__name__}: {e}",
                "sample_id": sample["id"],
                "bucket": sample["bucket"],
                "duration": sample["duration"],
            }


async def run_concurrency_benchmark(
    url: str,
    concurrency: int,
    samples_by_bucket: Dict[str, List[Dict[str, Any]]],
    requests_per_bucket: int = 60,
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
    # Interleave buckets round-robin (15-30s, 5-15s, 2-5s, ...), cycling through
    # each bucket's clips to reach requests_per_bucket requests per bucket.
    active = []
    for b_name in POOL_ORDER:
        b_samples = samples_by_bucket.get(b_name, [])
        if not b_samples:
            logger.warning("Bucket %s has no samples and will be skipped.", b_name)
            continue
        active.append(b_samples)

    measurement_pool: List[Dict[str, Any]] = []
    for i in range(requests_per_bucket):
        for b_samples in active:
            measurement_pool.append(b_samples[i % len(b_samples)])

    if not measurement_pool:
        raise RuntimeError("Measurement pool is empty: no samples in the 2-5s / 5-15s / 15-30s buckets.")

    logger.info("Measurement pool: %d requests", len(measurement_pool))

    semaphore = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=180.0)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        # 1. Warm-up Phase (not measured)
        if warmup_requests > 0:
            logger.info("Warming up server with %d requests...", warmup_requests)
            warmup_tasks = [
                send_transcribe_request(session, url, measurement_pool[i % len(measurement_pool)], semaphore)
                for i in range(warmup_requests)
            ]
            warmup_results = await asyncio.gather(*warmup_tasks)

            warmup_failed = [r for r in warmup_results if not r.get("success")]
            if warmup_failed:
                logger.error(
                    "Warmup: %d/%d requests FAILED. First error: %s",
                    len(warmup_failed), len(warmup_results), warmup_failed[0].get("error"),
                )
                if len(warmup_failed) == len(warmup_results):
                    raise RuntimeError(
                        f"All warmup requests failed; is the server running at {url}? "
                        f"Error: {warmup_failed[0].get('error')}"
                    )
            logger.info("Warmup complete.")

        # 2. Start GPU Monitoring
        if gpu_monitor:
            gpu_monitor.start()

        # 3. Measurement Phase
        logger.info("Dispatching %d benchmark requests at concurrency %d...", len(measurement_pool), concurrency)
        t_bench_start = time.perf_counter()
        bench_start_epoch = time.time()  # same clock as the GPU/CPU monitor

        tasks = [
            asyncio.create_task(send_transcribe_request(session, url, s, semaphore))
            for s in measurement_pool
        ]
        results = await asyncio.gather(*tasks)

        t_bench_end = time.perf_counter()
        wall_time_s = t_bench_end - t_bench_start

        # Convert per-request timestamps to seconds since the start of the measurement
        for r in results:
            if r.get("success"):
                r["t_start_s"] = round(r.pop("t_start_perf") - t_bench_start, 4)
                r["t_done_s"] = round(r.pop("t_done_perf") - t_bench_start, 4)

        # 4. Stop GPU Monitoring
        gpu_stats = gpu_monitor.stop() if gpu_monitor else {}

    # 5. Process Metrics
    successful_results = [r for r in results if r.get("success")]
    failed_results = [r for r in results if not r.get("success")]

    if failed_results:
        logger.warning(
            "%d/%d requests FAILED at C=%d. First error: %s",
            len(failed_results), len(results), concurrency, failed_results[0].get("error"),
        )

    total_audio_s = sum(r["audio_duration_s"] for r in successful_results)
    server_rtfs = [r["rtf_processing"] for r in successful_results if r.get("rtf_processing") is not None]
    client_rtfs = [r["client_rtf"] for r in successful_results]
    model_rtfs = [r["server_rtf_model"] for r in successful_results if r.get("server_rtf_model") is not None]

    # Only count server queue wait / latency when the server actually reported it
    # (defaulting to 0.0 would drag the averages down).
    queue_waits = [
        r["server_timing"]["queue_wait_time"]
        for r in successful_results
        if r["server_timing"].get("queue_wait_time") is not None
    ]
    server_latencies = [
        r["server_timing"]["total_latency"]
        for r in successful_results
        if r["server_timing"].get("total_latency") is not None
    ]
    client_latencies = [r["client_e2e_latency_s"] for r in successful_results]
    client_semaphore_waits = [r["client_semaphore_wait_s"] for r in successful_results]

    # WER (same rule as Phase 2: only successful requests that have a reference)
    wer_items = [r for r in successful_results if r.get("reference")]
    overall_wer = (
        compute_wer([r["text"] for r in wer_items], [r["reference"] for r in wer_items])
        if wer_items
        else 0.0
    )
    overall_cer = (
        compute_cer([r["text"] for r in wer_items], [r["reference"] for r in wer_items])
        if wer_items
        else 0.0
    )

    # Per-bucket metrics
    bucket_breakdown: Dict[str, Any] = {}
    for b_name in BUCKET_ORDER:
        b_res = [r for r in successful_results if r["bucket"] == b_name]
        b_rtfs = [r["rtf_processing"] for r in b_res if r.get("rtf_processing") is not None]
        b_client_rtfs = [r["client_rtf"] for r in b_res]
        b_audio_s = sum(r["audio_duration_s"] for r in b_res)
        b_wer_items = [r for r in b_res if r.get("reference")]
        bucket_breakdown[b_name] = {
            "n_requests": len(b_res),
            "total_audio_seconds": round(b_audio_s, 2),
            "rtf": compute_percentiles(b_rtfs),
            "rtf_client_e2e": compute_percentiles(b_client_rtfs),
            "wer": (
                round(compute_wer([r["text"] for r in b_wer_items],
                                  [r["reference"] for r in b_wer_items]), 4)
                if b_wer_items
                else None
            ),
            "cer": (
                round(compute_cer([r["text"] for r in b_wer_items],
                                  [r["reference"] for r in b_wer_items]), 4)
                if b_wer_items
                else None
            ),
        }

    throughput_audio_s_per_wall_s = total_audio_s / max(wall_time_s, 1e-4)
    requests_per_s = len(successful_results) / max(wall_time_s, 1e-4)

    # Steady-state throughput + GPU/CPU averages over the same trimmed window
    # (excludes ramp-up and tail; independent of how many requests were sent)
    win = steady_state_window(
        [(r["t_done_s"], r["audio_duration_s"]) for r in successful_results],
        concurrency,
    )
    if win is None:
        steady: Dict[str, Any] = {
            "available": False,
            "reason": (
                f"needs at least 3*C successful requests "
                f"(C={concurrency}, successful={len(successful_results)})"
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
                bench_start_epoch + win["t_start_s"],
                bench_start_epoch + win["t_end_s"],
            )

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
        "client_semaphore_wait_s": compute_percentiles(client_semaphore_waits),
        "latency_e2e_s": compute_percentiles(client_latencies),
        "latency_server_s": compute_percentiles(server_latencies),
        "wer": {
            "overall": round(overall_wer, 4),
            "n_evaluated": len(wer_items),
        },
        "cer": {
            "overall": round(overall_cer, 4),
            "n_evaluated": len(wer_items),
        },
        "bucket_breakdown": bucket_breakdown,
        "gpu": gpu_stats,
        "steady_state": steady,
        "per_sample": results,
    }

    # Console display
    logger.info("Results for Concurrency = %d:", concurrency)
    logger.info("  Processing RTF: avg %.4f | P50 %.4f | P95 %.4f",
                summary["rtf_server"]["avg"], summary["rtf_server"]["p50"], summary["rtf_server"]["p95"])
    logger.info("  Client e2e RTF: avg %.4f | P50 %.4f | P95 %.4f",
                summary["rtf_client_e2e"]["avg"], summary["rtf_client_e2e"]["p50"], summary["rtf_client_e2e"]["p95"])
    logger.info("  Throughput: %.2f audio-s/s (%.2f req/s) | failed: %d",
                throughput_audio_s_per_wall_s, requests_per_s, len(failed_results))
    if steady.get("available"):
        sg = steady.get("gpu", {})
        logger.info("  Steady state (%.2fs window, %d reqs): %.2f audio-s/s (%.2f req/s) | GPU %.1f%%",
                    steady["window_s"], steady["n_requests"], steady["audio_s_per_wall_s"],
                    steady["requests_per_s"], sg.get("gpu_util_avg_pct", 0.0))
        if sg:
            logger.info("  Steady-state CPU: %s", format_cpu_summary(sg))
    else:
        logger.info("  Steady state: not available (%s)", steady.get("reason"))
    logger.info("  WER: %.2f%% | CER: %.2f%% (%d samples)",
                overall_wer * 100, overall_cer * 100, len(wer_items))
    logger.info("  Server queue wait avg: %.3fs (P95: %.3fs)",
                summary["queue_wait_time_s"]["avg"], summary["queue_wait_time_s"]["p95"])
    logger.info("  GPU Util: %.1f%% (Max: %.1f%%) | Peak VRAM: %s MB",
                gpu_stats.get("gpu_util_avg_pct", 0.0),
                gpu_stats.get("gpu_util_max_pct", 0.0),
                gpu_stats.get("gpu_mem_used_peak_mb", 0.0))
    logger.info("  CPU Util: %s", format_cpu_summary(gpu_stats))

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
    # Original columns first (unchanged order), new columns appended at the end.
    headers = [
        "Concurrency", "Avg RTF", "P50 RTF", "P95 RTF", "Throughput", "GPU Util.", "VRAM",
        "Failed", "Client Avg RTF", "Client P95 RTF", "WER", "CER",
        "CPU Sys %", "CPU Client %", "CPU Server %",
        "Steady Audio-s/s", "Steady GPU %",
    ]
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
        failed = str(r["status"]["failed_requests"])
        c_avg = f"{r['rtf_client_e2e']['avg']:.4f}"
        c_p95 = f"{r['rtf_client_e2e']['p95']:.4f}"
        wer = f"{r['wer']['overall'] * 100:.2f}%"
        cer = f"{r['cer']['overall'] * 100:.2f}%"
        g = r["gpu"]
        cpu_sys = g.get("cpu_sys_util_avg_pct")
        cpu_cli = g.get("cpu_proc_util_avg_pct")
        cpu_srv = g.get("cpu_watched_util_avg_pct")
        steady = r.get("steady_state") or {}
        st_ok = bool(steady.get("available"))
        st_tp = steady.get("audio_s_per_wall_s") if st_ok else None
        st_gpu = (steady.get("gpu") or {}).get("gpu_util_avg_pct") if st_ok else None

        rows.append([str(c), avg_rtf, p50_rtf, p95_rtf, tp, gpu_u, vram, failed, c_avg, c_p95, wer, cer,
                     fmt_pct(cpu_sys), fmt_pct(cpu_cli), fmt_pct(cpu_srv),
                     "n/a" if st_tp is None else f"{st_tp:.2f}", fmt_pct(st_gpu)])
        csv_lines.append(
            f"{c},{avg_rtf},{p50_rtf},{p95_rtf},{r['throughput']['audio_s_per_wall_s']:.2f},"
            f"{r['gpu'].get('gpu_util_avg_pct', 0.0)},{r['gpu'].get('gpu_mem_used_peak_mb', 0.0)},"
            f"{failed},{c_avg},{c_p95},{r['wer']['overall'] * 100:.2f},{r['cer']['overall'] * 100:.2f},"
            f"{fmt_num(cpu_sys)},{fmt_num(cpu_cli)},{fmt_num(cpu_srv)},"
            f"{'' if st_tp is None else f'{st_tp:.2f}'},{fmt_num(st_gpu)}"
        )

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
            cfg = yaml.safe_load(f) or {}   # empty YAML file returns None

    bench_cfg = cfg.get("benchmark", {})
    url = args.url or bench_cfg.get("server_url", "http://127.0.0.1:8000/transcribe")
    test_file = args.test_file or bench_cfg.get("test_file", "data/test.jsonl")
    audio_dir = args.audio_dir or bench_cfg.get("audio_dir", "data/audio_clips/test")
    output_dir = args.output_dir or bench_cfg.get("output_dir", "results")
    warmup = args.warmup if args.warmup is not None else bench_cfg.get("warmup_requests", 5)
    reqs_per_bucket = (
        args.requests_per_bucket
        if args.requests_per_bucket is not None
        else bench_cfg.get("requests_per_bucket", 60)
    )

    if args.concurrency:
        concurrencies = [int(c.strip()) for c in args.concurrency.split(",") if c.strip()]
    else:
        concurrencies = bench_cfg.get("concurrencies", [1, 2, 4, 8, 16, 32, 64])

    logger.info("Initializing baseline benchmark with:")
    logger.info("  Server URL: %s", url)
    logger.info("  Test File: %s", test_file)
    logger.info("  Audio Directory: %s", audio_dir)
    logger.info("  Concurrencies: %s", concurrencies)
    logger.info("  Requests per bucket: %d", reqs_per_bucket)
    logger.info("  Warmup requests: %d", warmup)
    logger.info("  Output Directory: %s", output_dir)

    samples = load_dataset_samples(test_file=test_file, audio_dir=audio_dir)
    preload_audio_bytes(samples)
    server_pid = await fetch_server_pid(url)
    gpu_monitor = GPUMonitor(
        device_index=cfg.get("monitoring", {}).get("gpu_device_index", 0),
        watch_pids=[server_pid] if server_pid else None,
    )

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

        # Let the GPU / server settle between concurrency levels.
        await asyncio.sleep(1.0)

    generate_summary_table(all_results, output_dir=output_dir)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()