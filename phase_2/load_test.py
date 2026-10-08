"""
Phase 2: Offline batch load tester for the official qwen-asr vLLM wrapper.

Phase 1: sequential HF server over HTTP
Phase 2: Qwen3ASRModel.LLM (offline vllm.LLM) used as shipped:
         one transcribe(list_of_C_audios) call per batch, batches run back to back

The "concurrency level" C is the BATCH SIZE. The workload per level is
    total requests = requests_per_slot * C        (default 10 * C)
split into consecutive batches of C requests (so 10 batches per level), taken
from the same interleaved pool as the other phases (15-30s, 5-15s, 2-5s, ...).

Metric definitions
------------------
All requests of a batch finish together (vllm.LLM.generate returns once the whole
list is done), so:

  per-request processing time = wall time of the batch call it belongs to
  rtf            = batch wall time / OWN audio duration       (per request)
                   -> pessimistic for short clips batched with long ones, because a
                      short clip waits for the longest clip in its batch
  rtf_aggregate  = batch wall time / TOTAL audio of the batch (per batch)
                   -> the inverse of the batch's throughput; fair across clip lengths

Throughput (audio-s per wall-s) is the most comparable number with the other phases.
Steady-state throughput and GPU/CPU averages exclude the first and last batch
(same helper as the other phases: monitoring.steady_state_window).
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent

# common.py lives in phase2/, monitoring.py at the repo root or in phase2/.
for _p in (CURRENT_DIR, REPO_ROOT, REPO_ROOT / "phase2"):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))
if str(CURRENT_DIR) in sys.path:
    sys.path.remove(str(CURRENT_DIR))
sys.path.insert(0, str(CURRENT_DIR))

from common import (
    compute_cer,
    compute_percentiles,
    compute_wer,
)
from monitoring import GPUMonitor, format_cpu_summary, steady_state_window

logger = logging.getLogger("phase_2.load_test")

DEFAULT_CONCURRENCIES = [1, 2, 4, 8, 16, 32, 64]
REQUESTS_PER_SLOT = 10          # total requests per level = REQUESTS_PER_SLOT * C
DEFAULT_WARMUP_BATCHES = 2      # warmup = this many batches of size C
BUCKET_ORDER = ["2-5s", "5-15s", "15-30s"]   # used for logging / reporting
# Request order: longest -> shortest, interleaved round-robin (identical to Phase 1/2).
POOL_ORDER = ["15-30s", "5-15s", "2-5s"]


# ----------------------------------------------------------------------
# Workload
# ----------------------------------------------------------------------


def preload_audio(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Decode every clip once (path -> (np.ndarray, 16 kHz)) so the benchmark measures
    the engine, not disk reads / resampling. The wrapper's normalize_audios()
    accepts (waveform, sr) tuples.
    """
    from qwen_asr.inference.utils import SAMPLE_RATE, normalize_audios

    n = 0
    for s in samples:
        if isinstance(s["audio"], str):
            s["audio"] = (normalize_audios(s["audio"])[0], SAMPLE_RATE)
            n += 1
    logger.info("Preloaded %d audio clips into memory.", n)
    return samples


def build_measurement_pool(
    samples_by_bucket: Dict[str, List[Dict[str, Any]]],
    concurrency: int,
    requests_per_bucket: Optional[int] = None,
    requests_per_slot: int = REQUESTS_PER_SLOT,
    min_total_requests: int = 0,
) -> List[Dict[str, Any]]:
    """
    Build the measurement workload.

    Sizing:
      * requests_per_bucket given  -> that many requests per bucket (fixed size)
      * otherwise (auto)           -> total = max(requests_per_slot * C, min_total_requests),
                                      split as evenly as possible over the buckets
                                      (the remainder goes to the longest buckets first)
    Order: INTERLEAVED round-robin in POOL_ORDER
        15-30s[0], 5-15s[0], 2-5s[0], 15-30s[1], ...
    Each bucket cycles through its own clips if it has fewer clips than requests.
    """
    active = []
    for bucket_name in POOL_ORDER:
        bucket_samples = samples_by_bucket.get(bucket_name, [])
        if not bucket_samples:
            logger.warning("Bucket %s has no samples and will be skipped.", bucket_name)
            continue
        active.append((bucket_name, bucket_samples))
    if not active:
        return []

    k = len(active)
    if requests_per_bucket is not None:
        quotas = [int(requests_per_bucket)] * k
    else:
        total = max(int(requests_per_slot) * int(concurrency), int(min_total_requests))
        base, rem = divmod(total, k)
        quotas = [base + (1 if i < rem else 0) for i in range(k)]

    pool: List[Dict[str, Any]] = []
    for i in range(max(quotas)):
        for (_, bucket_samples), quota in zip(active, quotas):
            if i < quota:
                pool.append(bucket_samples[i % len(bucket_samples)])
    return pool


def _contexts(batch: List[Dict[str, Any]], mode: str) -> List[str]:
    """System-message text per request: '' (Phase 2) or the sample prompt (Phase 1)."""
    if mode == "sample":
        return [s.get("prompt", "") or "" for s in batch]
    return [""] * len(batch)


# ----------------------------------------------------------------------
# One concurrency (= batch size) level
# ----------------------------------------------------------------------


def run_batched_benchmark(
    backend: Any,
    samples_by_bucket: Dict[str, List[Dict[str, Any]]],
    batch_size: int,
    requests_per_bucket: Optional[int] = None,
    requests_per_slot: int = REQUESTS_PER_SLOT,
    min_total_requests: int = 0,
    warmup_batches: int = DEFAULT_WARMUP_BATCHES,
    context_mode: str = "empty",
    language: Optional[str] = None,
    gpu_monitor: Optional[GPUMonitor] = None,
) -> Dict[str, Any]:

    logger.info("=" * 70)
    logger.info("RUNNING PHASE 3 BENCHMARK - BATCH SIZE (C): %d", batch_size)
    logger.info("=" * 70)

    pool = build_measurement_pool(
        samples_by_bucket, batch_size, requests_per_bucket, requests_per_slot, min_total_requests
    )
    if not pool:
        raise RuntimeError("Measurement pool is empty: no samples in the 2-5s / 5-15s / 15-30s buckets.")

    batches = [pool[i : i + batch_size] for i in range(0, len(pool), batch_size)]
    sizing = "fixed" if requests_per_bucket is not None else "auto"
    logger.info(
        "Measurement pool: %d requests in %d batches of %d (sizing: %s)",
        len(pool), len(batches), batch_size, sizing,
    )
    for bucket_name in BUCKET_ORDER:
        logger.info("  %s: %d requests", bucket_name, sum(1 for s in pool if s["bucket"] == bucket_name))

    if batch_size > backend.max_num_seqs:
        logger.warning(
            "batch size %d > engine max_num_seqs=%d: extra requests queue inside vLLM.",
            batch_size, backend.max_num_seqs,
        )

    # ------------------------------------------------------------
    # 1. Warm-up (batches of the same size, taken from the front of the pool)
    # ------------------------------------------------------------
    for w in range(warmup_batches):
        wb = batches[w % len(batches)]
        logger.info("Warmup batch %d/%d (%d requests)...", w + 1, warmup_batches, len(wb))
        try:
            backend.transcribe_batch([s["audio"] for s in wb], _contexts(wb, context_mode), language)
        except Exception as e:
            raise RuntimeError(f"Warmup batch failed; the engine is not working: {e}") from e
    if warmup_batches > 0:
        logger.info("Warmup complete.")

    # ------------------------------------------------------------
    # 2. Measurement
    # ------------------------------------------------------------
    if gpu_monitor is not None:
        gpu_monitor.start()

    logger.info("Running %d batches of %d requests...", len(batches), batch_size)
    wall_start = time.perf_counter()
    wall_start_epoch = time.time()  # same clock as the GPU/CPU monitor

    per_batch: List[Dict[str, Any]] = []
    raw_results: List[Dict[str, Any]] = []
    req_index = 0

    for bi, batch in enumerate(batches):
        audios = [s["audio"] for s in batch]
        contexts = _contexts(batch, context_mode)

        outs = None
        error = None
        t_b0 = time.perf_counter()
        try:
            outs = backend.transcribe_batch(audios, contexts, language)
            if len(outs) != len(batch):
                raise RuntimeError(f"expected {len(batch)} outputs, got {len(outs)}")
        except Exception as e:
            logger.exception("Batch %d failed", bi)
            outs, error = None, f"{type(e).__name__}: {e}"
        t_b1 = time.perf_counter()

        batch_time = t_b1 - t_b0
        batch_audio = sum(max(float(s["duration"]), 1e-3) for s in batch)
        per_batch.append({
            "batch_index": bi,
            "n_requests": len(batch),
            "success": outs is not None,
            "batch_time_s": round(batch_time, 4),
            "batch_audio_s": round(batch_audio, 3),
            "rtf_aggregate": round(batch_time / max(batch_audio, 1e-3), 4),
            "audio_s_per_s": round(batch_audio / max(batch_time, 1e-6), 3),
            "t_start_s": round(t_b0 - wall_start, 4),
            "t_done_s": round(t_b1 - wall_start, 4),
        })

        for j, s in enumerate(batch):
            req_index += 1
            dur = max(float(s["duration"]), 1e-3)
            if outs is None:
                raw_results.append({
                    "success": False, "req_index": req_index, "batch_index": bi,
                    "bucket": s["bucket"], "audio_duration_s": dur, "error": error,
                })
                continue
            raw_results.append({
                "success": True,
                "req_index": req_index,
                "batch_index": bi,
                "batch_size": len(batch),
                "bucket": s["bucket"],
                "audio_duration_s": dur,
                "batch_time_s": round(batch_time, 4),
                "rtf": round(batch_time / dur, 4),        # batch wall time / own duration
                "t_start_s": round(t_b0 - wall_start, 4),
                "t_done_s": round(t_b1 - wall_start, 4),
                "language": getattr(outs[j], "language", ""),
                "prediction": getattr(outs[j], "text", ""),
                "reference": s.get("reference", ""),
            })

    wall_time_s = time.perf_counter() - wall_start
    gpu_stats = gpu_monitor.stop() if gpu_monitor is not None else {}

    # ------------------------------------------------------------
    # 3. Metrics
    # ------------------------------------------------------------
    successful = [r for r in raw_results if r.get("success")]
    failed = [r for r in raw_results if not r.get("success")]
    ok_batches = [b for b in per_batch if b["success"]]

    if failed:
        logger.warning(
            "%d/%d requests failed at C=%d. First error: %s",
            len(failed), len(raw_results), batch_size, failed[0].get("error"),
        )

    total_audio_s = sum(r["audio_duration_s"] for r in successful)
    rtfs = [r["rtf"] for r in successful]

    wer_items = [r for r in successful if r.get("reference")]
    preds = [r["prediction"] for r in wer_items]
    refs = [r["reference"] for r in wer_items]
    overall_wer = compute_wer(preds, refs) if refs else 0.0
    overall_cer = compute_cer(preds, refs) if refs else 0.0

    bucket_breakdown: Dict[str, Any] = {}
    for bucket_name in BUCKET_ORDER:
        b_res = [r for r in successful if r["bucket"] == bucket_name]
        b_items = [r for r in b_res if r.get("reference")]
        b_preds = [r["prediction"] for r in b_items]
        b_refs = [r["reference"] for r in b_items]
        bucket_breakdown[bucket_name] = {
            "n_requests": len(b_res),
            "total_audio_seconds": round(sum(r["audio_duration_s"] for r in b_res), 2),
            "rtf": compute_percentiles([r["rtf"] for r in b_res]),
            "wer": round(compute_wer(b_preds, b_refs), 4) if b_refs else None,
            "cer": round(compute_cer(b_preds, b_refs), 4) if b_refs else None,
        }

    throughput = total_audio_s / max(wall_time_s, 1e-4)
    requests_per_s = len(successful) / max(wall_time_s, 1e-4)

    # Steady state: excludes first/last batch (first has cold shapes, last has no successor)
    win = steady_state_window(
        [(r["t_done_s"], r["audio_duration_s"]) for r in successful],
        batch_size,
    )
    if win is None:
        steady: Dict[str, Any] = {
            "available": False,
            "reason": (
                f"needs at least 3*C successful requests "
                f"(C={batch_size}, successful={len(successful)})"
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

    summary = {
        "concurrency": batch_size,
        "mode": "offline-batch (one transcribe(list) call per batch)",
        "config": {
            "backend": "qwen-asr Qwen3ASRModel.LLM (offline vllm.LLM)",
            "model": backend.model_path,
            "batch_size": batch_size,
            "n_batches": len(batches),
            "sizing": sizing,
            "requests_per_slot": None if requests_per_bucket is not None else requests_per_slot,
            "requests_per_bucket": requests_per_bucket,
            "min_total_requests": min_total_requests,
            "total_requests": len(pool),
            "warmup_batches": warmup_batches,
            "context_mode": context_mode,
            "language": language,
            "max_num_seqs": backend.max_num_seqs,
            "max_inference_batch_size": backend.max_inference_batch_size,
            "max_new_tokens": backend.max_new_tokens,
            "gpu_memory_utilization": backend.gpu_memory_utilization,
            "engine_kwargs": getattr(backend, "engine_kwargs", {}),
            "engine_load_time_s": round(getattr(backend, "load_time_s", 0.0), 2),
            "buckets": {n: sum(1 for s in pool if s["bucket"] == n) for n in BUCKET_ORDER},
        },
        "status": {
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "failed_batches": len(per_batch) - len(ok_batches),
        },
        "throughput": {
            "wall_time_s": round(wall_time_s, 3),
            "total_audio_s": round(total_audio_s, 2),
            "audio_s_per_wall_s": round(throughput, 3),
            "requests_per_s": round(requests_per_s, 3),
        },
        "rtf": compute_percentiles(rtfs),
        "rtf_aggregate": compute_percentiles([b["rtf_aggregate"] for b in ok_batches]),
        "batch_time_s": compute_percentiles([b["batch_time_s"] for b in ok_batches]),
        "wer": {"overall": round(overall_wer, 4), "n_evaluated": len(refs)},
        "cer": {"overall": round(overall_cer, 4), "n_evaluated": len(refs)},
        "bucket_breakdown": bucket_breakdown,
        "gpu": gpu_stats,
        "steady_state": steady,
        "per_batch": per_batch,
        "per_sample": raw_results,
    }

    logger.info("Results for batch size C = %d:", batch_size)
    logger.info(
        "  Per-request RTF (batch time / own duration): avg %.4f | P50 %.4f | P95 %.4f",
        summary["rtf"]["avg"], summary["rtf"]["p50"], summary["rtf"]["p95"],
    )
    logger.info(
        "  Aggregate RTF (batch time / batch audio): avg %.4f | P95 %.4f | batch time avg %.2fs",
        summary["rtf_aggregate"]["avg"], summary["rtf_aggregate"]["p95"], summary["batch_time_s"]["avg"],
    )
    logger.info("  Throughput: %.2f audio-s/s (%.2f req/s)", throughput, requests_per_s)
    if steady.get("available"):
        sg = steady.get("gpu", {})
        logger.info(
            "  Steady state (%.2fs window, %d reqs): %.2f audio-s/s (%.2f req/s) | GPU %.1f%%",
            steady["window_s"], steady["n_requests"], steady["audio_s_per_wall_s"],
            steady["requests_per_s"], sg.get("gpu_util_avg_pct", 0.0),
        )
        if sg:
            logger.info("  Steady-state CPU: %s", format_cpu_summary(sg))
    else:
        logger.info("  Steady state: not available (%s)", steady.get("reason"))
    logger.info("  WER: %.2f%% | CER: %.2f%% (%d samples)", overall_wer * 100, overall_cer * 100, len(refs))
    logger.info(
        "  GPU Util: %.1f%% (Max: %.1f%%) | Peak VRAM: %s MB",
        gpu_stats.get("gpu_util_avg_pct", 0.0),
        gpu_stats.get("gpu_util_max_pct", 0.0),
        gpu_stats.get("gpu_mem_used_peak_mb", 0.0),
    )
    logger.info("  CPU Util: %s", format_cpu_summary(gpu_stats))

    return summary