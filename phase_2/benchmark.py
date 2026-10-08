"""
Phase 2: Batch-size sweep for the official qwen-asr vLLM wrapper (Qwen3ASRModel.LLM).

For every C in the sweep (default 1,2,4,8,16,32,64):
    total requests = requests_per_slot * C   (default 10 * C)
    run them as consecutive batches of C requests, one transcribe(list) call each.

See load_test.py for the exact metric definitions (per-request RTF = batch time /
own duration, aggregate RTF = batch time / batch audio, steady-state window).

Saves:
  phase_2/results/batch_{C}.json
  phase_2/results/summary.csv
  phase_2/results/summary.md

Run:
  python phase_2/benchmark.py --model models/lora_r16_merged
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
for _p in (CURRENT_DIR, REPO_ROOT, REPO_ROOT / "phase2"):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))
if str(CURRENT_DIR) in sys.path:
    sys.path.remove(str(CURRENT_DIR))
sys.path.insert(0, str(CURRENT_DIR))

from common import load_benchmark_dataset
from load_test import (
    BUCKET_ORDER,
    DEFAULT_CONCURRENCIES,
    DEFAULT_WARMUP_BATCHES,
    REQUESTS_PER_SLOT,
    preload_audio,
    run_batched_benchmark,
)
from monitoring import GPUMonitor, fmt_num, fmt_pct
from qwen_backend import QwenVLLMBackend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("phase_2.benchmark")


def save_result(result: Dict[str, Any], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"batch_{result['concurrency']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Saved result to %s", path)


def generate_summary_tables(results: List[Dict[str, Any]], output_dir: str) -> None:
    """
    CSV + Markdown summary. Maximum sustainable batch size = highest C reached before
    the FIRST level that fails the criterion (or has failed requests).
    """
    os.makedirs(output_dir, exist_ok=True)

    headers = [
        "Batch size (C)", "Requests", "Failed",
        "Avg RTF", "P50 RTF", "P95 RTF", "Agg RTF",
        "Audio sec/s", "Req/s", "Batch time (s)",
        "GPU %", "Steady Audio sec/s", "Steady GPU %",
        "CPU Sys %", "CPU Client %", "CPU Engine %",
        "VRAM (MB)", "WER", "CER", "Status (<=0.5)",
    ]
    rows: List[List[str]] = []
    csv_lines = [",".join(headers)]

    max_c_05 = max_c_03 = None
    broken_05 = broken_03 = False

    for r in sorted(results, key=lambda x: x["concurrency"]):
        c = r["concurrency"]
        g = r["gpu"]
        st = r.get("steady_state") or {}
        st_ok = bool(st.get("available"))
        st_tp = st.get("audio_s_per_wall_s") if st_ok else None
        st_gpu = (st.get("gpu") or {}).get("gpu_util_avg_pct") if st_ok else None

        p95 = r["rtf"]["p95"]
        healthy = r["status"]["successful_requests"] > 0 and r["status"]["failed_requests"] == 0
        ok_05, ok_03 = healthy and p95 <= 0.5, healthy and p95 <= 0.3
        if ok_05 and not broken_05:
            max_c_05 = c
        else:
            broken_05 = True
        if ok_03 and not broken_03:
            max_c_03 = c
        else:
            broken_03 = True

        wer, cer = r["wer"]["overall"] * 100, r["cer"]["overall"] * 100
        rows.append([
            str(c), str(r["config"]["total_requests"]), str(r["status"]["failed_requests"]),
            f"{r['rtf']['avg']:.4f}", f"{r['rtf']['p50']:.4f}", f"{p95:.4f}",
            f"{r['rtf_aggregate']['avg']:.4f}",
            f"{r['throughput']['audio_s_per_wall_s']:.2f}", f"{r['throughput']['requests_per_s']:.2f}",
            f"{r['batch_time_s']['avg']:.2f}",
            fmt_pct(g.get("gpu_util_avg_pct")),
            "n/a" if st_tp is None else f"{st_tp:.2f}", fmt_pct(st_gpu),
            fmt_pct(g.get("cpu_sys_util_avg_pct")), fmt_pct(g.get("cpu_proc_util_avg_pct")),
            fmt_pct(g.get("cpu_children_util_avg_pct")),
            f"{g.get('gpu_mem_used_peak_mb', 0.0):.1f}", f"{wer:.2f}%", f"{cer:.2f}%",
            "PASS" if ok_05 else "FAIL",
        ])
        csv_lines.append(",".join([
            str(c), str(r["config"]["total_requests"]), str(r["status"]["failed_requests"]),
            f"{r['rtf']['avg']:.4f}", f"{r['rtf']['p50']:.4f}", f"{p95:.4f}",
            f"{r['rtf_aggregate']['avg']:.4f}",
            f"{r['throughput']['audio_s_per_wall_s']:.2f}", f"{r['throughput']['requests_per_s']:.2f}",
            f"{r['batch_time_s']['avg']:.2f}",
            fmt_num(g.get("gpu_util_avg_pct")),
            "" if st_tp is None else f"{st_tp:.2f}", fmt_num(st_gpu),
            fmt_num(g.get("cpu_sys_util_avg_pct")), fmt_num(g.get("cpu_proc_util_avg_pct")),
            fmt_num(g.get("cpu_children_util_avg_pct")),
            f"{g.get('gpu_mem_used_peak_mb', 0.0):.1f}", f"{wer:.2f}", f"{cer:.2f}",
            "PASS" if ok_05 else "FAIL",
        ]))

    md = [
        "# Phase 2 — qwen-asr vLLM wrapper (offline batch) Benchmark Summary",
        "",
        "## Batch-size sweep",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join([" ---: " for _ in headers]) + "|",
    ]
    md += ["| " + " | ".join(row) + " |" for row in rows]
    md += [
        "",
        "## Success criterion (per-request RTF, informational)",
        "",
        f"- **P95 RTF ≤ 0.5:** maximum sustainable batch size = **{max_c_05 if max_c_05 is not None else 'None'}**",
        f"- **P95 RTF ≤ 0.3:** maximum sustainable batch size = **{max_c_03 if max_c_03 is not None else 'None'}**",
        "",
        "## How to read the metrics",
        "",
        "- One `transcribe(list)` call per batch; all requests of a batch finish together.",
        "- **Avg/P50/P95 RTF** = batch wall time / the request's OWN audio duration. Pessimistic for short "
        "clips batched with long ones (they wait for the longest clip in the batch).",
        "- **Agg RTF** = batch wall time / total audio of the batch (inverse of batch throughput).",
        "- **Steady** columns exclude the first and last batch.",
        "- Batch time includes everything inside the wrapper call: audio normalization, prompt building, "
        "vLLM generate and output parsing.",
        "",
    ]
    md_text = "\n".join(md)

    with open(os.path.join(output_dir, "summary.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines) + "\n")
    with open(os.path.join(output_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md_text + "\n")

    print("\n" + "=" * 80)
    print("Phase 2 SUMMARY")
    print("=" * 80)
    print(md_text)
    print("=" * 80 + "\n")


def run_benchmark_suite(
    model_path: str,
    test_file: str,
    audio_dir: str,
    output_dir: str,
    concurrencies: List[int],
    requests_per_bucket: Optional[int] = None,
    requests_per_slot: int = REQUESTS_PER_SLOT,
    min_total_requests: int = 0,
    warmup_batches: int = DEFAULT_WARMUP_BATCHES,
    context_mode: str = "empty",
    language: Optional[str] = None,
    gpu_mem: float = 0.9,
    max_num_seqs: Optional[int] = None,
    max_new_tokens: int = 512,
    max_inference_batch_size: int = -1,
    engine_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:

    if max_num_seqs is None:
        max_num_seqs = max(2048, max(concurrencies))

    # 1. Dataset -> buckets
    samples = load_benchmark_dataset(test_file=test_file, audio_dir=audio_dir)
    logger.info("Loaded %d benchmark utterances from %s.", len(samples), test_file)

    samples_by_bucket: Dict[str, List[Dict[str, Any]]] = {b: [] for b in BUCKET_ORDER}
    for s in samples:
        if s.get("bucket") in samples_by_bucket:
            samples_by_bucket[s["bucket"]].append(s)
    for b in BUCKET_ORDER:
        logger.info("  %-7s: %d source utterances", b, len(samples_by_bucket[b]))

    missing = [b for b in BUCKET_ORDER if not samples_by_bucket[b]]
    if missing:
        raise RuntimeError(f"Duration buckets contain no samples: {missing}")

    preload_audio(samples)

    if requests_per_bucket is None:
        logger.info("Request sizing: auto, %d requests per slot (total = %d * C, min total %d)",
                    requests_per_slot, requests_per_slot, min_total_requests)
    else:
        logger.info("Request sizing: fixed, %d requests per bucket at every level", requests_per_bucket)

    # 2. ONE engine for the whole sweep
    backend = QwenVLLMBackend(
        model_path=model_path,
        gpu_memory_utilization=gpu_mem,
        max_num_seqs=max_num_seqs,
        max_new_tokens=max_new_tokens,
        max_inference_batch_size=max_inference_batch_size,
        **(engine_kwargs or {}),
    )
    gpu_monitor = GPUMonitor(device_index=0, interval=0.05)

    results: List[Dict[str, Any]] = []
    try:
        for c in concurrencies:
            result = run_batched_benchmark(
                backend=backend,
                samples_by_bucket=samples_by_bucket,
                batch_size=c,
                requests_per_bucket=requests_per_bucket,
                requests_per_slot=requests_per_slot,
                min_total_requests=min_total_requests,
                warmup_batches=warmup_batches,
                context_mode=context_mode,
                language=language,
                gpu_monitor=gpu_monitor,
            )
            save_result(result, output_dir)
            results.append(result)
            time.sleep(1.0)  # let the GPU settle between levels
    finally:
        backend.shutdown()

    generate_summary_tables(results, output_dir)
    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 2: qwen-asr vLLM wrapper batch benchmark")
    parser.add_argument("--model", default="models/lora_r16_merged", help="Model path or HF model ID")
    parser.add_argument("--test_file", default="data/test.jsonl")
    parser.add_argument("--audio_dir", default="data/audio_clips/test")
    parser.add_argument("--output_dir", default="phase_2/results")
    parser.add_argument("--concurrencies", default="1,2,4,8,16,32,64",
                        help="Comma-separated batch sizes (C)")
    parser.add_argument("--requests_per_slot", type=int, default=REQUESTS_PER_SLOT,
                        help="Auto sizing: total requests per level = requests_per_slot * C")
    parser.add_argument("--requests_per_bucket", type=int, default=None,
                        help="Fixed sizing: requests per bucket at every level (disables auto sizing)")
    parser.add_argument("--min_total_requests", type=int, default=0,
                        help="Auto sizing floor, so small C still get enough samples (e.g. 90)")
    parser.add_argument("--warmup_batches", type=int, default=DEFAULT_WARMUP_BATCHES,
                        help="Warmup batches of size C before each level")
    parser.add_argument("--context_mode", choices=["empty", "sample"], default="empty",
                        help="System message: '' (like Phase 2) or the sample prompt (like Phase 1)")
    parser.add_argument("--language", default=None, help="Force a language (default: let the model decide)")
    parser.add_argument("--gpu_mem", type=float, default=0.9)
    parser.add_argument("--max_num_seqs", type=int, default=None,
                        help="vLLM max concurrent sequences (default: max(128, max C))")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_inference_batch_size", type=int, default=-1,
                        help="Wrapper-side splitting of a batch into sequential generate() calls "
                             "(-1 = no splitting, recommended)")
    parser.add_argument("--engine_args", type=str, default="{}",
                        help='JSON dict of extra vLLM args, e.g. \'{"dtype": "bfloat16"}\'')
    args = parser.parse_args()

    run_benchmark_suite(
        model_path=args.model,
        test_file=args.test_file,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        concurrencies=[int(c) for c in args.concurrencies.split(",") if c.strip()],
        requests_per_bucket=args.requests_per_bucket,
        requests_per_slot=args.requests_per_slot,
        min_total_requests=args.min_total_requests,
        warmup_batches=args.warmup_batches,
        context_mode=args.context_mode,
        language=args.language,
        gpu_mem=args.gpu_mem,
        max_num_seqs=args.max_num_seqs,
        max_new_tokens=args.max_new_tokens,
        max_inference_batch_size=args.max_inference_batch_size,
        engine_kwargs=json.loads(args.engine_args),
    )


if __name__ == "__main__":
    main()