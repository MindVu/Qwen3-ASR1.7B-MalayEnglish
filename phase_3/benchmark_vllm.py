"""
phase 3: Concurrency Benchmark Suite for vLLM Backend (continuous batching).

Executes concurrency sweep C = [1, 2, 4, 8, 16, 32, 64] using the
same workload construction as Phase 1:

    2-5s    : 30 requests  (configurable via --requests_per_bucket)
    5-15s   : 30 requests
    15-30s  : 30 requests
    Total   : 90 requests / concurrency level

The workload is constructed inside load_test.py from samples_by_bucket.

Collects:
  * Avg RTF, P50 RTF, P95 RTF
  * Audio throughput (sec/s), Requests/s
  * GPU utilization (%) & Peak VRAM (MB)
  * WER (%) using standardized text normalizer

Saves:
  phase_3/results/concurrency_{C}.json
  phase_3/results/summary.csv
  phase_3/results/summary.md
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from common import load_benchmark_dataset
from load_test import (
    DEFAULT_CONCURRENCIES,
    DEFAULT_WARMUP,
    REQUESTS_PER_BUCKET,
    preload_audio,
    run_concurrent_benchmark,
)
from monitoring import GPUMonitor, fmt_num, fmt_pct
from vllm_backend import AsyncVLLMBackend


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("phase_3.benchmark")


# Frozen Phase 1 HF baseline reference metrics
HF_BASELINE_REFERENCE = {
    "p95_rtf": 0.387,
    "avg_rtf": 0.251,
    "throughput_audio_s_per_wall_s": 4.71,
    "gpu_util_avg_pct": 33.5,
    "vram_peak_gb": 5.9,
    "wer_pct": 28.3,
}


BUCKET_NAMES = [
    "2-5s",
    "5-15s",
    "15-30s",
]


def save_concurrency_result(result: Dict[str, Any], output_dir: str) -> None:
    """Save one concurrency result as JSON."""

    os.makedirs(output_dir, exist_ok=True)

    concurrency = result["concurrency"]
    filepath = os.path.join(output_dir, f"concurrency_{concurrency}.json")

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info("Saved concurrency result to %s", filepath)


def generate_summary_tables(
    results_list: List[Dict[str, Any]],
    output_dir: str,
) -> None:
    """
    Generate CSV and Markdown summary tables.

    Success criteria (maximum sustainable concurrency = highest C reached
    before the FIRST failure of the criterion, with no failed requests):
      * Primary: P95 RTF <= 0.5
      * Strong : P95 RTF <= 0.3
    """

    os.makedirs(output_dir, exist_ok=True)

    headers = [
        "Concurrency",
        "Requests",
        "Failed",
        "Avg RTF",
        "P50 RTF",
        "P95 RTF",
        "Client Avg RTF",
        "Client P95 RTF",
        "Audio sec/s",
        "Req/s",
        "GPU %",
        "Steady Audio sec/s",
        "Steady GPU %",
        "CPU Sys %",
        "CPU Client %",
        "CPU Engine %",
        "VRAM (MB)",
        "WER",
        "CER",
        "Status (<=0.5)",
    ]

    rows = []
    csv_lines = [",".join(headers)]

    max_c_05 = None
    max_c_03 = None
    broken_05 = False
    broken_03 = False

    # Make sure results are processed in ascending concurrency order.
    for result in sorted(results_list, key=lambda r: r["concurrency"]):

        concurrency = result["concurrency"]

        avg_rtf = result["rtf"]["avg"]
        p50_rtf = result["rtf"]["p50"]
        p95_rtf = result["rtf"]["p95"]
        c_avg_rtf = result["rtf_client_e2e"]["avg"]
        c_p95_rtf = result["rtf_client_e2e"]["p95"]

        audio_throughput = result["throughput"]["audio_s_per_wall_s"]
        requests_per_s = result["throughput"]["requests_per_s"]

        gpu_util = result["gpu"].get("gpu_util_avg_pct", 0.0)
        vram_mb = result["gpu"].get("gpu_mem_used_peak_mb", 0.0)
        cpu_sys = result["gpu"].get("cpu_sys_util_avg_pct")
        cpu_cli = result["gpu"].get("cpu_proc_util_avg_pct")
        cpu_eng = result["gpu"].get("cpu_children_util_avg_pct")
        steady = result.get("steady_state") or {}
        st_ok = bool(steady.get("available"))
        st_tp = steady.get("audio_s_per_wall_s") if st_ok else None
        st_gpu = (steady.get("gpu") or {}).get("gpu_util_avg_pct") if st_ok else None

        wer_pct = result["wer"]["overall"] * 100
        cer_pct = result["cer"]["overall"] * 100

        total_requests = result["config"].get("total_requests", 0)
        n_failed = result["status"]["failed_requests"]
        n_ok = result["status"]["successful_requests"]

        # A level only counts as passing if it actually produced results
        # and had no failed requests.
        healthy = n_ok > 0 and n_failed == 0

        ok_05 = healthy and p95_rtf <= 0.5
        ok_03 = healthy and p95_rtf <= 0.3

        if ok_05 and not broken_05:
            max_c_05 = concurrency
        else:
            broken_05 = True

        if ok_03 and not broken_03:
            max_c_03 = concurrency
        else:
            broken_03 = True

        status_05 = "PASS" if ok_05 else "FAIL"

        rows.append([
            str(concurrency),
            str(total_requests),
            str(n_failed),
            f"{avg_rtf:.4f}",
            f"{p50_rtf:.4f}",
            f"{p95_rtf:.4f}",
            f"{c_avg_rtf:.4f}",
            f"{c_p95_rtf:.4f}",
            f"{audio_throughput:.2f}",
            f"{requests_per_s:.2f}",
            f"{gpu_util:.1f}%",
            "n/a" if st_tp is None else f"{st_tp:.2f}",
            fmt_pct(st_gpu),
            fmt_pct(cpu_sys),
            fmt_pct(cpu_cli),
            fmt_pct(cpu_eng),
            f"{vram_mb:.1f}",
            f"{wer_pct:.2f}%",
            f"{cer_pct:.2f}%",
            status_05,
        ])

        csv_lines.append(
            ",".join([
                str(concurrency),
                str(total_requests),
                str(n_failed),
                f"{avg_rtf:.4f}",
                f"{p50_rtf:.4f}",
                f"{p95_rtf:.4f}",
                f"{c_avg_rtf:.4f}",
                f"{c_p95_rtf:.4f}",
                f"{audio_throughput:.2f}",
                f"{requests_per_s:.2f}",
                f"{gpu_util:.1f}",
                "" if st_tp is None else f"{st_tp:.2f}",
                fmt_num(st_gpu),
                fmt_num(cpu_sys),
                fmt_num(cpu_cli),
                fmt_num(cpu_eng),
                f"{vram_mb:.1f}",
                f"{wer_pct:.2f}",
                f"{cer_pct:.2f}",
                status_05,
            ])
        )

    # ------------------------------------------------------------
    # Markdown table
    # ------------------------------------------------------------

    md_lines = [
        "# phase 3 — vLLM Concurrency Benchmark Summary (continuous batching)",
        "",
        "## Concurrency Sweep Results",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join([" ---: " for _ in headers]) + "|",
    ]

    for row in rows:
        md_lines.append("| " + " | ".join(row) + " |")

    # ------------------------------------------------------------
    # C=1 comparison
    # ------------------------------------------------------------

    c1_result = next((r for r in results_list if r["concurrency"] == 1), None)

    if c1_result:
        c1_p95 = f"{c1_result['rtf']['p95']:.4f}"
        c1_avg = f"{c1_result['rtf']['avg']:.4f}"
        c1_tp = f"{c1_result['throughput']['audio_s_per_wall_s']:.2f} sec/s"
        c1_gpu = f"{c1_result['gpu'].get('gpu_util_avg_pct', 0.0):.1f}%"
        c1_vram = f"{c1_result['gpu'].get('gpu_mem_used_peak_mb', 0.0) / 1024:.2f} GB"
        c1_wer = f"{c1_result['wer']['overall'] * 100:.2f}%"
    else:
        c1_p95 = c1_avg = c1_tp = c1_gpu = c1_vram = c1_wer = "N/A"

    # ------------------------------------------------------------
    # Best throughput
    # ------------------------------------------------------------

    best_tp_result = (
        max(results_list, key=lambda x: x["throughput"]["audio_s_per_wall_s"])
        if results_list
        else None
    )

    if best_tp_result:
        best_tp = (
            f"{best_tp_result['throughput']['audio_s_per_wall_s']:.2f} "
            f"sec/s (C={best_tp_result['concurrency']})"
        )
        best_p95 = f"{best_tp_result['rtf']['p95']:.4f}"
        best_avg = f"{best_tp_result['rtf']['avg']:.4f}"
        best_gpu = f"{best_tp_result['gpu'].get('gpu_util_avg_pct', 0.0):.1f}%"
        best_vram = f"{best_tp_result['gpu'].get('gpu_mem_used_peak_mb', 0.0) / 1024:.2f} GB"
        best_wer = f"{best_tp_result['wer']['overall'] * 100:.2f}%"
    else:
        best_tp = best_p95 = best_avg = best_gpu = best_vram = best_wer = "N/A"

    # ------------------------------------------------------------
    # Comparison with Phase 1
    # ------------------------------------------------------------

    c1_cer = f"{c1_result['cer']['overall'] * 100:.2f}%" if c1_result else "N/A"
    best_cer = f"{best_tp_result['cer']['overall'] * 100:.2f}%" if best_tp_result else "N/A"
    c1_cp95 = f"{c1_result['rtf_client_e2e']['p95']:.4f}" if c1_result else "N/A"
    best_cp95 = (
        f"{best_tp_result['rtf_client_e2e']['p95']:.4f}" if best_tp_result else "N/A"
    )

    md_lines.extend([
        "",
        "## Comparison with Frozen Phase 1 HF Baseline",
        "",
        "| Metric | HF Baseline (Phase 1) | vLLM (C=1) | vLLM Best Throughput |",
        "|---|---:|---:|---:|",
        f"| **P95 RTF** | {HF_BASELINE_REFERENCE['p95_rtf']} | {c1_p95} | {best_p95} |",
        f"| **Avg RTF** | {HF_BASELINE_REFERENCE['avg_rtf']} | {c1_avg} | {best_avg} |",
        f"| **Client P95 RTF** | see Phase 1 `baseline_summary` | {c1_cp95} | {best_cp95} |",
        (
            f"| **Audio Throughput** | "
            f"{HF_BASELINE_REFERENCE['throughput_audio_s_per_wall_s']} sec/s | "
            f"{c1_tp} | {best_tp} |"
        ),
        (
            f"| **GPU Utilization** | "
            f"{HF_BASELINE_REFERENCE['gpu_util_avg_pct']}% | {c1_gpu} | {best_gpu} |"
        ),
        (
            f"| **Peak VRAM** | "
            f"{HF_BASELINE_REFERENCE['vram_peak_gb']} GB | {c1_vram} | {best_vram} |"
        ),
        f"| **WER** | {HF_BASELINE_REFERENCE['wer_pct']}% | {c1_wer} | {best_wer} |",
        f"| **CER** | see Phase 1 `baseline_summary` | {c1_cer} | {best_cer} |",
        "",
        "## Success Criterion Assessment",
        "",
        (
            "- **Primary Target (P95 RTF ≤ 0.5):** "
            f"Maximum Sustainable Concurrency = "
            f"**{max_c_05 if max_c_05 is not None else 'None'}**"
        ),
        (
            "- **Strong Target (P95 RTF ≤ 0.3):** "
            f"Maximum Sustainable Concurrency = "
            f"**{max_c_03 if max_c_03 is not None else 'None'}**"
        ),
        "",
    ])

    md_table = "\n".join(md_lines)

    csv_path = os.path.join(output_dir, "summary.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines) + "\n")

    md_path = os.path.join(output_dir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_table + "\n")

    print()
    print("=" * 80)
    print("phase 3 vLLM CONCURRENCY BENCHMARK SUMMARY")
    print("=" * 80)
    print(md_table)
    print("=" * 80)
    print()


async def run_benchmark_suite(
    model_path: str = "Qwen/Qwen3-ASR-1.7B",
    test_file: str = "data/test.jsonl",
    audio_dir: str = "data/audio_clips/test",
    output_dir: str = "phase_3/results",
    concurrencies: Optional[List[int]] = None,
    warmup: int = DEFAULT_WARMUP,
    requests_per_bucket: int = REQUESTS_PER_BUCKET,
    gpu_mem: float = 0.9,
    max_num_seqs: Optional[int] = None,
    engine_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Run the complete phase 3 concurrency sweep.

    One shared AsyncLLM engine is reused for every concurrency level, so all
    in-flight requests are continuously batched by vLLM.
    """

    if concurrencies is None:
        concurrencies = DEFAULT_CONCURRENCIES

    # Extra vLLM engine arguments for ablations (e.g. {"dtype": "float16"}).
    engine_kwargs = dict(engine_kwargs or {})
    gpu_mem = engine_kwargs.pop("gpu_memory_utilization", gpu_mem)
    max_num_seqs = engine_kwargs.pop("max_num_seqs", max_num_seqs)
    if engine_kwargs:
        logger.info("Extra vLLM engine args: %s", engine_kwargs)

    if max_num_seqs is None:
        max_num_seqs = max(128, max(concurrencies))

    logger.info("Initializing phase 3 benchmark sweep: %s", concurrencies)

    # 1. Load dataset
    samples = load_benchmark_dataset(test_file=test_file, audio_dir=audio_dir)
    preload_audio(samples)

    logger.info("Loaded %d frozen benchmark utterances from %s.", len(samples), test_file)

    # 2. Group samples into the exact Phase 1 buckets
    samples_by_bucket: Dict[str, List[Dict[str, Any]]] = {b: [] for b in BUCKET_NAMES}

    for sample in samples:
        bucket = sample.get("bucket")
        if bucket in samples_by_bucket:
            samples_by_bucket[bucket].append(sample)
        else:
            logger.warning(
                "Sample %s has unknown/missing bucket: %r",
                sample.get("id"),
                bucket,
            )

    logger.info("Frozen dataset bucket distribution:")
    for bucket in BUCKET_NAMES:
        logger.info("  %-7s: %d source utterances", bucket, len(samples_by_bucket[bucket]))

    # 3. Validate workload
    missing_buckets = [b for b in BUCKET_NAMES if not samples_by_bucket[b]]
    if missing_buckets:
        raise RuntimeError(
            "Cannot reproduce the Phase 1 workload because these "
            f"duration buckets contain no samples: {missing_buckets}"
        )

    expected_requests = len(BUCKET_NAMES) * requests_per_bucket
    logger.info("Expected measured requests per concurrency level: %d", expected_requests)

    # 4. ONE shared async engine (created inside the running event loop)
    backend = await AsyncVLLMBackend.create(
        model_path=model_path,
        gpu_memory_utilization=gpu_mem,
        max_num_seqs=max_num_seqs,
        **engine_kwargs,
    )

    logger.info("Shared async vLLM backend initialized once (max_num_seqs=%d).", max_num_seqs)

    # 5. GPU monitor
    gpu_monitor = GPUMonitor(device_index=0, interval=0.05)

    all_results: List[Dict[str, Any]] = []

    try:
        # 6. Concurrency sweep
        for concurrency in concurrencies:

            logger.info("")
            logger.info("=" * 80)
            logger.info(">>> Benchmarking Concurrency C = %d <<<", concurrency)
            logger.info("=" * 80)

            result = await run_concurrent_benchmark(
                backend=backend,
                samples_by_bucket=samples_by_bucket,
                concurrency=concurrency,
                requests_per_bucket=requests_per_bucket,
                warmup_requests=warmup,
                gpu_monitor=gpu_monitor,
            )

            save_concurrency_result(result=result, output_dir=output_dir)
            all_results.append(result)

            await asyncio.sleep(1.0)
    finally:
        backend.shutdown()

    # 7. Final summary
    generate_summary_tables(results_list=all_results, output_dir=output_dir)


def main():

    parser = argparse.ArgumentParser(
        description="phase 3: vLLM Concurrency Benchmark Suite"
    )

    parser.add_argument("--model", default="models/lora_r16_merged",
                        help="Model path or Hugging Face model ID")
    parser.add_argument("--test_file", default="data/test.jsonl", help="Test dataset path")
    parser.add_argument("--audio_dir", default="data/audio_clips/test",
                        help="Test audio directory")
    parser.add_argument("--output_dir", default="phase_3/results",
                        help="Results output directory")
    parser.add_argument("--concurrencies", type=str, default="1,2,4,8,16,32,64",
                        help="Comma-separated concurrency levels")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help="Number of concurrent warmup requests")
    parser.add_argument("--requests_per_bucket", type=int, default=REQUESTS_PER_BUCKET,
                        help="Measured requests per duration bucket (default: matches Phase 1)")
    parser.add_argument("--gpu_mem", type=float, default=0.9,
                        help="vLLM GPU memory utilization")
    parser.add_argument("--max_num_seqs", type=int, default=None,
                        help="vLLM max concurrent sequences (default: max(128, max concurrency))")

    parser.add_argument("--engine_args", type=str, default="{}",
                        help='JSON dict of extra vLLM engine args, e.g. \'{"dtype": "float16"}\'')

    args = parser.parse_args()
    engine_kwargs = json.loads(args.engine_args)

    concurrencies = [int(c.strip()) for c in args.concurrencies.split(",") if c.strip()]

    asyncio.run(
        run_benchmark_suite(
            model_path=args.model,
            test_file=args.test_file,
            audio_dir=args.audio_dir,
            output_dir=args.output_dir,
            concurrencies=concurrencies,
            warmup=args.warmup,
            requests_per_bucket=args.requests_per_bucket,
            gpu_mem=args.gpu_mem,
            max_num_seqs=args.max_num_seqs,
            engine_kwargs=engine_kwargs,
        )
    )


if __name__ == "__main__":
    main()