"""
Phase 2: Concurrency Benchmark Suite for vLLM Backend.

Executes concurrency sweep C = [1, 2, 4, 8, 16, 32, 64] across the frozen
43 benchmark utterances, collecting:
  * Avg RTF, P50 RTF, P95 RTF
  * Audio throughput (sec/s), Requests/s, Generated tokens/s
  * GPU utilization (%) & Peak VRAM (MB)
  * WER (%) using standardized text normalizer

Saves machine-readable results:
  phase2/results/concurrency_{C}.json
  phase2/results/summary.csv
  phase2/results/summary.md
"""

import argparse
import asyncio
import json
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

from common import load_benchmark_dataset
from load_test import run_concurrent_benchmark
from monitoring import GPUMonitor
from vllm_backend import VLLMInferenceBackend, get_vllm_backend

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phase2.benchmark")

# Frozen Phase 1 HF baseline reference metrics
HF_BASELINE_REFERENCE = {
    "p95_rtf": 0.387,
    "avg_rtf": 0.251,
    "throughput_audio_s_per_wall_s": 4.71,
    "gpu_util_avg_pct": 33.5,
    "vram_peak_gb": 5.9,
    "wer_pct": 28.3,
}


def save_concurrency_result(result: Dict[str, Any], output_dir: str):
    """Saves individual concurrency result to results/concurrency_{C}.json."""
    os.makedirs(output_dir, exist_ok=True)
    c = result["concurrency"]
    filepath = os.path.join(output_dir, f"concurrency_{c}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Saved machine-readable concurrency result to %s", filepath)


def generate_summary_tables(results_list: List[Dict[str, Any]], output_dir: str):
    """
    Generates CSV and Markdown summary tables across concurrency levels
    and evaluates success criterion: max_concurrency(P95_RTF <= 0.5) and <= 0.3.
    """
    os.makedirs(output_dir, exist_ok=True)

    headers = [
        "Concurrency",
        "Avg RTF",
        "P50 RTF",
        "P95 RTF",
        "Audio sec/s",
        "Req/s",
        "GPU %",
        "VRAM (MB)",
        "WER",
        "Status (<=0.5)",
    ]

    rows = []
    csv_lines = [",".join(headers)]

    max_c_05 = None
    max_c_03 = None

    for r in results_list:
        c = r["concurrency"]
        avg_rtf = f"{r['rtf']['avg']:.4f}"
        p50_rtf = f"{r['rtf']['p50']:.4f}"
        p95_rtf_val = r["rtf"]["p95"]
        p95_rtf = f"{p95_rtf_val:.4f}"
        tp = f"{r['throughput']['audio_s_per_wall_s']:.2f}"
        req_s = f"{r['throughput']['requests_per_s']:.2f}"
        gpu_pct = f"{r['gpu'].get('gpu_util_avg_pct', 0.0):.1f}%"
        vram_mb = f"{r['gpu'].get('gpu_mem_used_peak_mb', 0.0):.1f}"
        wer_pct = f"{r['wer']['overall'] * 100:.2f}%"

        status_05 = "PASS" if p95_rtf_val <= 0.5 else "FAIL"
        if p95_rtf_val <= 0.5:
            max_c_05 = c
        if p95_rtf_val <= 0.3:
            max_c_03 = c

        rows.append([str(c), avg_rtf, p50_rtf, p95_rtf, tp, req_s, gpu_pct, vram_mb, wer_pct, status_05])
        csv_lines.append(f"{c},{avg_rtf},{p50_rtf},{p95_rtf},{tp},{req_s},{r['gpu'].get('gpu_util_avg_pct', 0.0)},{r['gpu'].get('gpu_mem_used_peak_mb', 0.0)},{r['wer']['overall'] * 100:.2f},{status_05}")

    # Build Markdown table
    md_lines = [
        "# Phase 2 — vLLM Concurrency Benchmark Summary",
        "",
        "## Concurrency Sweep Results",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join([" ---: " for _ in headers]) + "|",
    ]
    for row in rows:
        md_lines.append("| " + " | ".join(row) + " |")

    # Add Comparative Analysis against Phase 1 HF Baseline
    c1_res = next((r for r in results_list if r["concurrency"] == 1), None)
    c1_p95 = f"{c1_res['rtf']['p95']:.4f}" if c1_res else "N/A"
    c1_avg = f"{c1_res['rtf']['avg']:.4f}" if c1_res else "N/A"
    c1_tp = f"{c1_res['throughput']['audio_s_per_wall_s']:.2f} sec/s" if c1_res else "N/A"
    c1_gpu = f"{c1_res['gpu'].get('gpu_util_avg_pct', 0.0):.1f}%" if c1_res else "N/A"
    c1_vram = f"{c1_res['gpu'].get('gpu_mem_used_peak_mb', 0.0) / 1024:.2f} GB" if c1_res else "N/A"
    c1_wer = f"{c1_res['wer']['overall'] * 100:.2f}%" if c1_res else "N/A"

    best_tp_res = max(results_list, key=lambda x: x["throughput"]["audio_s_per_wall_s"]) if results_list else None
    best_tp = f"{best_tp_res['throughput']['audio_s_per_wall_s']:.2f} sec/s (C={best_tp_res['concurrency']})" if best_tp_res else "N/A"

    md_lines.extend([
        "",
        "## Comparison with Frozen Phase 1 HF Baseline",
        "",
        "| Metric | HF Baseline (Phase 1) | vLLM (C=1) | vLLM Best Peak |",
        "|---|---:|---:|---:|",
        f"| **P95 RTF** | {HF_BASELINE_REFERENCE['p95_rtf']} | {c1_p95} | {best_tp_res['rtf']['p95'] if best_tp_res else 'N/A'} |",
        f"| **Avg RTF** | {HF_BASELINE_REFERENCE['avg_rtf']} | {c1_avg} | {best_tp_res['rtf']['avg'] if best_tp_res else 'N/A'} |",
        f"| **Audio Throughput** | {HF_BASELINE_REFERENCE['throughput_audio_s_per_wall_s']} sec/s | {c1_tp} | {best_tp} |",
        f"| **GPU Utilization** | {HF_BASELINE_REFERENCE['gpu_util_avg_pct']}% | {c1_gpu} | {best_tp_res['gpu'].get('gpu_util_avg_pct', 0.0) if best_tp_res else 'N/A'}% |",
        f"| **Peak VRAM** | {HF_BASELINE_REFERENCE['vram_peak_gb']} GB | {c1_vram} | {best_tp_res['gpu'].get('gpu_mem_used_peak_mb', 0.0) / 1024 if best_tp_res else 'N/A':.2f} GB |",
        f"| **WER** | {HF_BASELINE_REFERENCE['wer_pct']}% | {c1_wer} | {best_tp_res['wer']['overall'] * 100 if best_tp_res else 'N/A':.2f}% |",
        "",
        "## Success Criterion Assessment",
        "",
        f"- **Primary Target (P95 RTF ≤ 0.5):** Maximum Sustainable Concurrency = **{max_c_05 if max_c_05 is not None else 'None'}**",
        f"- **Strong Target (P95 RTF ≤ 0.3):** Maximum Sustainable Concurrency = **{max_c_03 if max_c_03 is not None else 'None'}**",
        "",
    ])

    md_table = "\n".join(md_lines)

    # Save to disk
    with open(os.path.join(output_dir, "summary.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines) + "\n")

    with open(os.path.join(output_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md_table + "\n")

    print("\n" + "=" * 80)
    print("PHASE 2 vLLM CONCURRENCY BENCHMARK SUMMARY")
    print("=" * 80)
    print(md_table)
    print("=" * 80 + "\n")


async def run_benchmark_suite(
    model_path: str = "Qwen/Qwen3-ASR-1.7B",
        test_file: str = "data/test.jsonl",
    audio_dir: str = "data/audio_clips/test",
    output_dir: str = "phase2/results",
    concurrencies: Optional[List[int]] = None,
    warmup: int = 3,
    gpu_mem: float = 0.7,
):
    concurrencies = concurrencies or [1, 2, 4, 8, 16, 32, 64]
    logger.info("Initializing Phase 2 benchmark sweep across concurrencies: %s", concurrencies)

    # 1. Load dataset (43 utterances)
    samples = load_benchmark_dataset(test_file=test_file, audio_dir=audio_dir)
    logger.info("Loaded %d benchmark utterances from %s.", len(samples), test_file)

    # 2. Initialize shared vLLM backend strictly once
    backend = get_vllm_backend(
        model_path=model_path,
                gpu_memory_utilization=gpu_mem,
        max_inference_batch_size=1,
            )

    gpu_monitor = GPUMonitor(device_index=0, interval=0.05)
    all_results: List[Dict[str, Any]] = []

    # 3. Sweep concurrency levels
    for c in concurrencies:
        logger.info("\n>>> Benchmarking Concurrency C = %d <<<", c)
        res = await run_concurrent_benchmark(
            backend=backend,
            samples=samples,
            concurrency=c,
            warmup_requests=warmup,
            gpu_monitor=gpu_monitor,
        )
        save_concurrency_result(res, output_dir=output_dir)
        all_results.append(res)
        await asyncio.sleep(1.0)

    # 4. Generate summary tables
    generate_summary_tables(all_results, output_dir=output_dir)


def main():
    parser = argparse.ArgumentParser(description="Phase 2: vLLM Concurrency Benchmark Suite")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B", help="Model path")
    parser.add_argument("--test_file", default="data/test.jsonl", help="Test dataset path")
    parser.add_argument("--audio_dir", default="data/audio_clips/test", help="Audio directory")
    parser.add_argument("--output_dir", default="phase2/results", help="Results output directory")
    parser.add_argument("--concurrencies", type=str, default="1,2,4,8,16,32,64",
                        help="Comma-separated concurrency levels")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup requests")
    parser.add_argument("--gpu_mem", type=float, default=0.7, help="vLLM GPU memory utilization")
    args = parser.parse_args()

    c_list = [int(c.strip()) for c in args.concurrencies.split(",") if c.strip()]
    asyncio.run(
        run_benchmark_suite(
            model_path=args.model,
                        test_file=args.test_file,
            audio_dir=args.audio_dir,
            output_dir=args.output_dir,
            concurrencies=c_list,
            warmup=args.warmup,
            gpu_mem=args.gpu_mem,
        )
    )


if __name__ == "__main__":
    main()
