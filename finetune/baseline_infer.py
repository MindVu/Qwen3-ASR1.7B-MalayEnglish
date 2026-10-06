"""
Part 2 - Baseline Inference Benchmark for Qwen3-ASR 1.7B

Extends the original infer.py:
  * records hardware / software environment
  * buckets test audio by duration (2-5s, 5-15s, 15-30s, ...)
  * warm-up, then times every request (batch size 1, sequential = true baseline)
  * RTF = inference processing time / audio duration
      - rtf_e2e   : processor (feature extraction) + H2D + generate + decode
      - rtf_model : model.generate() only
    (audio file loading/resampling is timed separately and NOT counted, since a
     production server receives decoded PCM; use --include_load to count it)
  * average / P50 / P95 RTF (overall and per duration bucket)
  * GPU utilisation + VRAM sampled in a background thread (pynvml, or nvidia-smi fallback)
  * throughput: audio-seconds per wall-second, requests/s, generated tokens/s
  * optional WER if the test jsonl has a reference field (default key: "text")

Usage:
  python benchmark_baseline.py --test_file data/test.jsonl --output_dir bench_baseline
  python benchmark_baseline.py --adapter_path outputs/lora --output_dir bench_lora
"""

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import threading
import time
from collections import defaultdict

import librosa
import numpy as np
import torch
from peft import PeftModel
from qwen_asr import Qwen3ASRModel


# ----------------------------------------------------------------------
# Environment info
# ----------------------------------------------------------------------
def _pkg_version(name):
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "not installed"


def get_cpu_info():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
            else:
                model = platform.processor()
    except Exception:
        model = platform.processor() or "unknown"
    return f"{model} ({os.cpu_count()} logical cores)"


def get_ram_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return round(int(line.split()[1]) / 1024 / 1024, 1)
    except Exception:
        pass
    try:
        import psutil

        return round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        return None


def get_environment(dtype, attn_impl):
    env = {
        "gpu_model": None,
        "gpu_memory_gb": None,
        "cpu": get_cpu_info(),
        "ram_gb": get_ram_gb(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "cuda_version (torch)": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "pytorch_version": torch.__version__,
        "transformers_version": _pkg_version("transformers"),
        "qwen_asr_version": _pkg_version("qwen-asr"),
        "peft_version": _pkg_version("peft"),
        "flash_attn_version": _pkg_version("flash-attn"),
        "inference_framework": "HuggingFace transformers (eager generate, batch=1) via qwen_asr",
        "attention_implementation": attn_impl,
        "precision": str(dtype).replace("torch.", ""),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        env["gpu_model"] = p.name
        env["gpu_memory_gb"] = round(p.total_memory / 1024**3, 1)
        env["gpu_compute_capability"] = f"{p.major}.{p.minor}"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        )
        env["nvidia_driver"] = out.strip().splitlines()[0]
    except Exception:
        env["nvidia_driver"] = None
    return env


# ----------------------------------------------------------------------
# GPU monitor (utilisation + memory) running in the background
# ----------------------------------------------------------------------
class GPUMonitor:
    def __init__(self, device_index=0, interval=0.05):
        self.idx = device_index
        self.interval = interval
        self.util, self.mem_mb = [], []
        self._stop = threading.Event()
        self._thread = None
        self._nvml = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception:
            self._nvml = None  # fall back to nvidia-smi

    def _sample(self):
        if self._nvml is not None:
            u = self._nvml.nvmlDeviceGetUtilizationRates(self._handle).gpu
            m = self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used / 1024**2
            return u, m
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={self.idx}",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        u, m = out.strip().split(",")
        return float(u), float(m)

    def _run(self):
        while not self._stop.is_set():
            try:
                u, m = self._sample()
                self.util.append(u)
                self.mem_mb.append(m)
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self.util.clear()
        self.mem_mb.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()
        return {
            "gpu_util_avg_pct": round(float(np.mean(self.util)), 1) if self.util else None,
            "gpu_util_max_pct": round(float(np.max(self.util)), 1) if self.util else None,
            "gpu_mem_used_peak_mb": round(float(np.max(self.mem_mb)), 1) if self.mem_mb else None,
            "gpu_mem_used_avg_mb": round(float(np.mean(self.mem_mb)), 1) if self.mem_mb else None,
            "n_samples": len(self.util),
        }


# ----------------------------------------------------------------------
# Text normalization & WER evaluation
# ----------------------------------------------------------------------
try:
    from prepare_data import normalize_text, ASR_TEXT_TAG
except ImportError:
    try:
        from finetune.prepare_data import normalize_text, ASR_TEXT_TAG
    except ImportError:
        ASR_TEXT_TAG = "<asr_text>"

        def normalize_text(text: str) -> str:
            if text is None:
                return ""
            text = text.lower()
            text = re.sub(r"[^a-z0-9'\s]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text


def strip_language_tag(text):
    """
    Removes the "language {Lang}<asr_text>" prefix the fine-tuned model is
    trained to emit, leaving just the transcript. Safe to call on text that
    never had the tag (e.g. scoring the base/non-fine-tuned model, or an
    already-clean reference) -- returned unchanged in that case rather than
    mangled.
    """
    if text is None:
        return ""
    text = str(text)
    if ASR_TEXT_TAG in text:
        return text.split(ASR_TEXT_TAG, 1)[1].strip()
    return text.strip()


clean_prediction = strip_language_tag


def extract_reference_text(s, ref_key="text"):
    """
    Extracts the reference transcript from a sample dictionary, prioritizing
    clean/untagged fields ('transcript', 'reference') over 'text' or custom ref_key,
    and strips any language tags.
    """
    if isinstance(s, str):
        return strip_language_tag(s)
    if isinstance(s, dict):
        for key in ["transcript", "reference", ref_key, "text", "normalized_text", "raw_text"]:
            if key in s and s[key] is not None:
                return strip_language_tag(s[key])
    return None


def compute_wer_builtin(predictions, references):
    """
    Standard dynamic programming Word Error Rate (WER) computation.
    """
    total_words = 0
    total_edits = 0

    for pred, ref in zip(predictions, references):
        ref_words = ref.strip().split()
        pred_words = pred.strip().split()

        r_len = len(ref_words)
        p_len = len(pred_words)
        total_words += r_len

        # DP table: dp[i][j] = min edits between ref_words[:i] and pred_words[:j]
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
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # Deletion
                        dp[i][j - 1],      # Insertion
                        dp[i - 1][j - 1],  # Substitution
                    )

        total_edits += dp[r_len][p_len]

    if total_words == 0:
        return 0.0
    return total_edits / total_words


def compute_wer(predictions, references, normalize=True):
    """
    Calculates WER using evaluate, jiwer, or builtin DP fallback.
    """
    if len(predictions) == 0:
        return 0.0

    if normalize:
        preds = [normalize_text(strip_language_tag(p)) for p in predictions]
        refs = [normalize_text(strip_language_tag(r)) for r in references]
    else:
        preds = [strip_language_tag(p) for p in predictions]
        refs = [strip_language_tag(r) for r in references]

    try:
        import evaluate
        wer_metric = evaluate.load("wer")
        wer = wer_metric.compute(predictions=preds, references=refs)
    except Exception:
        try:
            import jiwer
            wer = jiwer.wer(reference=refs, hypothesis=preds)
        except Exception:
            wer = compute_wer_builtin(predictions=preds, references=refs)
    return float(wer)


def corpus_wer(refs, hyps):
    """Backwards-compatible wrapper for compute_wer."""
    return compute_wer(predictions=hyps, references=refs, normalize=True)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
BUCKETS = [
    ("<2s", 0, 2),
    ("2-5s", 2, 5),
    ("5-15s", 5, 15),
    ("15-30s", 15, 30),
    (">30s", 30, float("inf")),
]


def bucket_of(dur):
    for name, lo, hi in BUCKETS:
        if lo <= dur < hi:
            return name
    return ">30s"


def load_audio(path, sr=16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def pct(values, q):
    return float(np.percentile(values, q)) if len(values) else None


def rtf_stats(rtfs):
    if not rtfs:
        return {}
    return {
        "n": len(rtfs),
        "avg": round(float(np.mean(rtfs)), 4),
        "p50": round(pct(rtfs, 50), 4),
        "p95": round(pct(rtfs, 95), 4),
        "min": round(float(np.min(rtfs)), 4),
        "max": round(float(np.max(rtfs)), 4),
    }


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def to_device(inputs, model_dtype):
    out = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            v = v.cuda(non_blocking=True)
            if v.is_floating_point():
                v = v.to(dtype=model_dtype)
        out[k] = v
    return out


# ----------------------------------------------------------------------
# Single-request inference (same logic as original infer.py, but timed)
# ----------------------------------------------------------------------
def transcribe(model, processor, audio, prompt, max_new_tokens, model_dtype):
    """Returns (prediction, n_new_tokens, timings dict). Timings are GPU-synchronised."""
    sync()
    t0 = time.perf_counter()

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": [{"type": "audio", "audio": audio}]},
    ]
    text = processor.apply_chat_template(
        [messages], add_generation_prompt=True, tokenize=False
    )[0]
    inputs = processor(text=[text], audio=[audio], return_tensors="pt", padding=True)
    inputs = to_device(inputs, model_dtype)
    sync()
    t1 = time.perf_counter()

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    sync()
    t2 = time.perf_counter()

    output_ids = out.sequences if hasattr(out, "sequences") else out
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    pad_id = processor.tokenizer.pad_token_id
    n_tokens = int((generated_ids != pad_id).sum().item()) if pad_id is not None else int(generated_ids.numel())
    prediction = processor.tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0].strip()
    t3 = time.perf_counter()

    return prediction, n_tokens, {
        "t_preprocess": t1 - t0,
        "t_generate": t2 - t1,
        "t_decode": t3 - t2,
        "t_total": t3 - t0,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="Qwen/Qwen3-ASR-1.7B")
    ap.add_argument("--adapter_path", default=None, help="LoRA adapter (omit for base model)")
    ap.add_argument("--test_file", default="data/test.jsonl")
    ap.add_argument("--output_dir", default="bench/lora16")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=3, help="warm-up requests (not measured)")
    ap.add_argument("--repeats", type=int, default=1, help="passes over the sample set")
    ap.add_argument("--max_per_bucket", type=int, default=0,
                    help="cap samples per duration bucket (0 = use all)")
    ap.add_argument("--ref_key", default="text", help="reference transcript key for WER (prioritizes transcript/reference/text, strips language tags)")
    ap.add_argument("--include_load", action="store_true",
                    help="include audio file load/resample time in RTF")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------- model ----------------
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    print(f"Loading model: {args.model_path} ({dtype})")
    t_load0 = time.perf_counter()
    asr = Qwen3ASRModel.from_pretrained(args.model_path, dtype=dtype, device_map=None)
    model, processor = asr.model, asr.processor

    if args.adapter_path:
        print(f"Loading LoRA adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        sync()
    model_load_s = time.perf_counter() - t_load0
    model_dtype = next(model.parameters()).dtype

    attn_impl = getattr(getattr(model, "config", None), "_attn_implementation", None)
    if attn_impl is None:
        attn_impl = getattr(getattr(model, "base_model", None), "config", None)
        attn_impl = getattr(attn_impl, "_attn_implementation", "unknown")

    env = get_environment(model_dtype, attn_impl)
    env["model"] = args.model_path
    env["adapter"] = args.adapter_path
    env["model_load_time_s"] = round(model_load_s, 2)
    env["model_weights_vram_mb"] = (
        round(torch.cuda.memory_allocated() / 1024**2, 1) if torch.cuda.is_available() else None
    )
    print("\n=== Environment ===")
    for k, v in env.items():
        print(f"  {k}: {v}")

    # ---------------- data ----------------
    samples = []
    with open(args.test_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    print(f"\nLoaded {len(samples)} test samples. Reading audio...")

    items = []
    for s in samples:
        t = time.perf_counter()
        wav = load_audio(s["audio"], sr=args.sr)
        items.append({
            "sample": s,
            "audio": wav,
            "duration": len(wav) / args.sr,
            "load_time": time.perf_counter() - t,
        })
    for it in items:
        it["bucket"] = bucket_of(it["duration"])

    if args.max_per_bucket > 0:
        rng = np.random.RandomState(args.seed)
        per = defaultdict(list)
        for it in items:
            per[it["bucket"]].append(it)
        items = []
        for b in per.values():
            idx = rng.permutation(len(b))[: args.max_per_bucket]
            items.extend(b[i] for i in sorted(idx))

    dist = defaultdict(lambda: [0, 0.0])
    for it in items:
        dist[it["bucket"]][0] += 1
        dist[it["bucket"]][1] += it["duration"]
    print("Benchmark set:")
    for name, _, _ in BUCKETS:
        if name in dist:
            n, d = dist[name]
            print(f"  {name:>7}: {n:4d} clips, {d:8.1f}s audio")

    # ---------------- warm-up ----------------
    # First CUDA calls pay for kernel loading / cuDNN autotune / allocator growth.
    # Warm up across duration buckets so those costs are not in the measurement.
    print(f"\nWarm-up ({args.warmup} requests)...")
    warm_items = sorted(items, key=lambda x: x["duration"])
    step = max(1, len(warm_items) // max(args.warmup, 1))
    for it in warm_items[::step][: args.warmup]:
        transcribe(model, processor, it["audio"], it["sample"].get("prompt", ""),
                   args.max_new_tokens, model_dtype)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ---------------- measured run ----------------
    monitor = GPUMonitor() if torch.cuda.is_available() else None
    if monitor:
        monitor.start()

    records = []
    run_start = time.perf_counter()
    total = len(items) * args.repeats
    n_done = 0
    for rep in range(args.repeats):
        for it in items:
            s = it["sample"]
            pred, n_tok, tm = transcribe(
                model, processor, it["audio"], s.get("prompt", ""),
                args.max_new_tokens, model_dtype,
            )
            proc_time = tm["t_total"] + (it["load_time"] if args.include_load else 0.0)
            rec = {
                "id": s.get("id"),
                "audio": s["audio"],
                "bucket": it["bucket"],
                "repeat": rep,
                "audio_duration_s": round(it["duration"], 3),
                "t_load_s": round(it["load_time"], 4),
                "t_preprocess_s": round(tm["t_preprocess"], 4),
                "t_generate_s": round(tm["t_generate"], 4),
                "t_decode_s": round(tm["t_decode"], 4),
                "t_total_s": round(proc_time, 4),
                "rtf_e2e": proc_time / it["duration"],
                "rtf_model": tm["t_generate"] / it["duration"],
                "new_tokens": n_tok,
                "prediction": pred,
            }
            ref_val = extract_reference_text(s, args.ref_key)
            if ref_val is not None:
                rec["reference"] = ref_val
                rec["transcript"] = ref_val
            records.append(rec)
            n_done += 1
            print(f"[{n_done}/{total}] {it['bucket']:>6} dur={it['duration']:5.1f}s "
                  f"t={proc_time:5.2f}s RTF={rec['rtf_e2e']:.3f} tok={n_tok}")
    wall = time.perf_counter() - run_start
    gpu = monitor.stop() if monitor else {}

    # ---------------- aggregate ----------------
    total_audio = sum(r["audio_duration_s"] for r in records)
    total_proc = sum(r["t_total_s"] for r in records)
    total_tokens = sum(r["new_tokens"] for r in records)

    summary = {
        "environment": env,
        "config": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "batch_size": 1,
            "concurrency": 1,
            "max_new_tokens": args.max_new_tokens,
            "include_load_in_rtf": args.include_load,
            "n_requests": len(records),
        },
        "rtf_e2e": rtf_stats([r["rtf_e2e"] for r in records]),
        "rtf_model_only": rtf_stats([r["rtf_model"] for r in records]),
        "rtf_e2e_by_bucket": {
            name: rtf_stats([r["rtf_e2e"] for r in records if r["bucket"] == name])
            for name, _, _ in BUCKETS
            if any(r["bucket"] == name for r in records)
        },
        "latency_breakdown_avg_s": {
            k: round(statistics.mean(r[k] for r in records), 4)
            for k in ["t_load_s", "t_preprocess_s", "t_generate_s", "t_decode_s"]
        },
        "latency_breakdown_pct_of_total": {
            "preprocess": round(100 * sum(r["t_preprocess_s"] for r in records) / total_proc, 1),
            "generate": round(100 * sum(r["t_generate_s"] for r in records) / total_proc, 1),
            "decode": round(100 * sum(r["t_decode_s"] for r in records) / total_proc, 1),
        },
        "gpu": {
            **gpu,
            "torch_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1)
            if torch.cuda.is_available() else None,
            "torch_peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 1)
            if torch.cuda.is_available() else None,
        },
        "throughput": {
            "wall_time_s": round(wall, 2),
            "audio_seconds_total": round(total_audio, 1),
            # audio-seconds transcribed per wall-second (= 1 / aggregate RTF); >1 means faster than real time
            "audio_s_per_wall_s": round(total_audio / wall, 3),
            "requests_per_s": round(len(records) / wall, 3),
            "generated_tokens_per_s": round(total_tokens / total_proc, 2),
            "realtime_streams_equivalent_at_rtf1": round(total_audio / wall, 2),
        },
    }

    # WER (if references exist)
    with_ref = [r for r in records if ("reference" in r or "transcript" in r) and r["repeat"] == 0]
    if with_ref:
        refs = [strip_language_tag(r.get("transcript") or r.get("reference")) for r in with_ref]
        hyps = [strip_language_tag(r["prediction"]) for r in with_ref]
        summary["wer"] = {
            "overall": round(compute_wer(hyps, refs), 4),
            "n_utterances": len(with_ref),
            "by_bucket": {
                name: round(compute_wer(
                    [strip_language_tag(r["prediction"]) for r in with_ref if r["bucket"] == name],
                    [strip_language_tag(r.get("transcript") or r.get("reference")) for r in with_ref if r["bucket"] == name],
                ), 4)
                for name, _, _ in BUCKETS
                if any(r["bucket"] == name for r in with_ref)
            },
        }

    # ---------------- save ----------------
    with open(os.path.join(args.output_dir, "per_sample.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # markdown table for the report
    md = ["| Bucket | N | Avg RTF | P50 RTF | P95 RTF |", "|---|---|---|---|---|"]
    for name, st in summary["rtf_e2e_by_bucket"].items():
        md.append(f"| {name} | {st['n']} | {st['avg']} | {st['p50']} | {st['p95']} |")
    st = summary["rtf_e2e"]
    md.append(f"| **All** | {st['n']} | {st['avg']} | {st['p50']} | {st['p95']} |")
    with open(os.path.join(args.output_dir, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    # ---------------- print ----------------
    print("\n" + "=" * 60)
    print("BASELINE RESULTS (batch=1, sequential)")
    print("=" * 60)
    print("\n".join(md))
    print(f"\nRTF (model.generate only): avg={summary['rtf_model_only']['avg']} "
          f"p50={summary['rtf_model_only']['p50']} p95={summary['rtf_model_only']['p95']}")
    print(f"Latency split: {summary['latency_breakdown_pct_of_total']}")
    print(f"GPU: {summary['gpu']}")
    print(f"Throughput: {summary['throughput']}")
    if "wer" in summary:
        print(f"WER: {summary['wer']}")
    print(f"\nSaved to: {args.output_dir}/ (summary.json, summary.md, per_sample.jsonl)")


if __name__ == "__main__":
    main()