"""
phase 3 Common Utilities: text normalization, WER/CER computation, dataset loading, and metrics.
"""

import os
import platform
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ASR_TEXT_TAG = "<asr_text>"

BUCKETS: List[Tuple[str, float, float]] = [
    ("<2s", 0.0, 2.0),
    ("2-5s", 2.0, 5.0),
    ("5-15s", 5.0, 15.0),
    ("15-30s", 15.0, 30.0),
    (">30s", 30.0, float("inf")),
]


def bucket_of_duration(dur: float) -> str:
    """Classify duration in seconds into standard duration buckets."""
    for name, lo, hi in BUCKETS:
        if lo <= dur < hi:
            return name
    return ">30s"


def normalize_text(text: str) -> str:
    """
    Standard text normalization: lowercasing, punctuation removal, whitespace collapse.
    Preserves apostrophes inside words.
    """
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_language_tag(text: Any) -> str:
    """
    Removes the "language {Lang}<asr_text>" prefix emitted by fine-tuned model.
    Leaves clean text unchanged.
    """
    if text is None:
        return ""
    text = str(text)
    if ASR_TEXT_TAG in text:
        return text.split(ASR_TEXT_TAG, 1)[1].strip()
    return text.strip()


def extract_reference_transcript(item: Dict[str, Any], ref_key: str = "transcript") -> str:
    """
    Extracts the clean reference transcript from a dataset item, checking
    'transcript', 'reference', ref_key, and 'text', and strips any language tags.
    """
    for key in [ref_key, "transcript", "reference", "text", "normalized_text", "raw_text"]:
        if key in item and item[key] is not None:
            val = str(item[key]).strip()
            if val:
                return strip_language_tag(val)
    return ""


def compute_wer_builtin(predictions: List[str], references: List[str]) -> float:
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


def compute_wer(predictions: List[str], references: List[str], normalize: bool = True) -> float:
    """
    Calculates Word Error Rate using evaluate, jiwer, or builtin DP fallback.
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
        return float(wer)
    except Exception:
        try:
            import jiwer
            wer = jiwer.wer(reference=refs, hypothesis=preds)
            return float(wer)
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
            cur[j] = min(
                prev[j] + 1,                 # deletion
                cur[j - 1] + 1,              # insertion
                prev[j - 1] + (ca != cb),    # substitution
            )
        prev = cur
    return prev[-1]


def compute_cer_builtin(predictions: List[str], references: List[str]) -> float:
    """
    Character Error Rate: total character-level edits / total reference characters.
    Spaces count as characters (same convention as jiwer / evaluate "cer").
    """
    total_chars = 0
    total_edits = 0
    for pred, ref in zip(predictions, references):
        total_chars += len(ref)
        total_edits += _edit_distance(ref, pred)

    if total_chars == 0:
        return 0.0
    return total_edits / total_chars


def compute_cer(predictions: List[str], references: List[str], normalize: bool = True) -> float:
    """
    Calculates corpus-level Character Error Rate with the same normalization as WER.
    Chain: evaluate -> jiwer -> builtin DP. Spaces count as characters.
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
        cer_metric = evaluate.load("cer")
        return float(cer_metric.compute(predictions=preds, references=refs))
    except Exception:
        try:
            import jiwer
            return float(jiwer.cer(reference=refs, hypothesis=preds))
        except Exception:
            return float(compute_cer_builtin(predictions=preds, references=refs))


def compute_percentiles(values: List[float]) -> Dict[str, float]:
    """Calculate summary statistics and percentiles for a series of values."""
    if not values:
        return {"avg": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}

    arr = np.array(values, dtype=np.float64)
    return {
        "avg": round(float(np.mean(arr)), 4),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
    }


def load_benchmark_dataset(
    test_file: str = "data/test.jsonl",
    audio_dir: Optional[str] = "data/audio_clips/test",
) -> List[Dict[str, Any]]:
    """
    Loads benchmark dataset (43 utterances) and resolves paths and references.
    """
    import json
    import soundfile as sf

    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test dataset file not found: {test_file}")

    items: List[Dict[str, Any]] = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            audio_path = item.get("audio", "")

            # Resolve relative audio path if needed
            if not os.path.exists(audio_path) and audio_dir:
                candidate = os.path.join(audio_dir, os.path.basename(audio_path))
                if os.path.exists(candidate):
                    audio_path = candidate

            # Duration resolution
            dur = item.get("duration")
            if dur is None and os.path.exists(audio_path):
                try:
                    info = sf.info(audio_path)
                    dur = float(info.duration)
                except Exception:
                    dur = 0.0
            dur = float(dur or 0.0)

            ref = extract_reference_transcript(item)
            bucket = bucket_of_duration(dur)

            items.append({
                "id": str(item.get("id", os.path.basename(audio_path))),
                "audio": audio_path,
                "duration": dur,
                "prompt": item.get("prompt", "Transcribe the audio accurately."),
                "reference": ref,
                "bucket": bucket,
                "classification": item.get("classification", ""),
            })

    return items


def get_environment_info() -> Dict[str, Any]:
    """Capture environment and hardware details."""
    import torch

    env: Dict[str, Any] = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["gpu_count"] = torch.cuda.device_count()
        env["cuda_version"] = torch.version.cuda

    try:
        import vllm
        env["vllm_version"] = vllm.__version__
    except Exception:
        env["vllm_version"] = "unknown"

    try:
        import transformers
        env["transformers_version"] = transformers.__version__
    except Exception:
        pass

    return env