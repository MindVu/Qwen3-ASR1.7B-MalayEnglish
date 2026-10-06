"""
Phase 2: vLLM Inference Backend using official Qwen-ASR vLLM wrapper (Qwen3ASRModel.LLM).

Directly initializes Qwen3ASRModel.LLM via vLLM with zero HuggingFace model loading.
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Set multiprocessing method for vLLM worker processes before importing torch/vllm
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import strip_language_tag

logger = logging.getLogger("phase2.vllm_backend")


class VLLMInferenceBackend:
    """
    Singleton-style inference wrapper around official Qwen3ASRModel.LLM.
    Directly loads model once via vLLM. Zero HuggingFace model loading.
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-ASR-1.7B",
        gpu_memory_utilization: float = 0.7,
        max_inference_batch_size: int = 1,
        max_new_tokens: int = 512,
        forced_aligner: Optional[str] = None,
        **extra_vllm_kwargs: Any,
    ):
        self.model_path = model_path
        self.max_inference_batch_size = max_inference_batch_size
        self.max_new_tokens = max_new_tokens
        self.gpu_memory_utilization = gpu_memory_utilization

        logger.info(
            "Initializing Qwen3ASRModel.LLM backend (model=%s, batch_size=%d, max_tokens=%d, gpu_mem=%.2f)...",
            self.model_path,
            self.max_inference_batch_size,
            self.max_new_tokens,
            self.gpu_memory_utilization,
        )
        t0 = time.perf_counter()

        from qwen_asr import Qwen3ASRModel

        # Initialize official Qwen-ASR vLLM backend directly
        self.model = Qwen3ASRModel.LLM(
            model=self.model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_inference_batch_size=self.max_inference_batch_size,
            max_new_tokens=self.max_new_tokens,
            forced_aligner=forced_aligner,
            **extra_vllm_kwargs,
        )
        self.load_time_s = time.perf_counter() - t0
        logger.info("vLLM backend successfully initialized in %.2f seconds.", self.load_time_s)

    def transcribe(
        self,
        audio: Any,
        context: str = "",
        language: Optional[str] = None,
    ) -> str:
        """
        Simple, clean interface: transcribe(audio) -> text
        Safely strips any language tag prefix so callers receive clean transcripts.
        """
        results = self.model.transcribe(
            audio=audio,
            context=context,
            language=language,
            return_time_stamps=False,
        )
        if not results:
            return ""
        return strip_language_tag(results[0].text)

    def transcribe_detailed(
        self,
        audio: Any,
        context: str = "",
        language: Optional[str] = None,
        return_time_stamps: bool = False,
    ) -> Dict[str, Any]:
        """
        Detailed interface recording execution latency for benchmarking.
        """
        t0 = time.perf_counter()
        results = self.model.transcribe(
            audio=audio,
            context=context,
            language=language,
            return_time_stamps=return_time_stamps,
        )
        latency_s = time.perf_counter() - t0

        if not results:
            return {
                "text": "",
                "raw_text": "",
                "language": "",
                "latency_s": latency_s,
                "time_stamps": None,
            }

        res = results[0]
        raw_text = res.text
        clean_text = strip_language_tag(raw_text)

        return {
            "text": clean_text,
            "raw_text": raw_text,
            "language": getattr(res, "language", ""),
            "latency_s": latency_s,
            "time_stamps": getattr(res, "time_stamps", None),
        }


# Global backend instance for single-model sharing across requests
_GLOBAL_BACKEND: Optional[VLLMInferenceBackend] = None


def get_vllm_backend(
    model_path: str = "Qwen/Qwen3-ASR-1.7B",
    gpu_memory_utilization: float = 0.7,
    max_inference_batch_size: int = 1,
    max_new_tokens: int = 512,
    **kwargs: Any,
) -> VLLMInferenceBackend:
    """Returns or creates the shared global VLLMInferenceBackend."""
    global _GLOBAL_BACKEND
    if _GLOBAL_BACKEND is None:
        _GLOBAL_BACKEND = VLLMInferenceBackend(
            model_path=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
    return _GLOBAL_BACKEND


def transcribe(audio: Any) -> str:
    """
    Module-level transcribe function. Reuses the initialized model backend.
    Never reloads the model inside transcribe().
    """
    backend = get_vllm_backend()
    return backend.transcribe(audio)
