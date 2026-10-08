"""
Phase 2: vLLM Inference Backends for Qwen3-ASR.

Two backends:

1. VLLMInferenceBackend (sync, offline vllm.LLM via Qwen3ASRModel.LLM)
   - Single caller only. NOT thread-safe. Fine for C=1 / simple scripts.

2. AsyncVLLMBackend (AsyncLLM engine)
   - Many coroutines submit requests concurrently to ONE engine.
   - vLLM's scheduler does continuous batching + paged attention.
   - Each request returns as soon as it finishes (no waiting on the batch).
   - Must be created inside a running asyncio event loop.
"""

import asyncio
import itertools
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

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


# =====================================================================
# Sync backend (unchanged behavior)
# =====================================================================


class VLLMInferenceBackend:
    """
    Singleton-style inference wrapper around official Qwen3ASRModel.LLM.
    Directly loads model once via vLLM. Zero HuggingFace model loading.

    WARNING: the offline vllm.LLM is a blocking, single-caller API.
    Do not call it from multiple threads at once.
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
    """Module-level transcribe function. Reuses the initialized model backend."""
    backend = get_vllm_backend()
    return backend.transcribe(audio)


# =====================================================================
# Async backend: continuous batching + paged attention
# =====================================================================


class AsyncVLLMBackend:
    """
    Continuous-batching backend: many coroutines submit requests to one
    AsyncLLM engine. No threads, no locks.
    """

    def __init__(
        self,
        engine: Any,
        processor: Any,
        model_path: str,
        gpu_memory_utilization: float,
        max_num_seqs: int,
        max_new_tokens: int,
        load_time_s: float,
    ):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        self.engine = engine
        self.processor = processor
        self.model_path = model_path
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_seqs = max_num_seqs
        # kept so the summary config code in load_test.py works unchanged
        self.max_inference_batch_size = max_num_seqs
        self.max_new_tokens = max_new_tokens
        self.load_time_s = load_time_s
        self.sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, output_kind=RequestOutputKind.FINAL_ONLY)
        self._ids = itertools.count()

    @classmethod
    async def create(
        cls,
        model_path: str = "Qwen/Qwen3-ASR-1.7B",
        gpu_memory_utilization: float = 0.9,
        max_num_seqs: int = 128,
        max_new_tokens: int = 512,
        **engine_kwargs: Any,
    ) -> "AsyncVLLMBackend":
        """Must be awaited inside a running event loop."""
        t0 = time.perf_counter()

        # Importing qwen_asr registers the custom model class with vLLM.
        from qwen_asr import Qwen3ASRModel  # noqa: F401
        from qwen_asr.core.transformers_backend import Qwen3ASRProcessor
        from vllm import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        logger.info(
            "Initializing AsyncLLM engine (model=%s, gpu_mem=%.2f, max_num_seqs=%d)...",
            model_path,
            gpu_memory_utilization,
            max_num_seqs,
        )

        batched_tokens = max(10000, max_num_seqs)
        if "max_num_batched_tokens" in engine_kwargs:
            batched_tokens = max(batched_tokens, engine_kwargs.pop("max_num_batched_tokens"))

        args = AsyncEngineArgs(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,  # must be >= highest benchmark concurrency
            max_num_batched_tokens=batched_tokens,
            # enable_chunked_prefill=True,
            # enable_prefix_caching=True,
            **engine_kwargs,
        )
        engine = AsyncLLM.from_engine_args(args)
        processor = Qwen3ASRProcessor.from_pretrained(model_path, fix_mistral_regex=True)

        load_time_s = time.perf_counter() - t0
        logger.info("Async vLLM engine initialized in %.2fs", load_time_s)

        return cls(
            engine=engine,
            processor=processor,
            model_path=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            max_new_tokens=max_new_tokens,
            load_time_s=load_time_s,
        )

    def _build_prompt(self, context: str, language: Optional[str]) -> str:
        msgs = [
            {"role": "system", "content": context or ""},
            {"role": "user", "content": [{"type": "audio", "audio": ""}]},
        ]
        prompt = self.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )
        if language:
            prompt += f"language {language}<asr_text>"
        return prompt

    async def _infer_chunk(self, wav: Any, prompt: str) -> str:
        final = None
        async for out in self.engine.generate(
            {"prompt": prompt, "multi_modal_data": {"audio": [wav]}},
            self.sp,
            request_id=f"req-{next(self._ids)}",
        ):
            final = out
        return final.outputs[0].text

    async def transcribe_detailed(
        self,
        audio: Any,
        context: str = "",
        language: Optional[str] = None,
        return_time_stamps: bool = False,
    ) -> Dict[str, Any]:
        from qwen_asr.inference.utils import (
            MAX_ASR_INPUT_SECONDS,
            SAMPLE_RATE,
            merge_languages,
            normalize_audios,
            normalize_language_name,
            parse_asr_output,
            split_audio_into_chunks,
            validate_language,
        )

        if return_time_stamps:
            raise NotImplementedError("Timestamps are not supported in the async backend.")

        t0 = time.perf_counter()

        forced_lang = None
        if language and str(language).strip():
            forced_lang = normalize_language_name(str(language))
            validate_language(forced_lang)

        wav = normalize_audios(audio)[0]
        parts = split_audio_into_chunks(
            wav=wav,
            sr=SAMPLE_RATE,
            max_chunk_sec=MAX_ASR_INPUT_SECONDS,
        )
        prompt = self._build_prompt(context, forced_lang)

        raws = await asyncio.gather(*[self._infer_chunk(cw, prompt) for cw, _ in parts])

        langs, texts = [], []
        for raw in raws:
            lang, txt = parse_asr_output(raw, user_language=forced_lang)
            langs.append(lang)
            texts.append(txt)

        return {
            "text": strip_language_tag("".join(texts)),
            "raw_text": "".join(raws),
            "language": merge_languages(langs),
            "latency_s": time.perf_counter() - t0,
            # "time_stamps": None,
        }

    def shutdown(self) -> None:
        try:
            self.engine.shutdown()
        except Exception:
            logger.exception("Engine shutdown failed")