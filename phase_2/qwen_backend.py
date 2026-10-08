"""
Phase 2 backend: the OFFICIAL qwen-asr vLLM wrapper, used as shipped.

    Qwen3ASRModel.LLM(...)  ->  vllm.LLM  (offline, blocking engine)

How the wrapper behaves (from qwen_asr/inference/qwen3_asr.py):
  * transcribe(audio=[...], context=[...], language=...) takes a LIST of audios.
  * For each audio it builds the chat prompt, then _infer_asr_vllm() calls
    vllm.LLM.generate(...) on the list. vLLM schedules all items together
    (continuous batching + paged attention inside that single call).
  * generate() returns only when EVERY item of the call is finished, so all
    requests of a call share one completion time.
  * `max_inference_batch_size` splits the list into sequential generate() calls.
    Use -1 (no splitting) so a batch really reaches the engine as one batch.
  * The engine is blocking and single-caller: never call it from several threads.
"""

import gc
import logging
import os
import time
from typing import Any, List, Optional

# Same worker start method as Phase 2 (must be set before importing torch/vllm).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

logger = logging.getLogger("phase_2.qwen_backend")


class QwenVLLMBackend:
    """Thin holder around Qwen3ASRModel.LLM (no changes to its behavior)."""

    def __init__(
        self,
        model_path: str,
        gpu_memory_utilization: float = 0.9,
        max_num_seqs: int = 128,
        max_new_tokens: int = 512,
        max_inference_batch_size: int = -1,
        **engine_kwargs: Any,
    ):
        self.model_path = model_path
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_seqs = max_num_seqs
        self.max_new_tokens = max_new_tokens
        self.max_inference_batch_size = max_inference_batch_size
        self.engine_kwargs = dict(engine_kwargs)

        logger.info(
            "Initializing Qwen3ASRModel.LLM (model=%s, gpu_mem=%.2f, max_num_seqs=%d, "
            "max_new_tokens=%d, max_inference_batch_size=%d, extra=%s)...",
            model_path, gpu_memory_utilization, max_num_seqs, max_new_tokens,
            max_inference_batch_size, self.engine_kwargs,
        )
        t0 = time.perf_counter()

        # Importing qwen_asr also registers the custom model class with vLLM.
        from qwen_asr import Qwen3ASRModel

        # Extra kwargs (gpu_memory_utilization, max_num_seqs, dtype, ...) are
        # forwarded by the wrapper to vllm.LLM(...).
        self.model = Qwen3ASRModel.LLM(
            model=model_path,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            **self.engine_kwargs,
        )
        self.load_time_s = time.perf_counter() - t0
        logger.info("Qwen3ASRModel.LLM ready in %.2fs", self.load_time_s)

    def transcribe_batch(
        self,
        audios: List[Any],
        contexts: Optional[List[str]] = None,
        language: Optional[str] = None,
    ) -> List[Any]:
        """
        One wrapper call for a whole batch. Returns list[ASRTranscription]
        (fields: .language, .text) in the same order as `audios`.

        Everything the wrapper does is inside this call and therefore inside the
        timing the load tester measures: audio normalization, prompt building,
        vllm.LLM.generate, and output parsing (parse_asr_output).
        """
        if contexts is None:
            contexts = [""] * len(audios)
        return self.model.transcribe(
            audio=audios,
            context=contexts,
            language=language,
            return_time_stamps=False,
        )

    def shutdown(self) -> None:
        """Best-effort release of the engine (it runs a separate engine-core process)."""
        try:
            llm = getattr(self.model, "model", None)  # the underlying vllm.LLM
            engine = getattr(llm, "llm_engine", None)
            core = getattr(engine, "engine_core", None)
            if core is not None and hasattr(core, "shutdown"):
                core.shutdown()
        except Exception:
            logger.exception("Engine shutdown failed")
        self.model = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass