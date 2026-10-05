"""
Model initialization module for Qwen3-ASR 1.7B.

Features:
- Loads model and processor once at startup
- Configurable device ('cuda', 'cpu') and precision ('float32', 'float16', 'bfloat16')
- Optional LoRA adapter support (e.g. runs/lora_r16/checkpoint-42)
- Sets model to eval mode and verifies inference readiness
- Wraps model & processor in a clean, reusable container
"""

import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger("qwen_asr.model")


def parse_torch_dtype(dtype_str: str) -> torch.dtype:
    """Map string precision names to torch.dtype."""
    d = dtype_str.lower().strip()
    if d in ("float32", "fp32", "32"):
        return torch.float32
    elif d in ("float16", "fp16", "16"):
        return torch.float16
    elif d in ("bfloat16", "bf16"):
        return torch.bfloat16
    else:
        logger.warning("Unrecognized dtype '%s'; defaulting to torch.float32", dtype_str)
        return torch.float32


class Qwen3ASRModelWrapper:
    """
    Reusable container for loaded Qwen3-ASR model and processor.
    Ensures model is loaded strictly once at server startup and kept in memory.
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-ASR-1.7B",
        adapter_path: Optional[str] = None,
        device: str = "cuda",
        dtype: str = "float32",
    ):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.requested_device = device
        self.requested_dtype_str = dtype
        self.torch_dtype = parse_torch_dtype(dtype)

        # Resolve device
        if self.requested_device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.requested_device)

        self.model: Optional[torch.nn.Module] = None
        self.processor: Optional[Any] = None
        self.load_time_s: float = 0.0
        self.model_metadata: Dict[str, Any] = {}

    def load(self) -> "Qwen3ASRModelWrapper":
        """Load model, processor, and optional LoRA adapter into memory."""
        if self.model is not None:
            logger.info("Model already loaded. Skipping reload.")
            return self

        logger.info(
            "Loading Qwen3-ASR from '%s' on %s with dtype %s...",
            self.model_path,
            self.device,
            self.torch_dtype,
        )
        t0 = time.perf_counter()

        # Try loading via qwen_asr package first
        loaded = False
        try:
            from qwen_asr import Qwen3ASRModel

            logger.info("Using qwen_asr.Qwen3ASRModel for loading...")
            asr = Qwen3ASRModel.from_pretrained(
                self.model_path,
                dtype=self.torch_dtype,
                device_map=None,
            )
            self.model = asr.model
            self.processor = asr.processor
            loaded = True
        except ImportError:
            logger.info("qwen_asr package not found; falling back to transformers...")
        except Exception as e:
            logger.warning("qwen_asr loading failed: %s; falling back to transformers...", e)

        if not loaded:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

        # Attach LoRA adapter if specified
        if self.adapter_path:
            resolved_adapter = self.adapter_path
            if not os.path.exists(resolved_adapter):
                if "checkpoint_" in resolved_adapter:
                    alt = resolved_adapter.replace("checkpoint_", "checkpoint-")
                elif "checkpoint-" in resolved_adapter:
                    alt = resolved_adapter.replace("checkpoint-", "checkpoint_")
                else:
                    alt = resolved_adapter
                if os.path.exists(alt):
                    resolved_adapter = alt

            logger.info("Loading LoRA adapter from '%s'...", resolved_adapter)
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, resolved_adapter)
            logger.info("LoRA adapter attached successfully.")

        # Move model to device
        self.model.to(self.device)
        self.model.eval()

        # Synchronize if on GPU
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        self.load_time_s = time.perf_counter() - t0
        actual_dtype = next(self.model.parameters()).dtype
        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        vram_mb = None
        if self.device.type == "cuda":
            vram_mb = round(torch.cuda.memory_allocated(self.device) / (1024**2), 2)

        self.model_metadata = {
            "model_path": self.model_path,
            "adapter_path": self.adapter_path,
            "device": str(self.device),
            "dtype": str(actual_dtype).replace("torch.", ""),
            "parameters_total": num_params,
            "parameters_trainable": num_trainable,
            "load_time_seconds": round(self.load_time_s, 2),
            "vram_allocated_mb": vram_mb,
        }

        logger.info(
            "Model loaded successfully in %.2fs (adapter: %s). Total params: %d. VRAM: %s MB",
            self.load_time_s,
            self.adapter_path or "None",
            num_params,
            vram_mb,
        )
        return self


def load_model(config_dict: Optional[Dict[str, Any]] = None) -> Qwen3ASRModelWrapper:
    """
    Factory function to initialize and load model from config dictionary.
    """
    cfg = config_dict or {}
    model_cfg = cfg.get("model", {})
    wrapper = Qwen3ASRModelWrapper(
        model_path=model_cfg.get("model_path", "Qwen/Qwen3-ASR-1.7B"),
        adapter_path=model_cfg.get("adapter_path"),
        device=model_cfg.get("device", "cuda"),
        dtype=model_cfg.get("dtype", "float32"),
    )
    return wrapper.load()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    wrapper = Qwen3ASRModelWrapper()
    print("Model wrapper initialized (run with .load() to populate).")
