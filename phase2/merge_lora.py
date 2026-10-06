"""
Offline utility to merge a LoRA adapter into the base Qwen3-ASR model
using the official Qwen3ASRModel wrapper (matching Phase 1).

Produces a standalone checkpoint that vLLM can load directly.

Usage:
    python phase2/merge_lora.py \
        --base_model Qwen/Qwen3-ASR-1.7B \
        --adapter runs/lora_r16/checkpoint-42 \
        --output_dir models/lora_r16_merged
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import torch

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("merge_lora")


def _clean_generation_config(gc):
    """Make a GenerationConfig valid for greedy decoding (what ASR needs)."""
    if gc is None:
        return
    # Remove any monkeypatched callables (they break JSON serialization).
    for k in [k for k, v in list(gc.__dict__.items()) if callable(v)]:
        gc.__dict__.pop(k, None)
    gc.do_sample = False
    gc.temperature = None
    gc.top_p = None
    gc.top_k = None


def sanitize_all_generation_configs(model):
    """Clean the generation_config on the model and on every submodule that has one."""
    _clean_generation_config(getattr(model, "generation_config", None))
    for _, module in model.named_modules():
        _clean_generation_config(getattr(module, "generation_config", None))


def save_with_clean_generation_config(model, output_dir):
    """save_pretrained, falling back to a fresh GenerationConfig if validation still fails."""
    try:
        model.save_pretrained(output_dir)
    except (ValueError, TypeError) as e:
        logger.warning("save_pretrained failed (%s). Retrying with a fresh GenerationConfig.", e)
        from transformers import GenerationConfig

        old = getattr(model, "generation_config", None)
        model.generation_config = GenerationConfig(
            do_sample=False,
            bos_token_id=getattr(old, "bos_token_id", None),
            eos_token_id=getattr(old, "eos_token_id", None),
            pad_token_id=getattr(old, "pad_token_id", None),
        )
        model.save_pretrained(output_dir)


def merge_lora_with_qwen_wrapper(
    base_model_path: str = "Qwen/Qwen3-ASR-1.7B",
    adapter_path: str = "runs/lora_r16/checkpoint-42",
    output_dir: str = "models/lora_r16_merged",
    dtype: str = "bfloat16",
    device: str = "cpu",
):
    """
    Loads base model and processor via official qwen_asr.Qwen3ASRModel wrapper
    (matching Phase 1), attaches the LoRA adapter with PEFT, and exports
    the standalone merged model.
    """
    torch_dtype = torch.bfloat16 if dtype in ["bfloat16", "bf16"] else torch.float16

    # 1. Resolve adapter path
    resolved_adapter = adapter_path
    if not os.path.exists(resolved_adapter):
        candidate = os.path.join(str(REPO_ROOT), adapter_path)
        if os.path.exists(candidate):
            resolved_adapter = candidate
        elif "checkpoint_" in resolved_adapter:
            alt = resolved_adapter.replace("checkpoint_", "checkpoint-")
            if os.path.exists(alt):
                resolved_adapter = alt

    if not os.path.exists(resolved_adapter):
        raise FileNotFoundError(f"Adapter checkpoint not found: {adapter_path}")

    logger.info("Loading Qwen3-ASR base model via qwen_asr.Qwen3ASRModel wrapper from '%s'...", base_model_path)
    from qwen_asr import Qwen3ASRModel
    from peft import PeftModel

    # Load base model using official qwen-asr wrapper (exactly like Phase 1)
    asr = Qwen3ASRModel.from_pretrained(
        base_model_path,
        dtype=torch_dtype,
        device_map=device if device != "cpu" else None,
    )
    base_model = asr.model
    processor = asr.processor

    # 2. Attach LoRA adapter (matching Phase 1)
    logger.info("Attaching LoRA adapter from '%s'...", resolved_adapter)
    lora_model = PeftModel.from_pretrained(base_model, resolved_adapter)

    # 3. Merge weights into standalone model
    logger.info("Merging LoRA weights into base model (merge_and_unload)...")
    merged_model = lora_model.merge_and_unload()

    # 4. Save to target directory for vLLM
    logger.info("Saving standalone merged checkpoint to '%s'...", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Transformers validates generation_config strictly in save_pretrained.
    # Qwen's default has do_sample=False with temperature=1e-6, which fails.
    # Make it valid for greedy decoding on the model and all submodules.
    sanitize_all_generation_configs(merged_model)

    save_with_clean_generation_config(merged_model, output_dir)
    processor.save_pretrained(output_dir)

    # 5. Ensure all necessary ancillary configs are present (chat_template, etc.)
    extra_files = [
        "chat_template.json",
        "preprocessor_config.json",
        "generation_config.json",
        "merges.txt",
        "vocab.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    for fn in extra_files:
        dst = os.path.join(output_dir, fn)
        if not os.path.exists(dst):
            src = os.path.join(resolved_adapter, fn)
            if os.path.exists(src):
                shutil.copy2(src, dst)
            elif os.path.isdir(base_model_path):
                src_base = os.path.join(base_model_path, fn)
                if os.path.exists(src_base):
                    shutil.copy2(src_base, dst)

    # 6. Ensure generation_config.json on disk has no conflicting temperature
    gen_cfg_path = os.path.join(output_dir, "generation_config.json")
    if os.path.exists(gen_cfg_path):
        try:
            with open(gen_cfg_path, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
            if not cfg_data.get("do_sample", False) and "temperature" in cfg_data:
                del cfg_data["temperature"]
                with open(gen_cfg_path, "w", encoding="utf-8") as f:
                    json.dump(cfg_data, f, indent=2)
        except Exception as e:
            logger.warning("Could not sanitize generation_config.json: %s", e)

    logger.info("LoRA merge complete! Standalone checkpoint ready for vLLM at '%s'.", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into Qwen3-ASR model using official qwen_asr wrapper")
    parser.add_argument("--base_model", default="Qwen/Qwen3-ASR-1.7B", help="Base model name or path")
    parser.add_argument("--adapter", default="runs/lora_r16/checkpoint-42", help="LoRA checkpoint path")
    parser.add_argument("--output_dir", default="models/lora_r16_merged", help="Output directory for merged model")
    parser.add_argument("--dtype", default="bfloat16", help="Precision (bfloat16 or float16)")
    parser.add_argument("--device", default="cpu", help="Device for merge ('cpu' or 'cuda')")
    args = parser.parse_args()

    merge_lora_with_qwen_wrapper(
        base_model_path=args.base_model,
        adapter_path=args.adapter,
        output_dir=args.output_dir,
        dtype=args.dtype,
        device=args.device,
    )


if __name__ == "__main__":
    main()