# coding=utf-8
# Adapted from the official QwenLM/Qwen3-ASR finetuning script.
# Adds an optional LoRA path (--use_lora) on top of the reference full
# fine-tuning recipe. Everything else (collator, checkpoint-copy callback,
# dtype-casting trainer, resume logic) is unchanged from the reference.
#
# WHY LORA HERE: the reference script trains the full `thinker` end-to-end
# (it has no freezing/PEFT). With only 30-60 min of data, full fine-tuning
# of the entire audio-to-text pathway risks catastrophic forgetting of the
# model's general ASR ability. LoRA restricts updates to low-rank adapters
# on the thinker's linear layers, which is far more appropriate at this
# data scale -- see the fine-tuning report for the full justification.
import argparse
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import librosa
import torch
import torch.nn as nn
from datasets import load_dataset
from qwen_asr import Qwen3ASRModel
from torch.utils.tensorboard import SummaryWriter
from transformers import (GenerationConfig, Trainer, TrainerCallback,
                          TrainingArguments)


def patch_outer_forward(model):
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return

    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError(
            "Cannot patch forward: model has no `.thinker.forward`. "
            "Your qwen3_asr model may be incompatible."
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True


def scope_lora_targets_to_thinker(model, leaf_names: List[str]) -> List[str]:
    """
    Return fully-qualified dotted names of Linear layers under `thinker`
    whose leaf name matches one of `leaf_names`. Scoping this way (rather
    than passing bare leaf names to LoraConfig) prevents PEFT from matching
    same-named Linear layers anywhere else in the model -- if the audio
    front-end inside thinker happens to reuse names like q_proj/v_proj for
    its own attention, printing the matched list below lets you sanity
    check what actually got selected before training.
    """
    matched = []
    for full_name, module in model.named_modules():
        if not full_name.startswith("thinker."):
            continue
        if not isinstance(module, nn.Linear):
            continue
        if full_name.split(".")[-1] in leaf_names:
            matched.append(full_name)

    if not matched:
        raise ValueError(
            f"No Linear layers under `thinker` matched {leaf_names}. "
            "Run `print(model.thinker)` and adjust --lora_target_names."
        )

    print(f"[lora] {len(matched)} Linear layers matched under thinker:")
    for m in matched[:10]:
        print(f"  {m}")
    if len(matched) > 10:
        print(f"  ... and {len(matched) - 10} more")
    return matched


_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    best_step = None
    best_path = None
    for name in os.listdir(output_dir):
        m = _CKPT_RE.match(name)
        if not m:
            continue
        step = int(m.group(1))
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and (best_step is None or step > best_step):
            best_step = step
            best_path = path
    return best_path


def load_audio(path: str, sr: int = 16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def build_prefix_messages(prompt: str, audio_array):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def make_preprocess_fn_prefix_only(processor):
    def _preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ex.get("prompt", "")
        dummy_audio = None
        prefix_msgs = build_prefix_messages(prompt, dummy_audio)
        prefix_text = processor.apply_chat_template(
            [prefix_msgs], add_generation_prompt=True, tokenize=False
        )[0]
        return {
            "prompt": prompt,
            "audio": ex["audio"],
            "target": ex["text"],
            "prefix_text": prefix_text,
        }

    return _preprocess


@dataclass
class DataCollatorForQwen3ASRFinetuning:
    processor: Any
    sampling_rate: int = 16000

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_paths = [f["audio"] for f in features]
        prefix_texts = [f["prefix_text"] for f in features]
        targets = [f["target"] for f in features]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]
        audios = [load_audio(p, sr=self.sampling_rate) for p in audio_paths]

        full_inputs = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_inputs = self.processor(
            text=prefix_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            labels[i, :pl] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        full_inputs["labels"] = labels
        return full_inputs


class CastFloatInputsTrainer(Trainer):
    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=model_dtype)
        return inputs


def copy_required_hf_files_for_qwen_asr(src_dir: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    required = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "chat_template.json",
        "merges.txt",
        "vocab.json",
    ]
    for fn in required:
        src = os.path.join(src_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, fn))


class MakeEveryCheckpointInferableCallback(TrainerCallback):
    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path

    def on_save(self, args: TrainingArguments, state, control, **kwargs):
        if args.process_index != 0:
            return control

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            ckpt_dir = kwargs.get("checkpoint", ckpt_dir)

        copy_required_hf_files_for_qwen_asr(self.base_model_path, ckpt_dir)
        return control


class ExtraTensorBoardMetricsCallback(TrainerCallback):
    """
    Logs GPU memory usage and run config (trainable params, LoRA settings)
    to TensorBoard, in addition to the loss/lr/epoch Trainer already reports
    via report_to=["tensorboard"]. Uses its own SummaryWriter pointed at the
    same logging_dir, so it shows up as extra scalar tags in the same run
    rather than depending on callback ordering with HF's own TensorBoardCallback.
    """

    def __init__(self, logging_dir: str, run_metadata: Dict[str, Any]):
        self.writer = SummaryWriter(log_dir=logging_dir)
        self.run_metadata = run_metadata

    def on_train_begin(self, args: TrainingArguments, state, control, **kwargs):
        if args.process_index != 0:
            return control
        text = "\n".join(f"- **{k}**: {v}" for k, v in self.run_metadata.items())
        self.writer.add_text("run_config", text, global_step=0)
        return control

    def on_log(self, args: TrainingArguments, state, control, **kwargs):
        if args.process_index != 0 or not torch.cuda.is_available():
            return control
        step = state.global_step
        self.writer.add_scalar("gpu/memory_allocated_gb", torch.cuda.memory_allocated() / 1e9, step)
        self.writer.add_scalar("gpu/memory_reserved_gb", torch.cuda.memory_reserved() / 1e9, step)
        self.writer.add_scalar("gpu/max_memory_allocated_gb", torch.cuda.max_memory_allocated() / 1e9, step)
        return control

    def on_train_end(self, args: TrainingArguments, state, control, **kwargs):
        if args.process_index == 0:
            self.writer.flush()
            self.writer.close()
        return control


def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR Finetuning (full FT or LoRA)")

    # Paths
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--train_file", type=str, default="train.jsonl")
    p.add_argument("--eval_file", type=str, default="")
    p.add_argument("--output_dir", type=str, default="./qwen3-asr-finetuning-out")

    # Audio
    p.add_argument("--sr", type=int, default=16000)

    # Train hyper-params
    # NB: the reference defaults (batch_size=32) assume a large-hour dataset
    # on a big GPU. For a 30-60 min assessment dataset, start much smaller
    # (e.g. 2-4) and rely on grad_acc for effective batch size.
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_acc", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--lr_scheduler_type", type=str, default="linear")
    p.add_argument("--warmup_ratio", type=float, default=0.02)

    # LoRA
    p.add_argument("--use_lora", action="store_true",
                    help="Fine-tune with LoRA adapters on thinker's linear layers "
                         "instead of full fine-tuning (recommended at this data scale).")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_names", nargs="+",
                    default=["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"])

    # DataLoader
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--persistent_workers", type=int, default=1)
    p.add_argument("--prefetch_factor", type=int, default=2)

    # Save
    p.add_argument("--save_strategy", type=str, default="steps")
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--save_total_limit", type=int, default=5)

    # Logging
    p.add_argument("--logging_dir", type=str, default=None,
                    help="TensorBoard log dir. Defaults to <output_dir>/tensorboard.")
    p.add_argument("--report_to", type=str, default="tensorboard",
                    help="Passed to TrainingArguments.report_to. Use 'none' to disable.")

    # Resume
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume", type=int, default=0)

    return p.parse_args()


def main():
    args_cli = parse_args()

    if not args_cli.train_file:
        raise ValueError("TRAIN_FILE is required (json/jsonl). Needs fields: audio, text, optional prompt")

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args_cli.model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    patch_outer_forward(model)

    # Set generation_config from the pre-LoRA model config, before any PEFT
    # wrapping (PeftModel's .config proxying can be ambiguous depending on
    # PEFT version).
    model.generation_config = GenerationConfig.from_model_config(model.config)

    if args_cli.use_lora:
        from peft import LoraConfig, get_peft_model

        target_modules = scope_lora_targets_to_thinker(model, args_cli.lora_target_names)
        lora_config = LoraConfig(
            r=args_cli.lora_r,
            lora_alpha=args_cli.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args_cli.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[full-ft] {n_params:,} trainable parameters (entire model).")

    raw_ds = load_dataset(
        "json",
        data_files={
            "train": args_cli.train_file,
            **({"validation": args_cli.eval_file} if args_cli.eval_file else {}),
        },
    )
    ds = raw_ds.map(make_preprocess_fn_prefix_only(processor), num_proc=1)

    keep = {"prompt", "audio", "target", "prefix_text"}
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    collator = DataCollatorForQwen3ASRFinetuning(processor=processor, sampling_rate=args_cli.sr)

    logging_dir = args_cli.logging_dir or os.path.join(args_cli.output_dir, "tensorboard")

    training_args = TrainingArguments(
        output_dir=args_cli.output_dir,
        per_device_train_batch_size=args_cli.batch_size,
        gradient_accumulation_steps=args_cli.grad_acc,
        learning_rate=args_cli.lr,
        num_train_epochs=args_cli.epochs,
        logging_steps=args_cli.log_steps,
        logging_dir=logging_dir,
        report_to=[args_cli.report_to] if args_cli.report_to != "none" else [],
        lr_scheduler_type=args_cli.lr_scheduler_type,
        warmup_ratio=args_cli.warmup_ratio,
        dataloader_num_workers=args_cli.num_workers,
        dataloader_pin_memory=(args_cli.pin_memory == 1),
        dataloader_persistent_workers=(args_cli.persistent_workers == 1),
        dataloader_prefetch_factor=args_cli.prefetch_factor if args_cli.num_workers > 0 else None,
        save_strategy=args_cli.save_strategy,
        save_steps=args_cli.save_steps,
        save_total_limit=args_cli.save_total_limit,
        save_safetensors=True,
        eval_strategy="steps",
        eval_steps=args_cli.save_steps,
        do_eval=bool(args_cli.eval_file),
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    run_metadata = {
        "model_path": args_cli.model_path,
        "mode": "lora" if args_cli.use_lora else "full_finetune",
        "lora_r": args_cli.lora_r if args_cli.use_lora else None,
        "lora_alpha": args_cli.lora_alpha if args_cli.use_lora else None,
        "lora_target_names": args_cli.lora_target_names if args_cli.use_lora else None,
        "batch_size": args_cli.batch_size,
        "grad_acc": args_cli.grad_acc,
        "effective_batch_size": args_cli.batch_size * args_cli.grad_acc,
        "learning_rate": args_cli.lr,
        "epochs": args_cli.epochs,
        "precision": "bf16" if use_bf16 else "fp16",
    }

    callbacks = [MakeEveryCheckpointInferableCallback(base_model_path=args_cli.model_path)]
    if args_cli.report_to != "none":
        callbacks.append(ExtraTensorBoardMetricsCallback(logging_dir=logging_dir, run_metadata=run_metadata))

    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", None),
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=callbacks,
    )

    if args_cli.report_to != "none":
        print(f"TensorBoard logging to: {logging_dir}")
        print(f"View with: tensorboard --logdir {logging_dir}")

    resume_from = (args_cli.resume_from or "").strip()
    if not resume_from and args_cli.resume == 1:
        resume_from = find_latest_checkpoint(training_args.output_dir) or ""

    if resume_from:
        if trainer.args.process_index == 0:
            print(f"[resume] resume_from_checkpoint = {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    if args_cli.use_lora:
        # Save adapter only; merge separately at inference time if a
        # standalone checkpoint is preferred.
        trainer.save_model(args_cli.output_dir)
        print(f"LoRA adapter saved to {args_cli.output_dir}. "
              "Load with PeftModel.from_pretrained(base_model, adapter_dir) "
              "or merge_and_unload() for a standalone checkpoint.")


if __name__ == "__main__":
    main()