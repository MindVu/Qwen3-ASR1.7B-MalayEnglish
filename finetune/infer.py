import argparse
import json

import librosa
import torch
from qwen_asr import Qwen3ASRModel
from peft import PeftModel


def load_audio(path, sr=16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen3-ASR-1.7B",
    )

    # Optional: if omitted, use default/base model
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="LoRA adapter path. If omitted, use base model.",
    )

    parser.add_argument(
        "--test_file",
        type=str,
        default="data/test.jsonl",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="test_output.jsonl",
    )

    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Load base model
    # --------------------------------------------------
    use_bf16 = (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(0)[0] >= 8
    )

    dtype = torch.bfloat16 if use_bf16 else torch.float16

    print(f"Loading model: {args.model_path}")

    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=dtype,
        device_map=None,
    )

    model = asr_wrapper.model
    processor = asr_wrapper.processor

    # --------------------------------------------------
    # Optional LoRA
    # --------------------------------------------------
    if args.adapter_path:
        print(f"Loading LoRA adapter: {args.adapter_path}")

        model = PeftModel.from_pretrained(
            model,
            args.adapter_path,
        )

        print("Using fine-tuned LoRA weights.")

    else:
        print("No adapter provided.")
        print("Using original pretrained weights.")

    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    # --------------------------------------------------
    # Read test JSONL
    # --------------------------------------------------
    samples = []

    with open(args.test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                samples.append(json.loads(line))

    print(f"Loaded {len(samples)} test samples.")

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    with open(args.output_file, "w", encoding="utf-8") as fout:

        for i, sample in enumerate(samples):

            audio_path = sample["audio"]
            prompt = sample.get("prompt", "")

            print(
                f"[{i + 1}/{len(samples)}] "
                f"{audio_path}"
            )

            audio = load_audio(
                audio_path,
                sr=args.sr,
            )

            messages = [
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio",
                            "audio": audio,
                        }
                    ],
                },
            ]

            text = processor.apply_chat_template(
                [messages],
                add_generation_prompt=True,
                tokenize=False,
            )[0]

            inputs = processor(
                text=[text],
                audio=[audio],
                return_tensors="pt",
                padding=True,
            )

            model_dtype = next(model.parameters()).dtype

            inputs = {
                k: (
                    v.cuda().to(dtype=model_dtype)
                    if torch.is_tensor(v) and v.is_floating_point()
                    else v.cuda()
                    if torch.is_tensor(v)
                    else v
                )
                for k, v in inputs.items()
            }

            with torch.inference_mode():
                generation_output = model.generate(
                    **inputs,
                    max_new_tokens=512,
                )
                
                output_ids = generation_output.sequences
                
                input_len = inputs["input_ids"].shape[1]
                
                generated_ids = output_ids[:, input_len:]

            prediction = processor.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0].strip()

            result = dict(sample)
            result["prediction"] = prediction

            fout.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

            print(f"  prediction: {prediction}")

    print()
    print(f"Saved to: {args.output_file}")


if __name__ == "__main__":
    main()
