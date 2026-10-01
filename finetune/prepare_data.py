"""
Prepares the 60-minute Malay/English ASR dataset and writes it directly in
the format train_finetune.py expects: a flat JSONL file per split, with
`audio` as a file path string, plus the wav files themselves on disk.

This merges the old two-step pipeline (prepare_data.py -> save_to_disk,
then convert_to_jsonl.py -> jsonl + wavs) into one step. The intermediate
Arrow dataset is no longer written by default; pass --save_arrow_dataset if
you still want it for inspection/debugging.

Usage:
    python prepare_data.py \
        --dataset_dir ../data/Revolab-ASR-Benchmark-Public \
        --jsonl_path ../data/Revolab-ASR-Benchmark-Public-60min.jsonl \
        --output_dir ../data

Produces:
    ../data/train.jsonl
    ../data/eval.jsonl
    ../data/audio_clips/train/<id>.wav
    ../data/audio_clips/eval/<id>.wav
"""

import os
import json
import re
import argparse
from collections import Counter
from pathlib import Path

import soundfile as sf
from datasets import load_from_disk, Audio, DatasetDict

DEFAULT_PROMPT = "Transcribe the audio accurately."


def normalize_text(text):
    """
    Standard text normalization: lowercasing, punctuation removal, and whitespace collapse.
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def resolve_path(provided_path, candidate_paths):
    if provided_path and Path(provided_path).exists():
        return Path(provided_path).resolve()
    for cand in candidate_paths:
        if cand.exists():
            return cand.resolve()
    return candidate_paths[0].resolve()


def load_and_prepare_60min_dataset(
    dataset_dir=None,
    jsonl_path=None,
    test_size=0.1,
    max_duration=30.0,
    seed=42,
    eval_classification="malay+english",
):
    """
    Loads the 60-minute dataset defined by Revolab-ASR-Benchmark-Public-60min.jsonl,
    links it with the audio features in Revolab-ASR-Benchmark-Public, resamples to 16kHz,
    normalizes text, filters long clips, and splits into train/eval sets.

    Returns a DatasetDict with decoded `audio` (array + sampling_rate) --
    callers that need file-path-based JSONL (e.g. train_finetune.py) should
    use write_jsonl_split() below rather than consuming this directly.
    """
    current_dir = Path(__file__).resolve().parent
    repo_root = current_dir.parent

    resolved_dataset_dir = resolve_path(
        dataset_dir,
        [
            repo_root / "data" / "Revolab-ASR-Benchmark-Public",
            current_dir / ".." / "data" / "Revolab-ASR-Benchmark-Public",
            Path("data/Revolab-ASR-Benchmark-Public"),
        ],
    )

    resolved_jsonl_path = resolve_path(
        jsonl_path,
        [
            repo_root / "data" / "Revolab-ASR-Benchmark-Public-60min.jsonl",
            current_dir / ".." / "data" / "Revolab-ASR-Benchmark-Public-60min.jsonl",
            Path("data/Revolab-ASR-Benchmark-Public-60min.jsonl"),
        ],
    )

    print(f"Loading 60min manifest from: {resolved_jsonl_path}")
    if not resolved_jsonl_path.exists():
        raise FileNotFoundError(f"60min dataset jsonl not found at {resolved_jsonl_path}")

    selected_metadata = {}
    with open(resolved_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            selected_metadata[item["id"]] = item

    selected_ids = set(selected_metadata.keys())
    print(f"Found {len(selected_ids)} samples in 60min manifest.")

    print(f"Loading base dataset from disk: {resolved_dataset_dir}")
    if not resolved_dataset_dir.exists():
        raise FileNotFoundError(f"Base dataset directory not found at {resolved_dataset_dir}")

    raw_ds = load_from_disk(str(resolved_dataset_dir))
    full_ds = raw_ds["train"] if isinstance(raw_ds, dict) or hasattr(raw_ds, "keys") else raw_ds

    # 1. Filter dataset to 60min subset IDs
    print("Filtering dataset to 60min subset...")
    subset_ds = full_ds.filter(lambda example: example["id"] in selected_ids)

    # 2. Resample audio to 16kHz mono
    print("Resampling audio to 16kHz mono...")
    subset_ds = subset_ds.cast_column("audio", Audio(sampling_rate=16000))

    # 3. Enrich with metadata and normalized text
    def enrich_and_normalize(example):
        sample_id = example["id"]
        meta = selected_metadata.get(sample_id, {})
        norm_txt = meta.get("normalized_text") or example.get("normalized_text")
        if not norm_txt:
            raw_txt = meta.get("text") or example.get("text") or ""
            norm_txt = normalize_text(raw_txt)
        else:
            norm_txt = normalize_text(norm_txt)

        duration = meta.get("duration")
        if duration is None:
            audio = example["audio"]
            duration = len(audio["array"]) / audio["sampling_rate"]

        return {
            "id": sample_id,
            "text": norm_txt,
            "raw_text": example.get("text", ""),
            "classification": meta.get("classification", "unknown"),
            "duration": float(duration),
        }

    subset_ds = subset_ds.map(enrich_and_normalize)

    # 4. Filter clips > max_duration
    if max_duration is not None and max_duration > 0:
        before_count = len(subset_ds)
        subset_ds = subset_ds.filter(lambda ex: ex["duration"] <= max_duration)
        print(f"Filtered clips > {max_duration}s: kept {len(subset_ds)} of {before_count} samples.")

    # 5. Split train/eval: eval = ALL Malay+English code-switch samples,
    #    train = everything else. Not a random split -- intentional, so the
    #    eval set isolates the hardest/most relevant category (per the
    #    assessment's stated preference for English/Malay code-switching)
    #    rather than diluting it across a random mix.
    classification_counts = Counter(subset_ds["classification"])
    print(f"Classification counts in 60min subset: {dict(classification_counts)}")

    target = eval_classification.strip().lower()

    def is_eval_target(example):
        return (example.get("classification") or "").strip().lower() == target

    eval_ds = subset_ds.filter(is_eval_target)
    train_ds = subset_ds.filter(lambda ex: not is_eval_target(ex))

    if len(eval_ds) == 0:
        raise ValueError(
            f"No samples matched eval_classification='{eval_classification}'. "
            f"Available classification values: {sorted(set(subset_ds['classification']))}. "
            "Pass the correct label via --eval_classification."
        )

    print(
        f"Split by classification='{eval_classification}': "
        f"{len(eval_ds)} eval samples, {len(train_ds)} train samples."
    )

    split_dataset = DatasetDict({"train": train_ds, "test": eval_ds})

    train_duration = sum(train_ds["duration"]) / 60.0
    eval_duration = sum(eval_ds["duration"]) / 60.0
    print(f"Train split: {len(train_ds)} samples ({train_duration:.2f} min)")
    print(f"Eval split:  {len(eval_ds)} samples ({eval_duration:.2f} min)")

    return split_dataset


def write_jsonl_split(ds, split_name: str, audio_root: Path, jsonl_path: Path, prompt: str):
    """
    Writes one split out as .wav files + a JSONL manifest, in the format
    train_finetune.py expects (audio as a file path string).
    """
    audio_dir = audio_root / split_name
    audio_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for example in ds:
            sample_id = example["id"]
            audio = example["audio"]  # {"array": np.ndarray, "sampling_rate": int}

            wav_path = audio_dir / f"{sample_id}.wav"
            sf.write(str(wav_path), audio["array"], audio["sampling_rate"])

            record = {
                "id": sample_id,
                "audio": str(wav_path.resolve()),
                "text": example["text"],          # already normalized above
                "raw_text": example.get("raw_text", ""),
                "prompt": prompt,
                "duration": example.get("duration"),
                "classification": example.get("classification", "unknown"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[{split_name}] wrote {n_written} examples -> {jsonl_path}")
    print(f"[{split_name}] wav files -> {audio_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare 60-minute Malay/English ASR dataset for fine-tuning")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Path to Revolab-ASR-Benchmark-Public directory")
    parser.add_argument("--jsonl_path", type=str, default=None, help="Path to Revolab-ASR-Benchmark-Public-60min.jsonl")
    parser.add_argument("--output_dir", type=str, default=None,
                         help="Directory to write train.jsonl / eval.jsonl / audio_clips/ into")
    parser.add_argument("--test_size", type=float, default=0.1,
                         help="Unused now that eval is classification-based; kept for backward compatibility.")
    parser.add_argument("--max_duration", type=float, default=30.0, help="Maximum clip duration in seconds (default: 30.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (no longer affects the split, kept for compatibility)")
    parser.add_argument("--eval_classification", type=str, default="malay+english",
                         help="Classification label whose samples ALL go into the eval split "
                              "(everything else goes to train). Run once and check the printed "
                              "'Classification counts' line if this default doesn't match your data.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT,
                         help="System prompt written into every record's 'prompt' field")
    parser.add_argument("--save_arrow_dataset", action="store_true",
                         help="Also save the intermediate DatasetDict via save_to_disk "
                              "(data/60min_dataset_split) for inspection/debugging.")
    args = parser.parse_args()

    current_dir = Path(__file__).resolve().parent
    repo_root = current_dir.parent

    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_root = output_dir / "audio_clips"

    split_dataset = load_and_prepare_60min_dataset(
        dataset_dir=args.dataset_dir,
        jsonl_path=args.jsonl_path,
        test_size=args.test_size,
        max_duration=args.max_duration,
        seed=args.seed,
        eval_classification=args.eval_classification,
    )

    if args.save_arrow_dataset:
        arrow_dir = repo_root / "data" / "60min_dataset_split"
        print(f"Saving intermediate Arrow dataset to: {arrow_dir}")
        split_dataset.save_to_disk(str(arrow_dir))

    print(f"Writing JSONL + wav output to: {output_dir}")
    write_jsonl_split(split_dataset["train"], "train", audio_root, output_dir / "train.jsonl", args.prompt)
    write_jsonl_split(split_dataset["test"], "eval", audio_root, output_dir / "eval.jsonl", args.prompt)

    print("\n60-minute dataset preparation complete. Point train_finetune.py at:")
    print(f"  --train_file {output_dir / 'train.jsonl'}")
    print(f"  --eval_file {output_dir / 'eval.jsonl'}")


if __name__ == "__main__":
    main()