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

Split: random but STRATIFIED by classification -- --test_size and
--valid_size are applied independently within each classification group
(english / malay / malay+english / ...), then the groups are recombined.
This keeps the same language mix in train/valid/test instead of leaving it
to chance, which matters at this data scale where a plain random split
could put an entire class into one split by luck:
    - train.jsonl: majority of samples, used for gradient updates.
    - valid.jsonl: stratified held-out fraction, used as train_finetune.py's
                   --eval_file for in-training loss monitoring.
    - test.jsonl:  stratified held-out fraction, untouched by training, for
                   final before/after WER reporting.

Transcript format: each record's "text" field (the training target) is
tagged with a language-control prefix, per classification:

    language English<asr_text>{transcript}
    language Malay<asr_text>{transcript}
    language None<asr_text>{transcript}        # code-switch (or unmapped)

A separate "transcript" field holds the plain, untagged transcript for WER
scoring -- the tag's "<" ">" characters strip to nothing under
normalize_text() without restoring the missing whitespace, so scoring
against the tagged "text" field directly would corrupt word boundaries
(e.g. "English<asr_text>This" -> "englishasrtextthis" as one word).

    ../data/train.jsonl
    ../data/valid.jsonl
    ../data/test.jsonl
    ../data/audio_clips/train/<id>.wav
    ../data/audio_clips/valid/<id>.wav
    ../data/audio_clips/test/<id>.wav
"""

import os
import json
import re
import argparse
from collections import Counter
from pathlib import Path

import soundfile as sf
from datasets import load_from_disk, Audio, DatasetDict, concatenate_datasets

DEFAULT_PROMPT = "Transcribe the audio accurately."
ASR_TEXT_TAG = "<asr_text>"

# classification (lowercased) -> language tag. Anything not listed here
# falls back to "None" (same as code-switch) -- a missing/unexpected
# classification value should degrade to "no language hint" rather than
# silently mislabel the sample as a specific language.
DEFAULT_LANGUAGE_MAP = {
    "english": "English",
    "malay": "Malay",
    "malay+english": "None",  # code-switch
}


def classification_to_language_tag(classification, language_map):
    key = (classification or "").strip().lower()
    return language_map.get(key, "None")


def build_tagged_target(classification, transcript, language_map):
    lang = classification_to_language_tag(classification, language_map)
    return f"language {lang}{ASR_TEXT_TAG}{transcript}"


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


def stratified_three_way_split(subset_ds, test_size, valid_size, seed):
    """
    Splits subset_ds into train/valid/test, applying test_size and
    valid_size independently within each `classification` group so every
    split gets the same language mix, then recombines the groups.

    Uses explicit shuffle + select (rather than per-group train_test_split)
    so tiny groups degrade gracefully -- rounding to 0 for a given split
    just means that class isn't represented there, with a warning printed,
    rather than raising on a group too small for sklearn's constraints.
    """
    classifications = sorted(set(subset_ds["classification"]))
    train_parts, valid_parts, test_parts = [], [], []

    print(f"Stratifying split across {len(classifications)} classification group(s): {classifications}")

    for c in classifications:
        group = subset_ds.filter(lambda ex: ex["classification"] == c)
        group = group.shuffle(seed=seed)
        n = len(group)

        n_test = round(n * test_size)
        n_remainder = n - n_test
        n_valid = round(n_remainder * valid_size)
        n_train = n_remainder - n_valid

        if n > 0 and (n_test == 0 or n_valid == 0):
            print(f"  WARNING: classification='{c}' has only {n} sample(s) -- "
                  f"train={n_train}, valid={n_valid}, test={n_test}. Too few to "
                  "proportionally represent in every split; consider collecting more "
                  "data for this class or accepting it's train-only / under-represented.")
        else:
            print(f"  classification='{c}': n={n} -> train={n_train}, valid={n_valid}, test={n_test}")

        test_parts.append(group.select(range(0, n_test)))
        valid_parts.append(group.select(range(n_test, n_test + n_valid)))
        train_parts.append(group.select(range(n_test + n_valid, n)))

    final_train_ds = concatenate_datasets(train_parts).shuffle(seed=seed)
    valid_ds = concatenate_datasets(valid_parts).shuffle(seed=seed)
    test_ds = concatenate_datasets(test_parts).shuffle(seed=seed)

    return final_train_ds, valid_ds, test_ds


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
    valid_size=0.1,
    max_duration=30.0,
    seed=42,
):
    """
    Loads the 60-minute dataset defined by Revolab-ASR-Benchmark-Public-60min.jsonl,
    links it with the audio features in Revolab-ASR-Benchmark-Public, resamples to 16kHz,
    normalizes text, filters long clips, and splits randomly into train/valid/test.

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

    # 5. Classification distribution, printed for visibility -- this also
    #    drives the language tag applied later in write_jsonl_split.
    classification_counts = Counter(subset_ds["classification"])
    print(f"Classification counts in 60min subset: {dict(classification_counts)}")

    # 6. Stratified three-way split: test_size and valid_size applied within
    #    each classification group so the language mix is preserved across
    #    train/valid/test rather than left to chance.
    final_train_ds, valid_ds, eval_ds = stratified_three_way_split(
        subset_ds, test_size=test_size, valid_size=valid_size, seed=seed
    )

    split_dataset = DatasetDict({"train": final_train_ds, "validation": valid_ds, "test": eval_ds})

    train_duration = sum(final_train_ds["duration"]) / 60.0
    valid_duration = sum(valid_ds["duration"]) / 60.0
    test_duration = sum(eval_ds["duration"]) / 60.0
    print(f"Train split: {len(final_train_ds)} samples ({train_duration:.2f} min)")
    print(f"Valid split: {len(valid_ds)} samples ({valid_duration:.2f} min)")
    print(f"Test split:  {len(eval_ds)} samples ({test_duration:.2f} min)")

    return split_dataset


def write_jsonl_split(ds, split_name: str, audio_root: Path, jsonl_path: Path, prompt: str, language_map: dict):
    """
    Writes one split out as .wav files + a JSONL manifest, in the format
    train_finetune.py expects (audio as a file path string).

    "text" is the tagged training target: "language {Lang}<asr_text>{transcript}".
    "transcript" is the plain, untagged version -- use this for WER scoring.
    """
    audio_dir = audio_root / split_name
    audio_dir.mkdir(parents=True, exist_ok=True)

    tag_counts = Counter()
    n_written = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for example in ds:
            sample_id = example["id"]
            audio = example["audio"]  # {"array": np.ndarray, "sampling_rate": int}

            wav_path = audio_dir / f"{sample_id}.wav"
            sf.write(str(wav_path), audio["array"], audio["sampling_rate"])

            transcript = example["text"]  # already normalized above
            classification = example.get("classification", "unknown")
            lang_tag = classification_to_language_tag(classification, language_map)
            tag_counts[lang_tag] += 1

            record = {
                "id": sample_id,
                "audio": str(wav_path.resolve()),
                "text": build_tagged_target(classification, transcript, language_map),
                "transcript": transcript,         # untagged -- use for WER scoring
                "raw_text": example.get("raw_text", ""),
                "prompt": prompt,
                "duration": example.get("duration"),
                "classification": classification,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[{split_name}] wrote {n_written} examples -> {jsonl_path}")
    print(f"[{split_name}] language tag counts: {dict(tag_counts)}")
    print(f"[{split_name}] wav files -> {audio_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare 60-minute Malay/English ASR dataset for fine-tuning")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Path to Revolab-ASR-Benchmark-Public directory")
    parser.add_argument("--jsonl_path", type=str, default=None, help="Path to Revolab-ASR-Benchmark-Public-60min.jsonl")
    parser.add_argument("--output_dir", type=str, default=None,
                         help="Directory to write train.jsonl / valid.jsonl / test.jsonl / audio_clips/ into")
    parser.add_argument("--test_size", type=float, default=0.1,
                         help="Fraction of the full subset randomly held out as the final test split.")
    parser.add_argument("--valid_size", type=float, default=0.1,
                         help="Fraction of the remainder (after test) randomly held out as the dev/validation "
                              "split (used as train_finetune.py's --eval_file for in-training monitoring).")
    parser.add_argument("--max_duration", type=float, default=30.0, help="Maximum clip duration in seconds (default: 30.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split")
    parser.add_argument("--language_map", type=str, default=None,
                         help="JSON string mapping lowercased classification -> language tag, e.g. "
                              '\'{"english": "English", "malay": "Malay", "malay+english": "None"}\'. '
                              "Any classification not listed maps to \"None\". Defaults to "
                              f"{DEFAULT_LANGUAGE_MAP}. Check the printed 'Classification counts' "
                              "and '[split] language tag counts' lines to confirm this matches your data.")
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

    language_map = dict(DEFAULT_LANGUAGE_MAP)
    if args.language_map:
        language_map = json.loads(args.language_map)
    print(f"Language map: {language_map} (anything else -> 'None')")

    split_dataset = load_and_prepare_60min_dataset(
        dataset_dir=args.dataset_dir,
        jsonl_path=args.jsonl_path,
        test_size=args.test_size,
        valid_size=args.valid_size,
        max_duration=args.max_duration,
        seed=args.seed,
    )

    if args.save_arrow_dataset:
        arrow_dir = repo_root / "data" / "60min_dataset_split"
        print(f"Saving intermediate Arrow dataset to: {arrow_dir}")
        split_dataset.save_to_disk(str(arrow_dir))

    print(f"Writing JSONL + wav output to: {output_dir}")
    write_jsonl_split(split_dataset["train"], "train", audio_root, output_dir / "train.jsonl", args.prompt, language_map)
    write_jsonl_split(split_dataset["validation"], "valid", audio_root, output_dir / "valid.jsonl", args.prompt, language_map)
    write_jsonl_split(split_dataset["test"], "test", audio_root, output_dir / "test.jsonl", args.prompt, language_map)

    print("\n60-minute dataset preparation complete.")
    print("For training:")
    print(f"  --train_file {output_dir / 'train.jsonl'}")
    print(f"  --eval_file {output_dir / 'valid.jsonl'}")
    print("For final before/after WER reporting (score against the 'transcript' field, not 'text'):")
    print(f"  {output_dir / 'test.jsonl'}")


if __name__ == "__main__":
    main()