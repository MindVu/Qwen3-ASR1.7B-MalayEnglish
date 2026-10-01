import json
import random

from datasets import load_from_disk


# ============================================================
# Config
# ============================================================

DATASET_DIR = "./Revolab-ASR-Benchmark-Public"

CLASSIFIED_JSONL = (
    "./Revolab-ASR-Benchmark-Public-classified.jsonl"
)

OUTPUT_JSONL = (
    "./Revolab-ASR-Benchmark-Public-60min.jsonl"
)

TARGET_DURATION = 60 * 60  # 60 minutes

RANDOM_SEED = 42


# ============================================================
# Load original dataset
# ============================================================

dataset = load_from_disk(DATASET_DIR)
ds = dataset["train"]

print(f"Original dataset: {len(ds)} samples")


# ============================================================
# Build ID -> duration mapping
# ============================================================

duration_by_id = {}

for example in ds:
    sample_id = example["id"]

    # HuggingFace Audio feature normally provides:
    # {
    #     "path": ...,
    #     "array": ...,
    #     "sampling_rate": ...
    # }
    audio = example["audio"]

    duration = len(audio["array"]) / audio["sampling_rate"]

    duration_by_id[sample_id] = duration


print(
    f"Loaded durations for "
    f"{len(duration_by_id)} samples"
)


# ============================================================
# Read classification results
# ============================================================

classified = []

with open(
    CLASSIFIED_JSONL,
    "r",
    encoding="utf-8",
) as f:

    for line in f:
        item = json.loads(line)

        sample_id = item["id"]

        if sample_id not in duration_by_id:
            print(
                f"WARNING: duration not found for {sample_id}"
            )
            continue

        item["duration"] = duration_by_id[sample_id]

        classified.append(item)


print(f"Classification records: {len(classified)}")


# ============================================================
# Separate classes
# ============================================================

english = [
    x for x in classified
    if x["classification"] == "english"
]

codeswitch = [
    x for x in classified
    if x["classification"] == "malay+english"
]

malay = [
    x for x in classified
    if x["classification"] == "malay"
]


# ============================================================
# Keep ALL English + code-switching
# ============================================================

selected = english + codeswitch

current_duration = sum(
    x["duration"]
    for x in selected
)

print(
    f"English:       {len(english):4d} "
    f"{sum(x['duration'] for x in english) / 60:.2f} min"
)

print(
    f"Malay+English: {len(codeswitch):4d} "
    f"{sum(x['duration'] for x in codeswitch) / 60:.2f} min"
)

print(
    f"Initial total:  {len(selected):4d} "
    f"{current_duration / 60:.2f} min"
)


# ============================================================
# Randomly add Malay samples
# ============================================================

random.seed(RANDOM_SEED)
random.shuffle(malay)

for sample in malay:

    duration = sample["duration"]

    # Don't exceed 60 minutes
    if current_duration + duration > TARGET_DURATION:
        continue

    selected.append(sample)
    current_duration += duration

    if current_duration >= TARGET_DURATION:
        break


# ============================================================
# Shuffle final dataset
# ============================================================

random.shuffle(selected)


# ============================================================
# Save
# ============================================================

with open(
    OUTPUT_JSONL,
    "w",
    encoding="utf-8",
) as f:

    for item in selected:
        f.write(
            json.dumps(
                item,
                ensure_ascii=False,
            ) + "\n"
        )


# ============================================================
# Statistics
# ============================================================

counts = {
    "english": 0,
    "malay+english": 0,
    "malay": 0,
}

durations = {
    "english": 0.0,
    "malay+english": 0.0,
    "malay": 0.0,
}

for item in selected:

    label = item["classification"]

    counts[label] += 1
    durations[label] += item["duration"]


print("\n========== FINAL DATASET ==========")

print(f"Total samples: {len(selected)}")
print(f"Total duration: {current_duration / 60:.2f} minutes")

for label in [
    "english",
    "malay+english",
    "malay",
]:
    print(
        f"{label:15s}: "
        f"{counts[label]:4d} samples, "
        f"{durations[label] / 60:.2f} min"
    )

print(f"\nSaved to: {OUTPUT_JSONL}")
