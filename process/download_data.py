from datasets import load_dataset

TOKEN = ""

dataset = load_dataset(
    "Revolab/ASR-Benchmark-Public",
    token=TOKEN,
)

dataset.save_to_disk("./Revolab-ASR-Benchmark-Public")
