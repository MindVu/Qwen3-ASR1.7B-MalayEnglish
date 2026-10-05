import os
from pathlib import Path
from dotenv import load_dotenv
from datasets import load_dataset

# Load .env from project root or current working directory
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

TOKEN = os.getenv("HF_TOKEN")

dataset = load_dataset(
    "Revolab/ASR-Benchmark-Public",
    token=TOKEN,
)

dataset.save_to_disk("./data/Revolab-ASR-Benchmark-Public")
