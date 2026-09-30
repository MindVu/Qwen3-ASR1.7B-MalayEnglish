import os
from datasets import load_dataset
import librosa
import soundfile as sf
from pathlib import Path
import re

def normalize_text(text):
    # Basic normalization: lowercasing and punctuation removal
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def main():
    print("Preparing dataset...")
    # NOTE: Qwen3-ASR might need specific formatting.
    # Replace with actual dataset you want to use, e.g., 'mozilla-foundation/common_voice_15_0'
    DATASET_NAME = "google/fleurs" 
    LANGUAGE = "ms_my" # Malay
    
    OUTPUT_DIR = Path("../data")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Example logic to load and process
    print(f"Loading {DATASET_NAME} for {LANGUAGE}...")
    # dataset = load_dataset(DATASET_NAME, LANGUAGE, split="train", streaming=True)
    
    # 1. Iterate over dataset
    # 2. Resample to 16kHz
    # 3. Filter clips > 30s
    # 4. Normalize text
    # 5. Save to OUTPUT_DIR with metadata.jsonl
    
    print("Data preparation script scaffold complete.")

if __name__ == "__main__":
    main()
