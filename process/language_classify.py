import os
import json
from pathlib import Path
from dotenv import load_dotenv

from datasets import load_from_disk
from vllm import LLM, SamplingParams


# Load environment variables (such as HF_TOKEN)
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()


DATASET_DIR = (
    "./data/Revolab-ASR-Benchmark-Public"
    if os.path.exists("./data/Revolab-ASR-Benchmark-Public")
    else "./Revolab-ASR-Benchmark-Public"
)
MODEL_NAME = "Qwen/Qwen3-8B"
MODELS_DIR = "./models"
OUTPUT_FILE = "./data/Revolab-ASR-Benchmark-Public-classified.jsonl"


SYSTEM_PROMPT = """You are a language identification classifier for Malaysian speech transcripts.

Classify the transcript into exactly ONE of these labels:

malay
english
malay+english

Definitions:

- malay:
  The utterance is entirely or overwhelmingly Malay.
  Commonly used English loanwords alone do not make it code-switching.

- english:
  The utterance is entirely or overwhelmingly English.
  Malay names or unavoidable proper nouns alone do not make it code-switching.

- malay+english:
  The utterance contains genuine use of BOTH Malay and English.
  This includes Malay-English code-switching within the same utterance.

Return ONLY one label:
malay
english
malay+english
"""


def classify_output(output):
    prediction = output.outputs[0].text.strip().lower()

    # Remove common formatting
    prediction = prediction.replace("`", "").strip()

    if prediction in {
        "malay",
        "english",
        "malay+english",
    }:
        return prediction

    # Fallback
    if "malay+english" in prediction:
        return "malay+english"

    if "english" in prediction:
        return "english"

    if "malay" in prediction:
        return "malay"

    return prediction


def main():

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    dataset = load_from_disk(DATASET_DIR)

    print(dataset)

    ds = dataset["train"]

    print(f"Number of samples: {len(ds)}")


    # --------------------------------------------------------
    # Build prompts
    # --------------------------------------------------------

    prompts = []

    for example in ds:

        text = example.get("normalized_text")

        if not text:
            text = example.get("text", "")

        text = str(text).strip()

        prompts.append(
            [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": f"Transcript:\n{text}",
                },
            ]
        )


    # --------------------------------------------------------
    # Load Qwen
    # --------------------------------------------------------

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Use local directory if already downloaded, otherwise load MODEL_NAME with download_dir
    local_model_path = os.path.join(MODELS_DIR, "Qwen3-8B")
    if not os.path.exists(local_model_path):
        local_model_path = os.path.join(MODELS_DIR, "Qwen/Qwen3-8B")
    model_to_load = local_model_path if os.path.exists(local_model_path) else MODEL_NAME

    llm = LLM(
        model=model_to_load,
        download_dir=MODELS_DIR,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
    )


    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=20,
    )


    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    print("Starting inference...")

    outputs = llm.chat(
        prompts,
        sampling_params=sampling_params,
        chat_template_kwargs={
        "enable_thinking": False
    },
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        for example, output in zip(ds, outputs):

            label = classify_output(output)

            result = {
                "id": example["id"],
                "text": example.get("text"),
                "normalized_text": example.get("normalized_text"),
                "classification": label,
            }

            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                ) + "\n"
            )


    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
