# Fine-Tuning Report: Qwen3-ASR 1.7B

## 1. Dataset Description
* **Dataset Used:** [Insert Dataset Name, e.g., Common Voice 15.0 - Malay + Fleurs]
* **Dataset Duration:** [e.g., 45 minutes]
* **Language/Composition:** [e.g., 70% Malay, 30% Code-switched Malaysian English]
* **Preprocessing Steps:**
  1. Resampled to 16kHz mono.
  2. Filtered out clips > 30 seconds.
  3. Text normalization (lowercasing, punctuation removal).
  4. Splitting (90/10 Train/Eval split by speaker).

## 2. Training Configuration
* **Technique:** LoRA (Parameter-Efficient Fine-Tuning)
* **Number of Trainable Parameters:** [e.g., ~4.5M (0.26% of total)]
* **Batch Size:** [e.g., 4]
* **Gradient Accumulation Steps:** [e.g., 4]
* **Learning Rate:** [e.g., 1e-4]
* **Epochs/Steps:** [e.g., 3 epochs / 500 steps]
* **Precision:** BF16
* **Hardware:** [e.g., 1x NVIDIA L4 (24GB)]
* **Peak GPU Memory Usage:** [e.g., 16.5 GB]
* **Training Duration:** [e.g., 1 hour 15 mins]

## 3. Results & Evaluation
* **Final Training Loss:** [e.g., 0.12]
* **Before Fine-Tuning WER:** [e.g., 28.4%]
* **After Fine-Tuning WER:** [e.g., 19.2%]

### Transcript Examples (Before vs After)
| Audio File | Ground Truth | Baseline Transcript | Fine-Tuned Transcript |
|---|---|---|---|
| `sample_01.wav` | [Text] | [Text] | [Text] |
| `sample_02.wav` | [Text] | [Text] | [Text] |

## 4. Explanation of Strategy
*Why LoRA?* 
Given the small dataset size (30-60 minutes) and hardware constraints, full fine-tuning would likely overfit and require excessive VRAM. LoRA on the decoder attention blocks provides a highly efficient way to adapt the language modeling capabilities of the ASR model to Malaysian accents and code-switching without catastrophic forgetting of the acoustic representations.
