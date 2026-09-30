import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import time
import json
import argparse
import librosa
from pathlib import Path
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing eval audio files")
    parser.add_argument("--output", type=str, default="../results/bench_single.json", help="Output JSON file")
    args = parser.parse_args()

    MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Loading {MODEL_ID} on {DEVICE} with {TORCH_DTYPE}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID, 
        torch_dtype=TORCH_DTYPE, 
        low_cpu_mem_usage=True,
    ).to(DEVICE)

    # Note: Enable FlashAttention/SDPA if supported
    # model.to_bettertransformer() or pass attn_implementation="flash_attention_2" in from_pretrained

    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=TORCH_DTYPE,
        device=DEVICE,
    )

    audio_paths = list(Path(args.audio_dir).glob("*.wav"))
    results = []

    print(f"Running single-stream benchmark on {len(audio_paths)} files...")
    
    # Warmup
    if audio_paths:
        asr_pipeline(str(audio_paths[0]))

    for p in audio_paths:
        audio, sr = librosa.load(str(p), sr=16000)
        audio_duration = len(audio) / sr
        
        start_time = time.time()
        res = asr_pipeline(str(p))
        processing_time = time.time() - start_time
        
        results.append({
            "file": p.name,
            "duration": audio_duration,
            "processing_time": processing_time,
            "rtf": processing_time / audio_duration,
            "text": res["text"]
        })

    rtfs = [r["rtf"] for r in results]
    print(f"\nAvg RTF: {np.mean(rtfs):.3f}")
    print(f"P50 RTF: {np.percentile(rtfs, 50):.3f}")
    print(f"P95 RTF: {np.percentile(rtfs, 95):.3f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.output}")

if __name__ == "__main__":
    main()
