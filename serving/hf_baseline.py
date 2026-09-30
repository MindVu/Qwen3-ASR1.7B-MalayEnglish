import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from fastapi import FastAPI, UploadFile, File
import uvicorn
import tempfile
import librosa
import time
import os

app = FastAPI()

# Configuration
MODEL_ID = "Qwen/Qwen3-ASR-1.7B" # Verify correct model ID
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Load model and processor globally
print(f"Loading {MODEL_ID} on {DEVICE} with {TORCH_DTYPE}...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID, 
    torch_dtype=TORCH_DTYPE, 
    low_cpu_mem_usage=True, 
    use_safetensors=True
).to(DEVICE)

asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=TORCH_DTYPE,
    device=DEVICE,
)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # 1. Save uploaded file temporarily
    start_time = time.time()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # 2. Get audio duration for RTF calculation
        audio, sr = librosa.load(tmp_path, sr=16000)
        audio_duration = len(audio) / sr
        
        # 3. Transcribe
        result = asr_pipeline(tmp_path)
        
        # 4. Clean up
        os.remove(tmp_path)
        
        processing_time = time.time() - start_time
        
        return {
            "text": result["text"],
            "audio_duration": audio_duration,
            "processing_time": processing_time
        }
        
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
