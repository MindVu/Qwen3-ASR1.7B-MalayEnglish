from fastapi import FastAPI, UploadFile, File
import uvicorn
import asyncio
import tempfile
import librosa
import time
import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

app = FastAPI()

MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID, torch_dtype=TORCH_DTYPE, low_cpu_mem_usage=True
).to(DEVICE)

# Queue for dynamic batching
request_queue = asyncio.Queue()
MAX_BATCH_SIZE = 16
BATCH_TIMEOUT = 0.1 # seconds

async def process_batch():
    while True:
        batch = []
        try:
            # Wait for at least one item
            item = await request_queue.get()
            batch.append(item)
            
            # Wait briefly for more items to form a batch
            try:
                while len(batch) < MAX_BATCH_SIZE:
                    item = await asyncio.wait_for(request_queue.get(), timeout=BATCH_TIMEOUT)
                    batch.append(item)
            except asyncio.TimeoutError:
                pass
            
            if batch:
                # 1. Extract audio features for all items in batch
                # 2. Pad features (DataCollator / processor logic)
                # 3. model.generate(batch_features)
                # 4. processor.batch_decode(outputs)
                # 5. Send results back to individual futures
                
                # Mock processing for scaffold
                for req in batch:
                    future, audio_path = req
                    # Simulate processing
                    future.set_result(f"Batched transcription placeholder for {audio_path}")
                    
        except Exception as e:
            print(f"Batch processing error: {e}")
            for req in batch:
                if not req[0].done():
                    req[0].set_exception(e)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_batch())

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    start_time = time.time()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    audio, sr = librosa.load(tmp_path, sr=16000)
    audio_duration = len(audio) / sr

    future = asyncio.Future()
    await request_queue.put((future, tmp_path))
    
    try:
        text = await future
        processing_time = time.time() - start_time
        os.remove(tmp_path)
        
        return {
            "text": text,
            "audio_duration": audio_duration,
            "processing_time": processing_time
        }
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
