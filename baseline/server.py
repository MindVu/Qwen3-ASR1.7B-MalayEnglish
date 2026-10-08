"""
Asynchronous HTTP Inference Server for Qwen3-ASR 1.7B.

Features:
- FastAPI async server with non-blocking POST /transcribe endpoint
- Model loaded ONCE at server startup in lifespan context
- Optional LoRA adapter loading (e.g. runs/lora_r16/checkpoint-42)
- Bounded inference queue (max_queue_size=64) with HTTP 429 overload shedding
- Single dedicated GPU worker processing requests sequentially (no dynamic batching)
- Detailed per-request timing breakdown:
    * queue_wait_time
    * preprocessing_time
    * inference_time
    * postprocessing_time
    * total_latency
    * RTF (Real-Time Factor)
- Configurable host/port and adapter_path via CLI flags or config.yaml
"""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging
import os
import sys
import time
from typing import Any, Dict, Optional
import uuid
import yaml

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
import uvicorn

# Ensure current directory / phase1 is in pythonpath
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from model import Qwen3ASRModelWrapper, load_model
from inference import transcribe

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("qwen_asr.server")


@dataclass
class InferenceRequest:
    """Internal container for queued inference work."""
    request_id: str
    audio_bytes: bytes
    prompt: str
    max_new_tokens: int
    received_at: float
    future: asyncio.Future


class InferenceServerState:
    """Global state holding model, bounded queue, and worker tasks."""
    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.model_wrapper: Optional[Qwen3ASRModelWrapper] = None
        self.queue: Optional[asyncio.Queue] = None
        self.max_queue_size: int = 64
        self.overload_status_code: int = 429
        self.worker_task: Optional[asyncio.Task] = None
        self.executor: Optional[ThreadPoolExecutor] = None
        self.is_running: bool = False
        
        # Server metrics
        self.total_received: int = 0
        self.total_processed: int = 0
        self.total_rejected: int = 0


server_state = InferenceServerState()


def load_server_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from yaml file, with environment variable and CLI overrides."""
    resolved_path = config_path or os.environ.get("SERVER_CONFIG_PATH", "config.yaml")
    if not os.path.exists(resolved_path):
        resolved_path = os.path.join(CURRENT_DIR, resolved_path)
    
    if os.path.exists(resolved_path):
        logger.info("Loading server configuration from %s", resolved_path)
        with open(resolved_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        logger.warning("Config file %s not found. Using defaults.", resolved_path)
        cfg = {
            "model": {"model_path": "Qwen/Qwen3-ASR-1.7B", "device": "cuda", "dtype": "float32"},
            "server": {"host": "0.0.0.0", "port": 8000, "max_queue_size": 64, "overload_status_code": 429},
        }

    # Environment variable overrides
    if "ADAPTER_PATH" in os.environ and os.environ["ADAPTER_PATH"]:
        cfg.setdefault("model", {})["adapter_path"] = os.environ["ADAPTER_PATH"]

    return cfg


async def gpu_worker_loop():
    """
    Dedicated single GPU worker loop.
    Fetches requests from the bounded queue one by one and executes inference.
    Phase 1: Sequential processing (batch_size = 1).
    """
    logger.info("GPU inference worker started. Ready to process queued requests.")
    loop = asyncio.get_running_loop()

    while server_state.is_running:
        try:
            req: InferenceRequest = await server_state.queue.get()
        except asyncio.CancelledError:
            break

        dequeued_at = time.perf_counter()
        queue_wait_time = dequeued_at - req.received_at

        try:
            # Execute model inference in thread pool executor so async loop is unblocked
            result = await loop.run_in_executor(
                server_state.executor,
                transcribe,
                server_state.model_wrapper,
                req.audio_bytes,
                req.prompt,
                req.max_new_tokens,
            )

            completed_at = time.perf_counter()
            total_latency = completed_at - req.received_at

            response_payload = {
                "request_id": req.request_id,
                "text": result["text"],
                "audio_duration": result["audio_duration"],
                "processing_time": result["processing_time"],
                "rtf": result["rtf"],
                "rtf_model": result["rtf_model"],
                "timing": {
                    "queue_wait_time": round(queue_wait_time, 4),
                    "preprocessing_time": result["timing"]["preprocessing_time"],
                    "inference_time": result["timing"]["inference_time"],
                    "postprocessing_time": result["timing"]["postprocessing_time"],
                    "total_processing_time": result["processing_time"],
                    "total_latency": round(total_latency, 4),
                },
                "tokens_generated": result["tokens_generated"],
            }
            if not req.future.cancelled():
                req.future.set_result(response_payload)
            server_state.total_processed += 1

        except Exception as e:
            logger.exception("Error during inference for request %s: %s", req.request_id, e)
            if not req.future.cancelled():
                req.future.set_exception(e)
        finally:
            server_state.queue.task_done()

    logger.info("GPU inference worker stopped.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager: load model at startup, cleanup at shutdown."""
    cfg = load_server_config()
    server_state.config = cfg
    
    server_cfg = cfg.get("server", {})
    server_state.max_queue_size = server_cfg.get("max_queue_size", 64)
    server_state.overload_status_code = server_cfg.get("overload_status_code", 429)

    # Initialize bounded queue
    server_state.queue = asyncio.Queue(maxsize=server_state.max_queue_size)
    server_state.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu_infer_worker")
    server_state.is_running = True

    # Load model ONCE
    logger.info("Initializing Qwen3-ASR model...")
    server_state.model_wrapper = load_model(cfg)

    # Start GPU worker task
    server_state.worker_task = asyncio.create_task(gpu_worker_loop())

    logger.info(
        "Server startup complete. Bounded queue size: %d, Device: %s, Adapter: %s",
        server_state.max_queue_size,
        server_state.model_wrapper.device,
        server_state.model_wrapper.adapter_path or "None",
    )

    yield

    # Shutdown
    logger.info("Initiating server shutdown...")
    server_state.is_running = False
    if server_state.worker_task is not None:
        server_state.worker_task.cancel()
        try:
            await server_state.worker_task
        except asyncio.CancelledError:
            pass
    if server_state.executor is not None:
        server_state.executor.shutdown(wait=False)
    logger.info("Server shutdown complete.")


app = FastAPI(
    title="Qwen3-ASR 1.7B Baseline Inference Server",
    description="Phase 1: Bounded Queue Async Inference Server with Sequential GPU Worker",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check reporting model status, queue utilization, and uptime."""
    is_ready = (
        server_state.model_wrapper is not None
        and server_state.model_wrapper.model is not None
        and server_state.is_running
    )
    current_qsize = server_state.queue.qsize() if server_state.queue else 0
    return {
        "status": "healthy" if is_ready else "initializing",
        "model_loaded": is_ready,
        "device": str(server_state.model_wrapper.device) if server_state.model_wrapper else None,
        "adapter_path": server_state.model_wrapper.adapter_path if server_state.model_wrapper else None,
        "queue": {
            "current_size": current_qsize,
            "max_size": server_state.max_queue_size,
            "utilization_pct": round(100.0 * current_qsize / max(server_state.max_queue_size, 1), 1),
        },
        "metrics": {
            "total_received": server_state.total_received,
            "total_processed": server_state.total_processed,
            "total_rejected": server_state.total_rejected,
        },
        "model_metadata": server_state.model_wrapper.model_metadata if server_state.model_wrapper else {},
    }


@app.post("/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(..., description="Audio file to transcribe (e.g. WAV, MP3, FLAC)"),
    prompt: str = Form("Transcribe the audio accurately."),
    max_new_tokens: int = Form(512),
):
    """
    Asynchronous transcription endpoint.
    Places request into bounded queue; returns 429/503 if queue is full.
    Awaits result asynchronously without blocking the event loop.
    """
    server_state.total_received += 1
    received_at = time.perf_counter()
    request_id = str(uuid.uuid4())

    # Check bounded queue capacity
    if server_state.queue.full():
        server_state.total_rejected += 1
        logger.warning(
            "Inference queue full (%d/%d). Rejecting request %s with HTTP %d.",
            server_state.queue.qsize(),
            server_state.max_queue_size,
            request_id,
            server_state.overload_status_code,
        )
        raise HTTPException(
            status_code=server_state.overload_status_code,
            detail={
                "error": "Server overloaded",
                "message": f"Inference queue is full ({server_state.queue.qsize()}/{server_state.max_queue_size}). Please retry later.",
                "request_id": request_id,
            },
        )

    try:
        audio_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded audio file: {e}",
        )

    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded audio file is empty.",
        )

    loop = asyncio.get_running_loop()
    future = loop.create_future()
    req = InferenceRequest(
        request_id=request_id,
        audio_bytes=audio_bytes,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        received_at=received_at,
        future=future,
    )

    try:
        server_state.queue.put_nowait(req)
    except asyncio.QueueFull:
        server_state.total_rejected += 1
        raise HTTPException(
            status_code=server_state.overload_status_code,
            detail="Inference queue became full while enqueuing.",
        )

    # Await result asynchronously
    try:
        result = await future
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {e}",
        )


def main():
    """Server entrypoint with CLI flag overrides."""
    parser = argparse.ArgumentParser(description="Qwen3-ASR Baseline Inference Server")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--host", default=None, help="Host to bind server (e.g. 0.0.0.0 or 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Port to bind server (e.g. 8000)")
    parser.add_argument("--adapter-path", default=None, help="Path to LoRA adapter checkpoint (e.g. runs/lora_r16/checkpoint-42)")
    args = parser.parse_args()

    if args.adapter_path:
        os.environ["ADAPTER_PATH"] = args.adapter_path
    if args.config:
        os.environ["SERVER_CONFIG_PATH"] = args.config

    cfg = load_server_config(args.config)
    server_cfg = cfg.get("server", {})
    host = args.host or server_cfg.get("host", "0.0.0.0")
    port = args.port or int(server_cfg.get("port", 8000))

    logger.info("Starting Uvicorn server on http://%s:%d", host, port)
    uvicorn.run("server:app", host=host, port=port, workers=1, access_log=False)


if __name__ == "__main__":
    main()
