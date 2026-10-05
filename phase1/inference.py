"""
Pure model inference module for Qwen3-ASR 1.7B.

Features:
- Single clean entry point: transcribe(...) -> dict
- Accepts audio file path, raw audio bytes, or numpy waveform
- Accurate sub-millisecond GPU-synchronized timing breakdown:
    * preprocessing_time (audio decode, resampling, feature extraction)
    * inference_time (model.generate)
    * postprocessing_time (tokenizer decoding, text cleaning)
    * total_latency
    * RTF (Real-Time Factor)
- Strictly isolated from HTTP/server framework logic
"""

import io
import logging
import os
import re
import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch

try:
    import librosa
except ImportError:
    librosa = None

logger = logging.getLogger("qwen_asr.inference")


def clean_prediction(text: str) -> str:
    """Extract clean transcript if Qwen3-ASR outputs language tokens like '<asr_text>'."""
    if "<asr_text>" in text:
        text = text.split("<asr_text>", 1)[1]
    return text.strip()


def sync_if_cuda(device: torch.device):
    """Synchronize CUDA device for accurate GPU timing measurements."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def load_audio_waveform(
    audio_input: Union[str, bytes, io.BytesIO, np.ndarray],
    target_sr: int = 16000,
) -> Tuple[np.ndarray, float]:
    """
    Convert various audio input types into a 1D mono float32 numpy array and duration.
    Returns: (waveform_numpy, duration_seconds)
    """
    if isinstance(audio_input, np.ndarray):
        wav = audio_input.astype(np.float32)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=-1)
        dur = len(wav) / target_sr
        return wav, dur

    if isinstance(audio_input, bytes):
        audio_stream = io.BytesIO(audio_input)
    elif isinstance(audio_input, io.BytesIO):
        audio_stream = audio_input
    elif isinstance(audio_input, str):
        if not os.path.exists(audio_input):
            raise FileNotFoundError(f"Audio file not found: {audio_input}")
        audio_stream = audio_input
    else:
        raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

    # Use soundfile if possible, fallback to librosa
    try:
        data, sr = sf.read(audio_stream, dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=-1)
        if sr != target_sr:
            if librosa is not None:
                data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            else:
                raise RuntimeError(
                    f"Audio sampling rate is {sr}Hz but target is {target_sr}Hz. "
                    "Please install librosa or provide 16kHz audio."
                )
        duration = len(data) / target_sr
        return data, duration
    except Exception as e:
        if librosa is not None:
            if isinstance(audio_stream, io.BytesIO):
                audio_stream.seek(0)
            data, sr = librosa.load(audio_stream, sr=target_sr, mono=True)
            duration = len(data) / target_sr
            return data, duration
        raise RuntimeError(f"Failed to decode audio: {e}")


def transcribe(
    model_wrapper: Any,
    audio: Union[str, bytes, io.BytesIO, np.ndarray],
    prompt: str = "Transcribe the audio accurately.",
    max_new_tokens: int = 512,
    target_sr: int = 16000,
) -> Dict[str, Any]:
    """
    Execute single-utterance speech-to-text inference with detailed timing.

    Args:
        model_wrapper: Instance of Qwen3ASRModelWrapper containing model & processor.
        audio: Audio file path, raw audio bytes, or numpy float32 array.
        prompt: System/task prompt for transcription.
        max_new_tokens: Maximum new tokens to generate.
        target_sr: Target sample rate for model feature extractor (16000 Hz).

    Returns:
        Dictionary with transcription, duration, timing breakdown, and RTF.
    """
    model = model_wrapper.model
    processor = model_wrapper.processor
    device = model_wrapper.device
    model_dtype = model_wrapper.torch_dtype

    if model is None or processor is None:
        raise RuntimeError("Model or processor is not loaded in model_wrapper.")

    # 1. Load audio and compute duration
    t_start = time.perf_counter()
    waveform, audio_duration = load_audio_waveform(audio, target_sr=target_sr)
    t_audio_loaded = time.perf_counter()

    # 2. Preprocessing (chat template formatting, feature extraction, tensor transfer)
    sync_if_cuda(device)
    t_prep_start = time.perf_counter()

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": [{"type": "audio", "audio": waveform}]},
    ]
    formatted_prompt = processor.apply_chat_template(
        [messages], add_generation_prompt=True, tokenize=False
    )[0]

    inputs = processor(
        text=[formatted_prompt],
        audio=[waveform],
        return_tensors="pt",
        padding=True,
    )

    # Move tensors to device & precision
    prepared_inputs = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            v = v.to(device, non_blocking=True)
            if v.is_floating_point():
                v = v.to(dtype=model_dtype)
        prepared_inputs[k] = v

    sync_if_cuda(device)
    t_prep_end = time.perf_counter()

    # 3. Model Generation (GPU inference)
    sync_if_cuda(device)
    t_gen_start = time.perf_counter()

    with torch.inference_mode():
        generated_outputs = model.generate(
            **prepared_inputs,
            max_new_tokens=max_new_tokens,
        )

    sync_if_cuda(device)
    t_gen_end = time.perf_counter()

    # 4. Postprocessing & Token Decoding
    output_ids = (
        generated_outputs.sequences
        if hasattr(generated_outputs, "sequences")
        else generated_outputs
    )
    prompt_len = prepared_inputs["input_ids"].shape[1]
    new_token_ids = output_ids[:, prompt_len:]

    pad_id = processor.tokenizer.pad_token_id
    if pad_id is not None:
        num_generated_tokens = int((new_token_ids != pad_id).sum().item())
    else:
        num_generated_tokens = int(new_token_ids.numel())

    raw_text = processor.tokenizer.batch_decode(
        new_token_ids,
        skip_special_tokens=True,
    )[0].strip()

    cleaned_text = clean_prediction(raw_text)
    t_decode_end = time.perf_counter()

    # 5. Timing computation
    t_audio_load = t_audio_loaded - t_start
    t_preprocess = t_prep_end - t_prep_start
    t_generate = t_gen_end - t_gen_start
    t_decode = t_decode_end - t_gen_end
    t_total_proc = t_preprocess + t_generate + t_decode
    t_e2e = t_decode_end - t_start

    safe_dur = max(audio_duration, 1e-4)
    rtf_model = t_generate / safe_dur
    rtf_processing = t_total_proc / safe_dur
    rtf_e2e = t_e2e / safe_dur

    return {
        "text": cleaned_text,
        "raw_text": raw_text,
        "audio_duration": round(audio_duration, 3),
        "processing_time": round(t_total_proc, 4),
        "rtf": round(rtf_processing, 4),
        "rtf_model": round(rtf_model, 4),
        "rtf_e2e": round(rtf_e2e, 4),
        "tokens_generated": num_generated_tokens,
        "timing": {
            "audio_load_time": round(t_audio_load, 4),
            "preprocessing_time": round(t_preprocess, 4),
            "inference_time": round(t_generate, 4),
            "postprocessing_time": round(t_decode, 4),
            "total_processing_time": round(t_total_proc, 4),
            "total_latency": round(t_e2e, 4),
        },
    }
