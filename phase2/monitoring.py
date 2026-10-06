"""
GPU monitoring module for Phase 1 baseline inference.

Features:
- Samples GPU utilization (%) and GPU VRAM (MB) in a background thread
- Primary backend: pynvml (fast, in-process C-library)
- Fallback backend: nvidia-smi command-line query
- Thread-safe start/stop with summary statistics:
    * gpu_util_avg_pct
    * gpu_util_max_pct
    * gpu_mem_used_peak_mb
    * gpu_mem_used_avg_mb
    * torch_peak_allocated_mb
"""

import logging
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("qwen_asr.monitoring")


class GPUMonitor:
    """
    Background GPU monitoring sampler.
    """

    def __init__(self, device_index: int = 0, interval: float = 0.05):
        self.device_index = device_index
        self.interval = interval
        self.util_samples: List[float] = []
        self.mem_samples: List[float] = []
        self.timestamps: List[float] = []

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Check pynvml availability
        self._nvml = None
        self._nvml_handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            logger.info("GPUMonitor initialized using pynvml on GPU %d", device_index)
        except Exception as e:
            logger.info("pynvml not available (%s); falling back to nvidia-smi", e)
            self._nvml = None

    def sample_now(self) -> Tuple[Optional[float], Optional[float]]:
        """Sample (utilization_pct, mem_used_mb) immediately."""
        if self._nvml is not None:
            try:
                util = float(self._nvml.nvmlDeviceGetUtilizationRates(self._nvml_handle).gpu)
                mem = float(self._nvml.nvmlDeviceGetMemoryInfo(self._nvml_handle).used / (1024**2))
                return util, mem
            except Exception:
                pass

        # Fallback to nvidia-smi
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={self.device_index}",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=1.0,
            )
            u_str, m_str = out.strip().split(",")
            return float(u_str.strip()), float(m_str.strip())
        except Exception:
            return None, None

    def _worker(self):
        while not self._stop_event.is_set():
            u, m = self.sample_now()
            if u is not None and m is not None:
                self.timestamps.append(time.time())
                self.util_samples.append(u)
                self.mem_samples.append(m)
            time.sleep(self.interval)

    def start(self):
        """Reset history and start background sampling thread."""
        self.util_samples.clear()
        self.mem_samples.clear()
        self.timestamps.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        """Stop background sampling thread and compute aggregate metrics."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        torch_peak_alloc = None
        torch_peak_resv = None
        try:
            import torch

            if torch.cuda.is_available():
                torch_peak_alloc = round(torch.cuda.max_memory_allocated(self.device_index) / (1024**2), 1)
                torch_peak_resv = round(torch.cuda.max_memory_reserved(self.device_index) / (1024**2), 1)
        except Exception:
            pass

        if not self.util_samples:
            return {
                "gpu_util_avg_pct": 0.0,
                "gpu_util_max_pct": 0.0,
                "gpu_mem_used_peak_mb": 0.0,
                "gpu_mem_used_avg_mb": 0.0,
                "torch_peak_allocated_mb": torch_peak_alloc,
                "torch_peak_reserved_mb": torch_peak_resv,
                "samples_count": 0,
            }

        return {
            "gpu_util_avg_pct": round(float(np.mean(self.util_samples)), 1),
            "gpu_util_max_pct": round(float(np.max(self.util_samples)), 1),
            "gpu_mem_used_peak_mb": round(float(np.max(self.mem_samples)), 1),
            "gpu_mem_used_avg_mb": round(float(np.mean(self.mem_samples)), 1),
            "torch_peak_allocated_mb": torch_peak_alloc,
            "torch_peak_reserved_mb": torch_peak_resv,
            "samples_count": len(self.util_samples),
        }
