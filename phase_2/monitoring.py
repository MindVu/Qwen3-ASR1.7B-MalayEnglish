"""
GPU + CPU monitoring module (shared by Phase 1).

Samples in a background thread:
  GPU: utilization (%) and VRAM (MB)
       backend: pynvml (fast), fallback: nvidia-smi
  CPU (needs `psutil`; reported as None if psutil is missing):
       * system-wide utilization (% of ALL cores on the node, 0-100)
       * this process (the benchmark client), in % of ONE core (100 = one core
         fully busy; can exceed 100 with several busy threads)
       * child processes of this process, summed, same unit
         (Phase 2: the vLLM engine-core process is a child of the benchmark)
       * explicitly watched PIDs, summed, same unit
         (Phase 1: the HTTP server process, found through /health)

Every sample is time-stamped (time.time()), so statistics can be computed either
over the whole run (stop()) or over a sub-window (get_window_stats()). The
sub-window is used for STEADY-STATE numbers that exclude the ramp-up and tail of a
run (see steady_state_window()).

Summary statistics (same keys for the full run and for a window):
    gpu_util_avg_pct, gpu_util_max_pct, gpu_mem_used_peak_mb, gpu_mem_used_avg_mb,
    samples_count, cpu_logical_cores_allowed,
    cpu_sys_util_avg_pct / _max_pct,
    cpu_proc_util_avg_pct / _max_pct,
    cpu_children_util_avg_pct / _max_pct,
    cpu_watched_util_avg_pct / _max_pct
and, for stop() only: torch_peak_allocated_mb, torch_peak_reserved_mb

Notes on reading the CPU numbers:
  * System-wide CPU covers the whole node. On a shared node it includes other
    jobs, so prefer the per-process numbers.
  * A process near 100% means one core is saturated. If the benchmark client or
    the engine/server process sits there while the GPU is not saturated, that
    process (Python-side work) is the bottleneck.
"""

import logging
import os
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import psutil
except ImportError:  # CPU metrics are optional
    psutil = None

logger = logging.getLogger("qwen_asr.monitoring")

CPU_SERIES = ("sys", "proc", "children", "watched")


# ----------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------


def fmt_pct(v: Optional[float]) -> str:
    """Format a percentage for tables ('n/a' when missing)."""
    return "n/a" if v is None else f"{v:.1f}%"


def fmt_num(v: Optional[float]) -> str:
    """Format a number for CSV (empty when missing)."""
    return "" if v is None else f"{v:.1f}"


def format_cpu_summary(stats: Dict[str, Any]) -> str:
    """One-line human-readable CPU summary from stop()/get_window_stats() output."""

    def f(key: str) -> str:
        v = stats.get(key)
        return "n/a" if v is None else f"{v:.0f}%"

    parts = [
        f"system {f('cpu_sys_util_avg_pct')} (max {f('cpu_sys_util_max_pct')})",
        f"benchmark proc {f('cpu_proc_util_avg_pct')} (max {f('cpu_proc_util_max_pct')})",
    ]
    if stats.get("cpu_children_util_avg_pct") is not None:
        parts.append(
            f"child procs/engine {f('cpu_children_util_avg_pct')} "
            f"(max {f('cpu_children_util_max_pct')})"
        )
    if stats.get("cpu_watched_util_avg_pct") is not None:
        parts.append(
            f"server proc {f('cpu_watched_util_avg_pct')} "
            f"(max {f('cpu_watched_util_max_pct')})"
        )
    cores = stats.get("cpu_logical_cores_allowed")
    return " | ".join(parts) + (f" | allowed cores: {cores}" if cores else "")


# ----------------------------------------------------------------------
# Steady-state window
# ----------------------------------------------------------------------


def steady_state_window(
    done: List[Tuple[float, float]],
    concurrency: int,
) -> Optional[Dict[str, float]]:
    """
    Find the steady-state part of a closed-loop run (C requests kept in flight).

    Args:
        done: (completion_time_s, audio_seconds) for every SUCCESSFUL request,
              completion times relative to the start of the run.
        concurrency: C, the number of concurrent slots.

    The run has a ramp-up (slots filling), a steady middle and a tail (queue empty,
    fewer than C in flight). With n completions sorted by time:
        * after the C-th completion, every slot has been refilled at least once;
        * at the (n-C)-th completion the last request is dispatched, so after it
          fewer than C requests remain in flight.
    The steady window runs from the C-th to the (n-C)-th completion. Throughput is
    the audio finished after the window start, divided by the window length.

    Returns None when there are too few requests (need n >= 3*C, so that the
    window holds at least C requests). The estimate is still noisy with only a
    few requests per slot; ~10 requests per slot is a safer minimum.
    """
    n = len(done)
    c = max(1, int(concurrency))
    if n < 3 * c:
        return None

    done = sorted(done, key=lambda x: x[0])
    mid = done[c - 1 : n - c]  # completions number C .. n-C
    t0, t1 = mid[0][0], mid[-1][0]
    window = t1 - t0
    if window <= 0:
        return None

    counted = mid[1:]
    return {
        "t_start_s": t0,
        "t_end_s": t1,
        "window_s": window,
        "n_requests": len(counted),
        "audio_s_per_s": sum(a for _, a in counted) / window,
        "requests_per_s": len(counted) / window,
    }


# ----------------------------------------------------------------------
# Monitor
# ----------------------------------------------------------------------


def _allowed_cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 1


class GPUMonitor:
    """
    Background GPU (+ CPU) monitoring sampler.

    Args:
        device_index: GPU index to sample.
        interval: sampling period in seconds.
        watch_pids: extra process IDs whose CPU usage is tracked and reported as
            `cpu_watched_*` (e.g. the Phase 1 server). Must be on this machine.
    """

    CHILD_REFRESH_S = 1.0  # how often the child-process list is re-scanned

    def __init__(
        self,
        device_index: int = 0,
        interval: float = 0.05,
        watch_pids: Optional[List[int]] = None,
    ):
        self.device_index = device_index
        self.interval = interval
        self.watch_pids: List[int] = list(watch_pids or [])

        # GPU samples (timestamps are aligned with util/mem samples)
        self.util_samples: List[float] = []
        self.mem_samples: List[float] = []
        self.timestamps: List[float] = []

        # CPU samples: one (values, timestamps) pair per series
        self.cpu_samples: Dict[str, List[float]] = {k: [] for k in CPU_SERIES}
        self.cpu_ts: Dict[str, List[float]] = {k: [] for k in CPU_SERIES}

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # CPU state
        self.cpu_enabled = psutil is not None
        self._proc = None
        self._children: Dict[int, Any] = {}
        self._watched: Dict[int, Any] = {}
        self._children_refreshed_at = 0.0
        if self.cpu_enabled:
            self._proc = psutil.Process()
            logger.info("CPU monitoring enabled (psutil), allowed cores: %d", _allowed_cores())
        else:
            logger.warning("psutil not installed; CPU utilization will not be reported (pip install psutil)")

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

    # ------------------------------------------------------------------
    # GPU
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # CPU
    # ------------------------------------------------------------------

    @staticmethod
    def _proc_pct(p: Any) -> Optional[float]:
        """CPU % of one process since its previous call (None if it is gone)."""
        try:
            return float(p.cpu_percent(interval=None))
        except Exception:
            return None

    def _sum_pct(self, procs: List[Any]) -> Optional[float]:
        if not procs:
            return None
        return float(sum(v for v in (self._proc_pct(p) for p in procs) if v is not None))

    def _refresh_children(self) -> None:
        """Track new child processes; keep existing Process objects so their
        cpu_percent() deltas stay valid."""
        try:
            current = {c.pid: c for c in self._proc.children(recursive=True)}
        except Exception:
            return
        for pid, c in current.items():
            if pid not in self._children:
                self._children[pid] = c
                self._proc_pct(c)  # prime
        for pid in list(self._children):
            if pid not in current:
                del self._children[pid]

    def _record_cpu(self, series: str, ts: float, value: Optional[float]) -> None:
        if value is not None:
            self.cpu_samples[series].append(value)
            self.cpu_ts[series].append(ts)

    def _sample_cpu(self) -> None:
        if not self.cpu_enabled:
            return
        now = time.time()
        self._record_cpu("sys", now, float(psutil.cpu_percent(interval=None)))
        self._record_cpu("proc", now, self._proc_pct(self._proc))

        if now - self._children_refreshed_at >= self.CHILD_REFRESH_S:
            self._refresh_children()
            self._children_refreshed_at = now
        self._record_cpu("children", now, self._sum_pct(list(self._children.values())))
        self._record_cpu("watched", now, self._sum_pct(list(self._watched.values())))

    def _prime_cpu(self) -> None:
        """Reset CPU history and take the baseline readings cpu_percent() needs."""
        for k in CPU_SERIES:
            self.cpu_samples[k].clear()
            self.cpu_ts[k].clear()
        self._children.clear()
        self._watched.clear()
        if not self.cpu_enabled:
            return

        psutil.cpu_percent(interval=None)
        self._proc_pct(self._proc)

        self._refresh_children()
        self._children_refreshed_at = time.time()

        for pid in self.watch_pids:
            try:
                p = psutil.Process(pid)
                self._proc_pct(p)
                self._watched[pid] = p
            except Exception as e:
                logger.warning("Cannot watch PID %s for CPU usage: %s", pid, e)

    # ------------------------------------------------------------------
    # Summaries (full run or sub-window)
    # ------------------------------------------------------------------

    @staticmethod
    def _select(
        values: List[float],
        stamps: List[float],
        window: Optional[Tuple[float, float]],
    ) -> List[float]:
        if window is None:
            return list(values)
        t0, t1 = window
        return [v for v, t in zip(values, stamps) if t0 <= t <= t1]

    @staticmethod
    def _avg_max(samples: List[float]) -> Tuple[Optional[float], Optional[float]]:
        if not samples:
            return None, None
        return round(float(np.mean(samples)), 1), round(float(np.max(samples)), 1)

    def _cpu_summary(self, window: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {"cpu_logical_cores_allowed": _allowed_cores() if self.cpu_enabled else None}
        for name in CPU_SERIES:
            sel = self._select(self.cpu_samples[name], self.cpu_ts[name], window)
            avg, mx = self._avg_max(sel)
            out[f"cpu_{name}_util_avg_pct"] = avg
            out[f"cpu_{name}_util_max_pct"] = mx
        return out

    def _gpu_summary(self, window: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        util = self._select(self.util_samples, self.timestamps, window)
        mem = self._select(self.mem_samples, self.timestamps, window)
        if not util:
            return {
                "gpu_util_avg_pct": 0.0,
                "gpu_util_max_pct": 0.0,
                "gpu_mem_used_peak_mb": 0.0,
                "gpu_mem_used_avg_mb": 0.0,
                "samples_count": 0,
            }
        return {
            "gpu_util_avg_pct": round(float(np.mean(util)), 1),
            "gpu_util_max_pct": round(float(np.max(util)), 1),
            "gpu_mem_used_peak_mb": round(float(np.max(mem)), 1),
            "gpu_mem_used_avg_mb": round(float(np.mean(mem)), 1),
            "samples_count": len(util),
        }

    def get_window_stats(self, t_start: float, t_end: float) -> Dict[str, Any]:
        """
        Same statistics as stop(), restricted to samples taken between t_start and
        t_end (epoch seconds, time.time()). Call after stop() and before the next
        start(); samples are kept until start() is called again.
        """
        return {**self._gpu_summary((t_start, t_end)), **self._cpu_summary((t_start, t_end))}

    # ------------------------------------------------------------------
    # Sampling loop / lifecycle
    # ------------------------------------------------------------------

    def _worker(self):
        while not self._stop_event.is_set():
            u, m = self.sample_now()
            if u is not None and m is not None:
                self.timestamps.append(time.time())
                self.util_samples.append(u)
                self.mem_samples.append(m)
            try:
                self._sample_cpu()
            except Exception:
                pass  # never let monitoring crash the benchmark
            time.sleep(self.interval)

    def start(self):
        """Reset history and start background sampling thread."""
        self.util_samples.clear()
        self.mem_samples.clear()
        self.timestamps.clear()
        self._prime_cpu()
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

        gpu = self._gpu_summary()
        return {
            "gpu_util_avg_pct": gpu["gpu_util_avg_pct"],
            "gpu_util_max_pct": gpu["gpu_util_max_pct"],
            "gpu_mem_used_peak_mb": gpu["gpu_mem_used_peak_mb"],
            "gpu_mem_used_avg_mb": gpu["gpu_mem_used_avg_mb"],
            "torch_peak_allocated_mb": torch_peak_alloc,
            "torch_peak_reserved_mb": torch_peak_resv,
            "samples_count": gpu["samples_count"],
            **self._cpu_summary(),
        }