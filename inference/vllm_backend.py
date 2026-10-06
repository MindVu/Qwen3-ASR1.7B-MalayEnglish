import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase2.vllm_backend import (
    VLLMInferenceBackend,
    get_vllm_backend,
    transcribe,
)

__all__ = [
    "VLLMInferenceBackend",
    "get_vllm_backend",
    "transcribe",
]
