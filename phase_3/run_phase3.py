"""
Phase 2 Single-Command Automated Runner.

Directly runs the full vLLM concurrency sweep C = [1, 2, 4, 8, 16, 32, 64] on the frozen
43 benchmark utterances and reports the maximum sustainable concurrency.
"""

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_vllm import main

if __name__ == "__main__":
    main()
