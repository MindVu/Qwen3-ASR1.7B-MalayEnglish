"""
Single-Terminal Automated Runner for Phase 1 (Ideal for NSCC interactive jobs).

Workflow:
1. Starts server.py as a background process (loads base model + LoRA adapter).
2. Polls GET /health until model is fully loaded into GPU VRAM.
3. Executes benchmark across concurrency levels (1, 2, 4, 8, 16, 32, 64).
4. Displays and saves all results to results/.
5. Gracefully terminates the background server process upon completion.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


def is_server_healthy(health_url: str) -> bool:
    try:
        with urllib.request.urlopen(health_url, timeout=2.0) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode())
                return data.get("status") == "healthy"
    except Exception:
        pass
    return False


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Single-Terminal Automated Runner")
    parser.add_argument("--port", type=int, default=8000, help="Port for server")
    parser.add_argument("--adapter-path", default=None, help="LoRA adapter path (defaults to config.yaml value)")
    parser.add_argument("--concurrency", default="1,2,4,8,16,32,64", help="Concurrency sweep")
    parser.add_argument("--test-file", default="data/test.jsonl", help="Test dataset file")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    args = parser.parse_args()

    server_script = os.path.join(CURRENT_DIR, "server.py")
    benchmark_script = os.path.join(CURRENT_DIR, "benchmark.py")
    log_file = "server.log"
    health_url = f"http://127.0.0.1:{args.port}/health"

    print("=" * 70)
    print("PHASE 1: AUTOMATED SINGLE-TERMINAL RUNNER (NSCC / HPC)")
    print("=" * 70)
    print(f"1. Launching server in background (logs: {log_file})...")
    log_handle = open(log_file, "w")
    server_cmd = [sys.executable, server_script, "--port", str(args.port)]
    if args.adapter_path:
        server_cmd.extend(["--adapter-path", args.adapter_path])

    proc = subprocess.Popen(
        server_cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        cwd=os.getcwd(),
    )

    try:
        print("2. Waiting for Qwen3-ASR model + LoRA adapter to load into GPU memory...")
        t0 = time.time()
        ready = False
        while time.time() - t0 < 300: # 5 min timeout
            if proc.poll() is not None:
                print(f"\n[ERROR] Server process died prematurely with code {proc.returncode}!")
                print(f"Check {log_file} for details:")
                with open(log_file) as f:
                    print(f.read())
                sys.exit(1)

            if is_server_healthy(health_url):
                ready = True
                break
            time.sleep(2.0)
            elapsed = int(time.time() - t0)
            print(f"   ...loading model ({elapsed}s elapsed)", end="\r", flush=True)

        if not ready:
            print("\n[ERROR] Server timed out during initialization.")
            sys.exit(1)

        print(f"\n3. Server is READY on http://127.0.0.1:{args.port}! Starting benchmark suite...\n")

        # Run benchmark
        bench_cmd = [
            sys.executable,
            benchmark_script,
            "--url", f"http://127.0.0.1:{args.port}/transcribe",
            "--concurrency", args.concurrency,
            "--test-file", args.test_file,
            "--output-dir", args.output_dir,
        ]
        res = subprocess.run(bench_cmd)
        if res.returncode != 0:
            print(f"[WARNING] Benchmark finished with return code {res.returncode}")

    finally:
        print("\n4. Shutting down background server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_handle.close()
        print("5. Server stopped cleanly. All tasks completed.")


if __name__ == "__main__":
    main()
