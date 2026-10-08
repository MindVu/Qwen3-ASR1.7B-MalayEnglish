"""
Single-Terminal Automated Runner for Phase 1 (Ideal for NSCC interactive jobs).

Workflow:
1. Refuses to start if something is already listening on the target port.
2. Starts server.py as a background process (loads base model + LoRA adapter).
3. Polls GET /health until model is fully loaded into GPU VRAM.
4. Executes benchmark across concurrency levels (1, 2, 4, 8, 16, 32, 64).
5. Displays and saves all results (paths/settings come from config.yaml
   unless overridden on the command line).
6. Gracefully terminates the background server (and its process group).
7. Exits with the benchmark's return code.
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Ignore http(s)_proxy env vars for localhost health checks (common on HPC nodes).
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def is_server_healthy(health_url: str) -> bool:
    try:
        with _OPENER.open(health_url, timeout=2.0) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode())
                return data.get("status") == "healthy"
    except Exception:
        pass
    return False


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", port)) == 0


def tail(path: str, n: int = 60) -> str:
    try:
        with open(path, "r", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except Exception:
        return "<could not read log>"


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Single-Terminal Automated Runner")
    parser.add_argument("--port", type=int, default=8000, help="Port for server")
    parser.add_argument("--config", default=None, help="Path to config.yaml (forwarded to server and benchmark)")
    parser.add_argument("--adapter-path", default=None, help="LoRA adapter path (defaults to config.yaml value)")
    parser.add_argument("--concurrency", default="1,2,4,8,16,32,64", help="Concurrency sweep")
    # The options below default to None so config.yaml values are used unless overridden.
    parser.add_argument("--test-file", default=None, help="Test dataset file")
    parser.add_argument("--audio-dir", default=None, help="Audio clip directory")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--requests-per-bucket", type=int, default=30, help="Requests per duration bucket")
    parser.add_argument("--warmup", type=int, default=None, help="Warmup requests per concurrency level")
    parser.add_argument("--startup-timeout", type=int, default=600, help="Seconds to wait for model load")
    parser.add_argument("--log-file", default="server.log", help="Server log file")
    args = parser.parse_args()

    server_script = os.path.join(CURRENT_DIR, "server.py")
    benchmark_script = os.path.join(CURRENT_DIR, "benchmark.py")
    health_url = f"http://127.0.0.1:{args.port}/health"

    print("=" * 70)
    print("PHASE 1: AUTOMATED SINGLE-TERMINAL RUNNER (NSCC / HPC)")
    print("=" * 70)

    if port_in_use(args.port):
        print(f"[ERROR] Port {args.port} is already in use. A stale server may be running; "
              f"stop it or pass a different --port.")
        sys.exit(2)

    print(f"1. Launching server in background (logs: {args.log_file})...")
    log_handle = open(args.log_file, "w")
    server_cmd = [sys.executable, server_script, "--port", str(args.port)]
    if args.config:
        server_cmd.extend(["--config", args.config])
    if args.adapter_path:
        server_cmd.extend(["--adapter-path", args.adapter_path])

    proc = subprocess.Popen(
        server_cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        cwd=os.getcwd(),
        start_new_session=True,   # own process group so we can stop it and any children
    )

    exit_code = 0
    try:
        print("2. Waiting for Qwen3-ASR model + LoRA adapter to load into GPU memory...")
        t0 = time.time()
        ready = False
        while time.time() - t0 < args.startup_timeout:
            if proc.poll() is not None:
                print(f"\n[ERROR] Server process died prematurely with code {proc.returncode}!")
                print(f"Last lines of {args.log_file}:")
                print(tail(args.log_file))
                exit_code = 1
                return
            if is_server_healthy(health_url):
                ready = True
                break
            time.sleep(2.0)
            print(f"   ...loading model ({int(time.time() - t0)}s elapsed)", end="\r", flush=True)

        if not ready:
            print(f"\n[ERROR] Server not healthy after {args.startup_timeout}s.")
            print(f"Last lines of {args.log_file}:")
            print(tail(args.log_file))
            exit_code = 1
            return

        print(f"\n3. Server is READY on http://127.0.0.1:{args.port}! Starting benchmark suite...\n")

        bench_cmd = [
            sys.executable,
            benchmark_script,
            "--url", f"http://127.0.0.1:{args.port}/transcribe",
            "--concurrency", args.concurrency,
        ]
        if args.config:
            bench_cmd += ["--config", args.config]
        if args.test_file:
            bench_cmd += ["--test-file", args.test_file]
        if args.audio_dir:
            bench_cmd += ["--audio-dir", args.audio_dir]
        if args.output_dir:
            bench_cmd += ["--output-dir", args.output_dir]
        if args.requests_per_bucket is not None:
            bench_cmd += ["--requests-per-bucket", str(args.requests_per_bucket)]
        if args.warmup is not None:
            bench_cmd += ["--warmup", str(args.warmup)]

        res = subprocess.run(bench_cmd)
        if res.returncode != 0:
            print(f"[ERROR] Benchmark failed with return code {res.returncode}")
            exit_code = res.returncode

    finally:
        print("\n4. Shutting down background server...")
        stop_server(proc)
        log_handle.close()
        print("5. Server stopped.")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()