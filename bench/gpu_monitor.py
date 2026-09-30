import subprocess
import time
import json
import argparse

def get_gpu_stats():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits"
            ],
            stdout=subprocess.PIPE,
            text=True
        )
        output = result.stdout.strip().split('\n')[0].split(', ')
        return {
            "gpu_util": float(output[0]),
            "mem_used_mb": float(output[1]),
            "mem_total_mb": float(output[2])
        }
    except Exception as e:
        print(f"Error querying nvidia-smi: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=0.1, help="Polling interval in seconds")
    parser.add_argument("--output", type=str, default="../results/gpu_stats.json", help="Output JSON file")
    args = parser.parse_args()

    stats_log = []
    print(f"Monitoring GPU... Press Ctrl+C to stop. Saving to {args.output}")
    
    try:
        while True:
            stats = get_gpu_stats()
            if stats:
                stats["timestamp"] = time.time()
                stats_log.append(stats)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopping monitor...")
        
    with open(args.output, "w") as f:
        json.dump(stats_log, f, indent=2)
    print("Saved.")

if __name__ == "__main__":
    main()
