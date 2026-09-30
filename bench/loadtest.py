import asyncio
import aiohttp
import time
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

async def send_request(session, url, audio_path, results_list):
    start_time = time.time()
    try:
        # Assuming the server expects a file upload
        with open(audio_path, 'rb') as f:
            data = {'file': f}
            async with session.post(url, data=data) as response:
                res = await response.json()
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Mock audio duration for now, replace with actual extraction
                audio_duration = res.get("audio_duration", 5.0) 
                
                results_list.append({
                    "audio_path": audio_path,
                    "processing_time": processing_time,
                    "audio_duration": audio_duration,
                    "rtf": processing_time / audio_duration,
                    "status": response.status
                })
    except Exception as e:
        print(f"Request failed: {e}")

async def run_load_test(concurrency, num_requests, url, audio_paths):
    async with aiohttp.ClientSession() as session:
        tasks = []
        results = []
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_request(audio_path):
            async with semaphore:
                await send_request(session, url, audio_path, results)

        start_time = time.time()
        for i in range(num_requests):
            audio_path = audio_paths[i % len(audio_paths)]
            tasks.append(asyncio.create_task(bounded_request(audio_path)))

        await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        return results, total_time

def main():
    parser = argparse.ArgumentParser(description="Concurrent Load Testing for ASR")
    parser.add_argument("--url", type=str, default="http://localhost:8000/transcribe", help="Server URL")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing eval audio files")
    parser.add_argument("--concurrency_levels", type=str, default="1,2,4,8,16,32,64", help="Comma-separated concurrency levels")
    parser.add_argument("--requests_per_level", type=int, default=100, help="Total requests per concurrency level")
    parser.add_argument("--output", type=str, default="../results/loadtest_results.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    audio_paths = [str(p) for p in Path(args.audio_dir).glob("*.wav")]
    if not audio_paths:
        print(f"No .wav files found in {args.audio_dir}")
        return

    concurrency_levels = [int(c) for c in args.concurrency_levels.split(",")]
    all_results = {}

    for c in concurrency_levels:
        print(f"\n--- Running load test with concurrency {c} ---")
        results, wall_time = asyncio.run(run_load_test(c, args.requests_per_level, args.url, audio_paths))
        
        rtfs = [r["rtf"] for r in results if "rtf" in r]
        
        if not rtfs:
            print("No successful requests.")
            continue
            
        avg_rtf = np.mean(rtfs)
        p50_rtf = np.percentile(rtfs, 50)
        p95_rtf = np.percentile(rtfs, 95)
        throughput = sum(r["audio_duration"] for r in results) / wall_time
        
        print(f"Avg RTF: {avg_rtf:.3f}")
        print(f"P50 RTF: {p50_rtf:.3f}")
        print(f"P95 RTF: {p95_rtf:.3f}")
        print(f"Throughput (audio sec / wall sec): {throughput:.2f}")
        
        all_results[c] = {
            "avg_rtf": avg_rtf,
            "p50_rtf": p50_rtf,
            "p95_rtf": p95_rtf,
            "throughput": throughput,
            "wall_time": wall_time,
            "raw_results": results
        }
        
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
