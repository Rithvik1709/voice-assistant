#!/usr/bin/env python3
"""Performance budget assertion script for VoiceAssistant benchmarks."""
from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Assert performance budget based on benchmark results")
    parser.add_argument("--res-file", default="bench_results.json", help="Path to json results file")
    parser.add_argument("--budget", type=float, default=300.0, help="Performance budget for net p95 in ms")
    parser.add_argument("--active-streaming-time", type=float, default=200.0, help="Active client streaming time in ms to subtract")
    
    args = parser.parse_args()
    
    try:
        with open(args.res_file, encoding="utf-8") as f:
            res = json.load(f)
            
        summary = res.get("summary", {})
        first_audio_ms = summary.get("first_audio_ms", {})
        p95_first = first_audio_ms.get("p95", 0.0)
        throughput = summary.get("throughput_responses_per_sec", 0.0)
        
        net_p95_processing = max(0.0, p95_first - args.active_streaming_time)
        
        print("=== PERFORMANCE METRICS ===")
        print(f"Throughput: {throughput:.2f} resp/sec")
        print(f"Raw p95 First Audio Latency: {p95_first:.2f}ms")
        print(f"Net p95 ASR/TTS Processing Latency: {net_p95_processing:.2f}ms (Budget: {args.budget:.2f}ms)")
        print("=============================")
        
        if net_p95_processing >= args.budget:
            print(
                f"FAILURE: Net p95 ASR/TTS processing latency ({net_p95_processing:.2f}ms) "
                f"exceeds budget of {args.budget:.2f}ms!",
                file=sys.stderr,
            )
            sys.exit(1)
            
        print("SUCCESS: All performance metrics are within the performance budget!")
        sys.exit(0)
        
    except Exception as e:
        print(f"FAILURE: Error running performance assertion: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
