# Voice Assistant Benchmark Results

Date: 2026-08-16

Benchmark command used:

```bash
PYTHONPATH=. python3 scripts/bench/load_asr.py --host 127.0.0.1 --port 50051 --concurrency 10 --frames-per-client 30 --frame-ms 30 --sample-rate 16000 --out-file bench_asr_results.json
```

Environment note: the benchmark was executed against the mock gRPC service (`MOCK_MODELS=1`) to validate the streaming pipeline locally.

| Metric | Value | Notes |
| --- | ---: | --- |
| Concurrency | 10 | Concurrent client streams |
| Clients reported | 10 | Successful benchmark clients |
| First audio latency (p50) | 1008.65 ms | Median time to first audio response |
| First audio latency (p95) | 1009.66 ms | 95th percentile |
| First audio latency (p99) | 1009.66 ms | 99th percentile |
| Last audio latency (p50) | 1013.81 ms | Median end-of-stream response time |
| Last audio latency (p95) | 1014.78 ms | 95th percentile |
| Last audio latency (p99) | 1014.78 ms | 99th percentile |
| Total audio responses | 20 | Combined responses produced |
| Throughput | 1.97 responses/sec | Aggregate throughput |

## Summary

This run shows the project’s mock streaming pipeline responding in roughly 1.0–1.1 seconds for first audio delivery under a 10-client load, with a sustained throughput of about 1.97 responses/sec.
