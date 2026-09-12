# Voice Assistant Benchmark Results

Date: 2026-09-12

Benchmark command used:

```bash
PYTHONPATH=. python scripts/bench/load_asr.py --host 127.0.0.1 --port 50051 --concurrency 10 --frames-per-client 30 --frame-ms 30 --sample-rate 16000 --out-file bench_asr_results.json
```

Environment note: the benchmark was executed against the mock gRPC service (`MOCK_MODELS=1`) to validate the streaming pipeline locally.

The clients stream 30 frames at 30 ms each, so roughly 900 ms of each raw latency is active input audio time. Net processing latency is calculated as `raw latency - 900 ms`.

| Metric | Value | Notes |
| --- | ---: | --- |
| Concurrency | 10 | Concurrent client streams |
| Clients reported | 10 | Successful benchmark clients |
| First audible response latency (p50) | 969.32 ms | Median time to acknowledgement or content audio |
| First audible response latency (p95) | 970.30 ms | 95th percentile |
| First audible response latency (p99) | 970.30 ms | 99th percentile |
| Net first audible processing latency (p95) | 70.30 ms | Raw p95 minus active input streaming time |
| First content audio latency (p50) | 1020.94 ms | Median time to non-ack assistant audio |
| First content audio latency (p95) | 1022.50 ms | 95th percentile |
| Net first content processing latency (p95) | 122.50 ms | Raw p95 minus active input streaming time |
| Last audio latency (p50) | 1025.63 ms | Median end-of-stream response time |
| Last audio latency (p95) | 1027.27 ms | 95th percentile |
| Last audio latency (p99) | 1027.27 ms | 99th percentile |
| Ack responses | 10 | One acknowledgement per client stream |
| Total audio responses | 30 | Combined responses produced |
| Throughput | 2.92 responses/sec | Aggregate throughput |

## Summary

This run shows the mock streaming pipeline producing audible acknowledgement about 70 ms after end-of-speech at p95 under a 10-client load. First content audio arrives about 122 ms after end-of-speech at p95, while throughput rises from the previous 1.97 responses/sec baseline to 2.92 responses/sec because the server now streams acknowledgement plus content events.
