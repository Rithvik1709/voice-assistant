# Voice Assistant Benchmark Results

Date: 2026-09-26 (Vaani 2.0.0)

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
| First audible response latency (p50) | 961.51 ms | Median time to acknowledgement or content audio |
| First audible response latency (p95) | 964.51 ms | 95th percentile |
| First audible response latency (p99) | 964.51 ms | 99th percentile |
| Net first audible processing latency (p95) | 64.51 ms | Raw p95 minus active input streaming time |
| First content audio latency (p50) | 1011.92 ms | Median time to non-ack assistant audio |
| First content audio latency (p95) | 1012.77 ms | 95th percentile |
| Net first content processing latency (p95) | 112.77 ms | Raw p95 minus active input streaming time |
| Last audio latency (p50) | 1017.64 ms | Median end-of-stream response time |
| Last audio latency (p95) | 1019.15 ms | 95th percentile |
| Last audio latency (p99) | 1019.15 ms | 99th percentile |
| Ack responses | 10 | One acknowledgement per client stream |
| Total audio responses | 30 | Combined responses produced |
| Throughput | 2.95 responses/sec | Aggregate throughput |

## Summary

This run shows the mock streaming pipeline producing audible acknowledgement about 65 ms after end-of-speech at p95 under a 10-client load, and first content audio about 113 ms after end-of-speech at p95 (down from 70 ms and 122 ms in the 2026-09-12 run).

The 2.0.0 server encodes audio with real protobuf instead of JSON + base64, feeds audio to the recognizer frame by frame instead of decoding the whole utterance at the end, and runs LLM generation off the event loop.

At 50 concurrent clients (`--concurrency 50 --frames-per-client 2`, the CI configuration) the same comparison measured first-audio p95 of 215.6 ms versus 236.8 ms before, and first-content p95 of 262.8 ms versus 293.8 ms.

These are mock-model numbers: they measure transport, ASR framing and scheduling overhead, not model inference speed.
