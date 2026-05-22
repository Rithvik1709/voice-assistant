"""Load tester for the VoiceAssistant StreamVoice gRPC endpoint.

Usage examples:
    python scripts/bench/load_asr.py --concurrency 10 --frames-per-client 30

This script will:
- start N concurrent clients
- each client streams synthetic audio frames (simulated speech)
- each client records time-to-first-audio-response and time-to-last-audio-response
- compute p50/p95/p99 and throughput
- write JSON results to --out-file
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from typing import Any

import grpc

from voice_assistant.transport import voice_assistant_pb2 as pb2
from voice_assistant.transport import voice_assistant_pb2_grpc as pb2_grpc


DEFAULT_SAMPLE_RATE = 16000
DEFAULT_FRAME_MS = 100


class ClientResult:
    def __init__(self) -> None:
        self.request_start: float = 0.0
        self.first_audio_ts: float | None = None
        self.last_audio_ts: float | None = None
        self.audio_responses: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_start": self.request_start,
            "first_audio_ts": self.first_audio_ts,
            "last_audio_ts": self.last_audio_ts,
            "audio_responses": self.audio_responses,
        }


async def audio_generator(frames_per_client: int, frame_ms: int, sample_rate: int) -> pb2.AudioChunk:
    """Async generator producing synthetic 'speech' frames followed by a short non-speech frame.

    Each speech frame will be `frame_ms` long of a high-energy pattern encoded as 16-bit PCM.
    """
    frame_samples = int(sample_rate * (frame_ms / 1000.0))
    # 16-bit PCM high energy wave pattern to trigger VAD
    frame_bytes = (b"\x12\x34" * frame_samples)

    # Send `frames_per_client` speech frames
    for _ in range(frames_per_client):
        yield pb2.AudioChunk(pcm16=frame_bytes, sample_rate=sample_rate, timestamp_ms=int(time.time() * 1000))
        await asyncio.sleep(frame_ms / 1000.0)

    # Send one short (non-empty) frame shorter than vad.frame_bytes to indicate end-of-speech
    yield pb2.AudioChunk(pcm16=b"\x00", sample_rate=sample_rate, timestamp_ms=int(time.time() * 1000))


async def run_client(
    host: str,
    port: int,
    frames_per_client: int,
    frame_ms: int,
    sample_rate: int,
    timeout_s: float,
) -> ClientResult:
    result = ClientResult()
    target = f"{host}:{port}"
    async with grpc.aio.insecure_channel(target) as channel:
        stub = pb2_grpc.VoiceAssistantStub(channel)

        async def gen() -> pb2.AudioChunk:
            async for chunk in audio_generator(frames_per_client, frame_ms, sample_rate):
                yield chunk

        call = stub.StreamVoice(gen())
        result.request_start = time.monotonic()

        try:
            async for resp in call:
                now = time.monotonic()
                result.audio_responses += 1
                if result.first_audio_ts is None:
                    result.first_audio_ts = now
                result.last_audio_ts = now
        except asyncio.CancelledError:
            raise
        except Exception:
            # In stress tests the server may cut connections; record what we have.
            pass

    return result


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    vals = sorted(values)
    def q(p: float) -> float:
        idx = int(p * (len(vals) - 1))
        return vals[idx]
    return {"p50": q(0.50), "p95": q(0.95), "p99": q(0.99)}


async def run_benchmark(
    host: str,
    port: int,
    concurrency: int,
    frames_per_client: int,
    frame_ms: int,
    sample_rate: int,
    timeout_s: float,
) -> dict[str, Any]:
    tasks = []
    for _ in range(concurrency):
        tasks.append(
            asyncio.create_task(run_client(host, port, frames_per_client, frame_ms, sample_rate, timeout_s))
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    client_results: list[ClientResult] = []
    for r in results:
        if isinstance(r, ClientResult):
            client_results.append(r)

    first_latencies = []
    last_latencies = []
    total_audio = 0
    durations = []

    for cr in client_results:
        total_audio += cr.audio_responses
        if cr.first_audio_ts is not None:
            first_latencies.append((cr.first_audio_ts - cr.request_start) * 1000.0)
        if cr.last_audio_ts is not None:
            last_latencies.append((cr.last_audio_ts - cr.request_start) * 1000.0)
        # approximate client duration
        if cr.last_audio_ts is not None:
            durations.append(cr.last_audio_ts - cr.request_start)

    first_q = quantiles(first_latencies)
    last_q = quantiles(last_latencies)

    total_time = sum(durations) if durations else 0.0
    throughput = (total_audio / total_time) if total_time > 0 else 0.0

    summary = {
        "concurrency": concurrency,
        "clients_reported": len(client_results),
        "first_audio_ms": first_q,
        "last_audio_ms": last_q,
        "total_audio_responses": total_audio,
        "throughput_responses_per_sec": throughput,
    }

    return {"summary": summary, "clients": [c.to_dict() for c in client_results]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--frames-per-client", type=int, default=30)
    parser.add_argument("--frame-ms", type=int, default=DEFAULT_FRAME_MS)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--out-file", default="bench_asr_results.json")

    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    out = loop.run_until_complete(
        run_benchmark(
            args.host,
            args.port,
            args.concurrency,
            args.frames_per_client,
            args.frame_ms,
            args.sample_rate,
            args.timeout,
        )
    )

    with open(args.out_file, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()
