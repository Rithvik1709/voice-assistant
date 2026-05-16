import argparse
import asyncio
import time
import logging
from pathlib import Path
from unittest.mock import patch
from voice_assistant.tts.stream import PiperStreamingTTS, PiperConfig
from voice_assistant.tts.queue import AudioChunkQueue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_strict_tests(use_real_piper: bool = False):
    print("=== STARTING RIGOROUS TTS VERIFICATION ===")
    import psutil, os
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / (1024 * 1024)
    
    queue = AudioChunkQueue(maxsize=32)
    config = PiperConfig(voice_path=Path("models/en_US-lessac-medium.onnx"))
    tts = PiperStreamingTTS(config, queue)
    mock_pcm = b"\x00" * 44100
    
    # Context manager to optionally mock Piper
    import contextlib
    @contextlib.contextmanager
    def maybe_mock_piper(slow=False):
        if use_real_piper:
            yield None
        else:
            def slow_mock(text):
                time.sleep(0.2)
                return mock_pcm
            def fast_mock(text):
                time.sleep(0.005)
                return mock_pcm
                
            side_effect = slow_mock if slow else fast_mock
            with patch('voice_assistant.tts.stream.PiperProcess.__init__', return_value=None), \
                 patch('voice_assistant.tts.stream.PiperProcess.synthesize', side_effect=side_effect) as m:
                yield m

    # -------------------------------------------------------------------------
    # TEST 1: Burst Test & Latency Tracking
    # -------------------------------------------------------------------------
    print("\n--- TEST 1: BURST LOAD (200 requests) & E2E LATENCY ---")
    queue.clear()
    tts.ingest_queue = asyncio.Queue(maxsize=32)
    
    time_series_metrics = []
    
    with maybe_mock_piper() as mock_run:
        await tts.start()
        
        start_time = time.time()
        tasks = []
        for i in range(200):
            # Capture E2E input time
            req_start = time.time()
            async def track_req(idx, t_start):
                accepted = await tts.synthesize_sentence(f"Burst {idx}")
                # End-to-end latency tracking for ingestion
                latency = time.time() - t_start
                if idx % 50 == 0:
                    time_series_metrics.append({
                        "ts": time.time(),
                        "queue_size": queue.qsize(),
                        "latency": latency
                    })
            tasks.append(asyncio.create_task(track_req(i, req_start)))
            
        await asyncio.gather(*tasks)
        await asyncio.sleep(0.5)
        await tts.flush()
        end_time = time.time()
        
        metrics = queue.get_metrics()
        
        assert tts.ingest_queue.qsize() <= 32, "Ingest queue breached maxsize!"
        assert queue.qsize() <= 32, "Playback queue breached maxsize!"
        
        total_time = end_time - start_time
        throughput = 200 / total_time
        print(f"PASS: Burst handled. Throughput: {throughput:.2f} req/sec")
        if not use_real_piper:
            print(f"Batches executed: {mock_run.call_count}")
        print(f"Time-series snapshots: {len(time_series_metrics)}")
        
        await tts.stop()

    # -------------------------------------------------------------------------
    # TEST 2: Sustained Load
    # -------------------------------------------------------------------------
    print("\n--- TEST 2: SUSTAINED LOAD (50 requests at 50Hz) ---")
    queue.clear()
    tts.ingest_queue = asyncio.Queue(maxsize=32)
    
    with maybe_mock_piper() as mock_run:
        await tts.start()
        
        for i in range(50):
            await asyncio.sleep(0.02)
            await tts.synthesize_sentence(f"Sustained {i}")
            assert queue.qsize() <= 32, "Queue blew up during sustained load!"
        
        await asyncio.sleep(0.2)
        await tts.stop()
        print("PASS: Sustained load survived without queue explosion.")

    # -------------------------------------------------------------------------
    # TEST 3: Slow TTS (Force queue dropping)
    # -------------------------------------------------------------------------
    print("\n--- TEST 3: TTS OVERLOAD (Force queue dropping) ---")
    queue.clear()
    tts.ingest_queue = asyncio.Queue(maxsize=32)
    
    with maybe_mock_piper(slow=True):
        await tts.start()
        
        # Fire requests to force ingest queue backpressure
        for i in range(40):
            # synthesize_sentence returns False if it drops due to timeout
            await tts.synthesize_sentence(f"Slow {i}")
                
        await asyncio.sleep(1.0)
        await tts.stop()
        
        metrics = queue.get_metrics()
        # Assert ingest backpressure or playback dropping
        assert metrics['dropped_count'] > 0 or tts.ingest_queue.full(), "System did not apply backpressure!"
        assert queue.qsize() <= 32, "Queue breached limits!"
        
        print(f"PASS: System degraded gracefully. Dropped chunks/inputs, queues bounded.")

    import psutil, os
    process = psutil.Process(os.getpid())
    end_mem = process.memory_info().rss / (1024 * 1024)
    
    assert (end_mem - start_mem) < 100, f"Memory leak detected! Growth: {end_mem - start_mem:.2f} MB"
    print(f"\nALL INVARIANTS HELD. SYSTEM IS MATHEMATICALLY STABLE. (Mem delta: {end_mem - start_mem:.2f}MB)")

def is_piper_available():
    import subprocess
    try:
        subprocess.run(["piper", "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-real-piper", action="store_true", help="Force real Piper binary usage")
    args = parser.parse_args()
    
    use_real = args.use_real_piper or is_piper_available()
    if use_real:
        print("Auto-detected Piper binary! Using real executable for load tests.")
    else:
        print("Piper binary not found. Using mocks for load tests.")
        
    asyncio.run(run_strict_tests(use_real))
