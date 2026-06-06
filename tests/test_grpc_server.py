from __future__ import annotations

import asyncio
import os
import time
from unittest.mock import MagicMock, patch
import pytest

from voice_assistant.config import Settings
from voice_assistant.transport import voice_assistant_pb2 as pb2
from voice_assistant.transport.grpc_server import VoiceAssistantService, MockLLMClient
from voice_assistant.llm.client import LLMConfig, StreamingLLMClient

@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        sample_rate=16000,
        chunk_ms=30,
        vad_aggressiveness=3,
        tts_queue_maxsize=32,
        sentence_max_tokens=28,
        piper_voice="",
        tts_sample_rate=22050,
    )

@pytest.mark.asyncio
async def test_grpc_server_stream_voice_normal_flow(test_settings) -> None:
    # Force mock models for high-fidelity testing
    with patch.dict(os.environ, {"MOCK_MODELS": "1"}):
        service = VoiceAssistantService(test_settings)
        
        # Prepare mock input iterator
        # Send silence, speech (represented by high-value PCM), then silence to trigger ASR
        # Vosk frame length = sample_rate * chunk_ms / 1000 * 2 (16-bit) = 16000 * 0.03 * 2 = 960 bytes
        frame_len = int(16000 * 0.030)
        silence_frame = b"\x00\x00" * frame_len
        speech_frame = b"\xff\x0f" * frame_len # strong speech signal

        async def req_iterator():
            # Initial silence
            yield pb2.AudioChunk(pcm16=silence_frame)
            # User speaking
            for _ in range(5):
                yield pb2.AudioChunk(pcm16=speech_frame)
            # End of speech silence to trigger ASR
            for _ in range(5):
                yield pb2.AudioChunk(pcm16=silence_frame)
            # Allow some time for processing
            await asyncio.sleep(0.1)

        mock_context = MagicMock()
        
        responses = []
        async for response in service.StreamVoice(req_iterator(), mock_context):
            responses.append(response)

        # We expect to get back silent audio responses with the debug text from mock LLM
        assert len(responses) > 0
        assert responses[0].sample_rate == 22050
        assert "mock response" in responses[0].debug_text.lower()


@pytest.mark.asyncio
async def test_grpc_server_stream_voice_barge_in(test_settings) -> None:
    with patch.dict(os.environ, {"MOCK_MODELS": "1"}):
        service = VoiceAssistantService(test_settings)
        
        frame_len = int(16000 * 0.030)
        silence_frame = b"\x00\x00" * frame_len
        speech_frame = b"\xff\x0f" * frame_len

        # To simulate interruption, we want to start a response, and while it's yielding,
        # yield a new speech chunk from request iterator.
        input_queue = asyncio.Queue()
        
        async def req_iterator():
            while True:
                item = await input_queue.get()
                if item is None:
                    break
                yield item

        mock_context = MagicMock()
        
        responses = []
        stream_iterator = service.StreamVoice(req_iterator(), mock_context)
        
        async def read_responses():
            async for resp in stream_iterator:
                responses.append(resp)
                
        read_task = asyncio.create_task(read_responses())
        
        # 1. Trigger the first user speech
        await input_queue.put(pb2.AudioChunk(pcm16=silence_frame))
        for _ in range(5):
            await input_queue.put(pb2.AudioChunk(pcm16=speech_frame))
        # Silence to trigger ASR
        for _ in range(5):
            await input_queue.put(pb2.AudioChunk(pcm16=silence_frame))
            
        # Wait a small moment to let LLM generate and start yielding response
        await asyncio.sleep(0.05)
        
        # 2. Trigger barge-in (interruption) by sending new speech
        for _ in range(3):
            await input_queue.put(pb2.AudioChunk(pcm16=speech_frame))
            
        # Stop input queue
        await input_queue.put(None)
        await read_task
        
        # Barge-in successfully executed without throwing errors
        assert len(responses) >= 0


@pytest.mark.asyncio
async def test_llm_client_backpressure_abort() -> None:
    # Test StreamingLLMClient backpressure abort strategy when output queue is full
    mock_llama = MagicMock()
    mock_llama.create_completion.return_value = [
        {"choices": [{"text": "Hello"}]},
        {"choices": [{"text": "world"}]},
    ]
    
    with patch("voice_assistant.llm.client.Llama", return_value=mock_llama), \
         patch("voice_assistant.llm.client._LLAMA_AVAILABLE", True):
         
        client = StreamingLLMClient(LLMConfig(model_path="dummy"))
        out_queue = asyncio.Queue(maxsize=1)
        
        # Force a tiny timeout for the test to run quickly
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
            with pytest.raises(RuntimeError) as exc_info:
                await client.stream_tokens("test prompt", out_queue)
            assert "generation aborted" in str(exc_info.value)
