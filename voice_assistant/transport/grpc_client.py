from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator

import grpc
import sounddevice as sd

from voice_assistant.tts.player import AudioPlayer
from voice_assistant.tts.queue import AudioChunk

try:
    from voice_assistant.transport import voice_assistant_pb2 as pb2
    from voice_assistant.transport import voice_assistant_pb2_grpc as pb2_grpc
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Protobuf stubs are missing. Run grpc_tools.protoc using voice_assistant/transport/voice_assistant.proto"
    ) from exc

logger = logging.getLogger(__name__)


class GRPCVoiceClient:
    def __init__(
        self,
        target: str,
        sample_rate: int = 16_000,
        chunk_size: int = 480,
        playback_rate: int = 22_050,
    ) -> None:
        self.target = target
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=256)
        # Server audio is resampled to this rate if the voice uses another one.
        self._player = AudioPlayer(sample_rate=playback_rate)
        self._loop: asyncio.AbstractEventLoop | None = None

    def _mic_callback(self, indata, frames, _time, _status) -> None:
        if frames <= 0:
            return
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self._put_audio_frame, bytes(indata))

    def _put_audio_frame(self, pcm16: bytes) -> None:
        try:
            self._audio_queue.put_nowait(pcm16)
        except asyncio.QueueFull:
            pass

    async def _request_stream(self) -> AsyncIterator[pb2.AudioChunk]:
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.chunk_size,
            callback=self._mic_callback,
        ):
            while True:
                pcm = await self._audio_queue.get()
                yield pb2.AudioChunk(pcm16=pcm, sample_rate=self.sample_rate, timestamp_ms=int(time.time() * 1000))

    async def handle_response(self, resp: pb2.AudioResponse) -> None:
        if resp.interrupt:
            # The user barged in: stop speaking the previous reply right away.
            self._player.interrupt()
            return
        if resp.transcript:
            print(f"You:   {resp.transcript}", flush=True)
            return
        if resp.debug_text and resp.debug_text != "[ack]":
            print(f"Vaani: {resp.debug_text}", flush=True)
        if resp.pcm16:
            await self._player.play(AudioChunk(pcm16=resp.pcm16, sample_rate=resp.sample_rate or self._player.sample_rate))

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        await self._player.start()
        logger.info("Connected to %s; start speaking. Press Ctrl+C to stop.", self.target)
        try:
            async with grpc.aio.insecure_channel(self.target) as channel:
                stub = pb2_grpc.VoiceAssistantStub(channel)
                async for resp in stub.StreamVoice(self._request_stream()):
                    await self.handle_response(resp)
        finally:
            await self._player.stop()
