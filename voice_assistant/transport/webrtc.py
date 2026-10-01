"""WebRTC transport: talk to Vaani from a browser.

`vaani web` serves a small web page and a signalling endpoint. The page
sends microphone audio over WebRTC and plays Vaani's voice from a WebRTC
audio track; a data channel carries what was heard and said, so the page
can show the conversation. The browser's own echo cancellation keeps Vaani
from hearing itself, so barge-in works on speakers.

Each browser tab gets its own conversation (a VoiceSession, the same one the
gRPC server uses), sharing the loaded models.
"""
from __future__ import annotations

import asyncio
import contextlib
import fractions
import json
import logging
import time
from importlib import resources
from typing import Any

import numpy as np

try:
    import av  # type: ignore
    from aiohttp import web  # type: ignore
    from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription  # type: ignore
    from aiortc.mediastreams import MediaStreamError  # type: ignore

    _WEBRTC_AVAILABLE = True
except ImportError:  # pragma: no cover - optional runtime import
    _WEBRTC_AVAILABLE = False
    MediaStreamTrack = object  # type: ignore[assignment,misc]

from voice_assistant.config import Settings

logger = logging.getLogger(__name__)

# WebRTC audio (Opus) runs at 48 kHz; frames are 20 ms.
WEBRTC_RATE = 48_000
FRAME_SAMPLES = WEBRTC_RATE // 50


def require_webrtc() -> None:
    if not _WEBRTC_AVAILABLE:
        raise RuntimeError("WebRTC mode needs aiortc and aiohttp. Install them with `pip install 'voice-assistant[webrtc]'`.")


def _pcm16_frame(pcm16: bytes, sample_rate: int) -> Any:
    frame = av.AudioFrame.from_ndarray(
        np.frombuffer(pcm16, dtype=np.int16).reshape(1, -1), format="s16", layout="mono"
    )
    frame.sample_rate = sample_rate
    return frame


class SpeechTrack(MediaStreamTrack):  # type: ignore[misc,valid-type]
    """The audio track Vaani speaks on: queued speech, or silence when there is none."""

    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        self._buffer = bytearray()
        self._resamplers: dict[int, Any] = {}
        self._start: float | None = None
        self._pts = 0

    def push(self, pcm16: bytes, sample_rate: int) -> None:
        """Queue 16-bit mono speech at any sample rate."""
        if not pcm16:
            return
        if sample_rate == WEBRTC_RATE:
            self._buffer.extend(pcm16)
            return
        resampler = self._resamplers.get(sample_rate)
        if resampler is None:
            resampler = self._resamplers[sample_rate] = av.AudioResampler(
                format="s16", layout="mono", rate=WEBRTC_RATE
            )
        for frame in resampler.resample(_pcm16_frame(pcm16, sample_rate)):
            self._buffer.extend(frame.to_ndarray().tobytes())

    def clear(self) -> None:
        """Drop queued speech immediately (barge-in)."""
        self._buffer.clear()
        self._resamplers.clear()

    @property
    def buffered_s(self) -> float:
        return len(self._buffer) / 2 / WEBRTC_RATE

    async def recv(self) -> Any:
        if self.readyState != "live":
            raise MediaStreamError
        # Pace frames in real time; the browser plays whatever arrives.
        if self._start is None:
            self._start = time.monotonic()
        else:
            delay = self._start + self._pts / WEBRTC_RATE - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)

        size = FRAME_SAMPLES * 2
        data = bytes(self._buffer[:size])
        del self._buffer[:size]
        frame = _pcm16_frame(data.ljust(size, b"\x00"), WEBRTC_RATE)
        frame.pts = self._pts
        frame.time_base = fractions.Fraction(1, WEBRTC_RATE)
        self._pts += FRAME_SAMPLES
        return frame


class WebPeer:
    """One browser connection: its peer connection, conversation and tasks."""

    def __init__(self, service: Any, pc: Any) -> None:
        from voice_assistant.transport.grpc_server import VoiceSession

        self.pc = pc
        self.session = VoiceSession(service)
        self.speech = SpeechTrack()
        self.channel: Any = None
        self.tasks: list[asyncio.Task[None]] = []
        self.session.on_partial = lambda text: self.send({"type": "partial", "text": text})
        self._closed = False

    def send(self, message: dict[str, Any]) -> None:
        channel = self.channel
        if channel is not None and channel.readyState == "open":
            channel.send(json.dumps(message, ensure_ascii=False))

    async def start(self) -> None:
        await self.session.start()
        self.tasks.append(asyncio.create_task(self._pump_responses()))

    def listen_to(self, track: Any) -> None:
        self.tasks.append(asyncio.create_task(self._consume_microphone(track)))

    async def _consume_microphone(self, track: Any) -> None:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=self.session.settings.sample_rate)
        while True:
            try:
                frame = await track.recv()
            except MediaStreamError:
                return
            for out in resampler.resample(frame):
                await self.session.feed(out.to_ndarray().tobytes())

    async def _pump_responses(self) -> None:
        while True:
            response = await self.session.responses.get()
            if response is None:
                return
            if response.interrupt:
                self.speech.clear()
                self.send({"type": "interrupt"})
            if response.transcript:
                self.send({"type": "transcript", "text": response.transcript})
            if response.pcm16:
                self.speech.push(response.pcm16, response.sample_rate)
                if response.debug_text and response.debug_text != "[ack]":
                    self.send({"type": "reply", "text": response.debug_text})

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.session.close()
        self.speech.stop()
        await self.pc.close()


class WebServer:
    """Serves the page and answers WebRTC offers."""

    def __init__(self, settings: Settings, service: Any | None = None) -> None:
        require_webrtc()
        if service is None:
            from voice_assistant.transport.grpc_server import VoiceAssistantService

            service = VoiceAssistantService(settings)
        self.settings = settings
        self.service = service
        self.peers: set[WebPeer] = set()

    def app(self) -> Any:
        app = web.Application()
        app.router.add_get("/", self.index)
        app.router.add_post("/offer", self.offer)
        app.on_shutdown.append(self._shutdown)
        return app

    async def index(self, _request: Any) -> Any:
        page = resources.files("voice_assistant.transport").joinpath("web/index.html").read_text(encoding="utf-8")
        return web.Response(text=page, content_type="text/html")

    async def offer(self, request: Any) -> Any:
        try:
            params = await request.json()
            description = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        except (ValueError, KeyError, TypeError):
            return web.json_response({"error": "expected JSON with sdp and type"}, status=400)

        pc = RTCPeerConnection()
        peer = WebPeer(self.service, pc)
        self.peers.add(peer)

        async def close_peer() -> None:
            self.peers.discard(peer)
            await peer.close()

        @pc.on("datachannel")
        def on_datachannel(channel: Any) -> None:
            peer.channel = channel

        @pc.on("track")
        def on_track(track: Any) -> None:
            if track.kind == "audio":
                peer.listen_to(track)

        @pc.on("connectionstatechange")
        async def on_state() -> None:
            logger.info("Browser connection %s", pc.connectionState)
            if pc.connectionState in {"failed", "closed"}:
                await close_peer()

        try:
            await peer.start()
            await pc.setRemoteDescription(description)
            pc.addTrack(peer.speech)
            await pc.setLocalDescription(await pc.createAnswer())
        except Exception:
            logger.exception("Could not answer the browser's offer")
            await close_peer()
            return web.json_response({"error": "could not start a session"}, status=500)
        return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

    async def _shutdown(self, _app: Any) -> None:
        peers, self.peers = list(self.peers), set()
        await asyncio.gather(*(peer.close() for peer in peers), return_exceptions=True)


async def serve_web(host: str, port: int, settings: Settings) -> None:
    from voice_assistant.llm.client import warm_up_llm

    server = WebServer(settings)
    await warm_up_llm(server.service.llm, settings.assistant_system_prompt)
    runner = web.AppRunner(server.app())
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    shown = "localhost" if host in {"127.0.0.1", "0.0.0.0", "::"} else host
    logger.info("Vaani web is running: open http://%s:%s in your browser", shown, port)
    if host not in {"127.0.0.1", "localhost", "::1"}:
        logger.info("Browsers only allow the microphone on localhost or HTTPS; put Vaani behind HTTPS for other devices")
    try:
        await asyncio.Event().wait()
    finally:
        with contextlib.suppress(Exception):
            await runner.cleanup()
