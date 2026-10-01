from __future__ import annotations

import asyncio
import fractions
import json
import os

import numpy as np
import pytest

aiortc = pytest.importorskip("aiortc")
pytest.importorskip("aiohttp")

import av  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription  # noqa: E402

from voice_assistant.config import Settings  # noqa: E402
from voice_assistant.transport.webrtc import FRAME_SAMPLES, WEBRTC_RATE, SpeechTrack, WebServer  # noqa: E402

# These open local sockets (an HTTP server, WebRTC over loopback), so they only
# run when asked: CI sets VAANI_NETWORK_TESTS=1.
needs_network = pytest.mark.skipif(
    os.getenv("VAANI_NETWORK_TESTS") != "1", reason="opens local sockets; set VAANI_NETWORK_TESTS=1"
)


class SpeakingMicrophone(MediaStreamTrack):
    """A browser microphone: half a second of 'speech' (loud tone), then silence."""

    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        self.pts = 0

    async def recv(self):
        await asyncio.sleep(0.02)
        loud = self.pts < WEBRTC_RATE // 2
        t = (self.pts + np.arange(FRAME_SAMPLES)) / WEBRTC_RATE
        samples = (np.sin(2 * np.pi * 300 * t) * (8000 if loud else 0)).astype(np.int16)
        frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = WEBRTC_RATE
        frame.pts = self.pts
        frame.time_base = fractions.Fraction(1, WEBRTC_RATE)
        self.pts += FRAME_SAMPLES
        return frame


async def test_speech_track_resamples_and_clears() -> None:
    track = SpeechTrack()
    track.push(b"\x10\x00" * 22_050, 22_050)  # one second at Piper's rate

    assert track.buffered_s == pytest.approx(1.0, abs=0.01)
    frame = await track.recv()
    assert frame.samples == FRAME_SAMPLES and frame.sample_rate == WEBRTC_RATE

    track.clear()
    assert track.buffered_s == 0
    silent = await track.recv()
    assert not silent.to_ndarray().any()
    track.stop()


@needs_network
async def test_page_is_served(monkeypatch) -> None:
    monkeypatch.setenv("MOCK_MODELS", "1")
    server = WebServer(Settings())
    async with TestClient(TestServer(server.app())) as client:
        res = await client.get("/")
        assert res.status == 200
        assert "<title>Vaani</title>" in await res.text()

        bad = await client.post("/offer", data="nope")
        assert bad.status == 400


@needs_network
async def test_browser_conversation_over_webrtc(monkeypatch) -> None:
    import aioice.ice

    # Connect over loopback only: no network interfaces, no firewall prompts.
    monkeypatch.setattr(aioice.ice, "get_host_addresses", lambda use_ipv4, use_ipv6: ["127.0.0.1"])
    monkeypatch.setenv("MOCK_MODELS", "1")
    monkeypatch.setenv("ASR_ENDPOINT_SILENCE_MS", "200")
    server = WebServer(Settings())

    async with TestClient(TestServer(server.app())) as client:
        pc = RTCPeerConnection()
        events: list[dict] = []
        received_frames = 0
        channel = pc.createDataChannel("events")
        channel.on("message", lambda message: events.append(json.loads(message)))

        @pc.on("track")
        def on_track(track):
            async def drain():
                nonlocal received_frames
                while True:
                    try:
                        await track.recv()
                    except Exception:
                        return
                    received_frames += 1

            asyncio.ensure_future(drain())

        pc.addTrack(SpeakingMicrophone())
        await pc.setLocalDescription(await pc.createOffer())
        res = await client.post("/offer", json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
        assert res.status == 200
        answer = await res.json()
        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))

        async def conversation_done() -> None:
            while not any(e["type"] == "reply" and "mock response" in e["text"] for e in events):
                await asyncio.sleep(0.05)

        try:
            await asyncio.wait_for(conversation_done(), timeout=15)
        finally:
            await pc.close()

        transcripts = [e["text"] for e in events if e["type"] == "transcript"]
        assert transcripts and transcripts[0] == "tell me something interesting"
        assert received_frames > 10  # Vaani's audio track is flowing
        await asyncio.sleep(0.2)
    assert server.peers == set()
