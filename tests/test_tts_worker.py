from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import numpy as np

from voice_assistant.tts.player import AudioPlayer, resample_linear
from voice_assistant.tts.queue import AudioChunk, AudioChunkQueue
from voice_assistant.tts.stream import PiperConfig, PiperStreamingTTS


class FakeSynth:
    def __init__(self, delay: float = 0.0, fail_on: str | None = None) -> None:
        self.delay = delay
        self.fail_on = fail_on
        self.calls: list[str] = []
        self.release = threading.Event()
        self.release.set()

    def synthesize(self, text: str) -> bytes:
        self.calls.append(text)
        self.release.wait(2)
        time.sleep(self.delay)
        if self.fail_on and self.fail_on in text:
            raise RuntimeError("synthesis exploded")
        return b"\x01\x00" * 100

    def close(self) -> None:
        pass


def make_tts(synth: FakeSynth) -> tuple[PiperStreamingTTS, AudioChunkQueue]:
    q = AudioChunkQueue(maxsize=64)
    tts = PiperStreamingTTS(PiperConfig(voice_path=Path("missing.onnx")), q)
    tts._synth = synth
    return tts, q


async def test_sentences_become_audio_after_flush() -> None:
    synth = FakeSynth()
    tts, q = make_tts(synth)
    await tts.start()

    assert await tts.synthesize_sentence("One.")
    assert await tts.synthesize_sentence("Two.")
    await asyncio.wait_for(tts.flush(), 2)
    await tts.stop()

    assert " ".join(synth.calls) == "One. Two."
    assert q.qsize() >= 1
    assert (await q.get()).sample_rate == 22050


async def test_worker_survives_synthesis_error() -> None:
    synth = FakeSynth(fail_on="bad")
    tts, q = make_tts(synth)
    await tts.start()

    await tts.synthesize_sentence("bad sentence.")
    await asyncio.wait_for(tts.flush(), 2)
    await tts.synthesize_sentence("good sentence.")
    await asyncio.wait_for(tts.flush(), 2)
    await tts.stop()

    assert q.qsize() == 1


async def test_cancel_pending_discards_in_flight_audio() -> None:
    synth = FakeSynth()
    synth.release.clear()
    tts, q = make_tts(synth)
    await tts.start()

    await tts.synthesize_sentence("first.")
    while not synth.calls:
        await asyncio.sleep(0.005)
    await tts.synthesize_sentence("queued.")
    tts.cancel_pending()
    synth.release.set()
    await asyncio.wait_for(tts.flush(), 2)
    await tts.stop()

    assert synth.calls == ["first."]
    assert q.empty()


async def test_full_ingest_queue_returns_false_instead_of_raising(monkeypatch) -> None:
    tts, _ = make_tts(FakeSynth())
    tts.ingest_queue = asyncio.Queue(maxsize=1)
    tts.ingest_queue.put_nowait("occupied")

    async def instant_timeout(coro, timeout):
        coro.close()
        raise TimeoutError

    monkeypatch.setattr("voice_assistant.tts.stream.asyncio.wait_for", instant_timeout)
    assert await tts.synthesize_sentence("dropped.") is False


def _render(player: AudioPlayer, frames: int) -> np.ndarray:
    out = np.zeros((frames, 1), dtype=np.float32)
    player._callback(out, frames, None, None)
    return out[:, 0]


async def test_player_keeps_playing_after_interrupt() -> None:
    player = AudioPlayer(sample_rate=16000)
    loud = AudioChunk(pcm16=(np.ones(64, dtype=np.int16) * 1000).tobytes(), sample_rate=16000)

    await player.play(loud)
    assert player.is_playing
    player.interrupt()
    assert not player.is_playing
    assert not _render(player, 32).any()

    await player.play(loud)
    assert _render(player, 32).any()


def test_resample_changes_length_proportionally() -> None:
    audio = np.linspace(-1, 1, 2205, dtype=np.float32)

    assert len(resample_linear(audio, 22050, 16000)) == 1600
    assert resample_linear(audio, 16000, 16000) is audio


async def test_markdown_is_removed_before_synthesis() -> None:
    synth = FakeSynth()
    tts, _ = make_tts(synth)
    await tts.start()

    await tts.synthesize_sentence("**Great** question! 😀")
    await asyncio.wait_for(tts.flush(), 2)
    await tts.stop()

    assert synth.calls == ["Great question!"]
