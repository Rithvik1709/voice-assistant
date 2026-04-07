from __future__ import annotations

import asyncio
from collections import deque

from voice_assistant.asr.vad import VoiceActivityDetector
from voice_assistant.tts.player import AudioPlayer


class BargeInController:
    def __init__(self, vad: VoiceActivityDetector, interrupt_event: asyncio.Event, required_frames: int = 3) -> None:
        self.vad = vad
        self.interrupt_event = interrupt_event
        self.required_frames = required_frames
        self._frames: deque[bytes] = deque(maxlen=required_frames)

    def push_frame(self, frame: bytes) -> bool:
        self._frames.append(frame)
        if len(self._frames) < self.required_frames:
            return False
        if self.vad.detect_barge_in(self._frames):
            self.interrupt_event.set()
            return True
        return False

    async def watch_and_interrupt(self, player: AudioPlayer, frame_stream: asyncio.Queue[bytes]) -> None:
        while True:
            frame = await frame_stream.get()
            if self.push_frame(frame):
                player.interrupt()
