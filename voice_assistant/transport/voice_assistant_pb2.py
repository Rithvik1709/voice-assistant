from __future__ import annotations

import base64
import json
from dataclasses import dataclass


@dataclass(slots=True)
class AudioChunk:
    pcm16: bytes = b""
    sample_rate: int = 16_000
    timestamp_ms: int = 0

    def SerializeToString(self) -> bytes:
        return json.dumps(
            {"pcm16": base64.b64encode(self.pcm16).decode("ascii"), "sample_rate": self.sample_rate, "timestamp_ms": self.timestamp_ms}
        ).encode("utf-8")

    @classmethod
    def FromString(cls, data: bytes) -> "AudioChunk":
        payload = json.loads(data.decode("utf-8"))
        return cls(
            pcm16=base64.b64decode(payload.get("pcm16", "")),
            sample_rate=int(payload.get("sample_rate", 16_000)),
            timestamp_ms=int(payload.get("timestamp_ms", 0)),
        )


@dataclass(slots=True)
class TextEvent:
    type: str = ""
    text: str = ""
    confidence: float = 0.0
    timestamp_ms: int = 0

    def SerializeToString(self) -> bytes:
        return json.dumps(
            {"type": self.type, "text": self.text, "confidence": self.confidence, "timestamp_ms": self.timestamp_ms}
        ).encode("utf-8")

    @classmethod
    def FromString(cls, data: bytes) -> "TextEvent":
        payload = json.loads(data.decode("utf-8"))
        return cls(
            type=payload.get("type", ""),
            text=payload.get("text", ""),
            confidence=float(payload.get("confidence", 0.0)),
            timestamp_ms=int(payload.get("timestamp_ms", 0)),
        )


@dataclass(slots=True)
class AudioResponse:
    pcm16: bytes = b""
    sample_rate: int = 22_050
    timestamp_ms: int = 0
    debug_text: str = ""

    def SerializeToString(self) -> bytes:
        return json.dumps(
            {
                "pcm16": base64.b64encode(self.pcm16).decode("ascii"),
                "sample_rate": self.sample_rate,
                "timestamp_ms": self.timestamp_ms,
                "debug_text": self.debug_text,
            }
        ).encode("utf-8")

    @classmethod
    def FromString(cls, data: bytes) -> "AudioResponse":
        payload = json.loads(data.decode("utf-8"))
        return cls(
            pcm16=base64.b64decode(payload.get("pcm16", "")),
            sample_rate=int(payload.get("sample_rate", 22_050)),
            timestamp_ms=int(payload.get("timestamp_ms", 0)),
            debug_text=payload.get("debug_text", ""),
        )
