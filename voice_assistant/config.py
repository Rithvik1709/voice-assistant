from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(slots=True)
class Settings:
    sample_rate: int = 16_000
    channels: int = 1
    chunk_ms: int = int(os.getenv("CHUNK_MS", "20"))
    chunk_size: int = 320
    vad_aggressiveness: int = int(os.getenv("VAD_AGGRESSIVENESS", "2"))
    vad_speech_frames_trigger: int = 3
    asr_endpoint_silence_ms: int = int(os.getenv("ASR_ENDPOINT_SILENCE_MS", "60"))

    model_path: str = os.getenv("MODEL_PATH", "")
    draft_model_path: str = os.getenv("DRAFT_MODEL_PATH", "")
    piper_voice: str = os.getenv("PIPER_VOICE", "")
    asr_model_path: str = os.getenv("ASR_MODEL_PATH", "")

    asr_backend: str = os.getenv("ASR_BACKEND", "vosk")
    quant_level: str = os.getenv("QUANT_LEVEL", "Q4_K_M")
    n_gpu_layers: int = int(os.getenv("N_GPU_LAYERS", "-1"))

    grpc_port: int = int(os.getenv("GRPC_PORT", "50051"))

    llm_max_tokens: int = 256
    llm_temperature: float = 0.7
    llm_context_size: int = 4096

    tts_sample_rate: int = 22_050
    sentence_max_tokens: int = int(os.getenv("TTS_SENTENCE_MAX_TOKENS", "8"))
    tts_eager_min_words: int = int(os.getenv("TTS_EAGER_MIN_WORDS", "3"))
    tts_queue_maxsize: int = 6
    player_blocksize: int = int(os.getenv("PLAYER_BLOCKSIZE", "128"))
    llm_queue_maxsize: int = 128
    asr_queue_maxsize: int = 32

    topic_similarity_threshold: float = 0.55

    def __post_init__(self) -> None:
        self.chunk_size = int(self.sample_rate * self.chunk_ms / 1000)

    def validate(self) -> None:
     required = {
        "MODEL_PATH": self.model_path,
        "PIPER_VOICE": self.piper_voice,
        "ASR_MODEL_PATH": self.asr_model_path,
    }

     missing = [k for k, v in required.items() if not v]

     if missing:
        raise ValueError(
            f"Missing required environment values: {', '.join(missing)}"
        )

     path_checks = {
        "MODEL_PATH": self.model_path,
        "PIPER_VOICE": self.piper_voice,
        "ASR_MODEL_PATH": self.asr_model_path,
    }

     for name, path in path_checks.items():
        if not Path(path).expanduser().exists():
            raise FileNotFoundError(
                f"{name} does not exist: {path}"
            )

     numeric_positive_checks = {
        "sample_rate": self.sample_rate,
        "chunk_ms": self.chunk_ms,
        "chunk_size": self.chunk_size,
        "grpc_port": self.grpc_port,
        "tts_sample_rate": self.tts_sample_rate,
        "sentence_max_tokens": self.sentence_max_tokens,
        "tts_eager_min_words": self.tts_eager_min_words,
        "tts_queue_maxsize": self.tts_queue_maxsize,
        "player_blocksize": self.player_blocksize,
        "llm_queue_maxsize": self.llm_queue_maxsize,
        "asr_queue_maxsize": self.asr_queue_maxsize,
    }

     for name, value in numeric_positive_checks.items():
        if value <= 0:
            raise ValueError(
                f"{name} must be greater than 0, got {value}"
            )

     if not 0 <= self.vad_aggressiveness <= 3:
        raise ValueError(
            "vad_aggressiveness must be between 0 and 3"
        )

     if not 0.0 <= self.topic_similarity_threshold <= 1.0:
        raise ValueError(
            "topic_similarity_threshold must be between 0.0 and 1.0"
        )

     if not 1024 <= self.grpc_port <= 65535:
        raise ValueError(
            f"grpc_port must be between 1024 and 65535, got {self.grpc_port}"
        )

    @property
    def piper_voice_path(self) -> Path:
        return Path(self.piper_voice).expanduser().resolve()
