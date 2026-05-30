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
        if os.getenv("MOCK_MODELS") == "1":
            return
        required = {
            "MODEL_PATH": self.model_path,
            "PIPER_VOICE": self.piper_voice,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(f"Missing required environment values: {', '.join(missing)}")

    @property
    def piper_voice_path(self) -> Path:
        return Path(self.piper_voice).expanduser().resolve()
