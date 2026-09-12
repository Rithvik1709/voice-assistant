from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(slots=True)
class Settings:
    sample_rate: int = 16_000
    channels: int = 1
    chunk_ms: int = field(default_factory=lambda: _env_int("CHUNK_MS", 20))
    chunk_size: int = 320
    vad_aggressiveness: int = field(default_factory=lambda: _env_int("VAD_AGGRESSIVENESS", 2))
    vad_speech_frames_trigger: int = 3
    asr_endpoint_silence_ms: int = field(default_factory=lambda: _env_int("ASR_ENDPOINT_SILENCE_MS", 60))
    ack_tone_ms: int = field(default_factory=lambda: _env_int("ACK_TONE_MS", 55))
    enable_ack_tone: bool = field(default_factory=lambda: _env_bool("ENABLE_ACK_TONE", True))

    model_path: str = field(default_factory=lambda: os.getenv("MODEL_PATH", ""))
    draft_model_path: str = field(default_factory=lambda: os.getenv("DRAFT_MODEL_PATH", ""))
    piper_voice: str = field(default_factory=lambda: os.getenv("PIPER_VOICE", ""))
    asr_model_path: str = field(default_factory=lambda: os.getenv("ASR_MODEL_PATH", ""))

    asr_backend: str = field(default_factory=lambda: os.getenv("ASR_BACKEND", "vosk"))
    quant_level: str = field(default_factory=lambda: os.getenv("QUANT_LEVEL", "Q4_K_M"))
    n_gpu_layers: int = field(default_factory=lambda: _env_int("N_GPU_LAYERS", -1))

    grpc_port: int = field(default_factory=lambda: _env_int("GRPC_PORT", 50051))

    llm_max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 256))
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.7))
    llm_context_size: int = field(default_factory=lambda: _env_int("LLM_CONTEXT_SIZE", 4096))
    assistant_system_prompt: str = field(
        default_factory=lambda: os.getenv(
            "ASSISTANT_SYSTEM_PROMPT",
            "You are Vaani, a concise voice assistant. Answer clearly in one or two short sentences unless the user asks for detail.",
        )
    )

    # Maximum number of user-assistant conversation turns
    # retained in memory before older history is pruned.
    conversation_history_turns: int = field(
        default_factory=lambda: _env_int("CONVERSATION_HISTORY_TURNS", 10)
    )
    conversation_memory_path: str = field(default_factory=lambda: os.getenv("CONVERSATION_MEMORY_PATH", ""))

    tts_sample_rate: int = 22_050
    sentence_max_tokens: int = field(default_factory=lambda: _env_int("TTS_SENTENCE_MAX_TOKENS", 8))
    tts_eager_min_words: int = field(default_factory=lambda: _env_int("TTS_EAGER_MIN_WORDS", 3))
    tts_queue_maxsize: int = 6
    player_blocksize: int = field(default_factory=lambda: _env_int("PLAYER_BLOCKSIZE", 128))
    llm_queue_maxsize: int = 128
    asr_queue_maxsize: int = 32

    topic_similarity_threshold: float = field(default_factory=lambda: _env_float("TOPIC_SIMILARITY_THRESHOLD", 0.55))

    def __post_init__(self) -> None:
        profile = os.getenv("VAANI_PROFILE", "").strip().lower()
        if profile in {"fast", "low_latency", "turbo"}:
            self.chunk_ms = min(self.chunk_ms, 20)
            self.asr_endpoint_silence_ms = min(self.asr_endpoint_silence_ms, 50)
            self.llm_max_tokens = min(self.llm_max_tokens, 128)
            self.sentence_max_tokens = min(self.sentence_max_tokens, 6)
            self.tts_eager_min_words = min(self.tts_eager_min_words, 2)
            self.player_blocksize = min(self.player_blocksize, 96)

        self.chunk_size = int(self.sample_rate * self.chunk_ms / 1000)

    def validate(self) -> None:
        if os.getenv("MOCK_MODELS") == "1":
            return

        required = {
            "MODEL_PATH": self.model_path,
            "PIPER_VOICE": self.piper_voice,
        }
        if self.asr_backend in {"vosk", "whispercpp"}:
            required["ASR_MODEL_PATH"] = self.asr_model_path

        missing = [k for k, v in required.items() if not v]

        if missing:
            raise ValueError(
                f"Missing required environment values: {', '.join(missing)}"
            )

        missing_paths = [
            f"{name}={value}"
            for name, value in required.items()
            if value and not Path(value).expanduser().exists()
        ]
        if missing_paths:
            raise ValueError(
                "Configured model paths do not exist: "
                + ", ".join(missing_paths)
            )

        if shutil.which("piper") is None:
            raise ValueError(
                "Piper executable not found on PATH. Install piper-tts or add "
                "the Piper binary to PATH."
            )

        piper_config = Path(f"{self.piper_voice}.json").expanduser()
        if not piper_config.exists():
            raise ValueError(
                f"Missing Piper voice config file: {piper_config}"
            )

    @property
    def piper_voice_path(self) -> Path:
        return Path(self.piper_voice).expanduser().resolve()
