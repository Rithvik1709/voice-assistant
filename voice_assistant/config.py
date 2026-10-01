from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from voice_assistant.lang import AUTO, LANGUAGE_NAMES, parse_language_list

load_dotenv()


class ConfigError(ValueError):
    """Raised for invalid or incomplete configuration."""


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def mock_models_enabled() -> bool:
    return os.getenv("MOCK_MODELS") == "1"


@dataclass(slots=True)
class Settings:
    sample_rate: int = 16_000
    channels: int = 1
    chunk_ms: int = field(default_factory=lambda: _env_int("CHUNK_MS", 20))
    chunk_size: int = 320
    vad_aggressiveness: int = field(default_factory=lambda: _env_int("VAD_AGGRESSIVENESS", 2))
    vad_speech_frames_trigger: int = 3
    # Trailing silence that ends an utterance. Natural pauses between words
    # are often 100-250ms, so values much lower than this split sentences.
    asr_endpoint_silence_ms: int = field(default_factory=lambda: _env_int("ASR_ENDPOINT_SILENCE_MS", 400))
    # Longer silence allowed when the words so far look unfinished ("and ...", "the ...").
    asr_hold_silence_ms: int = field(default_factory=lambda: _env_int("ASR_HOLD_SILENCE_MS", 1000))
    ack_tone_ms: int = field(default_factory=lambda: _env_int("ACK_TONE_MS", 55))
    enable_ack_tone: bool = field(default_factory=lambda: _env_bool("ENABLE_ACK_TONE", True))

    # Barge-in lets the user interrupt the assistant by speaking. Without
    # headphones or echo cancellation the assistant can hear itself, so it can
    # be disabled; the microphone is then muted while the assistant speaks.
    enable_barge_in: bool = field(default_factory=lambda: _env_bool("ENABLE_BARGE_IN", True))
    barge_in_ms: int = field(default_factory=lambda: _env_int("BARGE_IN_MS", 240))

    # "llama" runs a local GGUF file in-process; "openai" talks to any
    # OpenAI-compatible server (Ollama, LM Studio, vLLM, llama.cpp server, hosted APIs).
    llm_backend: str = field(default_factory=lambda: os.getenv("LLM_BACKEND", "llama").strip().lower())
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", ""))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""), repr=False)
    model_path: str = field(default_factory=lambda: os.getenv("MODEL_PATH", ""))
    draft_model_path: str = field(default_factory=lambda: os.getenv("DRAFT_MODEL_PATH", ""))
    piper_voice: str = field(default_factory=lambda: os.getenv("PIPER_VOICE", ""))
    # A folder of Piper voices, one per language, picked by the language of
    # each reply. PIPER_VOICE stays the default voice.
    piper_voices_dir: str = field(default_factory=lambda: os.getenv("PIPER_VOICES_DIR", ""))
    # What speaks a language that has no Piper voice: "espeak" (espeak-ng,
    # robotic but 100+ languages), "default" (the default voice), "none"
    # (stay silent), or "auto" (espeak-ng when installed, else the default voice).
    tts_fallback: str = field(default_factory=lambda: os.getenv("TTS_FALLBACK", "auto").strip().lower())
    asr_model_path: str = field(default_factory=lambda: os.getenv("ASR_MODEL_PATH", ""))

    # vosk: fast, streaming partials, weaker accuracy. whisper: faster-whisper,
    # far more accurate, transcribes each utterance at the endpoint.
    asr_backend: str = field(default_factory=lambda: os.getenv("ASR_BACKEND", "whisper").strip().lower())
    # A language code ("en", "hi", "ta") or "auto" to detect it on every
    # utterance; replies then follow the language the user spoke.
    asr_language: str = field(default_factory=lambda: os.getenv("ASR_LANGUAGE", "en").strip().lower() or "en")
    # With "auto": the languages detection may choose from. Short utterances
    # are easily misdetected, so listing the expected ones helps a lot.
    asr_languages: tuple[str, ...] = field(
        default_factory=lambda: parse_language_list(os.getenv("ASR_LANGUAGES", ""))
    )
    quant_level: str = field(default_factory=lambda: os.getenv("QUANT_LEVEL", "Q4_K_M"))
    n_gpu_layers: int = field(default_factory=lambda: _env_int("N_GPU_LAYERS", -1))

    grpc_port: int = field(default_factory=lambda: _env_int("GRPC_PORT", 50051))

    llm_max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 256))
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.7))
    llm_context_size: int = field(default_factory=lambda: _env_int("LLM_CONTEXT_SIZE", 4096))
    assistant_system_prompt: str = field(
        default_factory=lambda: os.getenv(
            "ASSISTANT_SYSTEM_PROMPT",
            "You are Vaani, a concise voice assistant. Your replies are spoken aloud, so answer in one or two short plain sentences unless the user asks for detail, and never use markdown, lists, code blocks or emoji.",
        )
    )

    # Maximum number of user-assistant conversation turns
    # retained in memory before older history is pruned.
    conversation_history_turns: int = field(
        default_factory=lambda: _env_int("CONVERSATION_HISTORY_TURNS", 10)
    )
    conversation_memory_path: str = field(default_factory=lambda: os.getenv("CONVERSATION_MEMORY_PATH", ""))

    # Live weather via Open-Meteo. Off by default: it contacts the internet.
    enable_weather: bool = field(default_factory=lambda: _env_bool("ENABLE_WEATHER", False))
    weather_default_city: str = field(default_factory=lambda: os.getenv("WEATHER_DEFAULT_CITY", ""))
    weather_units: str = field(default_factory=lambda: os.getenv("WEATHER_UNITS", "celsius"))

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
            self.asr_endpoint_silence_ms = min(self.asr_endpoint_silence_ms, 300)
            self.llm_max_tokens = min(self.llm_max_tokens, 128)
            self.sentence_max_tokens = min(self.sentence_max_tokens, 6)
            self.tts_eager_min_words = min(self.tts_eager_min_words, 2)
            self.player_blocksize = min(self.player_blocksize, 96)

        self.chunk_size = int(self.sample_rate * self.chunk_ms / 1000)

    def check_ranges(self) -> list[str]:
        """Return human-readable problems with numeric settings."""
        problems: list[str] = []
        if self.chunk_ms not in {10, 20, 30}:
            problems.append(f"CHUNK_MS must be 10, 20 or 30 (WebRTC VAD frame sizes), got {self.chunk_ms}")
        if not 0 <= self.vad_aggressiveness <= 3:
            problems.append(f"VAD_AGGRESSIVENESS must be between 0 and 3, got {self.vad_aggressiveness}")
        if self.asr_endpoint_silence_ms < self.chunk_ms:
            problems.append(
                f"ASR_ENDPOINT_SILENCE_MS must be at least one chunk ({self.chunk_ms}ms), "
                f"got {self.asr_endpoint_silence_ms}"
            )
        if self.asr_hold_silence_ms < self.asr_endpoint_silence_ms:
            problems.append(
                f"ASR_HOLD_SILENCE_MS ({self.asr_hold_silence_ms}) must be at least "
                f"ASR_ENDPOINT_SILENCE_MS ({self.asr_endpoint_silence_ms})"
            )
        if self.asr_backend not in {"vosk", "whisper", "whispercpp"}:
            problems.append(f"ASR_BACKEND must be 'vosk', 'whisper' or 'whispercpp', got {self.asr_backend!r}")
        problems.extend(self._language_problems())
        if self.llm_backend not in {"llama", "openai"}:
            problems.append(f"LLM_BACKEND must be 'llama' or 'openai', got {self.llm_backend!r}")
        if self.llm_max_tokens <= 0:
            problems.append(f"LLM_MAX_TOKENS must be positive, got {self.llm_max_tokens}")
        if self.llm_context_size <= 0:
            problems.append(f"LLM_CONTEXT_SIZE must be positive, got {self.llm_context_size}")
        if not 0.0 <= self.llm_temperature <= 2.0:
            problems.append(f"LLM_TEMPERATURE must be between 0 and 2, got {self.llm_temperature}")
        if self.sentence_max_tokens <= 0:
            problems.append(f"TTS_SENTENCE_MAX_TOKENS must be positive, got {self.sentence_max_tokens}")
        if self.barge_in_ms < self.chunk_ms:
            problems.append(f"BARGE_IN_MS must be at least one chunk ({self.chunk_ms}ms), got {self.barge_in_ms}")
        if self.weather_units.lower() not in {"celsius", "fahrenheit"}:
            problems.append(f"WEATHER_UNITS must be 'celsius' or 'fahrenheit', got {self.weather_units!r}")
        if not 0 < self.grpc_port < 65536:
            problems.append(f"GRPC_PORT must be a valid TCP port, got {self.grpc_port}")
        return problems

    def _language_problems(self) -> list[str]:
        problems: list[str] = []
        if self.asr_language != AUTO and self.asr_language not in LANGUAGE_NAMES:
            problems.append(f"ASR_LANGUAGE must be 'auto' or a language code such as 'en' or 'hi', got {self.asr_language!r}")
        unknown = [code for code in self.asr_languages if code not in LANGUAGE_NAMES]
        if unknown:
            problems.append(f"ASR_LANGUAGES has unknown language codes: {', '.join(unknown)}")
        if self.asr_language == AUTO and self.asr_backend != "whisper":
            problems.append(f"ASR_LANGUAGE=auto needs ASR_BACKEND=whisper; {self.asr_backend} models know one language")
        if self.asr_backend == "whisper" and self.asr_language != "en" and self.asr_model_path.rstrip("/").endswith(".en"):
            problems.append(
                f"ASR_MODEL_PATH {self.asr_model_path!r} is an English-only Whisper model; "
                "use a multilingual one (e.g. large-v3-turbo or small) for other languages"
            )
        if self.tts_fallback not in {"auto", "espeak", "default", "none"}:
            problems.append(f"TTS_FALLBACK must be 'auto', 'espeak', 'default' or 'none', got {self.tts_fallback!r}")
        return problems

    @property
    def multilingual(self) -> bool:
        """Whether replies may be in a language other than English."""
        return self.asr_language != "en"

    def validate(self, need_asr: bool = True, need_tts: bool = True) -> None:
        problems = self.check_ranges()
        if problems:
            raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(problems))

        if mock_models_enabled():
            return

        if self.llm_backend == "openai":
            required = {"LLM_BASE_URL": self.llm_base_url, "LLM_MODEL": self.llm_model}
        else:
            required = {"MODEL_PATH": self.model_path}
        if need_tts and not self.piper_voices_dir:
            required["PIPER_VOICE"] = self.piper_voice
        # Whisper takes a size name ("base.en") or a directory and has a default.
        if need_asr and self.asr_backend in {"vosk", "whispercpp"}:
            required["ASR_MODEL_PATH"] = self.asr_model_path

        missing = [k for k, v in required.items() if not v]

        if missing:
            raise ConfigError(
                f"Missing required environment values: {', '.join(missing)}. "
                "Run `vaani models --write-env --download` to fetch the open starter models."
            )

        paths = {k: v for k, v in required.items() if k not in {"LLM_BASE_URL", "LLM_MODEL"}}
        if need_tts:
            paths.update({"PIPER_VOICE": self.piper_voice, "PIPER_VOICES_DIR": self.piper_voices_dir})
        missing_paths = [
            f"{name}={value}"
            for name, value in paths.items()
            if value and not Path(value).expanduser().exists()
        ]
        if missing_paths:
            raise ConfigError(
                "Configured model paths do not exist: "
                + ", ".join(missing_paths)
            )

        if not need_tts:
            return

        from voice_assistant.tts.stream import piper_python_available

        if not piper_python_available() and shutil.which("piper") is None:
            raise ConfigError(
                "Piper is not installed. Run `pip install piper-tts` or add "
                "the Piper binary to PATH."
            )

        if self.piper_voice:
            piper_config = Path(f"{self.piper_voice}.json").expanduser()
            if not piper_config.exists():
                raise ConfigError(
                    f"Missing Piper voice config file: {piper_config}"
                )

        if self.piper_voices_dir:
            from voice_assistant.tts.voices import discover_voices

            if not discover_voices(Path(self.piper_voices_dir).expanduser()):
                raise ConfigError(
                    f"PIPER_VOICES_DIR {self.piper_voices_dir} has no Piper voices "
                    "(each needs a .onnx file and its .onnx.json config)"
                )

    @property
    def piper_voice_path(self) -> Path:
        return Path(self.piper_voice).expanduser().resolve()

    def piper_config(self):
        """TTS settings: the default voice, the per-language voices and the fallback."""
        from voice_assistant.tts.stream import PiperConfig

        return PiperConfig(
            self.piper_voice_path if self.piper_voice else None,
            self.tts_sample_rate,
            voices_dir=Path(self.piper_voices_dir).expanduser() if self.piper_voices_dir else None,
            fallback=self.tts_fallback,
        )

    def whisper_model(self) -> str:
        """The Whisper model to load: ASR_MODEL_PATH, or a default that knows the language."""
        from voice_assistant.asr.stream import default_whisper_model

        return self.asr_model_path or default_whisper_model(self.asr_language)

    def build_actions(self):
        """Intent actions configured from these settings."""
        from voice_assistant.actions import BasicIntentActions

        weather = None
        if self.enable_weather:
            from voice_assistant.weather import OpenMeteoWeather

            weather = OpenMeteoWeather(unit=self.weather_units)
        return BasicIntentActions(weather=weather, default_city=self.weather_default_city)

    @property
    def barge_in_frames(self) -> int:
        return max(1, self.barge_in_ms // self.chunk_ms)
