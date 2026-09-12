from __future__ import annotations

import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ModelAsset:
    name: str
    url: str
    target: str
    license: str
    source: str
    extract_to: str | None = None


RECOMMENDED_MODELS = [
    ModelAsset(
        name="LLM: Qwen2.5 0.5B Instruct GGUF Q4_K_M",
        url="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        target="Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
        license="Apache-2.0",
        source="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    ),
    ModelAsset(
        name="TTS: Piper en_US lessac medium",
        url="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        target="en_US-lessac-medium.onnx",
        license="MIT",
        source="rhasspy/piper-voices",
    ),
    ModelAsset(
        name="TTS config: Piper en_US lessac medium",
        url="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
        target="en_US-lessac-medium.onnx.json",
        license="MIT",
        source="rhasspy/piper-voices",
    ),
    ModelAsset(
        name="ASR: Vosk small English US",
        url="https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
        target="vosk-model-small-en-us-0.15.zip",
        license="Apache-2.0",
        source="Alpha Cephei Vosk models",
        extract_to="vosk-model-small-en-us-0.15",
    ),
]


def env_template(models_dir: Path) -> str:
    models = models_dir.as_posix()
    lines = [
        f'MODEL_PATH="{models}/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"',
        'DRAFT_MODEL_PATH=""',
        f'PIPER_VOICE="{models}/en_US-lessac-medium.onnx"',
        f'ASR_MODEL_PATH="{models}/vosk-model-small-en-us-0.15"',
        'ASR_BACKEND="vosk"',
        "VAD_AGGRESSIVENESS=2",
        "CHUNK_MS=20",
        "ASR_ENDPOINT_SILENCE_MS=60",
        "ACK_TONE_MS=55",
        "ENABLE_ACK_TONE=1",
        'VAANI_PROFILE="low_latency"',
        'ASSISTANT_SYSTEM_PROMPT="You are Vaani, a concise voice assistant. Answer clearly in one or two short sentences unless the user asks for detail."',
        "TTS_SENTENCE_MAX_TOKENS=8",
        "TTS_EAGER_MIN_WORDS=3",
        "PLAYER_BLOCKSIZE=128",
        "GRPC_PORT=50051",
        'CONVERSATION_MEMORY_PATH="data/session.jsonl"',
    ]
    return "\n".join(lines) + "\n"


def write_env_file(path: Path, models_dir: Path, overwrite: bool = False) -> bool:
    if path.exists() and not overwrite:
        return False

    path.write_text(env_template(models_dir), encoding="utf-8")
    return True


def format_model_plan(models_dir: Path) -> str:
    lines = ["Recommended open model set", f"Target directory: {models_dir}"]
    for asset in RECOMMENDED_MODELS:
        action = "download and extract" if asset.extract_to else "download"
        lines.append(f"- {asset.name}")
        lines.append(f"  {action}: {asset.target}")
        lines.append(f"  license: {asset.license}")
        lines.append(f"  source: {asset.source}")
    return "\n".join(lines)


def download_recommended_models(models_dir: Path) -> list[str]:
    models_dir.mkdir(parents=True, exist_ok=True)
    messages: list[str] = []

    for asset in RECOMMENDED_MODELS:
        target = models_dir / asset.target
        extract_to = models_dir / asset.extract_to if asset.extract_to else None

        if extract_to and extract_to.exists():
            messages.append(f"skip {asset.name}: {extract_to} already exists")
            continue
        if target.exists() and not extract_to:
            messages.append(f"skip {asset.name}: {target} already exists")
            continue

        urllib.request.urlretrieve(asset.url, target)
        messages.append(f"downloaded {asset.name}: {target}")

        if extract_to:
            with zipfile.ZipFile(target) as archive:
                archive.extractall(models_dir)
            messages.append(f"extracted {asset.name}: {extract_to}")

    return messages
