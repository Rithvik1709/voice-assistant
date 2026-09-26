from __future__ import annotations

import importlib.util
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from voice_assistant.config import Settings


@dataclass(slots=True)
class DoctorCheck:
    name: str
    ok: bool
    detail: str


def _path_check(name: str, value: str, required: bool = True) -> DoctorCheck:
    if not value:
        return DoctorCheck(name, not required, "not configured")

    path = Path(value).expanduser()
    if path.exists():
        return DoctorCheck(name, True, str(path))

    return DoctorCheck(name, False, f"missing: {path}")


def _port_check(port: int) -> DoctorCheck:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        result = sock.connect_ex(("127.0.0.1", port))

    if result == 0:
        return DoctorCheck("grpc_port", False, f"port {port} is already in use")

    return DoctorCheck("grpc_port", True, f"port {port} is available")


def _audio_check() -> DoctorCheck:
    try:
        import sounddevice as sd

        device = sd.query_devices(kind="input")
    except Exception as exc:
        return DoctorCheck("audio_input", False, f"unavailable: {exc}")

    name = str(device.get("name", "default input"))
    return DoctorCheck("audio_input", True, name)


def _module_check(name: str, module: str, hint: str, required: bool = True) -> DoctorCheck:
    try:
        found = importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        found = False
    if found:
        return DoctorCheck(name, True, f"{module} importable")
    return DoctorCheck(name, not required, f"{module} not installed ({hint})")


def _piper_check() -> DoctorCheck:
    from voice_assistant.tts.stream import piper_python_available

    if piper_python_available():
        return DoctorCheck("piper", True, "piper-tts Python package (in-process synthesis)")
    binary = shutil.which("piper")
    if binary:
        return DoctorCheck("piper", True, f"CLI binary {binary}")
    return DoctorCheck("piper", False, "not found: pip install piper-tts, or add the piper binary to PATH")


def _piper_smoke_check(voice: str) -> DoctorCheck | None:
    """Synthesize one word in a subprocess to prove Piper actually works.

    Broken installs (e.g. wheels with a missing espeak-ng data path) can abort
    the interpreter, so this must not run in-process.
    """
    from voice_assistant.tts.stream import piper_python_available

    if not voice or not Path(voice).expanduser().exists():
        return None
    if piper_python_available():
        cmd = [sys.executable, "-m", "piper"]
    elif shutil.which("piper"):
        cmd = ["piper"]
    else:
        return None
    try:
        proc = subprocess.run(
            [*cmd, "--model", str(Path(voice).expanduser()), "--output_raw"],
            input=b"test\n",
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return DoctorCheck("piper_synthesis", False, f"could not run Piper: {exc}")
    if proc.returncode == 0 and len(proc.stdout) > 1000:
        return DoctorCheck("piper_synthesis", True, f"synthesized {len(proc.stdout) // 2} samples")
    err = proc.stderr.decode("utf-8", errors="replace").strip().splitlines()
    return DoctorCheck("piper_synthesis", False, err[-1] if err else f"no audio produced (exit {proc.returncode})")


def _llm_server_check(settings: Settings) -> DoctorCheck:
    """Ask an OpenAI-compatible server for its model list."""
    import json
    import urllib.error
    import urllib.request

    url = settings.llm_base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"} if settings.llm_api_key else {}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=5) as response:
            data = json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        return DoctorCheck("llm_server", False, f"{url} returned HTTP {exc.code}")
    except (OSError, ValueError) as exc:
        return DoctorCheck("llm_server", False, f"cannot reach {url}: {exc}")
    models = [m.get("id", "") for m in data.get("data", []) if isinstance(m, dict)]
    if models and settings.llm_model not in models:
        return DoctorCheck("llm_server", False, f"model {settings.llm_model!r} not served; available: {', '.join(models[:8])}")
    return DoctorCheck("llm_server", True, f"{settings.llm_base_url} serving {settings.llm_model}")


def _config_check(settings: Settings) -> DoctorCheck:
    problems = settings.check_ranges()
    if problems:
        return DoctorCheck("config", False, "; ".join(problems))
    barge = f"barge-in after {settings.barge_in_ms} ms" if settings.enable_barge_in else "barge-in off (half-duplex)"
    return DoctorCheck(
        "config",
        True,
        f"{settings.sample_rate} Hz, {settings.chunk_ms} ms chunks, "
        f"{settings.asr_endpoint_silence_ms} ms endpoint, {barge}",
    )


def run_doctor(settings: Settings, check_audio: bool = True) -> list[DoctorCheck]:
    local_hint = "pip install 'voice-assistant[local]'"
    checks = [
        _config_check(settings),
        *(
            [_llm_server_check(settings)]
            if settings.llm_backend == "openai"
            else [_module_check("llm_runtime", "llama_cpp", local_hint), _path_check("llm_model", settings.model_path)]
        ),
        (
            _module_check("asr_runtime", "faster_whisper", "pip install 'voice-assistant[whisper]'")
            if settings.asr_backend == "whisper"
            else _module_check("asr_runtime", "vosk", local_hint, settings.asr_backend == "vosk")
        ),
        _piper_check(),
        (
            DoctorCheck("asr_model", True, f"whisper {settings.asr_model_path or 'base.en'}")
            if settings.asr_backend == "whisper" and not Path(settings.asr_model_path).expanduser().exists()
            else _path_check("asr_model", settings.asr_model_path, settings.asr_backend in {"vosk", "whispercpp"})
        ),
        _path_check("piper_voice", settings.piper_voice),
        _path_check("piper_config", f"{settings.piper_voice}.json" if settings.piper_voice else ""),
        _port_check(settings.grpc_port),
    ]

    smoke = _piper_smoke_check(settings.piper_voice)
    if smoke is not None:
        checks.append(smoke)

    if check_audio:
        checks.append(_audio_check())

    return checks


def format_doctor_report(checks: list[DoctorCheck]) -> str:
    lines = ["Vaani setup doctor"]
    for check in checks:
        marker = "OK" if check.ok else "FAIL"
        lines.append(f"[{marker}] {check.name}: {check.detail}")
    failures = sum(1 for check in checks if not check.ok)
    lines.append("All checks passed." if not failures else f"{failures} check(s) failed.")
    return "\n".join(lines)


def has_failures(checks: list[DoctorCheck]) -> bool:
    return any(not check.ok for check in checks)
