from __future__ import annotations

import shutil
import socket
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


def run_doctor(settings: Settings, check_audio: bool = True) -> list[DoctorCheck]:
    checks = [
        DoctorCheck("python_config", True, f"{settings.sample_rate} Hz, {settings.chunk_ms} ms chunks"),
        _path_check("llm_model", settings.model_path),
        _path_check("asr_model", settings.asr_model_path, settings.asr_backend in {"vosk", "whispercpp"}),
        _path_check("piper_voice", settings.piper_voice),
        _path_check("piper_config", f"{settings.piper_voice}.json" if settings.piper_voice else ""),
        DoctorCheck("piper_binary", shutil.which("piper") is not None, shutil.which("piper") or "not found on PATH"),
        _port_check(settings.grpc_port),
    ]

    if check_audio:
        checks.append(_audio_check())

    return checks


def format_doctor_report(checks: list[DoctorCheck]) -> str:
    lines = ["Vaani setup doctor"]
    for check in checks:
        marker = "OK" if check.ok else "FAIL"
        lines.append(f"[{marker}] {check.name}: {check.detail}")
    return "\n".join(lines)


def has_failures(checks: list[DoctorCheck]) -> bool:
    return any(not check.ok for check in checks)
