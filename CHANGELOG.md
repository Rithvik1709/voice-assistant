# Changelog

## Vaani 1.1.0 - Unreleased

### Added

- Setup doctor mode for checking model paths, Piper, gRPC port availability, and audio input.
- Basic intent action hooks for deterministic responses before LLM fallback.
- Optional JSONL session memory through `CONVERSATION_MEMORY_PATH`.

### Verified

- `pytest -q`

## Vaani 1.0.0 - 2026-08-16

First stable open-source release baseline for Vaani.

### Added

- MIT license for open-source distribution.
- Release package metadata for Python 3.11+.
- Source distribution and wheel build validation.
- Packaged gRPC `.proto` file.

### Changed

- Local mode now wires Piper TTS with the correct playback queue argument.
- ASR and gRPC mic callbacks now hand audio frames into the asyncio loop safely.
- gRPC server mode now sends chat-message history to the LLM client.
- TTS flushing now waits for queued and batched text to be synthesized before returning.
- Configuration validation now fails early for missing model paths, ASR model paths, Piper voice config, and Piper executable.

### Verified

- `pytest -q`
- `python -m compileall -q voice_assistant tests`
- `python -m build`
- `python -m twine check dist/*`
