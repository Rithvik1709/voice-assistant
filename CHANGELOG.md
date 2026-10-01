# Changelog

## Unreleased

### Added

- **Echo cancellation.** Vaani's own voice is removed from the microphone with WebRTC's AEC3 (through the LiveKit SDK), using everything it plays as the reference, so barge-in works on laptop speakers without headphones. `ECHO_CANCELLATION` is `auto` (on when installed), `on` or `off`. It is in the `[local]` extra, or `[aec]` on its own.
- **Wake word.** `WAKE_WORD="hey vaani"` answers only requests that start with the phrase (fuzzy-matched, and given to Whisper as hotwords). Saying just the wake word plays a chime and waits for the request, and follow-ups within `WAKE_WORD_FOLLOW_UP_S` need no wake word. `WAKE_WORD_MODEL` listens acoustically with openWakeWord instead (`[wakeword]` extra), so speech recognition only runs after the wake word.
- **Long-term memory.** With `USER_FACTS_PATH`, facts you mention ("my name is…", "I live in…", "I'm vegetarian", "I love cricket") are kept across sessions in a local JSON file and added to the system prompt. "Remember that…", "what do you know about me?", "forget that…" and "forget everything about me" manage them.
- **Browser client.** `vaani web` serves a page that streams your voice to Vaani over WebRTC and plays its reply, with the conversation and live transcripts on screen (`[webrtc]` extra). It replaces the unused WebRTC stub.
- **Faster Whisper turns and live transcripts.** Whisper starts decoding `ASR_EARLY_DECODE_MS` (150 ms) into a pause, and the endpoint reuses that result when you did not speak again, so the decode no longer adds to the response time. Periodic decodes (`ASR_PARTIAL_INTERVAL_MS`) give live partial transcripts, which also make the unfinished-sentence hold (`ASR_HOLD_SILENCE_MS`) work with Whisper.
- `vaani doctor` reports echo cancellation, the wake word and long-term memory.

- **Multilingual conversations.** `ASR_LANGUAGE=auto` detects the language of every utterance with Whisper (99 languages), asks the LLM to reply in it, and speaks the reply with that language's Piper voice. `ASR_LANGUAGES` limits detection to the languages you speak.
- `PIPER_VOICES_DIR`: a folder of Piper voices, one per language, loaded on first use. `PIPER_VOICE` stays the default voice.
- `TTS_FALLBACK` for languages without a Piper voice: espeak-ng, the default voice, or silence.
- `vaani models --languages hi,ta,ja` downloads multilingual Whisper `large-v3-turbo` and a Piper voice per language, and `--write-env` configures them.
- `vaani doctor` reports the configured languages and any without a voice.
- Text chat guesses the language from the writing system (Tamil, Devanagari, Han, kana, Hangul, Thai and others).

### Changed

- Sentences split at `।`, `。`, `！`, `？`, `؟`, `۔` and other non-Latin full stops, and Chinese and Japanese are chunked by characters, so speech starts early in those languages too.
- Speech cleanup only turns symbols into English words ("40%" to "40 percent") for English; other languages keep the symbol for their voice to read.
- Time, date, weather and greeting shortcuts skip requests spoken in languages other than English and Hindi, so the LLM answers them in the user's language.
- With no `ASR_MODEL_PATH`, Whisper loads `large-v3-turbo` for any language other than English (still `base.en` for English). faster-whisper 1.1 or newer is required.

## Vaani 2.0.0 - 2026-09-26

### Upgrading from 1.0

- **Install:** the model runtimes are now an extra. Use `pip install -e ".[local]"` for voice mode; a plain `pip install -e .` gives the core only (mock mode, gRPC server, benchmarks).
- **gRPC clients:** messages are now real protobuf, as defined in `voice_assistant.proto`. Clients built against the 1.0 JSON stubs must regenerate their stubs. `AudioResponse` gained `interrupt` and `transcript`; clients should drop buffered audio on `interrupt`.
- **Speech recognition:** Whisper is the default. To keep Vosk, set `ASR_BACKEND=vosk` and point `ASR_MODEL_PATH` at the Vosk model folder. Run `vaani models --download` to fetch Whisper `base.en`.
- **Endpointing:** `ASR_ENDPOINT_SILENCE_MS` now defaults to 400 (was 60). An explicit 60 in an old `.env` will still split sentences at natural pauses; remove it or raise it.
- **Tracing:** spans are no longer printed by default. Set `OTEL_EXPORTER_OTLP_ENDPOINT`, or `VAANI_TRACE_CONSOLE=1`, and install the `[otel]` extra.

### Fixed

- Speech recognition runs off the event loop in local mode, so a slow decode cannot stall playback.
- LLM generation no longer blocks the event loop. Tokens are produced in a worker thread, so microphone capture, TTS, and playback keep running while the model generates, and a reply can be cancelled at the next token.
- Barge-in now works in local mode. It was implemented but never wired up: speaking while the assistant talks cancels generation, drops queued speech, and silences playback.
- Playback no longer stays silent forever after an interruption (the player latched an `interrupted` flag that was never cleared).
- Piper TTS works with current Piper releases. The old code passed `--json-input` (not supported by `piper-tts`) and parsed a WAV header from stdout, so no audio was produced. Vaani now synthesizes in-process through `piper-tts`, or runs the Piper CLI with `--output_raw`.
- The TTS sample rate is read from the voice config instead of being assumed to be 22.05 kHz.
- Vosk transcripts no longer lose words when Kaldi detects an internal endpoint mid-utterance.
- Utterances are no longer split at every short pause: the default endpoint silence is 400 ms (300 ms with the low-latency profile) instead of 60 ms, frames inside an utterance are fed to the recognizer, and a short pre-roll keeps the first syllable.
- A full TTS queue returns `False` so the orchestrator's retry logic runs, instead of raising and stopping the pipeline.
- An LLM error speaks a short apology and the assistant keeps listening, instead of the whole pipeline exiting.
- `GRPC_PORT` is honoured; the `--port` default used to override it.
- The gRPC server ends utterances after real trailing silence (it used to end them at the first non-speech frame), checks every frame of a chunk instead of only the first, and answers the pending utterance when the client half-closes.
- Speculative decoding scored draft tokens against a context that duplicated already-accepted tokens, and could exceed `max_new_tokens`.
- `token_logprobs` ran one identical completion per candidate; it now runs one.
- Any Devanagari sentence without a keyword was classified as a greeting and answered "Hi, I am listening."; greetings inside longer questions and "play" in "how do I play chess" no longer trigger canned replies.
- `pip install -e .` works on macOS (the `vosk>=0.3.45` pin had no macOS wheels, and `numpy<2` blocked Python 3.13).

### Added

- OpenAI-compatible LLM backend (`LLM_BACKEND=openai`, `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`) for Ollama, LM Studio, vLLM, llama.cpp server, or hosted models, with streaming, cancellation, warm-up, and a doctor reachability check.
- faster-whisper speech recognition, now the default (`ASR_BACKEND=whisper`, `ASR_LANGUAGE`), shared across gRPC streams; `vaani models --download` fetches Whisper `base.en` instead of Vosk. Vosk remains available with `ASR_BACKEND=vosk`. On test questions it transcribed 5 of 5 correctly against 2 of 5 for Vosk small, at about 300 ms per utterance on CPU.
- Smarter endpointing: when the words so far end on a word like "and", "the" or "um", Vaani waits up to `ASR_HOLD_SILENCE_MS` before ending the turn.
- `vaani chat`: text chat through the full assistant pipeline (actions, LLM, memory), with `--speak` to also hear replies, `/reset` and `/quit` commands, and piped input. Needs only the LLM model, or no models with `MOCK_MODELS=1`.
- Speech normalization before TTS: markdown emphasis, headings, bullets, code blocks, links, URLs and emoji are removed, and symbols such as `21°C`, `40%`, `&` and `$1,200` are spoken as words. The default system prompt now tells the model its replies are spoken.
- Live weather answers from Open-Meteo (no API key), opt-in with `ENABLE_WEATHER=1`, with city extraction from English and Hinglish requests, `WEATHER_DEFAULT_CITY`, and `WEATHER_UNITS`.
- LLM warm-up on the system prompt at startup, so the first question does not pay for processing it.
- gRPC `AudioResponse.transcript`, and the bundled client prints the conversation.
- Graceful gRPC shutdown on `SIGTERM`.
- Session memory files are compacted once they exceed 2000 messages.
- `vaani` console command, with modes as positional arguments (`vaani doctor`), `--version`, and `--log-level`.
- gRPC `AudioResponse.interrupt`, sent when the user barges in so clients drop buffered audio.
- Half-duplex mode (`ENABLE_BARGE_IN=0`) that mutes the microphone while the assistant speaks, and `BARGE_IN_MS`.
- Time and date intents answered deterministically.
- Per-turn latency metrics logged in local and server modes.
- Doctor checks for installed runtimes, configuration ranges, and a real Piper synthesis smoke test.
- Configuration validation with readable errors for malformed or out-of-range values.
- CI job running ruff, the unit tests on Python 3.11 and 3.12, and a protobuf stub freshness check.
- Tests for barge-in, the ASR endpointing state machine, non-blocking LLM streaming, the TTS worker, player, speculative decoding, and gRPC interruption.

### Changed

- The gRPC stubs are real protobuf code generated from the `.proto` (previously hand-written JSON + base64 encoders that other-language clients could not talk to).
- Model runtimes moved to extras: install with `pip install -e ".[local]"`. The core install no longer compiles llama.cpp, and `webrtcvad-wheels` replaces `webrtcvad` so no C compiler is needed.
- Tracing is opt-in (`OTEL_EXPORTER_OTLP_ENDPOINT` or `VAANI_TRACE_CONSOLE=1`); spans are no longer printed to stdout on every turn.
- Model downloads are atomic and show progress; the Vosk zip is removed after extraction.
- The gRPC server shares one loaded Vosk model and Piper voice across streams, runs NLU actions like local mode, and serializes access to the shared LLM.
- Setup doctor mode for checking model paths, Piper, gRPC port availability, and audio input.
- Open-model bootstrap mode for previewing/downloading the starter model set and writing local env files.
- Basic intent action hooks for deterministic responses before LLM fallback.
- Optional JSONL session memory through `CONVERSATION_MEMORY_PATH`.
- Shared acknowledgement tone generation for local and gRPC response paths.
- Low-latency profile controls through `VAANI_PROFILE=low_latency`.
- gRPC streams emit immediate acknowledgement audio before full assistant content.
- Benchmark output separates first audible response from first content audio.
- Settings read environment values when `Settings()` is created.
- gRPC server shutdown stops the async server gracefully.

### Verified

- `pytest -q` (110 passed, 1 skipped)
- `ruff check voice_assistant tests scripts`
- `MOCK_MODELS=1` gRPC load benchmark at 10 and 50 concurrent streams, within the CI performance budget

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
