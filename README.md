# Vaani

Vaani is an open-source, low-latency voice assistant that runs entirely on your machine with open models. It listens through your microphone, understands what you said, thinks with a local LLM, and answers out loud, streaming every stage so it starts speaking before the reply is finished.

It can also run as a gRPC server for remote clients, or as a text chat when you have no microphone.

- **Streaming everything:** speech recognition, LLM tokens, and speech synthesis overlap instead of running one after another.
- **Barge-in:** start talking and Vaani stops mid-sentence, then answers your new question.
- **Local and open:** Qwen2.5 via llama.cpp, Whisper (or Vosk) for speech recognition, Piper for speech. No cloud APIs required, but any OpenAI-compatible server (Ollama, LM Studio, vLLM, hosted) can be plugged in for a stronger model.
- **Fast answers for simple things:** time, date, greetings, and (optionally) live weather are answered instantly without the LLM.
- **Speaks like a person, not a screen:** markdown, lists, code, links, and emoji are never read aloud.
- **English and Hinglish:** the intent layer understands requests like "gaana chala do" and "delhi ka mausam batao".
- **60+ languages, offline:** with `ASR_LANGUAGE=auto`, Vaani detects the language you speak on every turn, the LLM answers in it, and a matching Piper voice speaks it. See [Languages](#languages).

See [CHANGELOG.md](CHANGELOG.md) for what changed in 2.0.0, including how to upgrade from 1.0.

## Contents

- [Quickstart](#quickstart)
- [Ways to run Vaani](#ways-to-run-vaani)
- [Languages](#languages)
- [How it works](#how-it-works)
- [Configuration](#configuration)
- [Models](#models)
- [gRPC server and client](#grpc-server-and-client)
- [Performance](#performance)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

## Quickstart

You need Python 3.11+ and PortAudio for audio (`brew install portaudio` on macOS, `sudo apt-get install libportaudio2` on Debian/Ubuntu).

```bash
git clone https://github.com/Rithvik1709/voice-assistant.git
cd voice-assistant
pip install -e ".[local]"                # Vaani plus the model runtimes
vaani models --write-env --download      # fetch the open starter models (~700 MB) and write .env
vaani doctor                             # check everything is ready
vaani local                              # start talking
```

Speak after the "Vaani is listening" line. Each turn logs its latency (`ASR_latency_ms`, `TTFT_ms`, `E2E_ms`, and more). Press Ctrl+C to stop.

No microphone, or just want to look around first? Try the full pipeline with no models at all:

```bash
pip install -e .
MOCK_MODELS=1 vaani chat
```

## Ways to run Vaani

Every mode is a subcommand of `vaani` (`python -m voice_assistant.main` works too, as does `vaani --mode <mode>`).

| Command | What it does |
| --- | --- |
| `vaani local` | Voice assistant on this machine: microphone in, speakers out |
| `vaani chat` | Type messages and read replies; add `--speak` to also hear them |
| `vaani server` | gRPC server for remote clients (`--host`, `--port`) |
| `vaani client` | Voice client for a remote server (`--target host:port`) |
| `vaani doctor` | Check configuration, runtimes, models, port, microphone, and Piper |
| `vaani models` | Show the starter model set; `--download` and `--write-env` set it up |

Add `--log-level DEBUG` to any mode for detailed logs, and `--version` to print the version.

### Voice mode and barge-in

With barge-in on (the default), speaking for about a quarter of a second while Vaani is talking stops its reply immediately, and your new question is answered next. What you heard of the interrupted reply stays in the conversation history.

Laptop speakers feed Vaani's own voice back into the microphone, which can make it interrupt itself. Use headphones, or set `ENABLE_BARGE_IN=0`: the microphone is then muted while Vaani speaks, so it never hears itself.

### Chat mode

```bash
vaani chat                    # needs only MODEL_PATH
vaani chat --speak            # also speak replies through Piper
echo "what time is it" | vaani chat
```

Chat uses the same intent actions, LLM, and conversation memory as voice mode. Type `/reset` to forget the conversation and `/quit` to exit.

### Use a bigger or hosted model

The default 0.5B model is fast but not very knowledgeable. Point Vaani at any OpenAI-compatible server to use a stronger model, with no other changes:

```bash
# Ollama (ollama pull qwen2.5:7b)
LLM_BACKEND=openai LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen2.5:7b vaani local

# LM Studio, vLLM, or llama.cpp's llama-server work the same way; so do hosted APIs:
LLM_BACKEND=openai LLM_BASE_URL=https://api.example.com/v1 LLM_MODEL=<model> LLM_API_KEY=<key> vaani chat
```

`MODEL_PATH` is not needed with this backend, and `vaani doctor` checks the server is reachable and serves the model.

### Choosing a speech recognizer

Vaani uses Whisper (through faster-whisper) by default. On test questions Whisper `base.en` transcribed every one correctly in about 300 ms on a laptop CPU, while Vosk small turned "how a rainbow forms" into "how arena forums".

```bash
ASR_MODEL_PATH=small.en vaani local   # more accurate, about 4x slower
ASR_BACKEND=vosk ASR_MODEL_PATH=models/vosk-model-small-en-us-0.15 vaani local
```

Whisper transcribes each utterance when you stop talking, so its decode time is added to the response and there are no live partial transcripts. Vosk streams as you speak, adds almost nothing after you stop, and supports the unfinished-sentence hold (`ASR_HOLD_SILENCE_MS`), but mishears far more. Vosk models are at [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models).

### Mock mode

`MOCK_MODELS=1` replaces speech recognition, the LLM, and TTS with lightweight stand-ins. It works with `chat` and `server`, needs only the core install, and is what CI and the benchmarks use.

## Languages

Vaani speaks English out of the box. To talk to it in other languages, set up the languages you need once (this is the only step that needs the internet), then it runs offline:

```bash
vaani models --languages hi,ta,fr,ja --download --write-env
vaani doctor     # shows which languages have a voice
vaani local
```

This fetches multilingual Whisper `large-v3-turbo` and one Piper voice per language into `models/voices/`, and writes a `.env` with `ASR_LANGUAGE=auto` and `ASR_LANGUAGES=en,hi,ta,fr,ja`. Every turn then works like this:

1. **Listening.** Whisper detects the language of each utterance. `ASR_LANGUAGES` limits detection to the languages you actually speak, which matters because a short "ok" is easily mistaken for another language. Without it, any of Whisper's 99 languages can be detected.
2. **Thinking.** The LLM is asked to reply in the language you spoke. The default 0.5B model handles this poorly, so use a multilingual model: a Qwen2.5 7B or Gemma 3 GGUF as `MODEL_PATH`, or any server through `LLM_BACKEND=openai`.
3. **Speaking.** The reply is cut into sentences at that language's punctuation (`।`, `。`, `؟` and others). Chinese and Japanese are chunked by characters, since they have no spaces. The voice for the language in `PIPER_VOICES_DIR` speaks it, and `PIPER_VOICE` stays the default.

Piper has voices for about 40 languages. For the rest, `TTS_FALLBACK` decides what happens: `auto` (the default) uses [espeak-ng](https://github.com/espeak-ng/espeak-ng) if it is installed (robotic, but it speaks over 100 languages: `brew install espeak-ng` or `apt-get install espeak-ng`) and the default voice otherwise. Set `espeak`, `default`, or `none` (stay silent) to choose one.

Time, date, weather and greeting shortcuts answer in English and Hindi only. Requests in other languages go to the LLM. In `vaani chat`, the language is guessed from the writing system: Tamil script is Tamil, but Latin-script text could be many languages and is left to the LLM.

## How it works

```text
 Microphone ──▶ VAD + streaming ASR ──▶ final transcript
 (20 ms frames) (Whisper, WebRTC VAD)         │
      │                                      ▼
      │                           intent classifier (NLU)
      │                             │                 │
      │                   simple command          anything else
      │                   (time, weather…)             │
      │                             │                  ▼
      │                             │         LLM token stream
      │                             │     (llama.cpp, worker thread)
      │                             ▼                  │
      │                      sentence chunker ◀────────┘
      │                             │
      │                  speech cleanup + Piper TTS
      │                             │
      │                             ▼
      └── barge-in ──────────▶ audio playback
          (user speaks: cancel generation, drop queued audio)
```

1. **Listening.** Audio arrives in 20 ms frames. WebRTC VAD finds speech, and a short pre-roll keeps the first syllable. An utterance ends after `ASR_ENDPOINT_SILENCE_MS` of silence and is transcribed by Whisper. With Vosk, frames are transcribed as they stream in, and if your last words sound unfinished ("turn on the…") Vaani waits up to `ASR_HOLD_SILENCE_MS` so it does not cut you off.
2. **Understanding.** A lightweight intent classifier (English, Hindi, and Hinglish keywords) catches simple commands, which are answered directly. Everything else goes to the LLM with the recent conversation.
3. **Thinking.** llama.cpp generates tokens in a worker thread, so listening and speaking never stall. Generation can be cancelled at the next token.
4. **Speaking.** Tokens are cut into sentences (and short phrases, for a fast first word), cleaned of formatting, and synthesized by Piper while the LLM is still writing. A short acknowledgement tone plays the moment your question is understood.

The code is organised by stage:

```text
voice_assistant/
  asr/          VAD, streaming recognition, endpointing, partial transcripts
  nlu/          intent classifier
  actions.py    instant answers (time, date, weather, greetings)
  weather.py    Open-Meteo weather lookup
  llm/          llama.cpp and OpenAI-compatible streaming clients, speculative decoding
  tts/          Piper synthesis, speech text cleanup, playback queue, audio player
  pipeline/     orchestrator: turns, barge-in, history, per-turn metrics
  transport/    gRPC server, client, and protobuf definitions
  chat.py       text chat mode
  mocks.py      stand-in models for MOCK_MODELS=1
  doctor.py     setup checks
  config.py     settings and validation
```

## Configuration

Settings come from environment variables or a `.env` file. `vaani models --write-env` creates one for the starter models, and [.env.example](.env.example) documents every option. Invalid values (for example `CHUNK_MS=25`) are reported with a clear message at startup.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL_PATH` | | GGUF model for the LLM (required) |
| `PIPER_VOICE` | | Piper `.onnx` voice, with its `.onnx.json` next to it (required for speech) |
| `ASR_MODEL_PATH` | `base.en` (`large-v3-turbo` for other languages) | Whisper model folder or size name (`base.en`, `small.en`, `small`), or a Vosk model folder |
| `ASSISTANT_SYSTEM_PROMPT` | concise spoken style | Instructions for the LLM |
| `LLM_BACKEND` | `llama` | `llama` runs `MODEL_PATH` in-process; `openai` uses an OpenAI-compatible server |
| `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY` | | Server URL (ending in `/v1`), model name, and optional key for the `openai` backend |
| `ASR_BACKEND` | `whisper` | `whisper` (accurate) or `vosk` (fast, live partials, less accurate) |
| `ASR_LANGUAGE` | `en` | Language for Whisper, e.g. `hi`, or `auto` to detect it on every utterance and reply in it (see [Languages](#languages)) |
| `ASR_LANGUAGES` | | With `auto`: the languages detection may choose from, e.g. `en,hi,ta` |
| `PIPER_VOICES_DIR` | | Folder of Piper voices, one per language; the voice is picked by each reply's language |
| `TTS_FALLBACK` | `auto` | For languages without a voice: `espeak`, `default` (the default voice), `none`, or `auto` (espeak-ng if installed, else the default voice) |
| `ASR_ENDPOINT_SILENCE_MS` | `400` | Silence that ends an utterance. Lower is snappier but may cut sentences at natural pauses. |
| `ASR_HOLD_SILENCE_MS` | `1000` | Longer silence allowed when your words so far end on "and", "the", "um" and similar (Vosk only) |
| `ENABLE_BARGE_IN` | `1` | Let speech interrupt Vaani; `0` mutes the mic while it speaks |
| `BARGE_IN_MS` | `240` | How long you must speak before Vaani stops |
| `VAD_AGGRESSIVENESS` | `2` | Speech detection strictness, 0 (lenient) to 3 (strict) |
| `CHUNK_MS` | `20` | Audio frame size: 10, 20 or 30 |
| `VAANI_PROFILE` | | `low_latency` trims endpointing (to 300 ms), token budget, and chunk sizes |
| `LLM_MAX_TOKENS` | `256` | Longest reply, in tokens |
| `LLM_TEMPERATURE` | `0.7` | Sampling temperature |
| `LLM_CONTEXT_SIZE` | `4096` | LLM context window |
| `N_GPU_LAYERS` | `-1` | Layers offloaded to the GPU (`-1` is all) |
| `CONVERSATION_HISTORY_TURNS` | `10` | Turns kept in the LLM context |
| `CONVERSATION_MEMORY_PATH` | | JSONL file that remembers the conversation across sessions |
| `ENABLE_ACK_TONE` / `ACK_TONE_MS` | `1` / `55` | Short tone when your question is understood |
| `ENABLE_WEATHER` | `0` | Live weather from [Open-Meteo](https://open-meteo.com) (free, no key). Off by default because it sends the city name over the internet. |
| `WEATHER_DEFAULT_CITY` | | City used when a weather question names none |
| `WEATHER_UNITS` | `celsius` | `celsius` or `fahrenheit` |
| `GRPC_PORT` | `50051` | Server port when `--port` is not given |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | | Send OpenTelemetry traces to a collector (tracing is off otherwise) |
| `VAANI_TRACE_CONSOLE` | | `1` prints trace spans to the console |

## Models

`vaani models --download` fetches this starter set into `models/`. An interrupted download is retried on the next run instead of being mistaken for a finished file.

| Stage | Default model | Download | License |
| --- | --- | --- | --- |
| LLM | Qwen2.5 0.5B Instruct, GGUF Q4_K_M | ~490 MB | Apache-2.0 |
| Speech | Piper `en_US-lessac-medium` | ~63 MB | MIT |
| Recognition | Whisper `base.en` (faster-whisper) | ~140 MB | MIT |

**LLM.** Any chat GGUF model works with llama.cpp. For better answers try Llama 3 8B Instruct or Mistral 7B Instruct at Q4_K_M or Q5_K_M. The model is warmed up on the system prompt at startup, so the first question is fast.

**Speech.** Any [Piper voice](https://huggingface.co/rhasspy/piper-voices) works; the sample rate is read from its `.onnx.json`. With the `piper-tts` package installed, voices load once and run in-process; otherwise the `piper` binary on `PATH` is used.

**Recognition.** Any Whisper size works (`small.en` is more accurate; for other languages use a multilingual size such as `large-v3-turbo` or the faster but weaker `small`), and so does any [Vosk model](https://alphacephei.com/vosk/models) with `ASR_BACKEND=vosk` (one language per model, no `auto`).

With `vaani models --languages ...`, multilingual Whisper `large-v3-turbo` (~1.6 GB) replaces `base.en`, and one Piper voice per language (~60 MB each) is added under `models/voices/`.

Check the license of any replacement model before redistributing it.

### Install extras

```bash
pip install -e ".[local]"          # model runtimes: llama.cpp, Whisper, Vosk, Piper
pip install -e .                   # core only: mock mode, gRPC server, benchmarks
pip install -e ".[local,cuda]"     # llama.cpp with CUDA
pip install -e ".[local,metal]"    # llama.cpp with Metal (Apple Silicon)
pip install -e ".[otel]"           # OpenTelemetry trace export
pip install -e ".[all]"            # everything, including Silero VAD and WebRTC
```

## gRPC server and client

```bash
vaani server --host 0.0.0.0 --port 50051
vaani client --target localhost:50051
```

The server streams audio in and speech out over one bidirectional `StreamVoice` call, defined in [voice_assistant.proto](voice_assistant/transport/voice_assistant.proto). Each connection gets its own recognizer, speech queue, and conversation; the loaded models are shared.

- Send 16 kHz, 16-bit mono PCM in chunks of any size.
- An utterance ends after the configured silence, or when the client closes its side of the stream.
- `AudioResponse` messages carry speech audio, and a few fields carry control information:
  - `transcript` comes in its own message (no audio), once per turn, with what the server heard.
  - `interrupt = true` comes in its own message (no audio) when the user barges in; the client should drop any audio it still has buffered.
  - `debug_text` rides on the first audio chunk of each sentence and holds that sentence's text.
- The bundled client plays the audio, handles interrupts, and prints the conversation.
- On `SIGTERM`, open streams get 5 seconds to finish before the server exits.

After editing the `.proto`, regenerate the stubs (the committed ones are built with `grpcio-tools==1.66.2` so they work with protobuf 5.27 and newer):

```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. \
  voice_assistant/transport/voice_assistant.proto
python scripts/check_proto_stubs.py
```

## Performance

`scripts/bench/load_asr.py` streams synthetic speech from many concurrent gRPC clients and reports latency percentiles. Run it against a mock server to measure the pipeline's own overhead:

```bash
MOCK_MODELS=1 vaani server &
python scripts/bench/load_asr.py --concurrency 10 --frames-per-client 30 --frame-ms 30
```

In mock mode with 10 concurrent clients, the acknowledgement arrives about 65 ms after the end of speech (p95) and the first spoken content about 113 ms after (p95). Full results are in [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md). CI fails any change that pushes 50-client p95 processing latency over 300 ms.

With real models, total response time is dominated by the LLM's time to first token and the first sentence's synthesis, which depend on your hardware and model size.

## Development

```bash
pip install -e ".[dev]"
pytest -q
ruff check voice_assistant tests scripts
```

The tests use fakes for the microphone, speakers, and models, so they run in seconds with no downloads. They cover barge-in and cancellation, endpointing, non-blocking LLM streaming, the TTS worker and speech cleanup, gRPC streaming and interruption, chat mode, intents, actions, weather, and configuration.

CI runs lint and tests on Python 3.11 and 3.12, checks the protobuf stubs match the `.proto`, and runs the 50-client performance test. See [CONTRIBUTING.md](CONTRIBUTING.md) to get started.

## Troubleshooting

Start with `vaani doctor`: it names the failing piece and how to fix it.

- **`piper_synthesis` fails with an espeak-ng `phontab` error.** Some `piper-tts` wheels for Apple Silicon ship with a broken espeak-ng data path, so Piper cannot speak at all. Install the official Piper binary and put it on `PATH`, then `pip uninstall piper-tts` so Vaani uses the binary. Or run Vaani on Linux.
- **Vaani keeps interrupting itself.** Its voice is reaching the microphone. Use headphones or set `ENABLE_BARGE_IN=0`.
- **Sentences get cut off halfway.** Raise `ASR_ENDPOINT_SILENCE_MS`, for example to 500.
- **Vaani mishears you.** Use a larger Whisper model (`ASR_MODEL_PATH=small.en`), and check `vaani doctor` shows the right microphone.
- **Answers are shallow or wrong.** Use a bigger model, locally or through `LLM_BACKEND=openai`.
- **Vaani answers in the wrong language.** Set `ASR_LANGUAGES` to the languages you speak, so short utterances are not detected as something else. If the language is right but the reply is not, the LLM is too small for that language.
- **A language is spoken with the wrong accent, or not at all.** `vaani doctor` lists languages without a Piper voice. Fetch one with `vaani models --languages <code> --download`, or install espeak-ng.
- **`llama-cpp-python` build fails with `Undefined symbols ... X509`.** An x86_64 OpenSSL in `/usr/local/lib` (old Intel Homebrew) is being linked. Build without it: `CMAKE_ARGS="-DLLAMA_OPENSSL=OFF -DLLAMA_CURL=OFF" pip install llama-cpp-python`.
- **Replies are slow.** Offload the LLM to your GPU (`[cuda]` or `[metal]` extra, `N_GPU_LAYERS=-1`), use a smaller model, or set `VAANI_PROFILE=low_latency`.
- **`llama-cpp-python` fails to build.** Install a prebuilt wheel: `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu`.
- **No sound or wrong device.** `vaani doctor` shows the microphone in use; pick devices in your system sound settings.

## Roadmap

- Wake word, so Vaani listens only when called
- Echo cancellation for barge-in on laptop speakers
- Music and smart-home actions
- Translated replies for the time, date and weather shortcuts
- WebRTC transport for browser clients

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgements

Vaani builds on [llama.cpp](https://github.com/ggerganov/llama.cpp) and `llama-cpp-python`, [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and OpenAI Whisper, [Piper](https://github.com/rhasspy/piper), [Vosk](https://alphacephei.com/vosk/), [Open-Meteo](https://open-meteo.com), and the wider open-source voice AI community.
