# Contributing to Voice Assistant 🎙️

First off, thank you for considering contributing to this project! 🚀

Whether you're fixing bugs, improving documentation, optimizing latency, or adding features, every contribution is appreciated.

This project focuses on building a **low-latency real-time streaming voice assistant pipeline** using:

* Streaming ASR
* LLM token streaming
* Streaming TTS
* gRPC communication
* Async orchestration

We welcome contributors of all experience levels, especially beginners participating in open-source programs like GSSoC.

---

# 🤝 Code of Conduct

Please be respectful and professional while interacting with maintainers and contributors.

We aim to maintain a positive and inclusive environment for everyone.

---

# 🚀 Getting Started

## 1. Fork the Repository

Click the **Fork** button at the top-right corner of this repository.

---

## 2. Clone Your Fork

```bash
git clone https://github.com/YOUR_USERNAME/voice-assistant.git
```

---

## 3. Navigate to the Project Directory

```bash
cd voice-assistant
```

---

# ⚙️ Project Setup

Please follow the setup instructions provided in the README before contributing.

The README includes:

* Dependency installation
* Environment configuration
* Model downloads
* Running local mode
* gRPC setup
* Testing instructions

[README.md](./README.md)

---

# 🌿 Contribution Workflow

## Step 1: Create a New Branch

```bash
git checkout -b feature/your-feature-name
```

Examples:

* `feature/add-dark-mode`
* `fix/tts-streaming-bug`
* `docs/update-readme`

---

## Step 2: Make Your Changes

Keep your changes focused and well-structured.

---

## Step 3: Commit Your Changes

```bash
git commit -m "docs: add beginner-friendly contribution guidelines"
```

Use meaningful commit messages.

---

## Step 4: Push Changes

```bash
git push origin feature/your-feature-name
```

---

## Step 5: Open a Pull Request

While creating a PR:

* Clearly describe your changes
* Mention the related issue number
* Add screenshots/logs if applicable

---

# 🧹 Coding Standards

Please follow these guidelines:

* Write clean and modular code
* Use meaningful variable and function names
* Maintain existing project structure
* Avoid unnecessary comments
* Remove unused imports/files

---

# 🧪 Running Tests

Run the tests and linter before submitting a PR:

```bash
pip install -e ".[dev]"
pytest -q
ruff check voice_assistant tests scripts
```

The tests use fakes for models, the microphone, and speakers, so no downloads are needed. Current coverage includes:

* Barge-in, cancellation, and error recovery in the orchestrator
* ASR endpointing, pre-roll, and Vosk segment handling
* Non-blocking LLM streaming
* The TTS worker, Piper process handling, and audio playback
* gRPC streaming and interruption (mock mode)
* NLU, intent actions, configuration, and speculative decoding

If you change `voice_assistant.proto`, regenerate the stubs (see the README) and run `python scripts/check_proto_stubs.py`.

---

# 🌱 Beginner-Friendly Contributions

If you're new to open source, you can start with:

* Documentation improvements
* README enhancements
* UI/UX improvements
* Accessibility fixes
* Refactoring duplicated code
* Adding tests
* Improving error handling

---

# ⭐ Need Help?

Feel free to ask questions through issues or discussions if you need help getting started.

Happy Contributing 🚀
