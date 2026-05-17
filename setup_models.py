#!/usr/bin/env python3
import os
import sys
import ssl
import shutil
import zipfile
import argparse
from pathlib import Path
import urllib.request
from huggingface_hub import hf_hub_download

# Configure paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
ENV_FILE = BASE_DIR / ".env"
ENV_EXAMPLE = BASE_DIR / ".env.example"

# LLM Quantized Options
LLM_OPTIONS = {
    "1": {
        "name": "Qwen2.5-0.5B-Instruct (Fast Download / Testing ~398MB)",
        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    },
    "2": {
        "name": "Llama-3-8B-Instruct (High Quality / Heavy ~4.9GB)",
        "repo_id": "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
    }
}

class DownloadProgressBar:
    """Tqdm-like simple progress bar using urllib.request"""
    def __init__(self, description="Downloading"):
        self.description = description
        self.last_percent = -1

    def __call__(self, block_num, block_size, total_size):
        if total_size <= 0:
            return
        percent = int(block_num * block_size * 100 / total_size)
        if percent != self.last_percent:
            self.last_percent = percent
            # Print a neat text-based progress bar
            bar_length = 40
            filled_length = int(round(bar_length * percent / 100))
            bar = '=' * filled_length + '-' * (bar_length - filled_length)
            sys.stdout.write(f"\r{self.description}: [{bar}] {percent}% completed")
            sys.stdout.flush()
            if percent >= 100:
                sys.stdout.write("\n")

def setup_directories():
    """Create models directory if it doesn't exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[*] Verified models directory at: {MODELS_DIR}")

def download_llm(selection="1"):
    """Download the selected LLM GGUF model from Hugging Face."""
    selected = LLM_OPTIONS.get(selection, LLM_OPTIONS["1"])
    print(f"\n[*] Preparing to download LLM: {selected['name']}...")
    try:
        # Use huggingface_hub to fetch the file securely (handles resuming, chunking automatically)
        dest_path = hf_hub_download(
            repo_id=selected["repo_id"],
            filename=selected["filename"],
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False
        )
        print(f"[+] LLM successfully downloaded to: {dest_path}")
        return Path(dest_path).relative_to(BASE_DIR)
    except Exception as e:
        print(f"[-] Error downloading LLM: {e}")
        sys.exit(1)

def download_tts():
    """Download Piper ONNX model and config from Hugging Face."""
    print("\n[*] Downloading Piper TTS voice model (en_US-lessac-medium)...")
    try:
        onnx_path = hf_hub_download(
            repo_id="rhasspy/piper-voices",
            filename="en/en_US/lessac/medium/en_US-lessac-medium.onnx",
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False
        )
        json_path = hf_hub_download(
            repo_id="rhasspy/piper-voices",
            filename="en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False
        )
        # Move the downloaded voice files to the root of the models folder for easier reference
        dest_onnx = MODELS_DIR / "en_US-lessac-medium.onnx"
        dest_json = MODELS_DIR / "en_US-lessac-medium.onnx.json"
        
        # Copy or move
        shutil.move(onnx_path, dest_onnx)
        shutil.move(json_path, dest_json)
        
        # Clean up empty parent folders left by local_dir
        shutil.rmtree(MODELS_DIR / "en", ignore_errors=True)

        print(f"[+] TTS Model successfully downloaded to: {dest_onnx}")
        return dest_onnx.relative_to(BASE_DIR)
    except Exception as e:
        print(f"[-] Error downloading TTS voice: {e}")
        sys.exit(1)

def download_asr():
    """Download and extract Vosk English model."""
    print("\n[*] Downloading Vosk ASR model (vosk-model-small-en-us-0.15)...")
    url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    zip_path = MODELS_DIR / "vosk-model-small-en-us-0.15.zip"
    dest_dir = MODELS_DIR / "vosk-model-small-en-us-0.15"

    if dest_dir.exists():
        print(f"[+] Vosk model already exists at: {dest_dir}")
        return dest_dir.relative_to(BASE_DIR)

    try:
        # Download the zip file with bypass SSL validation context to prevent expired cert failures
        context = ssl._create_unverified_context()
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        
        with urllib.request.urlopen(req, context=context) as response:
            total_size = int(response.info().get('Content-Length', 0))
            block_size = 1024 * 8
            downloaded = 0
            
            with open(zip_path, 'wb') as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = int(downloaded * 100 / total_size)
                        bar_length = 40
                        filled_length = int(round(bar_length * percent / 100))
                        bar = '=' * filled_length + '-' * (bar_length - filled_length)
                        sys.stdout.write(f"\rASR Model Zip: [{bar}] {percent}% completed")
                        sys.stdout.flush()
                sys.stdout.write("\n")

        print("[*] Extracting ASR model...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(MODELS_DIR)
        
        # Clean up the zip file
        zip_path.unlink()
        print(f"[+] ASR Model extracted successfully to: {dest_dir}")
        return dest_dir.relative_to(BASE_DIR)
    except Exception as e:
        print(f"[-] Error downloading or extracting ASR: {e}")
        if zip_path.exists():
            zip_path.unlink()
        sys.exit(1)

def update_env(llm_path, tts_path, asr_path):
    """Generate or update the .env file with the downloaded model paths."""
    print("\n[*] Configuring .env file...")
    
    # Read existing env if any
    env_lines = {}
    if ENV_FILE.exists():
        with open(ENV_FILE, "r") as f:
            for line in f:
                if "=" in line:
                    key, val = line.strip().split("=", 1)
                    env_lines[key] = val

    # Update paths (normalized for multi-platform support)
    env_lines["MODEL_PATH"] = str(llm_path).replace("\\", "/")
    env_lines["PIPER_VOICE"] = str(tts_path).replace("\\", "/")
    env_lines["ASR_MODEL_PATH"] = str(asr_path).replace("\\", "/")
    
    # Set default variables if missing
    defaults = {
        "GRPC_PORT": "50051",
        "VAD_AGGRESSIVENESS": "2",
        "CHUNK_MS": "20",
        "ASR_ENDPOINT_SILENCE_MS": "60",
        "ASR_BACKEND": "vosk",
        "TTS_SENTENCE_MAX_TOKENS": "8",
        "TTS_EAGER_MIN_WORDS": "3",
        "PLAYER_BLOCKSIZE": "128"
    }
    for k, v in defaults.items():
        if k not in env_lines:
            env_lines[k] = v

    # Write back to .env
    with open(ENV_FILE, "w") as f:
        for k, v in env_lines.items():
            f.write(f"{k}={v}\n")
    
    print("[+] Successfully generated .env file!")

def main():
    parser = argparse.ArgumentParser(description="Automated model downloader for voice-assistant pipeline.")
    parser.add_argument("--llm", choices=["1", "2"], help="Pre-select LLM: 1 for Qwen0.5B, 2 for Llama3-8B")
    parser.add_argument("--yes", action="store_true", help="Automatic yes to all defaults (non-interactive mode)")
    args = parser.parse_args()

    print("==============================================")
    print("      Voice Assistant - Model Setup CLI       ")
    print("==============================================")

    setup_directories()

    # Determine LLM Choice
    if args.llm:
        llm_choice = args.llm
    elif args.yes:
        llm_choice = "1"  # Default to fast Qwen for automated runs
    else:
        print("\nChoose an LLM model to download:")
        for k, v in LLM_OPTIONS.items():
            print(f"  [{k}] {v['name']}")
        try:
            choice = input("Enter choice [1 or 2, default 1]: ").strip()
            llm_choice = choice if choice in LLM_OPTIONS else "1"
        except (KeyboardInterrupt, EOFError):
            print("\n[-] Cancelled setup.")
            sys.exit(0)

    # Downloads
    llm_path = download_llm(llm_choice)
    tts_path = download_tts()
    asr_path = download_asr()

    # Configuration
    update_env(llm_path, tts_path, asr_path)

    print("\n==============================================")
    print("       Setup Completed Successfully!         ")
    print("==============================================")
    print("You can now run your voice assistant using:")
    print("  python -m voice_assistant.main --mode local")
    print("==============================================")

if __name__ == "__main__":
    main()
