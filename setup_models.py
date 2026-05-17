#!/usr/bin/env python3
import os
import sys
import ssl
import shutil
import zipfile
import argparse
from pathlib import Path
import urllib.request
import urllib.error
from tqdm import tqdm
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

def setup_directories():
    """Create models directory if it doesn't exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[*] Verified models directory at: {MODELS_DIR}")

def download_url_with_progress(url, dest_path, description, context=None):
    """Download a URL to a file path showing progress with tqdm."""
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, context=context) as response:
        total_size = int(response.info().get('Content-Length', 0))
        block_size = 1024 * 8
        
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=description, leave=True) as pbar:
            with open(dest_path, 'wb') as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))

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
        )
        json_path = hf_hub_download(
            repo_id="rhasspy/piper-voices",
            filename="en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
            local_dir=MODELS_DIR,
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
        # 1. Try standard verified secure download first
        download_url_with_progress(url, zip_path, "ASR Model Zip")
    except Exception as e:
        # Check if the failure looks like an SSL verification/certificate issue
        err_msg = str(e).lower()
        if "ssl" in err_msg or "certificate" in err_msg or "verify" in err_msg:
            print("\n" + "=" * 70)
            print("[WARNING] Secure SSL connection failed due to certificate verification issues.")
            print("[WARNING] (This is typically caused by an expired certificate on the hosting domain.)")
            print("[WARNING] Falling back to bypass SSL verification context safely...")
            print("=" * 70 + "\n")
            
            try:
                # 2. Safely fall back to unverified context with warning shown to user
                unverified_context = ssl._create_unverified_context()
                download_url_with_progress(url, zip_path, "ASR Model Zip (Bypass-SSL)", context=unverified_context)
            except Exception as inner_e:
                print(f"[-] Error downloading ASR model even after SSL bypass: {inner_e}")
                sys.exit(1)
        else:
            print(f"[-] Error downloading ASR: {e}")
            sys.exit(1)

    try:
        print("[*] Extracting ASR model...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(MODELS_DIR)
        
        # Clean up the zip file
        zip_path.unlink()
        print(f"[+] ASR Model extracted successfully to: {dest_dir}")
        return dest_dir.relative_to(BASE_DIR)
    except Exception as e:
        print(f"[-] Error extracting ASR zip file: {e}")
        if zip_path.exists():
            zip_path.unlink()
        sys.exit(1)

def update_env(llm_path, tts_path, asr_path):
    """Non-destructively generate or update the .env file with the downloaded model paths."""
    print("\n[*] Configuring .env file...")
    
    updates = {
        "MODEL_PATH": str(llm_path).replace("\\", "/"),
        "PIPER_VOICE": str(tts_path).replace("\\", "/"),
        "ASR_MODEL_PATH": str(asr_path).replace("\\", "/"),
    }
    
    # Defaults template if the .env file does not exist or has no variables
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

    # Load existing file lines
    lines = []
    existing_keys = set()
    
    if ENV_FILE.exists():
        with open(ENV_FILE, "r") as f:
            lines = f.readlines()
        
        # Parse currently set keys cleanly (stripping whitespace to prevent duplicates)
        for line in lines:
            if "=" in line and not line.strip().startswith("#"):
                key, _ = line.split("=", 1)
                existing_keys.add(key.strip())
    else:
        # Initialize with .env.example template if available to preserve comments
        if ENV_EXAMPLE.exists():
            print("[*] Initializing .env from .env.example template...")
            with open(ENV_EXAMPLE, "r") as f:
                lines = f.readlines()
            for line in lines:
                if "=" in line and not line.strip().startswith("#"):
                    key, _ = line.split("=", 1)
                    existing_keys.add(key.strip())
        else:
            # Otherwise use default keys
            for k, v in defaults.items():
                if k not in updates:
                    lines.append(f"{k}={v}\n")
                    existing_keys.add(k)

    # Process and build new lines non-destructively
    new_lines = []
    keys_updated = set()
    
    for line in lines:
        if "=" in line and not line.strip().startswith("#"):
            key, val = line.split("=", 1)
            stripped_key = key.strip()
            if stripped_key in updates:
                new_lines.append(f"{stripped_key}={updates[stripped_key]}\n")
                keys_updated.add(stripped_key)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    # Append any variables we have updates for that weren't already present in the file
    for k, v in updates.items():
        if k not in keys_updated:
            new_lines.append(f"{k}={v}\n")

    # Write back to .env non-destructively
    with open(ENV_FILE, "w") as f:
        f.writelines(new_lines)
    
    print("[+] Successfully updated .env file non-destructively!")

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
