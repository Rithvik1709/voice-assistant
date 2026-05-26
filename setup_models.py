from pathlib import Path
import urllib.request
import zipfile
import os
import ssl

MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
MODEL_NAME = "vosk-model-small-en-us-0.15"
MODELS_DIR = Path("models")
ZIP_PATH = MODELS_DIR / f"{MODEL_NAME}.zip"
MODEL_PATH = MODELS_DIR / MODEL_NAME
ENV_FILE = Path(".env")


def show_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent = min(downloaded * 100 / total_size, 100)

    print(f"\rDownloading: {percent:.2f}%", end="")


def download_model():
    print("\nDownloading ASR model...\n")

    ssl._create_default_https_context = ssl._create_unverified_context

    urllib.request.urlretrieve(
    MODEL_URL,
    ZIP_PATH,
    reporthook=show_progress
    )

    print("\nDownload completed.")


def extract_model():
    print("\nExtracting model...\n")

    if not ZIP_PATH.exists():
        raise FileNotFoundError("Downloaded ZIP file not found.")

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(MODELS_DIR)

    print("Extraction completed.")


def cleanup():
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
        print("Removed ZIP file.")


def update_env():
    env_line = f"ASR_MODEL_PATH=models/{MODEL_NAME}\n"

    if not ENV_FILE.exists():
        ENV_FILE.write_text(env_line)
        print(".env file created.")
        return

    content = ENV_FILE.read_text()

    if "ASR_MODEL_PATH" not in content:
        with ENV_FILE.open("a") as f:
            f.write(f"\n{env_line}")

        print("Updated .env file.")


def main():
    print("=" * 50)
    print("VOICE ASSISTANT MODEL SETUP")
    print("=" * 50)

    try:
        MODELS_DIR.mkdir(exist_ok=True)

        if MODEL_PATH.exists():
            print("\nASR model already exists.")
        else:
            download_model()
            extract_model()
            cleanup()

        update_env()

        print("\nSetup completed successfully!")

    except Exception as e:
        print(f"\nSetup failed: {e}")


if __name__ == "__main__":
    main()