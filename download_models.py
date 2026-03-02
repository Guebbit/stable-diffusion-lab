import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

# ------------------------------------------------------------
# CONFIGURATION SECTION
# ------------------------------------------------------------
# Host-mounted folder where models are stored.
MODELS_DIR = Path("/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Optional access tokens loaded from .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
CIV_TOKEN = os.getenv("CIV_TOKEN")


# ------------------------------------------------------------
# MODEL LIST
# ------------------------------------------------------------
# Keep this dictionary simple so adding/removing models is easy.
models = {
    "dreamshaperXL": {
        "type": "hf",
        "url": "Lykon/DreamShaper",
    },
    "RealisticVisionV51": {
        "type": "civ",
        "url": "https://civitai.com/api/download/models/4201",
    },
    "AOM5": {
        "type": "civ",
        "url": "https://civitai.com/api/download/models/9942",
    },
}

success = []
failed = []


def _download_file(url: str, destination: Path, token: str | None = None) -> None:
    """Stream a file to disk to avoid loading the full model in memory."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        with requests.get(url, headers=headers, stream=True, timeout=120) as response:
            response.raise_for_status()
            with destination.open("wb") as file_obj:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file_obj.write(chunk)
    except requests.Timeout as error:
        raise RuntimeError(f"Request timed out for {url}") from error


print(f"\n======= START DOWNLOAD TO {MODELS_DIR} =======\n")

for name, cfg in models.items():
    model_type = cfg["type"]
    model_url = cfg["url"]

    try:
        if model_type == "hf":
            target_dir = MODELS_DIR / name
            if target_dir.exists():
                print(f"✓ {name} already exists")
                success.append(name)
                continue

            print(f"Downloading {name} from Hugging Face...")
            snapshot_download(
                repo_id=model_url,
                token=HF_TOKEN,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
            )

        elif model_type in {"civ", "http"}:
            # Store single-file models directly in /models with the model name.
            target_file = MODELS_DIR / f"{name}.safetensors"
            if target_file.exists():
                print(f"✓ {name} already exists")
                success.append(name)
                continue

            source_label = "Civitai" if model_type == "civ" else "HTTP"
            print(f"Downloading {name} from {source_label}...")
            _download_file(model_url, target_file, CIV_TOKEN if model_type == "civ" else None)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        print(f"✓ Downloaded {name}")
        success.append(name)

    except (requests.RequestException, OSError, RuntimeError, ValueError) as error:
        print(f"⚠️ Skipping {name}: {error}")
        failed.append(name)

print("Downloaded:", success)
print("Failed:", failed)
print("\n======= DOWNLOAD FINISH =======")

# Keep docker command successful even if one optional model fails.
sys.exit(0)
