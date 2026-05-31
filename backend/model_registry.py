"""
Centralized model registry — persists known models to a JSON file.

Responsibilities:
 - CRUD operations on the model catalog (add, list, remove)
 - Detecting whether a model is actually downloaded on disk
 - Triggering background downloads for HuggingFace / CivitAI models

The registry file lives in MODELS_CACHE_DIR/registry.json so it survives
container restarts (volume-mounted alongside the model weights).
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional

from logging_config import get_logger

logger = get_logger(__name__)

# ─── Configuration ─────────────────────────────────────────────────────────

MODELS_CACHE_DIR = Path(os.environ.get("MODELS_CACHE_DIR", "/app/models_cache"))
REGISTRY_FILE = MODELS_CACHE_DIR / "registry.json"

# HuggingFace token and CivitAI token — reused from model_service env vars
HF_TOKEN = os.environ.get("HF_TOKEN", "")
CIV_TOKEN = os.environ.get("CIV_TOKEN", "")


# ─── Registry entry shape ──────────────────────────────────────────────────

def _default_registry() -> list[dict]:
    """Seed registry with well-known models on first run."""
    return [
        {
            "id": "runwayml/stable-diffusion-v1-5",
            "name": "Stable Diffusion v1.5",
            "source": "huggingface",
            "family": "sd15",
            "description": "The classic SD 1.5 base — fast, widely compatible",
            "long_description": (
                "The canonical Stable Diffusion 1.5 checkpoint, fine-tuned by Runway from the CompVis SD 1.4 base. "
                "Native resolution is 512 × 512. It is the most widely supported model: virtually every "
                "LoRA, ControlNet, and community extension targets it. A good all-purpose starting point."
            ),
            "tags": ["general", "fast", "classic"],
            "source_url": "https://huggingface.co/runwayml/stable-diffusion-v1-5",
            "size": "~4.3 GB",
        },
        {
            "id": "stabilityai/stable-diffusion-2-1",
            "name": "Stable Diffusion v2.1",
            "source": "huggingface",
            "family": "sd15",
            "description": "SD 2.1 — improved anatomy, 768 px native resolution",
            "long_description": (
                "Stability AI's second-generation base model. Trained with a new OpenCLIP text encoder "
                "(instead of CLIP), which handles long, complex prompts better. Native resolution is "
                "768 × 768. Note: LoRAs trained for SD 1.5 are NOT compatible with this model family."
            ),
            "tags": ["general", "detailed"],
            "source_url": "https://huggingface.co/stabilityai/stable-diffusion-2-1",
            "size": "~5.2 GB",
        },
        {
            "id": "CompVis/stable-diffusion-v1-4",
            "name": "Stable Diffusion v1.4",
            "source": "huggingface",
            "family": "sd15",
            "description": "The original SD 1.4 by CompVis — lightweight and historical",
            "long_description": (
                "The original public release of Stable Diffusion by CompVis (LMU Munich). "
                "Trained on LAION-Aesthetics v2 at 512 × 512. Lighter than 1.5 and useful for "
                "historical comparisons or resource-limited machines. Most community content now "
                "targets v1.5 instead."
            ),
            "tags": ["classic", "fast", "lightweight"],
            "source_url": "https://huggingface.co/CompVis/stable-diffusion-v1-4",
            "size": "~4.3 GB",
        },
        {
            "id": "prompthero/openjourney",
            "name": "OpenJourney v4",
            "source": "huggingface",
            "family": "sd15",
            "description": "Midjourney-inspired style — vivid, painterly outputs",
            "long_description": (
                "A fine-tune of SD 1.5 on Midjourney v4 outputs by PromptHero. "
                "Trigger the style with the prefix \"mdjrny-v4 style\" in your prompt. "
                "Produces vivid, painterly images with the characteristic Midjourney look "
                "— great for concept art and stylized illustrations."
            ),
            "tags": ["artistic", "stylized", "midjourney"],
            "source_url": "https://huggingface.co/prompthero/openjourney",
            "size": "~2.4 GB",
        },
        {
            "id": "dreamlike-art/dreamlike-photoreal-2.0",
            "name": "Dreamlike Photoreal 2.0",
            "source": "huggingface",
            "family": "sd15",
            "description": "Photorealistic fine-tune — detailed skin tones and lighting",
            "long_description": (
                "A photorealism-focused fine-tune of SD 1.5 by Dreamlike Art. "
                "Excels at realistic portraits, landscapes, and product shots. "
                "Best results at 768 × 512 or higher. Add \"photo\" or \"photograph\" in the "
                "prompt to steer the model toward photographic output."
            ),
            "tags": ["photorealistic", "portraits", "detailed"],
            "source_url": "https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0",
            "size": "~2.4 GB",
        },
        {
            "id": "Lykon/dreamshaper-8",
            "name": "DreamShaper 8",
            "source": "huggingface",
            "family": "sd15",
            "description": "Highly versatile — photos, art, fantasy at 512 px",
            "long_description": (
                "DreamShaper 8 by Lykon is one of the most popular community fine-tunes of SD 1.5. "
                "Covers a wide range of styles: photorealistic portraits, fantasy art, and concept design. "
                "Also available as a CivitAI checkpoint with the same weights. "
                "Works best with DPM++ 2M Karras sampler at 20–30 steps."
            ),
            "tags": ["versatile", "photorealistic", "artistic"],
            "source_url": "https://huggingface.co/Lykon/dreamshaper-8",
            "size": "~2.2 GB",
        },
        {
            "id": "stabilityai/stable-diffusion-xl-base-1.0",
            "name": "Stable Diffusion XL 1.0",
            "source": "huggingface",
            "family": "sdxl",
            "description": "SDXL base — high detail and color fidelity at 1024 px",
            "long_description": (
                "Stability AI's SDXL 1.0 base model. Native resolution is 1024 × 1024. "
                "Uses a two-stage architecture (base + optional refiner) and a much larger "
                "U-Net than SD 1.5, producing significantly more detailed and coherent images. "
                "Requires more VRAM (≈8 GB for float16). Recommended GPU: RTX 3080 or better."
            ),
            "tags": ["high-quality", "detailed", "large"],
            "source_url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0",
            "size": "~6.9 GB",
        },
        {
            "id": "stabilityai/sdxl-turbo",
            "name": "SDXL Turbo",
            "source": "huggingface",
            "family": "sdxl",
            "description": "Distilled SDXL — near-instant results in 1–4 steps",
            "long_description": (
                "SDXL Turbo uses Adversarial Diffusion Distillation (ADD) to compress "
                "the full SDXL sampling process into 1–4 steps without a major quality loss. "
                "Ideal for rapid iteration and real-time applications. "
                "Use guidance_scale=0 (or very low) and num_inference_steps=1–4 for best results. "
                "Not recommended for sketch2ink (ControlNet needs more steps)."
            ),
            "tags": ["fast", "real-time", "sdxl"],
            "source_url": "https://huggingface.co/stabilityai/sdxl-turbo",
            "size": "~6.7 GB",
        },
        {
            "id": "128713",
            "name": "DreamShaper 8",
            "source": "civitai",
            "family": "sd15",
            "description": "Versatile art/photo model — version 128713",
            "long_description": (
                "DreamShaper 8 by Lykon (CivitAI version 128713). "
                "One of the most downloaded models on CivitAI. Handles portraits, fantasy, "
                "concept art and landscapes equally well. "
                "For best results use DPM++ 2M Karras, 20–30 steps, CFG 4–7."
            ),
            "tags": ["versatile", "photorealistic", "artistic"],
            "source_url": "https://civitai.com/models/4384?modelVersionId=128713",
            "size": "~2.1 GB",
        },
        {
            "id": "130072",
            "name": "Realistic Vision V5.1",
            "source": "civitai",
            "family": "sd15",
            "description": "Hyper-photorealistic portraits and scenes — version 130072",
            "long_description": (
                "Realistic Vision V5.1 by SG_161222 (CivitAI version 130072). "
                "Focused on extreme photorealism — skin pores, fabric textures, natural lighting. "
                "Pairs well with a VAE (Variational Autoencoder) for sharper colors: vae-ft-mse-840000-ema-pruned.safetensors. "
                "Negative prompt should include \"cartoon, painting, illustration\" to steer away from art styles."
            ),
            "tags": ["photorealistic", "portraits", "detailed"],
            "source_url": "https://civitai.com/models/4201?modelVersionId=130072",
            "size": "~2.1 GB",
        },
        {
            "id": "403131",
            "name": "majicMIX Realistic v7",
            "source": "civitai",
            "family": "sd15",
            "description": "Asian-beauty–focused photorealism — version 403131",
            "long_description": (
                "majicMIX Realistic v7 by Merjic (CivitAI version 403131). "
                "Specialized in realistic East Asian portrait photography. "
                "Renders fine facial details, hair strands and natural skin tones exceptionally well. "
                "Works well at 512 × 768 (portrait) with 25–35 steps."
            ),
            "tags": ["photorealistic", "portraits", "asian-style"],
            "source_url": "https://civitai.com/models/43331?modelVersionId=403131",
            "size": "~2.1 GB",
        },
    ]


# ─── Persistence helpers ───────────────────────────────────────────────────

def _load_registry() -> list[dict]:
    """Read the registry JSON from disk, or create it with defaults."""
    MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not REGISTRY_FILE.exists():
        registry = _default_registry()
        _save_registry(registry)
        return registry

    with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_registry(registry: list[dict]) -> None:
    """Write the registry list to disk as pretty-printed JSON."""
    MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


# ─── Download status detection ─────────────────────────────────────────────

def _safe_resolve_path(base: Path, relative_suffix: str) -> Path:
    """Resolve a path and ensure it stays inside the base directory.

    Security: prevents path-traversal attacks where a crafted model_id
    like "../../etc/passwd" could escape the cache directory.
    """
    resolved = (base / relative_suffix).resolve()
    cache_root = base.resolve()
    # Ensure the resolved path is inside the cache directory
    if not str(resolved).startswith(str(cache_root)):
        raise RuntimeError(f"Path traversal detected: {relative_suffix}")
    return resolved


def _is_huggingface_model_downloaded(model_id: str) -> bool:
    """Check if a HuggingFace model snapshot exists in the local cache.

    HuggingFace stores downloaded repos under:
      <cache_dir>/models--<org>--<repo>/snapshots/<hash>/
    We check if at least one snapshot directory exists with content.
    """
    # HuggingFace transforms "org/repo" → "models--org--repo"
    safe_name = model_id.replace("/", "--")
    model_dir = _safe_resolve_path(MODELS_CACHE_DIR, f"models--{safe_name}")

    if not model_dir.exists():
        return False

    # Check for a non-empty snapshots folder
    # (ensures snapshot contains actual model files, not just empty directory)
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        return False

    for snapshot in snapshots_dir.iterdir():
        if snapshot.is_dir() and any(snapshot.iterdir()):
            return True

    return False


def _is_civitai_model_downloaded(model_version_id: str) -> bool:
    """Check if a CivitAI .safetensors checkpoint exists on disk."""
    import re
    normalized = model_version_id.strip()
    # Only allow numeric IDs to prevent path injection
    if not re.fullmatch(r"\d+", normalized):
        return False
    checkpoint_path = _safe_resolve_path(MODELS_CACHE_DIR, f"civitai_{normalized}.safetensors")
    return checkpoint_path.exists()


def is_model_downloaded(model_id: str, source: str) -> bool:
    """Unified check — routes to the correct detection logic by source."""
    if source == "huggingface":
        return _is_huggingface_model_downloaded(model_id)
    elif source == "civitai":
        return _is_civitai_model_downloaded(model_id)
    return False


# ─── Public API ────────────────────────────────────────────────────────────

def list_models() -> list[dict]:
    """Return all registered models with their current download status.

    While a background download thread is active we force downloaded=False so
    the frontend never shows a model as "Ready" before the files are complete.
    """
    registry = _load_registry()
    for entry in registry:
        # True when a background thread is actively downloading this specific model
        currently_downloading = is_downloading(entry["id"], entry["source"])
        entry["downloading"] = currently_downloading
        # A partial HuggingFace snapshot can pass the filesystem check early,
        # so we rely on the in-progress flag as the authoritative gate.
        entry["downloaded"] = (not currently_downloading) and is_model_downloaded(
            entry["id"], entry["source"]
        )
    return registry


def list_downloaded_models() -> list[dict]:
    """Return only models that are actually present on disk."""
    return [m for m in list_models() if m["downloaded"]]


def add_model(
    model_id: str,
    name: str,
    source: str,
    family: str,
    description: str = "",
    long_description: str = "",
    tags: Optional[list[str]] = None,
    source_url: str = "",
    size: str = "",
) -> dict:
    """Register a new model in the catalog. Returns the created entry."""
    registry = _load_registry()

    # Prevent duplicates (same id + source combo)
    for entry in registry:
        if entry["id"] == model_id and entry["source"] == source:
            raise ValueError(f"Model '{model_id}' from '{source}' already registered")

    new_entry = {
        "id": model_id,
        "name": name,
        "source": source,
        "family": family,
        "description": description,
        "long_description": long_description,
        "tags": tags or [],
        "source_url": source_url,
        "size": size,
    }
    registry.append(new_entry)
    _save_registry(registry)

    new_entry["downloaded"] = is_model_downloaded(model_id, source)
    new_entry["downloading"] = is_downloading(model_id, source)
    return new_entry


def remove_model(model_id: str, source: str) -> bool:
    """Remove a model from the registry. Returns True if found and removed."""
    registry = _load_registry()
    original_len = len(registry)
    registry = [e for e in registry if not (e["id"] == model_id and e["source"] == source)]

    if len(registry) == original_len:
        return False

    _save_registry(registry)
    return True


# ─── Background download helpers ───────────────────────────────────────────

# Tracks model IDs currently being downloaded in background threads.
# Format: "source:model_id" (e.g. "huggingface:runwayml/stable-diffusion-v1-5").
# Protected by _download_lock for thread-safe access from concurrent downloads.
_downloading: set[str] = set()
_download_lock = threading.Lock()


def is_downloading(model_id: str, source: str) -> bool:
    """Check if a model is currently being downloaded in the background."""
    key = f"{source}:{model_id}"
    return key in _downloading


def download_model_background(model_id: str, source: str) -> None:
    """Trigger a background download for the given model.

    For HuggingFace: uses from_pretrained with download_only semantics.
    For CivitAI: downloads the .safetensors file via their API.
    """
    key = f"{source}:{model_id}"

    with _download_lock:
        if key in _downloading:
            logger.info("Download already in progress for %s", key)
            return
        _downloading.add(key)

    def _do_download():
        try:
            if source == "huggingface":
                _download_huggingface_model(model_id)
            elif source == "civitai":
                _download_civitai_model(model_id)
            logger.info("Download complete: %s", key)
        except Exception:
            logger.exception("Download failed: %s", key)
        finally:
            with _download_lock:
                _downloading.discard(key)

    thread = threading.Thread(target=_do_download, daemon=True)
    thread.start()


def _download_huggingface_model(model_id: str) -> None:
    """Download a HuggingFace model to cache without loading into GPU memory."""
    from huggingface_hub import snapshot_download

    logger.info("Downloading HuggingFace model: %s", model_id)
    # snapshot_download fetches all model files (weights, config, tokenizer)
    # to the local cache directory without loading them into GPU/CPU memory.
    snapshot_download(
        repo_id=model_id,
        cache_dir=str(MODELS_CACHE_DIR),
        token=HF_TOKEN or None,
    )


def _download_civitai_model(model_version_id: str) -> None:
    """Download a CivitAI checkpoint .safetensors file."""
    import re
    import requests

    normalized = model_version_id.strip()
    if not re.fullmatch(r"\d+", normalized):
        raise RuntimeError(f"CivitAI model version ID must be numeric, got: {model_version_id}")

    MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODELS_CACHE_DIR / f"civitai_{normalized}.safetensors"

    if checkpoint_path.exists():
        logger.info("CivitAI model already cached at %s", checkpoint_path)
        return

    url = f"https://civitai.com/api/download/models/{normalized}"
    headers: dict[str, str] = {}
    if CIV_TOKEN:
        headers["Authorization"] = "Bearer " + CIV_TOKEN

    logger.info("Downloading CivitAI model version %s …", model_version_id)
    # 600s timeout: model checkpoints can be 2–8+ GB, needs generous time for slow connections
    response = requests.get(url, headers=headers, stream=True, timeout=600)
    if response.status_code != 200:
        raise RuntimeError(
            f"CivitAI download failed with status {response.status_code}: {response.text[:200]}"
        )

    with open(checkpoint_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)

    logger.info("CivitAI model downloaded to %s", checkpoint_path)
