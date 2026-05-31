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
            "tags": ["general", "fast", "classic"],
        },
        {
            "id": "stabilityai/stable-diffusion-2-1",
            "name": "Stable Diffusion v2.1",
            "source": "huggingface",
            "family": "sd15",
            "description": "SD 2.1 — improved anatomy, 768 px native resolution",
            "tags": ["general", "detailed"],
        },
        {
            "id": "CompVis/stable-diffusion-v1-4",
            "name": "Stable Diffusion v1.4",
            "source": "huggingface",
            "family": "sd15",
            "description": "The original SD 1.4 by CompVis — lightweight and historical",
            "tags": ["classic", "fast", "lightweight"],
        },
        {
            "id": "prompthero/openjourney",
            "name": "OpenJourney v4",
            "source": "huggingface",
            "family": "sd15",
            "description": "Midjourney-inspired style — vivid, painterly outputs",
            "tags": ["artistic", "stylized", "midjourney"],
        },
        {
            "id": "dreamlike-art/dreamlike-photoreal-2.0",
            "name": "Dreamlike Photoreal 2.0",
            "source": "huggingface",
            "family": "sd15",
            "description": "Photorealistic fine-tune — detailed skin tones and lighting",
            "tags": ["photorealistic", "portraits", "detailed"],
        },
        {
            "id": "Lykon/dreamshaper-8",
            "name": "DreamShaper 8",
            "source": "huggingface",
            "family": "sd15",
            "description": "Highly versatile — photos, art, fantasy at 512 px",
            "tags": ["versatile", "photorealistic", "artistic"],
        },
        {
            "id": "stabilityai/stable-diffusion-xl-base-1.0",
            "name": "Stable Diffusion XL 1.0",
            "source": "huggingface",
            "family": "sdxl",
            "description": "SDXL base — high detail and color fidelity at 1024 px",
            "tags": ["high-quality", "detailed", "large"],
        },
        {
            "id": "stabilityai/sdxl-turbo",
            "name": "SDXL Turbo",
            "source": "huggingface",
            "family": "sdxl",
            "description": "Distilled SDXL — near-instant results in 1–4 steps",
            "tags": ["fast", "real-time", "sdxl"],
        },
        {
            "id": "128713",
            "name": "DreamShaper 8",
            "source": "civitai",
            "family": "sd15",
            "description": "Versatile art/photo model — version 128713",
            "tags": ["versatile", "photorealistic", "artistic"],
        },
        {
            "id": "130072",
            "name": "Realistic Vision V5.1",
            "source": "civitai",
            "family": "sd15",
            "description": "Hyper-photorealistic portraits and scenes — version 130072",
            "tags": ["photorealistic", "portraits", "detailed"],
        },
        {
            "id": "403131",
            "name": "majicMIX Realistic v7",
            "source": "civitai",
            "family": "sd15",
            "description": "Asian-beauty–focused photorealism — version 403131",
            "tags": ["photorealistic", "portraits", "asian-style"],
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
    """Return all registered models with their current download status."""
    registry = _load_registry()
    for entry in registry:
        entry["downloaded"] = is_model_downloaded(entry["id"], entry["source"])
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
    tags: Optional[list[str]] = None,
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
        "tags": tags or [],
    }
    registry.append(new_entry)
    _save_registry(registry)

    new_entry["downloaded"] = is_model_downloaded(model_id, source)
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
    response = requests.get(url, headers=headers, stream=True, timeout=600)
    if response.status_code != 200:
        raise RuntimeError(
            f"CivitAI download failed with status {response.status_code}: {response.text[:200]}"
        )

    with open(checkpoint_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)

    logger.info("CivitAI model downloaded to %s", checkpoint_path)
