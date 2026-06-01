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
from download_history import save_download_event, list_download_events, clear_download_events

logger = get_logger(__name__)

# ─── Configuration ─────────────────────────────────────────────────────────

MODELS_CACHE_DIR = Path(os.environ.get("MODELS_CACHE_DIR", "/app/models_cache"))
REGISTRY_FILE = MODELS_CACHE_DIR / "registry.json"

# HuggingFace token and CivitAI token — reused from model_service env vars
HF_TOKEN = os.environ.get("HF_TOKEN", "")
CIV_TOKEN = os.environ.get("CIV_TOKEN", "")


# ─── Registry entry shape ──────────────────────────────────────────────────

def _default_registry() -> list[dict]:
    """Seed registry with models from MODELS_TO_LIST."""
    return [
        # ─── HUGGINGFACE MODELS ───────────────────────────────────────
        # MULTIPURPOSE (best overall) — FLUX.1 Dev
        {
            "id": "black-forest-labs/FLUX.1-dev",
            "name": "FLUX.1 Dev",
            "source": "huggingface",
            "family": "flux",
            "description": "Multipurpose — best overall quality, highly versatile",
            "long_description": (
                "FLUX.1 Dev by Black Forest Labs is a 12 billion parameter rectified flow model "
                "conditioned on T5-XXL + CLIP-L. It produces highly detailed, photorealistic images "
                "with excellent text rendering and prompt adherence. Use with Euler solver, 20–40 steps, "
                "CFG 1.0–3.5. Native resolution 1024×1024. Recommended for general-purpose generation."
            ),
            "tags": ["general", "flux", "best-quality"],
            "source_url": "https://huggingface.co/black-forest-labs/FLUX.1-dev",
            "size": "~23 GB",
        },
        # FASTEST TEST MODEL — FLUX.1 Schnell
        {
            "id": "black-forest-labs/FLUX.1-schnell",
            "name": "FLUX.1 Schnell",
            "source": "huggingface",
            "family": "flux",
            "description": "Fastest model — 1–4 steps for rapid iteration",
            "long_description": (
                "FLUX.1 Schnell is the distilled, faster variant of FLUX.1 Dev. "
                "Uses rectified flow to produce high-quality images in just 1–4 steps. "
                "Ideal for rapid prompt testing and real-time generation. "
                "Works best with Euler solver and guidance=0–2. "
                "Slightly lower detail than Dev but much faster."
            ),
            "tags": ["general", "flux", "fast", "real-time"],
            "source_url": "https://huggingface.co/black-forest-labs/FLUX.1-schnell",
            "size": "~23 GB",
        },
        # FASTEST SDXL TEST MODEL — SDXL Lightning
        {
            "id": "ByteDance/SDXL-Lightning",
            "name": "SDXL Lightning",
            "source": "huggingface",
            "family": "sdxl",
            "description": "SDXL distilled — near-instant results in 2–4 steps",
            "long_description": (
                "SDXL-Lightning by ByteDance uses adversarial diffusion distillation "
                "to compress SDXL into a 2–4 step sampler. Ideal for rapid iteration on SDXL architecture. "
                "Use with Euler A or UniPC solver, CFG 0–2. "
                "Native resolution 1024×1024. Requires ≈8 GB VRAM."
            ),
            "tags": ["general", "sdxl", "fast", "lightweight"],
            "source_url": "https://huggingface.co/ByteDance/SDXL-Lightning",
            "size": "~6.9 GB",
        },
        # INKING / CLEAN LINEART — Mistoline SDXL
        {
            "id": "XorAIS/mistoline-sdxl-fp16",
            "name": "Mistoline SDXL",
            "source": "huggingface",
            "family": "sdxl",
            "description": "Inking / clean lineart from sketches — ControlNet-compatible",
            "long_description": (
                "Mistoline is a ControlNet-based line-art model for SDXL that turns rough sketches "
                "into clean, professional lineart. Ideal for converting rough concept sketches into polished "
                "inked lines. Compatible with standard ControlNet workflows. "
                "Use with CFG 3–5, DPM++ 2M Karras, 20–30 steps."
            ),
            "tags": ["lineart", "controlnet", "sdxl", "inking"],
            "source_url": "https://huggingface.co/XorAIS/mistoline-sdxl-fp16",
            "size": "~6.7 GB",
        },
        # ─── CIVITAI MODELS ─────────────────────────────────────────────
        # PHOTOREALISM — Juggernaut XL
        {
            "id": "2144",
            "name": "Juggernaut XL",
            "source": "civitai",
            "family": "sdxl",
            "description": "Photorealism — best model for realistic photography",
            "long_description": (
                "Juggernaut XL by Khaoz A.I. is a photorealism-focused SDXL checkpoint. "
                "Excels at landscapes, portraits, and product photography with natural lighting "
                "and skin tones. Use CFG 4–7, DPM++ 2M Karras, 20–30 steps. "
                "Add 'photograph' or 'photo' to prompt for best results."
            ),
            "tags": ["photorealistic", "sdxl", "photography"],
            "source_url": "https://civitai.com/models/136245/juggernaut-xl",
            "size": "~6.7 GB",
        },
        # REALISTIC PORTRAITS — RealVisXL
        {
            "id": "7331",
            "name": "RealVisXL",
            "source": "civitai",
            "family": "sdxl",
            "description": "Realistic portraits — high-fidelity facial details",
            "long_description": (
                "RealVisXL by SG_161222 is a SDXL fine-tune optimized for realistic portrait generation. "
                "Produces lifelike skin textures, natural poses, and accurate facial proportions. "
                "Use CFG 3–5, DPM++ 2M Karras, 25–35 steps. "
                "Negative prompt: 'cartoon, painting, illustration, drawing' to stay photorealistic."
            ),
            "tags": ["photorealistic", "portraits", "sdxl"],
            "source_url": "https://civitai.com/models/7331/realvisxl",
            "size": "~6.7 GB",
        },
        # CINEMATIC PHOTOGRAPHY — Colossus Project Flux
        {
            "id": "860",
            "name": "Colossus Project Flux",
            "source": "civitai",
            "family": "flux",
            "description": "Cinematic photography — film-like composition and lighting",
            "long_description": (
                "Colossus Project Flux is a FLUX.1 fine-tune specialized for cinematic photography. "
                "Produces dramatic lighting, film-like color grading, and professional composition. "
                "Works with FLUX's native 1024 resolution. Use CFG 1–3, Euler, 20–40 steps. "
                "Add 'cinematic', 'film still' to prompt for best results."
            ),
            "tags": ["photorealistic", "cinematic", "flux", "dramatic"],
            "source_url": "https://civitai.com/models/860/colossus-project",
            "size": "~23 GB",
        },
        # ART / CONCEPT ART — DreamShaper XL
        {
            "id": "39282",
            "name": "DreamShaper XL",
            "source": "civitai",
            "family": "sdxl",
            "description": "Art / Concept art — vibrant stylized and fantasy art",
            "long_description": (
                "DreamShaper XL by Lykon is an SDXL fine-tune optimized for concept art, "
                "fantasy illustrations, and stylized portraits. Covers a wide range of artistic styles "
                "from semi-realistic to highly stylized. Use CFG 5–7, DPM++ 2M Karras, 20–30 steps."
            ),
            "tags": ["artistic", "concept-art", "fantasy", "sdxl"],
            "source_url": "https://civitai.com/models/39282/dreamshaper-xl",
            "size": "~6.7 GB",
        },
        # ANIME (general) — Illustrious XL
        {
            "id": "31969",
            "name": "Illustrious XL",
            "source": "civitai",
            "family": "sdxl",
            "description": "Anime — general-purpose anime and illustration style",
            "long_description": (
                "Illustrious XL by CyberC. A high-quality anime-focused SDXL checkpoint. "
                "Excels at character design, illustrations, and anime-style portraits. "
                "Use CFG 5–7, DPM++ 2M Karras, 20–30 steps. "
                "Pairs well with anime-specific LoRAs and ControlNets."
            ),
            "tags": ["anime", "illustration", "sdxl"],
            "source_url": "https://civitai.com/models/31969/illustrious-xl",
            "size": "~6.7 GB",
        },
        # ANIME (largest ecosystem) — Pony Diffusion V6 XL
        {
            "id": "53806",
            "name": "Pony Diffusion V6 XL",
            "source": "civitai",
            "family": "sdxl",
            "description": "Anime — largest ecosystem, extensive LoRA support",
            "long_description": (
                "Pony Diffusion V6 XL is a custom SDXL-based model that has spawned the largest "
                "community for anime/furry-focused generation. Huge ecosystem of LoRAs, embeddings, "
                "and ControlNets built specifically for this model. Use CFG 3.5–5, DPM++ 2M Karras, "
                "20–30 steps. Has its own prompt tags (e.g. '1boy', '1girl' tags system)."
            ),
            "tags": ["anime", "pony", "large-ecosystem", "sdxl"],
            "source_url": "https://civitai.com/models/53806/pony-diffusion-v6-xl",
            "size": "~6.7 GB",
        },
        # FURRY — Furry Diffusion XL
        {
            "id": "226",
            "name": "Furry Diffusion XL",
            "source": "civitai",
            "family": "sdxl",
            "description": "Furry art — dedicated furry-style generation",
            "long_description": (
                "Furry Diffusion XL by Neklader. A specialized SDXL checkpoint for furry-style "
                "character art and illustration. Supports anthropomorphic characters with natural "
                "fur rendering, expressive anatomy, and stylized proportions. "
                "Use CFG 5–7, DPM++ 2M Karras, 20–30 steps."
            ),
            "tags": ["furry", "anime", "sdxl"],
            "source_url": "https://civitai.com/models/226/furry-diffusion-xl",
            "size": "~6.7 GB",
        },
        # NSFW ANIME — NoobAI XL
        {
            "id": "26556",
            "name": "NoobAI XL",
            "source": "civitai",
            "family": "sdxl",
            "description": "NSFW anime — adult-oriented anime generation",
            "long_description": (
                "NoobAI XL is an SDXL fine-tune specialized for NSFW anime-style generation. "
                "Strong character consistency and detailed anatomical rendering. "
                "Use CFG 6–7, DPM++ 2M Karras, 25–30 steps. "
                "Works well with anime-specific embeddings."
            ),
            "tags": ["nsfw", "anime", "sdxl"],
            "source_url": "https://civitai.com/models/26556/noobai-xl",
            "size": "~6.7 GB",
        },
        # NSFW FURRY — Indigo Furry Mix
        {
            "id": "3495",
            "name": "Indigo Furry Mix",
            "source": "civitai",
            "family": "sdxl",
            "description": "NSFW furry — detailed furry-style adult art",
            "long_description": (
                "Indigo Furry Mix by SFW. A high-quality SDXL checkpoint for furry-style "
                "generation with excellent fur rendering, natural anatomy, and stylized proportions. "
                "One of the most popular furry models on CivitAI. "
                "Use CFG 5–7, DPM++ 2M Karras, 25–35 steps."
            ),
            "tags": ["nsfw", "furry", "sdxl"],
            "source_url": "https://civitai.com/models/3495/indigo-furry-mix",
            "size": "~6.7 GB",
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

    Models that are actively being downloaded will have downloaded=False
    (even if a partial snapshot exists on disk) so the UI doesn't
    prematurely show them as available.
    """
    registry = _load_registry()
    for entry in registry:
        entry["downloaded"] = is_model_downloaded(entry["id"], entry["source"])
        # Override: if downloading, report not-yet-downloaded
        if is_downloading(entry["id"], entry["source"]):
            entry["downloaded"] = False
    return registry


def get_download_progress(model_id: str, source: str) -> dict:
    """Return download progress for a model: {downloaded_bytes, total_bytes, percentage}.

    Returns None if no download is in progress or no progress data exists.
    """
    key = f"{source}:{model_id}"
    with _progress_lock:
        progress = _download_progress.get(key)
    if not progress:
        return None
    downloaded = progress.get("downloaded_bytes", 0)
    total = progress.get("total_bytes")
    percentage = 0
    if total and total > 0:
        percentage = min(100, int(downloaded / total * 100))
    return {
        "downloaded_bytes": downloaded,
        "total_bytes": total,
        "percentage": percentage,
    }


def list_downloaded_models() -> list[dict]:
    """Return only models that are actually present on disk."""
    return [m for m in list_models() if m["downloaded"]]


# ─── Download Events API ──────────────────────────────────────────────────────

def list_download_events_api() -> list[dict]:
    """Return all persisted download events, newest first."""
    return list_download_events()


def clear_download_events_api() -> int:
    """Clear all download events. Returns number of files removed."""
    return clear_download_events()


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

# Tracks download progress: key -> {"downloaded_bytes": int, "total_bytes": int | None}
_download_progress: dict[str, dict] = {}
_progress_lock = threading.Lock()


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
            save_download_event(source, model_id, "started", "Download already in progress")
            return
        _downloading.add(key)

    # Log download start event
    save_download_event(source, model_id, "started", f"Download started for {source} model {model_id}")

    def _do_download():
        try:
            if source == "huggingface":
                _download_huggingface_model(model_id)
            elif source == "civitai":
                _download_civitai_model(model_id)
            logger.info("Download complete: %s", key)
            save_download_event(source, model_id, "completed", f"Download completed for {model_id}")
        except Exception:
            logger.exception("Download failed: %s", key)
            save_download_event(source, model_id, "failed", f"Download failed: {str(__import__('sys').exc_info()[1])}")
        finally:
            with _download_lock:
                _downloading.discard(key)

    thread = threading.Thread(target=_do_download, daemon=True)
    thread.start()


def _download_huggingface_model(model_id: str) -> None:
    """Download a HuggingFace model to cache without loading into GPU memory."""
    from huggingface_hub import snapshot_download, HfFileSystem

    logger.info("Downloading HuggingFace model: %s", model_id)
    key = f"huggingface:{model_id}"

    # Get total size for progress tracking
    total_bytes = 0
    try:
        fs = HfFileSystem()
        repo_files = fs.ls(model_id, recursive=True)
        total_bytes = sum(fs.info(f)["size"] for f in repo_files if fs.info(f).get("size"))
    except Exception:
        pass

    with _progress_lock:
        _download_progress[key] = {
            "downloaded_bytes": 0,
            "total_bytes": total_bytes if total_bytes > 0 else None,
        }

    # Use tqdm callback to track download progress
    downloaded_bytes = [0]  # use list for mutability in closure

    def _progress_callback(bytes_downloaded: int) -> None:
        downloaded_bytes[0] += bytes_downloaded
        with _progress_lock:
            _download_progress[key] = {
                "downloaded_bytes": downloaded_bytes[0],
                "total_bytes": total_bytes if total_bytes > 0 else None,
            }

    logger.info("Downloading HuggingFace model: %s", model_id)
    snapshot_download(
        repo_id=model_id,
        cache_dir=str(MODELS_CACHE_DIR),
        token=HF_TOKEN or None,
        callback=_progress_callback,
    )

    # Mark 100% complete
    with _progress_lock:
        _download_progress[key] = {
            "downloaded_bytes": total_bytes if total_bytes > 0 else 1,
            "total_bytes": total_bytes if total_bytes > 0 else 1,
        }


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
    key = f"civitai:{model_version_id}"

    # Initialize progress tracking
    with _progress_lock:
        _download_progress[key] = {
            "downloaded_bytes": 0,
            "total_bytes": None,  # Content-Length may not be reliable for large downloads
        }

    # 600s timeout: model checkpoints can be 2–8+ GB, needs generous time for slow connections
    response = requests.get(url, headers=headers, stream=True, timeout=600)
    if response.status_code != 200:
        raise RuntimeError(
            f"CivitAI download failed with status {response.status_code}: {response.text[:200]}"
        )

    # Use Content-Length if available
    total_bytes = None
    content_length = response.headers.get("content-length")
    if content_length:
        try:
            total_bytes = int(content_length)
            with _progress_lock:
                _download_progress[key] = {
                    "downloaded_bytes": 0,
                    "total_bytes": total_bytes,
                }
        except ValueError:
            pass

    downloaded_bytes = [0]

    with open(checkpoint_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded_bytes[0] += len(chunk)
                with _progress_lock:
                    _download_progress[key] = {
                        "downloaded_bytes": downloaded_bytes[0],
                        "total_bytes": total_bytes,
                    }

    # Mark 100% complete
    actual_total = total_bytes if total_bytes else downloaded_bytes[0]
    with _progress_lock:
        _download_progress[key] = {
            "downloaded_bytes": downloaded_bytes[0],
            "total_bytes": actual_total,
        }

    logger.info("CivitAI model downloaded to %s", checkpoint_path)
