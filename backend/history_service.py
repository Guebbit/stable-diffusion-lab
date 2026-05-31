"""
Generation history persistence.

Saves every generated image (full metadata + base64 data URL) to an individual
JSON file on disk so the gallery survives container restarts.

Storage layout:
  <MODELS_CACHE_DIR>/history/<image-uuid>.json   ← one file per image

Why one file per image?
  - Deletion is a single os.remove() — no need to load and rewrite a master list.
  - Reads are parallelisable and there is no giant JSON blob to parse.
  - Ordering is done in-memory at list time (sort by created_at descending).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from logging_config import get_logger
from schemas import GeneratedImage

logger = get_logger(__name__)

# ─── Configuration ──────────────────────────────────────────────────────────

MODELS_CACHE_DIR = Path(os.environ.get("MODELS_CACHE_DIR", "/app/models_cache"))
HISTORY_DIR = MODELS_CACHE_DIR / "history"


# ─── Internal helpers ────────────────────────────────────────────────────────

def _ensure_history_dir() -> None:
    """Create the history directory if it does not exist yet."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _entry_path(image_id: str) -> Path:
    """Return the JSON file path for a given image UUID.

    Security: strip any path separators so a crafted ID like "../../etc/passwd"
    cannot escape the history directory.
    """
    safe_id = os.path.basename(image_id)
    return HISTORY_DIR / f"{safe_id}.json"


# ─── Public API ─────────────────────────────────────────────────────────────

def save_generation(images: list[GeneratedImage]) -> None:
    """Persist each image from a batch to its own JSON file.

    Called right after _build_generation_response() in main.py.
    Errors are logged but never raised — a history write failure must not
    prevent the API from returning the generated image to the user.
    """
    _ensure_history_dir()
    for image in images:
        try:
            path = _entry_path(image.id)
            with open(path, "w", encoding="utf-8") as f:
                # model_dump() serialises the Pydantic model to a plain dict
                json.dump(image.model_dump(), f, ensure_ascii=False)
            logger.debug("History saved: %s", image.id)
        except Exception:
            logger.exception("Failed to save history entry %s", image.id)


def list_history() -> list[GeneratedImage]:
    """Return all persisted images, newest first.

    Scans the history directory, loads each JSON file into a GeneratedImage,
    and sorts by created_at (ISO-8601 strings sort lexicographically).
    Corrupted files are skipped with a warning.
    """
    _ensure_history_dir()
    entries: list[GeneratedImage] = []

    for json_file in HISTORY_DIR.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries.append(GeneratedImage(**data))
        except Exception:
            logger.warning("Skipping corrupted history file: %s", json_file.name)

    # Sort newest → oldest (ISO-8601 timestamps compare correctly as strings)
    entries.sort(key=lambda img: img.created_at, reverse=True)
    return entries


def delete_history_entry(image_id: str) -> bool:
    """Delete a single history entry by image UUID.

    Returns True if the file was found and removed, False if it didn't exist.
    """
    path = _entry_path(image_id)
    if not path.exists():
        return False
    try:
        path.unlink()
        logger.info("History entry deleted: %s", image_id)
        return True
    except Exception:
        logger.exception("Failed to delete history entry %s", image_id)
        return False


def clear_history() -> int:
    """Delete ALL history entries. Returns the number of files removed."""
    _ensure_history_dir()
    count = 0
    for json_file in HISTORY_DIR.glob("*.json"):
        try:
            json_file.unlink()
            count += 1
        except Exception:
            logger.warning("Could not delete history file: %s", json_file.name)
    logger.info("History cleared — %d entries removed", count)
    return count


def get_history_entry(image_id: str) -> Optional[GeneratedImage]:
    """Load and return a single history entry by UUID, or None if not found."""
    path = _entry_path(image_id)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return GeneratedImage(**data)
    except Exception:
        logger.exception("Failed to load history entry %s", image_id)
        return None
