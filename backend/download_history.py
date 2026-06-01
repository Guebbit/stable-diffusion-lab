"""
Download events persistence.

Saves every download event (start, progress, success, error) to JSON files on disk
so the gallery survives container restarts.

Storage layout:
  <MODELS_CACHE_DIR>/download_events/<source>-<model_id>-<timestamp>.json
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from logging_config import get_logger

logger = get_logger(__name__)

# ─── Configuration ──────────────────────────────────────────────────────────

MODELS_CACHE_DIR = Path(os.environ.get("MODELS_CACHE_DIR", "/app/models_cache"))
DOWNLOAD_EVENTS_DIR = MODELS_CACHE_DIR / "download_events"

# ─── Internal helpers ────────────────────────────────────────────────────────

def _ensure_events_dir() -> None:
    DOWNLOAD_EVENTS_DIR.mkdir(parents=True, exist_ok=True)


def _event_filename(source: str, model_id: str, timestamp: datetime) -> str:
    safe_source = source.replace(":", "_").replace("/", "_")
    safe_model_id = model_id.replace(":", "_").replace("/", "_")
    ts = timestamp.strftime("%Y%m%dT%H%M%S%f")
    return f"{safe_source}-{safe_model_id}-{ts}.json"


def _event_path(source: str, model_id: str, timestamp: datetime) -> Path:
    return DOWNLOAD_EVENTS_DIR / _event_filename(source, model_id, timestamp)


# ─── Public API ─────────────────────────────────────────────────────────────

def save_download_event(source: str, model_id: str, status: str, detail: str = "") -> dict:
    """Save a download event and return the event dict."""
    _ensure_events_dir()
    now = datetime.now(timezone.utc)
    event = {
        "source": source,
        "model_id": model_id,
        "status": status,
        "detail": detail,
        "timestamp": now.isoformat(),
    }
    try:
        path = _event_path(source, model_id, now)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(event, f, ensure_ascii=False)
        logger.debug("Download event saved: %s", event)
    except Exception:
        logger.exception("Failed to save download event for %s", model_id)
    return event


def list_download_events() -> list[dict]:
    """Return all persisted download events, newest first."""
    _ensure_events_dir()
    events: list[dict] = []
    for json_file in sorted(DOWNLOAD_EVENTS_DIR.glob("*.json"), reverse=True):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            events.append(data)
        except Exception:
            logger.warning("Skipping corrupted download event file: %s", json_file.name)
    return events


def clear_download_events() -> int:
    """Delete ALL download events. Returns the number of files removed."""
    _ensure_events_dir()
    count = 0
    for json_file in DOWNLOAD_EVENTS_DIR.glob("*.json"):
        try:
            json_file.unlink()
            count += 1
        except Exception:
            logger.warning("Could not delete download event file: %s", json_file.name)
    return count