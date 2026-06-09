"""
Shared utilities for model source providers.

Provides common functionality used across HuggingFace, Civitai,
and Local source providers.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


def compute_file_sha256(file_path: Path, chunk_size: int = 8 * 1024 * 1024) -> str | None:
    """
    Compute SHA256 hash of a file.

    Args:
        file_path: Path to the file to hash.
        chunk_size: Read chunk size in bytes (default 8MB).

    Returns:
        Hex-encoded SHA256 digest, or None if file cannot be read.
    """
    try:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except OSError as e:
        logger.warning("Could not compute SHA256 for %s: %s", file_path, e)
        return None