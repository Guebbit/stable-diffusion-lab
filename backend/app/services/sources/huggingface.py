"""
Hugging Face model source provider.

Handles model metadata resolution and file downloads from the
Hugging Face Hub. Supports multi-file models with resume capability.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class HuggingFaceSourceProvider:
    """
    Downloads models from the Hugging Face Hub.

    Uses the huggingface_hub library for:
    - Listing files in a repository (model_info API)
    - Downloading individual files with resume support
    - Fetching commit hashes for integrity verification
    """

    def __init__(self, token: str | None = None) -> None:
        """
        Initialize with optional HF token for private/gated models.

        Args:
            token: Hugging Face API token. None for public models only.
        """
        self._token = token

    async def resolve_model_info(self, model_id: str) -> dict[str, Any]:
        """
        Fetch file manifest from HF Hub API.

        Returns a dict with:
        - files: list of {relative_path, size_bytes, sha256}
        - total_size_bytes: sum of all file sizes
        """
        # TODO: Implement using huggingface_hub.HfApi
        raise NotImplementedError("HuggingFace source provider not yet implemented")

    async def download_file(
        self,
        model_id: str,
        file_path: str,
        destination: Path,
        resume_from_byte: int = 0,
        on_progress: Any = None,
    ) -> dict[str, Any]:
        """
        Download a single file with resume support.

        Args:
            model_id: HF repo ID (e.g., "stabilityai/stable-diffusion-xl-base-1.0")
            file_path: Relative path within the repo
            destination: Local path to write the file
            resume_from_byte: Byte offset to resume from (0 for fresh download)
            on_progress: Callback for progress updates

        Returns:
            Dict with {sha256, size_bytes} for verification
        """
        # TODO: Implement using huggingface_hub.hf_hub_download with resume
        raise NotImplementedError("HuggingFace file download not yet implemented")

    def supports_resume(self) -> bool:
        """HuggingFace Hub supports byte-range resume."""
        return True
