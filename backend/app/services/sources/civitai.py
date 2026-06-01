"""
Civitai model source provider.

Handles model metadata resolution and file downloads from Civitai.
Civitai models are typically single-file (safetensors/ckpt).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CivitaiSourceProvider:
    """
    Downloads models from the Civitai platform.

    Uses the Civitai API for:
    - Fetching model metadata and download URLs
    - Downloading single-file checkpoints
    - Handling Civitai's authentication for restricted models
    """

    def __init__(self, api_key: str | None = None) -> None:
        """
        Initialize with optional Civitai API key.

        Args:
            api_key: Civitai API key for authenticated downloads.
        """
        self._api_key = api_key

    async def resolve_model_info(self, model_id: str) -> dict[str, Any]:
        """
        Fetch model metadata from Civitai API.

        Returns a dict with:
        - files: list of {relative_path, size_bytes, sha256}
        - total_size_bytes: total download size
        """
        # TODO: Implement using Civitai REST API
        raise NotImplementedError("Civitai source provider not yet implemented")

    async def download_file(
        self,
        model_id: str,
        file_path: str,
        destination: Path,
        resume_from_byte: int = 0,
        on_progress: Any = None,
    ) -> dict[str, Any]:
        """
        Download a model file from Civitai.

        Civitai models are usually single files, but the interface stays
        consistent with multi-file sources for uniformity.
        """
        # TODO: Implement using httpx with range headers for resume
        raise NotImplementedError("Civitai file download not yet implemented")

    def supports_resume(self) -> bool:
        """Civitai supports byte-range resume for most files."""
        return True
