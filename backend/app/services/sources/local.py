"""
Local model source provider.

Handles importing models from local filesystem paths.
No download needed — validates and registers existing files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LocalSourceProvider:
    """
    Imports models from local filesystem.

    For locally-stored models:
    - Validates file existence and format
    - Computes checksums for integrity tracking
    - Moves or symlinks files into the managed storage directory
    """

    async def resolve_model_info(self, model_id: str) -> dict[str, Any]:
        """
        Resolve local model metadata by scanning the filesystem.

        Args:
            model_id: Path or identifier for the local model.

        Returns:
            Dict with file list and sizes (no download URLs needed).
        """
        # TODO: Scan local path, enumerate files, compute sizes
        raise NotImplementedError("Local source provider not yet implemented")

    async def import_model(
        self,
        source_path: Path,
        destination: Path,
        copy: bool = True,
    ) -> dict[str, Any]:
        """
        Import a local model file into managed storage.

        Args:
            source_path: Path to the model file(s) to import.
            destination: Target directory in managed storage.
            copy: If True, copy files. If False, create symlinks.

        Returns:
            Dict with {files_imported, total_bytes}
        """
        # TODO: Implement file copy/symlink logic
        raise NotImplementedError("Local model import not yet implemented")

    def supports_resume(self) -> bool:
        """Local imports don't need resume — they're filesystem operations."""
        return False
