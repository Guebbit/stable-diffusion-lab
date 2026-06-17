"""
Storage manager — centralized filesystem path resolution and directory management.

ALL file path operations go through this class. No other module should construct
storage paths manually with os.path.join or Path concatenation. This ensures:
- Consistent directory structure
- Automatic directory creation
- Single place to change if storage layout evolves
"""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import UUID

from app.infrastructure.config.settings import get_settings


class StorageManager:
    """
    Manages the local filesystem layout for the application.

    Directory structure under storage_root:
        storage/
        ├── models/          # Downloaded model weights
        │   ├── huggingface/ # HuggingFace model repos (mirrored structure)
        │   ├── civitai/     # CivitAI single-file checkpoints
        │   └── local/       # Manually imported models
        ├── artifacts/       # Generated outputs (images, videos)
        │   └── {job_id}/    # One subdirectory per job
        ├── thumbnails/      # Downscaled previews for gallery
        ├── temp/            # Temporary processing files (auto-cleaned)
        ├── imports/         # User-uploaded source files
        └── logs/            # Application logs (if file-based logging enabled)
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        for path in [
            self._settings.models_path,
            self._settings.models_path / "huggingface",
            self._settings.models_path / "civitai",
            self._settings.models_path / "local",
            self._settings.artifacts_path,
            self._settings.thumbnails_path,
            self._settings.temp_path,
            self._settings.imports_path,
            self._settings.logs_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    # --- Model paths ---

    def model_dir(self, source: str, model_id: str) -> Path:
        """Return the directory where a model's files are stored."""
        path = self._settings.models_path / source / Path(*model_id.split("/"))
        path.mkdir(parents=True, exist_ok=True)
        return path

    # --- Artifact paths ---

    def artifact_dir(self, job_id: UUID) -> Path:
        """Return the directory for a job's output artifacts."""
        path = self._settings.artifacts_path / str(job_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def thumbnail_path(self, artifact_id: UUID, extension: str = ".webp") -> Path:
        """Return the file path for an artifact's thumbnail."""
        return self._settings.thumbnails_path / f"{artifact_id}{extension}"

    # --- Temp paths ---

    def temp_file(self, filename: str) -> Path:
        """Return a path in the temp directory for intermediate processing."""
        return self._settings.temp_path / filename

    # --- Import paths ---

    def import_path(self, filename: str) -> Path:
        """Return a path for a user-uploaded source file."""
        return self._settings.imports_path / filename

    # --- Cleanup ---

    def cleanup_temp(self) -> None:
        """Remove all temporary files."""
        if self._settings.temp_path.exists():
            shutil.rmtree(self._settings.temp_path)
            self._settings.temp_path.mkdir(parents=True, exist_ok=True)

    def delete_model_files(self, source: str, model_id: str) -> None:
        """Remove all files for a model from disk."""
        path = self._settings.models_path / source / Path(*model_id.split("/"))
        if path.exists():
            shutil.rmtree(path)

    def delete_artifact_files(self, job_id: UUID) -> None:
        """Remove all artifact files for a job."""
        path = self.artifact_dir(job_id)
        if path.exists():
            shutil.rmtree(path)
