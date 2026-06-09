"""
Local model source provider.

Handles importing models from local filesystem paths.
No download needed — validates and registers existing files.
"""

from __future__ import annotations

import logging
import hashlib
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Common model file extensions
MODEL_EXTENSIONS = {
    ".safetensors",
    ".ckpt",
    ".bin",
    ".pt",
    ".pth",
    ".onnx",
    ".gguf",
    ".gorilla",
}


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
                     If it points to a file, returns single-file info.
                     If it points to a directory, scans for model files.

        Returns:
            Dict with file list and sizes (no download URLs needed).
        """
        source = Path(model_id)

        if not source.exists():
            raise FileNotFoundError(f"Local model path does not exist: {model_id}")

        files_info: list[dict[str, Any]] = []
        total_size_bytes = 0

        if source.is_file():
            # Single file mode
            file_info = _scan_file(source)
            files_info.append(file_info)
            total_size_bytes = file_info["size_bytes"]
        elif source.is_dir():
            # Directory mode: scan for model files
            for file_path in source.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in MODEL_EXTENSIONS:
                    rel_path = str(file_path.relative_to(source))
                    file_info = _scan_file(file_path, relative_to=source)
                    files_info.append(file_info)
                    total_size_bytes += file_info["size_bytes"]

            # If no model files found, scan all files
            if not files_info:
                for file_path in source.rglob("*"):
                    if file_path.is_file():
                        rel_path = str(file_path.relative_to(source))
                        file_info = _scan_file(file_path, relative_to=source)
                        files_info.append(file_info)
                        total_size_bytes += file_info["size_bytes"]
        else:
            raise ValueError(
                f"Local model path is neither a file nor a directory: {model_id}"
            )

        return {
            "files": files_info,
            "total_size_bytes": total_size_bytes,
            "model_id": model_id,
            "source": "local",
        }

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
        if not source_path.exists():
            raise FileNotFoundError(f"Source path does not exist: {source_path}")

        destination.mkdir(parents=True, exist_ok=True)

        files_imported: list[str] = []
        total_bytes = 0

        if source_path.is_file():
            dest_file = destination / source_path.name
            imported = _import_file(source_path, dest_file, copy=copy)
            files_imported.append(imported["destination"])
            total_bytes = imported["size_bytes"]

        elif source_path.is_dir():
            for file_path in source_path.rglob("*"):
                if not file_path.is_file():
                    continue
                rel = file_path.relative_to(source_path)
                dest_file = destination / rel
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                imported = _import_file(file_path, dest_file, copy=copy)
                files_imported.append(imported["destination"])
                total_bytes += imported["size_bytes"]
        else:
            raise ValueError(f"Invalid source path: {source_path}")

        logger.info(
            "Imported %d files (%d bytes) from %s to %s",
            len(files_imported),
            total_bytes,
            source_path,
            destination,
        )

        return {
            "files_imported": files_imported,
            "total_bytes": total_bytes,
        }

    async def download_file(
        self,
        model_id: str,
        file_path: str,
        destination: Path,
        resume_from_byte: int = 0,
        on_progress: Any = None,
    ) -> dict[str, Any]:
        """
        Copy a local file to the destination.

        Acts as a compatibility shim for the uniform download_file interface.
        """
        source = Path(model_id)
        if source.is_dir():
            source = source / file_path
        elif source.is_file():
            source = source

        if not source.exists():
            raise FileNotFoundError(f"Source file does not exist: {source}")

        destination.parent.mkdir(parents=True, exist_ok=True)

        # If file already exists and is the same size, skip
        if destination.exists():
            existing_size = destination.stat().st_size
            source_size = source.stat().st_size
            if existing_size == source_size:
                sha256 = _compute_file_sha256(destination)
                return {"sha256": sha256, "size_bytes": existing_size}

        shutil.copy2(str(source), str(destination))
        final_size = destination.stat().st_size
        sha256 = _compute_file_sha256(destination)

        if on_progress is not None:
            on_progress({
                "bytes_downloaded": final_size,
                "total_size": final_size,
                "percent": 100.0,
            })

        return {"sha256": sha256, "size_bytes": final_size}

    def supports_resume(self) -> bool:
        """Local imports don't need resume — they're filesystem operations."""
        return False


def _scan_file(
    file_path: Path,
    relative_to: Path | None = None,
) -> dict[str, Any]:
    """Scan a single file and return metadata."""
    rel = str(file_path.relative_to(relative_to)) if relative_to else str(file_path)
    size = file_path.stat().st_size
    info: dict[str, Any] = {
        "relative_path": rel,
        "size_bytes": size,
    }
    try:
        info["sha256"] = _compute_file_sha256(file_path)
    except OSError as e:
        logger.warning("Could not compute SHA256 for %s: %s", file_path, e)
    return info


def _import_file(
    source: Path,
    destination: Path,
    copy: bool = True,
) -> dict[str, Any]:
    """Import a single file via copy or symlink."""
    # Skip if destination already exists
    if destination.exists():
        return {
            "destination": str(destination),
            "size_bytes": destination.stat().st_size,
        }

    if copy:
        shutil.copy2(str(source), str(destination))
    else:
        destination.symlink_to(source.resolve())

    return {
        "destination": str(destination),
        "size_bytes": destination.stat().st_size,
    }


def _compute_file_sha256(file_path: Path, chunk_size: int = 8 * 1024 * 1024) -> str | None:
    """Compute SHA256 hash of a file."""
    try:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except OSError as e:
        logger.warning("Could not compute SHA256 for %s: %s", file_path, e)
        return None
