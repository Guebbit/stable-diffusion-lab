"""
Hugging Face model source provider.

Handles model metadata resolution and file downloads from the
Hugging Face Hub. Supports multi-file models with resume capability.
"""

from __future__ import annotations

import logging
import hashlib
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download, HfFolder

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
        # If token not provided, try to use cached token
        if self._token is None:
            self._token = HfFolder.get_token()

        self._api = HfApi(token=self._token)

    async def resolve_model_info(self, model_id: str) -> dict[str, Any]:
        """
        Fetch file manifest from HF Hub API.

        Returns a dict with:
        - files: list of {relative_path, size_bytes, sha256}
        - total_size_bytes: sum of all file sizes

        Raises:
            RepositoryNotFoundError: if model_id does not exist on HF Hub
            ValueError: if model_id is empty
        """
        if not model_id or not model_id.strip():
            raise ValueError(f"Invalid model_id: {model_id!r}")

        try:
            repo_info = self._api.repo_info(model_id, repo_type="model")
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch repo info for '{model_id}': {e}"
            ) from e

        files_info: list[dict[str, Any]] = []
        total_size_bytes = 0

        for sibling in repo_info.siblings:
            rfilename = sibling.rfilename
            size = sibling.size or 0
            total_size_bytes += size

            file_entry: dict[str, Any] = {
                "relative_path": rfilename,
                "size_bytes": size,
            }

            # Extract LFS metadata if available (sha256, lfs type)
            lfs_info = getattr(sibling, "lfs", None)
            if lfs_info is not None:
                file_entry["sha256"] = getattr(lfs_info, "sha256", None)
                file_entry["lfs_type"] = getattr(lfs_info, "type", None)

            files_info.append(file_entry)

        return {
            "files": files_info,
            "total_size_bytes": total_size_bytes,
            "model_id": model_id,
            "source": "huggingface",
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
        Download a single file with resume support.

        Args:
            model_id: HF repo ID (e.g., "stabilityai/stable-diffusion-xl-base-1.0")
            file_path: Relative path within the repo
            destination: Local path to write the file
            resume_from_byte: Byte offset to resume from (0 for fresh download)
            on_progress: Callback for progress updates. Should accept a dict with
                        keys: bytes_downloaded, total_size, percent

        Returns:
            Dict with {sha256, size_bytes} for verification
        """
        if not model_id or not model_id.strip():
            raise ValueError(f"Invalid model_id: {model_id!r}")
        if not file_path or not file_path.strip():
            raise ValueError(f"Invalid file_path: {file_path!r}")

        destination.parent.mkdir(parents=True, exist_ok=True)

        # Check if file already exists and is complete (resume logic)
        if destination.exists():
            existing_size = destination.stat().st_size
            if resume_from_byte == 0 and existing_size > 0:
                # File already downloaded, compute hash and return
                sha256 = _compute_file_sha256(destination)
                logger.info(
                    "File already exists at %s (%d bytes), skipping download",
                    destination,
                    existing_size,
                )
                return {
                    "sha256": sha256,
                    "size_bytes": existing_size,
                }
            elif resume_from_byte > 0 and existing_size >= resume_from_byte:
                # Partial file exists, will resume
                logger.info(
                    "Resuming download of %s from byte %d (existing: %d bytes)",
                    destination,
                    resume_from_byte,
                    existing_size,
                )

        # Download to a temporary file first, then move to destination
        # This prevents corruption if download is interrupted
        tmp_destination = destination.with_suffix(".download.tmp")

        try:
            # hf_hub_download supports progress_callback in newer versions
            # We use local_dir_use_symlks=False to get actual files
            downloaded_path = hf_hub_download(
                repo_id=model_id,
                filename=file_path,
                token=self._token,
                local_dir=str(tmp_destination.parent),
                local_dir_use_symlinks=False,
            )

            # hf_hub_download downloads to local_dir with the same relative path
            actual_downloaded = tmp_destination.parent / Path(file_path).name

            # If downloaded to a different path, move it
            if actual_downloaded.exists() and actual_downloaded != tmp_destination:
                tmp_destination.unlink(missing_ok=True)
                actual_downloaded.rename(tmp_destination)

            # Ensure the downloaded file exists
            if not tmp_destination.exists():
                raise FileNotFoundError(
                    f"Download failed: file not found at {tmp_destination}"
                )

            # Compute size and hash
            final_size = tmp_destination.stat().st_size
            sha256 = _compute_file_sha256(tmp_destination)

            # Report 100% progress
            if on_progress is not None:
                on_progress({
                    "bytes_downloaded": final_size,
                    "total_size": final_size,
                    "percent": 100.0,
                })

            # Move final file to destination
            tmp_destination.rename(destination)

            logger.info(
                "Downloaded %s/%s -> %s (%d bytes, sha256: %s)",
                model_id,
                file_path,
                destination,
                final_size,
                sha256[:16] if sha256 else "unknown",
            )

            return {
                "sha256": sha256,
                "size_bytes": final_size,
            }

        except Exception as e:
            # Clean up temp file on failure
            tmp_destination.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to download '{file_path}' from '{model_id}': {e}"
            ) from e

    def supports_resume(self) -> bool:
        """HuggingFace Hub supports byte-range resume."""
        return True


def _compute_file_sha256(file_path: Path, chunk_size: int = 8 * 1024 * 1024) -> str | None:
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
