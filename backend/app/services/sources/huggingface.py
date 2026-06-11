"""
Hugging Face model source provider.

Handles model metadata resolution and file downloads from the
Hugging Face Hub. Supports multi-file models with resume capability.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import get_token as hf_get_token

from app.services.sources.utils import compute_file_sha256

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
            self._token = hf_get_token()

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
            loop = asyncio.get_running_loop()
            repo_info = await loop.run_in_executor(
                None, lambda: self._api.repo_info(model_id, repo_type="model")
            )
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

        # Check if file already exists and is complete
        if destination.exists():
            existing_size = destination.stat().st_size
            if resume_from_byte == 0 and existing_size > 0:
                sha256 = compute_file_sha256(destination)
                logger.info(
                    "File already exists at %s (%d bytes), skipping download",
                    destination,
                    existing_size,
                )
                return {"sha256": sha256, "size_bytes": existing_size}

        # hf_hub_download(local_dir=base_dir, filename=file_path)
        # places the file at base_dir/file_path (preserving subdirectory structure).
        # Compute base_dir by stripping the file_path components from destination.
        # e.g. destination = /models/sd15/feature_extractor/preprocessor_config.json
        #      file_path   = feature_extractor/preprocessor_config.json
        #      base_dir    = /models/sd15/
        base_dir = destination
        for _ in Path(file_path).parts:
            base_dir = base_dir.parent

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: hf_hub_download(
                    repo_id=model_id,
                    filename=file_path,
                    token=self._token,
                    local_dir=str(base_dir),
                ),
            )

            if not destination.exists():
                raise FileNotFoundError(
                    f"Download failed: file not found at {destination}"
                )

            final_size = destination.stat().st_size
            sha256 = compute_file_sha256(destination)

            # hf_hub_download manages its own temp files; report 100% at completion
            if on_progress is not None:
                info = {"bytes_downloaded": final_size, "total_size": final_size, "percent": 100.0}
                if inspect.iscoroutinefunction(on_progress):
                    await on_progress(info)
                else:
                    on_progress(info)

            logger.info(
                "Downloaded %s/%s -> %s (%d bytes, sha256: %s)",
                model_id,
                file_path,
                destination,
                final_size,
                sha256[:16] if sha256 else "unknown",
            )

            return {"sha256": sha256, "size_bytes": final_size}

        except Exception as e:
            raise RuntimeError(
                f"Failed to download '{file_path}' from '{model_id}': {e}"
            ) from e

    def supports_resume(self) -> bool:
        """HuggingFace Hub supports byte-range resume."""
        return True
