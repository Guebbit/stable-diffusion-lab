"""
Civitai model source provider.

Handles model metadata resolution and file downloads from Civitai.
Civitai models are typically single-file (safetensors/ckpt).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from pathlib import Path
from typing import Any

import httpx

from app.services.sources.utils import compute_file_sha256

logger = logging.getLogger(__name__)

CIVITAI_API_BASE = "https://civitai.com/api/v1"


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
        self._headers = {}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    async def resolve_model_info(self, model_id: str) -> dict[str, Any]:
        """
        Fetch model metadata from Civitai API.

        Args:
            model_id: Civitai model ID (numeric) or model slug.

        Returns a dict with:
            - files: list of {relative_path, size_bytes, sha256}
            - total_size_bytes: total download size
            - model_id: the original model_id passed in
            - source: "civitai"
        """
        if not model_id or not model_id.strip():
            raise ValueError(f"Invalid model_id: {model_id!r}")

        # Parse model_id: might be "12345" or "12345/version-id" or just a slug
        parts = model_id.split("/")
        model_ref = parts[0]
        version_ref = parts[1] if len(parts) > 1 else None

        async with httpx.AsyncClient(timeout=30) as client:
            # Step 1: Fetch model info
            if model_ref.isdigit():
                model_response = await client.get(
                    f"{CIVITAI_API_BASE}/models/{model_ref}",
                    headers=self._headers,
                )
            else:
                # Try to find by search
                search_response = await client.get(
                    f"{CIVITAI_API_BASE}/models",
                    params={"tags": model_ref, "limit": 1},
                    headers=self._headers,
                )
                if search_response.status_code != 200:
                    raise RuntimeError(
                        f"Civitai search failed: {search_response.status_code}"
                    )
                search_data = search_response.json()
                models = search_data.get("models", [])
                if not models:
                    raise ValueError(f"No Civitai model found matching '{model_id}'")
                model_data = models[0]
            if model_ref.isdigit():
                if model_response.status_code != 200:
                    raise RuntimeError(
                        f"Civitai API error: {model_response.status_code} - {model_response.text}"
                    )
                model_data = model_response.json()

            model_name = model_data.get("name", "unknown")
            versions = model_data.get("versions", [])

            if not versions:
                raise ValueError(f"Model '{model_id}' has no versions on Civitai")

            # Step 2: Find the right version
            target_version = None
            if version_ref:
                # Try to match by version id or name
                for v in versions:
                    v_id = str(v.get("id", ""))
                    v_name = v.get("name", "")
                    if v_id == version_ref or v_name == version_ref:
                        target_version = v
                        break
                if target_version is None:
                    raise ValueError(
                        f"Version '{version_ref}' not found in model '{model_id}'"
                    )
            else:
                # Use the latest version (first in list usually)
                target_version = versions[0]

            version_id = target_version.get("id")
            files = target_version.get("files", [])

            if not files:
                raise ValueError(
                    f"Version {version_id} of model '{model_id}' has no downloadable files"
                )

            # Step 3: Map files to our schema
            files_info: list[dict[str, Any]] = []
            total_size_bytes = 0

            for f in files:
                meta = f.get("meta", {})
                size = meta.get("sizeMd5", 0) if isinstance(meta, dict) else meta.get("size", 0) if isinstance(meta, dict) else 0
                # Handle both dict and non-dict meta
                if isinstance(meta, dict):
                    size = meta.get("size", 0) or 0
                elif hasattr(meta, "get"):
                    size = meta.get("size", 0) or 0
                else:
                    size = 0

                filename = f.get("name", "model.safetensors")
                hash_str = meta.get("hash", "") if isinstance(meta, dict) else None

                file_entry: dict[str, Any] = {
                    "relative_path": filename,
                    "size_bytes": size,
                    "download_url": f.get("url", ""),
                    "type": f.get("type", "Model"),
                }
                if hash_str:
                    file_entry["sha256"] = hash_str

                # Only download primary model files, not metadata
                file_type = f.get("type", "Model")
                if file_type == "Model":
                    total_size_bytes += size
                    files_info.append(file_entry)
                else:
                    logger.debug("Skipping non-model file: %s (type: %s)", filename, file_type)

        return {
            "files": files_info,
            "total_size_bytes": total_size_bytes,
            "model_id": model_id,
            "source": "civitai",
            "model_name": model_name,
            "version_id": target_version.get("id"),
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
        Download a model file from Civitai.

        Civitai models are usually single files, but the interface stays
        consistent with multi-file sources for uniformity.

        Args:
            model_id: Civitai model ID, optionally with /version-id
            file_path: Not used for Civitai (file is determined by model metadata)
            destination: Local path to write the file
            resume_from_byte: Byte offset to resume from (0 for fresh download)
            on_progress: Callback for progress updates

        Returns:
            Dict with {sha256, size_bytes} for verification
        """
        if not model_id or not model_id.strip():
            raise ValueError(f"Invalid model_id: {model_id!r}")

        destination.parent.mkdir(parents=True, exist_ok=True)

        # Check if file already exists
        if destination.exists():
            existing_size = destination.stat().st_size
            if existing_size > 0:
                sha256 = compute_file_sha256(destination)
                logger.info(
                    "File already exists at %s (%d bytes), skipping download",
                    destination,
                    existing_size,
                )
                return {"sha256": sha256, "size_bytes": existing_size}

        # Resolve the download URL from Civitai API
        parts = model_id.split("/")
        model_ref = parts[0]
        version_ref = parts[1] if len(parts) > 1 else None

        download_url = None
        async with httpx.AsyncClient(timeout=30) as client:
            if model_ref.isdigit():
                resp = await client.get(
                    f"{CIVITAI_API_BASE}/models/{model_ref}",
                    headers=self._headers,
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"Civitai API error: {resp.status_code}")
                model_data = resp.json()
            else:
                resp = await client.get(
                    f"{CIVITAI_API_BASE}/models",
                    params={"tags": model_ref, "limit": 1},
                    headers=self._headers,
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"Civitai search failed: {resp.status_code}")
                data = resp.json()
                models = data.get("models", [])
                if not models:
                    raise ValueError(f"No Civitai model found matching '{model_id}'")
                model_data = models[0]

            versions = model_data.get("versions", [])
            target_version = None
            if version_ref:
                for v in versions:
                    if str(v.get("id", "")) == version_ref or v.get("name", "") == version_ref:
                        target_version = v
                        break
            if target_version is None:
                target_version = versions[0] if versions else None
            if target_version is None:
                raise ValueError(f"No version found for model '{model_id}'")

            files = target_version.get("files", [])
            for f in files:
                if f.get("type") == "Model":
                    url = f.get("url", "")
                    if url:
                        download_url = url
                        break

        if not download_url:
            raise RuntimeError(f"No download URL found for model '{model_id}'")

        # Download with streaming and progress reporting
        tmp_destination = destination.with_suffix(".download.tmp")

        try:
            async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
                headers = dict(self._headers)
                if resume_from_byte > 0 and destination.exists():
                    headers["Range"] = f"bytes={resume_from_byte}-"

                async with client.stream("GET", download_url, headers=headers) as resp:
                    if resp.status_code == 416:
                        # Range not satisfiable - file already fully downloaded
                        sha256 = compute_file_sha256(destination)
                        return {"sha256": sha256, "size_bytes": destination.stat().st_size}

                    resp.raise_for_status()

                    total_size = int(resp.headers.get("content-range", "").split("/")[-1]) if "content-range" in resp.headers else int(resp.headers.get("content-length", 0))
                    downloaded = resume_from_byte

                    mode = "ab" if resume_from_byte > 0 else "wb"
                    with open(tmp_destination, mode) as f:
                        async for chunk in resp.aiter_bytes(chunk_size=1024 * 1024):
                            f.write(chunk)
                            downloaded += len(chunk)
                            if on_progress is not None and total_size > 0:
                                info = {
                                    "bytes_downloaded": downloaded,
                                    "total_size": total_size,
                                    "percent": round(downloaded / total_size * 100, 2),
                                }
                                if inspect.iscoroutinefunction(on_progress):
                                    await on_progress(info)
                                else:
                                    on_progress(info)

            # Verify and finalize
            final_size = tmp_destination.stat().st_size
            sha256 = compute_file_sha256(tmp_destination)

            if on_progress is not None:
                info = {"bytes_downloaded": final_size, "total_size": final_size, "percent": 100.0}
                if inspect.iscoroutinefunction(on_progress):
                    await on_progress(info)
                else:
                    on_progress(info)

            tmp_destination.rename(destination)

            logger.info(
                "Downloaded Civitai model %s -> %s (%d bytes, sha256: %s)",
                model_id,
                destination,
                final_size,
                sha256[:16] if sha256 else "unknown",
            )

            return {"sha256": sha256, "size_bytes": final_size}

        except Exception as e:
            tmp_destination.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to download Civitai model '{model_id}': {e}"
            ) from e

    def supports_resume(self) -> bool:
        """Civitai supports byte-range resume for most files."""
        return True
