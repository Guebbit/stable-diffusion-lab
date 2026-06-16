"""
GitHub Releases model source provider.

Model ID format: owner/repo/tag  (e.g. "TencentARC/GFPGAN/v1.3.4")
Use "owner/repo/latest" to always pull the most recent release.

Only assets whose extension matches a known model format are downloaded;
source archives (.zip, .tar.gz, etc.) are skipped automatically.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any

import httpx

from app.services.sources.utils import compute_file_sha256

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"

_MODEL_EXTENSIONS = {
    ".safetensors", ".ckpt", ".bin", ".pt", ".pth", ".onnx", ".gguf",
}


def _parse_model_id(model_id: str) -> tuple[str, str, str]:
    parts = model_id.strip().split("/")
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            f"Invalid GitHub model_id {model_id!r}. "
            "Expected 'owner/repo/tag' (e.g. 'TencentARC/GFPGAN/v1.3.4')."
        )
    return parts[0], parts[1], parts[2]


class GitHubSourceProvider:
    """
    Downloads model weight files from GitHub release assets.

    Skips source archives; only downloads assets with recognised model
    file extensions (.pth, .safetensors, .ckpt, .bin, .onnx, .gguf).
    Supports byte-range resume.
    """

    def __init__(self, token: str | None = None) -> None:
        """
        Args:
            token: Optional GitHub Personal Access Token.
                   Raises API rate limit from 60 to 5000 req/hr.
        """
        self._token = token
        self._api_headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            self._api_headers["Authorization"] = f"Bearer {token}"

    async def resolve_model_info(self, model_id: str) -> dict[str, Any]:
        """
        Fetch release asset list from the GitHub API.

        Returns:
            {
                "files": [{"relative_path": str, "size_bytes": int, "download_url": str}],
                "total_size_bytes": int,
                "model_id": str,
                "source": "github",
            }

        Raises:
            RuntimeError: if the release is not found (404) or the API call fails.
            ValueError: if no downloadable model assets exist in the release.
        """
        owner, repo, tag = _parse_model_id(model_id)

        if tag == "latest":
            url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/releases/latest"
        else:
            url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/releases/tags/{tag}"

        async with httpx.AsyncClient(timeout=30, headers=self._api_headers) as client:
            resp = await client.get(url)

        if resp.status_code == 404:
            raise RuntimeError(
                f"GitHub release not found for '{model_id}'. "
                f"Verify the tag at https://github.com/{owner}/{repo}/releases"
            )
        resp.raise_for_status()
        release = resp.json()

        assets = release.get("assets", [])
        files_info: list[dict[str, Any]] = []
        total_size_bytes = 0

        for asset in assets:
            name: str = asset.get("name", "")
            if Path(name).suffix.lower() not in _MODEL_EXTENSIONS:
                logger.debug("Skipping non-model asset '%s'", name)
                continue

            size: int = asset.get("size", 0)
            download_url: str = asset.get("browser_download_url", "")
            files_info.append({
                "relative_path": name,
                "size_bytes": size,
                "download_url": download_url,
            })
            total_size_bytes += size

        if not files_info:
            names = [a.get("name") for a in assets]
            raise ValueError(
                f"No downloadable model assets found in '{owner}/{repo}' release '{tag}'. "
                f"Available assets: {names}"
            )

        return {
            "files": files_info,
            "total_size_bytes": total_size_bytes,
            "model_id": model_id,
            "source": "github",
        }

    async def download_file(
        self,
        model_id: str,
        file_path: str,
        destination: Path,
        resume_from_byte: int = 0,
        on_progress: Any = None,
        download_url: str | None = None,
    ) -> dict[str, Any]:
        """
        Stream-download a single release asset with resume support.

        Args:
            model_id: GitHub model_id (owner/repo/tag) — used for logging.
            file_path: Asset filename — used only when download_url is absent.
            destination: Local path to write the file.
            resume_from_byte: Byte offset to resume from (0 = fresh).
            on_progress: Async or sync callback:
                         {"bytes_downloaded": int, "total_size": int, "percent": float}.
            download_url: browser_download_url from resolve_model_info().
                          When absent, resolve_model_info() is called to look it up.

        Returns:
            {"sha256": str | None, "size_bytes": int}
        """
        if not download_url:
            model_info = await self.resolve_model_info(model_id)
            matched = next(
                (f for f in model_info["files"] if f["relative_path"] == file_path),
                None,
            )
            if matched is None:
                raise RuntimeError(
                    f"Asset '{file_path}' not found in release for '{model_id}'"
                )
            download_url = matched["download_url"]

        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists() and resume_from_byte == 0:
            existing_size = destination.stat().st_size
            if existing_size > 0:
                sha256 = compute_file_sha256(destination)
                logger.info(
                    "Asset already exists at %s (%d bytes), skipping", destination, existing_size
                )
                return {"sha256": sha256, "size_bytes": existing_size}

        _progress_is_async = on_progress is not None and inspect.iscoroutinefunction(on_progress)

        async def _emit(info: dict) -> None:
            if on_progress is None:
                return
            if _progress_is_async:
                await on_progress(info)
            else:
                on_progress(info)

        tmp = destination.with_suffix(destination.suffix + ".download.tmp")

        try:
            req_headers = dict(self._api_headers)
            # GitHub asset redirects to S3 — do not send the API Accept header there.
            # Use a plain dict with only Range if needed.
            dl_headers: dict[str, str] = {}
            if resume_from_byte > 0 and destination.exists():
                dl_headers["Range"] = f"bytes={resume_from_byte}-"

            async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
                async with client.stream("GET", download_url, headers=dl_headers) as resp:
                    if resp.status_code == 416:
                        sha256 = compute_file_sha256(destination)
                        return {"sha256": sha256, "size_bytes": destination.stat().st_size}

                    resp.raise_for_status()

                    content_range = resp.headers.get("content-range", "")
                    if content_range:
                        total_size = int(content_range.split("/")[-1])
                    else:
                        total_size = int(resp.headers.get("content-length", 0) or 0)

                    downloaded = resume_from_byte
                    mode = "ab" if resume_from_byte > 0 else "wb"

                    with open(tmp, mode) as f:
                        async for chunk in resp.aiter_bytes(chunk_size=1024 * 1024):
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                await _emit({
                                    "bytes_downloaded": downloaded,
                                    "total_size": total_size,
                                    "percent": round(downloaded / total_size * 100, 2),
                                })

            final_size = tmp.stat().st_size
            sha256 = compute_file_sha256(tmp)

            await _emit({"bytes_downloaded": final_size, "total_size": final_size, "percent": 100.0})

            tmp.rename(destination)

            logger.info(
                "Downloaded GitHub asset %s/%s -> %s (%d bytes, sha256: %s)",
                model_id, file_path, destination, final_size,
                sha256[:16] if sha256 else "unknown",
            )

            return {"sha256": sha256, "size_bytes": final_size}

        except Exception as e:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to download GitHub asset '{file_path}' from '{model_id}': {e}"
            ) from e

    def supports_resume(self) -> bool:
        """GitHub CDN (S3-backed) supports byte-range resume."""
        return True
