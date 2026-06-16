"""
Model reconciliation — syncs DB status against actual files on disk.

Runs once at startup as a non-blocking background task. Three outcomes
per model:

  RECOVERED  — files present on disk, DB said not_downloaded / error /
               downloading (stale after a container restart) → mark downloaded
  RESET      — DB said downloaded or downloading but files are gone → reset to
               not_downloaded so the user can re-trigger the download
  UNCHANGED  — status already matches reality, no action needed
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.infrastructure.config.settings import get_settings
from app.infrastructure.database.repositories import ModelRepository

logger = logging.getLogger(__name__)

_STALE_STATUSES = {"downloaded", "downloading"}


def _expected_path(source: str, model_id: str, models_root: Path) -> Path:
    """Compute where a model's files should live, mirroring the download handler."""
    return models_root / source / Path(*model_id.split("/"))


def _scan_directory(path: Path) -> tuple[int, int]:
    """Return (total_size_bytes, file_count) for all regular files under path."""
    total_size = 0
    file_count = 0
    for f in path.rglob("*"):
        if f.is_file():
            total_size += f.stat().st_size
            file_count += 1
    return total_size, file_count


async def reconcile_models(session_factory: async_sessionmaker[AsyncSession]) -> None:
    """
    Scan model storage and reconcile every DB record.

    Designed to run as asyncio.create_task() so startup is never blocked.
    Disk scanning is offloaded to a thread pool to avoid blocking the event loop.
    """
    settings = get_settings()
    models_root = settings.models_path

    async with session_factory() as session:
        models = await ModelRepository(session).list_all()

    logger.info("[reconcile] START — %d models, root=%s", len(models), models_root)

    recovered = reset = 0

    for model in models:
        # Prefer the stored path; fall back to the computed one for fresh DB installs.
        candidate = (
            Path(model.local_path)
            if model.local_path
            else _expected_path(model.source, model.model_id, models_root)
        )

        # Run blocking filesystem work off the event loop.
        files: list[Path] = await asyncio.to_thread(
            lambda p=candidate: [f for f in p.rglob("*") if f.is_file()] if p.exists() else []
        )
        has_files = bool(files)

        if has_files and model.status != "downloaded":
            total_size, file_count = await asyncio.to_thread(_scan_directory, candidate)
            async with session_factory() as session:
                await ModelRepository(session).update_status(
                    model.model_id,
                    "downloaded",
                    local_path=str(candidate),
                    total_size_bytes=total_size,
                    file_count=file_count,
                )
                await session.commit()
            logger.info(
                "[reconcile] RECOVERED  %s → %s (%d files, %.1f MB)",
                model.model_id,
                candidate,
                file_count,
                total_size / 1024 / 1024,
            )
            recovered += 1

        elif not has_files and model.status in _STALE_STATUSES:
            async with session_factory() as session:
                await ModelRepository(session).update_status(
                    model.model_id, "not_downloaded"
                )
                await session.commit()
            logger.warning(
                "[reconcile] RESET      %s — no files at %s (was: %s)",
                model.model_id,
                candidate,
                model.status,
            )
            reset += 1

    unchanged = len(models) - recovered - reset
    logger.info(
        "[reconcile] DONE — %d recovered, %d reset, %d unchanged",
        recovered,
        reset,
        unchanged,
    )
