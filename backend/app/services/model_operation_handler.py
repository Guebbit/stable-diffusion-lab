"""
Model Operation Handler — processes model lifecycle jobs (download, delete, refresh).

Single responsibility: given a model-related job type, delegate to the
appropriate source provider, update job/model progress in the DB, and
emit structured log lines for container monitoring.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.enums import JobType
from app.domain.events import JobEvent, ModelEvent
from app.infrastructure.config.settings import get_settings
from app.infrastructure.database.repositories import JobRepository, ModelRepository
from app.orchestrator.event_bus import event_bus
from app.services.sources.civitai import CivitaiSourceProvider
from app.services.sources.huggingface import HuggingFaceSourceProvider

logger = logging.getLogger(__name__)


class ModelOperationHandler:
    """Handle model lifecycle operations (download, delete, refresh)."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._session_factory = session_factory

    async def handle(
        self,
        job_id: UUID,
        job_type: str,
        params: dict,
    ) -> None:
        """
        Process a model lifecycle job.

        Raises:
            ValueError: If the job type is not a model operation type.
        """
        job_type_enum = JobType(job_type)

        handlers = {
            JobType.MODEL_DOWNLOAD: self._handle_download,
            JobType.MODEL_DELETE: self._handle_delete,
            JobType.MODEL_REFRESH: self._handle_refresh,
        }

        handler = handlers.get(job_type_enum)
        if handler is None:
            raise ValueError(
                f"ModelOperationHandler does not handle job type '{job_type}'. "
                f"Supported: {list(handlers.keys())}"
            )

        await handler(job_id, params)

    # ── Private operation handlers ───────────────────────────────────

    async def _handle_download(self, job_id: UUID, params: dict) -> None:
        model_id: str = params.get("model_id", "")
        source: str = params.get("source", "")
        settings = get_settings()

        # ── Load model record ────────────────────────────────────────
        async with self._session_factory() as session:
            model = await ModelRepository(session).get_by_model_id(model_id)
            if model is None:
                raise ValueError(f"Model '{model_id}' not found in registry")
            await ModelRepository(session).update_status(model_id, "downloading")
            await session.commit()

        logger.info(
            "[download] START   job=%s | model=%s | source=%s",
            job_id, model_id, source,
        )

        try:
            await self._do_download(job_id, model_id, source, settings)
        except Exception:
            async with self._session_factory() as session:
                await ModelRepository(session).update_status(model_id, "error")
                await session.commit()
            logger.error("[download] FAILED  job=%s | model=%s", job_id, model_id)
            await event_bus.publish_event(
                ModelEvent(
                    event_type="model.download_failed",
                    job_id=str(job_id),
                    level="error",
                    message=f"Download failed for model '{model_id}'",
                    payload={"model_id": model_id, "source": source},
                )
            )
            raise

    async def _do_download(
        self,
        job_id: UUID,
        model_id: str,
        source: str,
        settings,
    ) -> None:
        """Core download logic — separated for clean error-handling in the caller."""
        # ── Pick source provider ─────────────────────────────────────
        if source == "huggingface":
            provider: HuggingFaceSourceProvider | CivitaiSourceProvider = (
                HuggingFaceSourceProvider(token=settings.huggingface_token or None)
            )
        elif source == "civitai":
            provider = CivitaiSourceProvider(
                api_key=settings.civitai_token or None
            )
        else:
            raise ValueError(f"Unsupported model source: '{source}'")

        # ── Resolve file manifest ────────────────────────────────────
        logger.info("[download] RESOLVE job=%s | resolving file manifest…", job_id)
        model_info = await provider.resolve_model_info(model_id)
        files: list[dict] = model_info.get("files", [])
        total_size: int = model_info.get("total_size_bytes", 0)
        total_files = len(files)

        if total_files == 0:
            raise ValueError(f"No downloadable files found for model '{model_id}'")

        logger.info(
            "[download] MANIFEST job=%s | %d file(s), %.1f MB",
            job_id, total_files,
            total_size / 1024 / 1024 if total_size else 0,
        )

        # Update job with known file count + record download-start milestone
        async with self._session_factory() as session:
            job_repo = JobRepository(session)
            await job_repo.update_progress(
                job_id, 0,
                total_steps=total_files,
                message=f"Preparing {total_files} file(s)",
            )
            await job_repo.record_milestone(
                job_id,
                message=f"Download started: {total_files} file(s), {round(total_size / 1024 / 1024, 1)} MB",
                metadata={"model_id": model_id, "total_files": total_files, "total_size_bytes": total_size},
            )
            await session.commit()

        # ── Prepare destination directory ────────────────────────────
        # e.g. /app/storage/models/huggingface/stabilityai/sdxl-base-1.0
        dest_dir = settings.models_path / source / Path(*model_id.split("/"))
        dest_dir.mkdir(parents=True, exist_ok=True)

        # ── Download each file ───────────────────────────────────────
        for file_idx, file_info in enumerate(files):
            relative_path: str = file_info["relative_path"]
            file_name = Path(relative_path).name  # basename for log display only
            size_mb = round(file_info.get("size_bytes", 0) / 1024 / 1024, 1)
            dest_file = dest_dir / relative_path  # preserve subdirectory structure

            logger.info(
                "[download] FILE    job=%s | %d/%d '%s' (%.1f MB)",
                job_id, file_idx + 1, total_files, file_name, size_mb,
            )

            # Build a per-file async progress handler
            progress_fn = self._make_progress_handler(
                job_id=job_id,
                model_id=model_id,
                file_idx=file_idx,
                total_files=total_files,
                file_name=file_name,
            )

            await provider.download_file(
                model_id=model_id,
                file_path=relative_path,
                destination=dest_file,
                on_progress=progress_fn,
                download_url=file_info.get("download_url"),
            )

            # Commit per-file completion into DB
            done_pct = int((file_idx + 1) / total_files * 100)
            async with self._session_factory() as session:
                job_repo = JobRepository(session)
                await job_repo.update_progress(
                    job_id, done_pct,
                    current_step=file_idx + 1,
                    total_steps=total_files,
                    message=f"Downloaded {file_name}",
                )
                await job_repo.record_milestone(
                    job_id,
                    message=f"File {file_idx + 1}/{total_files} complete: {file_name} ({size_mb} MB)",
                    metadata={"file": file_name, "file_index": file_idx + 1, "total_files": total_files, "size_mb": size_mb},
                )
                await session.commit()

            logger.info(
                "[download] DONE    job=%s | %d/%d '%s' complete",
                job_id, file_idx + 1, total_files, file_name,
            )

        # ── Mark model as downloaded ─────────────────────────────────
        async with self._session_factory() as session:
            await ModelRepository(session).update_status(
                model_id, "downloaded",
                local_path=str(dest_dir),
                total_size_bytes=total_size,
                file_count=total_files,
            )
            await session.commit()

        await event_bus.publish_event(
            ModelEvent(
                event_type="model.downloaded",
                job_id=str(job_id),
                message=f"Model '{model_id}' downloaded successfully",
                payload={"model_id": model_id, "source": source, "files": total_files},
            )
        )

        logger.info(
            "[download] COMPLETE job=%s | model=%s → %s",
            job_id, model_id, dest_dir,
        )

    def _make_progress_handler(
        self,
        job_id: UUID,
        model_id: str,
        file_idx: int,
        total_files: int,
        file_name: str,
    ):
        """
        Return an async callable that accepts a progress dict and:
        - Logs at 10 % milestones
        - Updates job + model progress in the DB (throttled: every 1 % change)
        - Publishes a job.progress event to the event bus
        """
        last_logged: list[int] = [-10]
        last_db_pct: list[int] = [-1]

        async def _on_progress(info: dict) -> None:
            file_pct = int(info.get("percent", 0))
            # Weight per-file progress into overall progress
            base = int(file_idx / total_files * 100)
            contribution = int(file_pct / total_files)
            overall_pct = base + contribution

            # Log every 10 %
            milestone = (file_pct // 10) * 10
            if milestone > last_logged[0]:
                last_logged[0] = milestone
                logger.info(
                    "[download] PROGRESS job=%s | %d/%d '%s': %d%%  (overall ~%d%%)",
                    job_id, file_idx + 1, total_files, file_name,
                    file_pct, overall_pct,
                )

            # DB write throttled to every 1 % change
            if overall_pct > last_db_pct[0]:
                last_db_pct[0] = overall_pct
                async with self._session_factory() as s:
                    await JobRepository(s).update_progress(
                        job_id, overall_pct,
                        current_step=file_idx,
                        total_steps=total_files,
                        message=f"Downloading {file_name} ({file_pct}%)",
                    )
                    await s.commit()

                await event_bus.publish_event(
                    JobEvent(
                        event_type="job.progress",
                        job_id=str(job_id),
                        message=f"Downloading {file_name}: {file_pct}%",
                        payload={
                            "model_id": model_id,
                            "progress_percent": overall_pct,
                            "file": file_name,
                            "file_index": file_idx + 1,
                            "total_files": total_files,
                        },
                    )
                )

        return _on_progress

    async def _handle_delete(self, job_id: UUID, params: dict) -> None:
        logger.info(
            "[model_op] DELETE  job=%s | model=%s",
            job_id,
            params.get("model_id"),
        )

    async def _handle_refresh(self, job_id: UUID, params: dict) -> None:
        logger.info(
            "[model_op] REFRESH job=%s | model=%s",
            job_id,
            params.get("model_id"),
        )
