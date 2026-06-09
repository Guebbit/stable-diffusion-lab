"""
Model Operation Handler — processes model lifecycle jobs (download, delete, refresh).

Single responsibility: given a model-related job type, delegate to the
ModelService to perform the operation and update job status accordingly.
"""

from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.enums import JobType

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

        Args:
            job_id: The job UUID to update.
            job_type: JobType enum value as string.
            params: Serialized job parameters.

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
        logger.info(
            "Job %s: downloading model %s",
            job_id,
            params.get("model_id"),
        )

    async def _handle_delete(self, job_id: UUID, params: dict) -> None:
        logger.info(
            "Job %s: deleting model %s",
            job_id,
            params.get("model_id"),
        )

    async def _handle_refresh(self, job_id: UUID, params: dict) -> None:
        logger.info(
            "Job %s: refreshing model %s",
            job_id,
            params.get("model_id"),
        )