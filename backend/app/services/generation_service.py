"""
Generation service — thin facade for submitting generation jobs.

Delegates model resolution to :class:`ModelResolver` and job creation to
:class:`JobCreator`. Keeps the API router code clean and testable.
"""

from __future__ import annotations

import logging
from uuid import UUID

from app.domain.value_objects import GenerationParams
from app.infrastructure.database.repositories import JobRepository, ModelRepository

from .job_creator import JobCreator
from .model_resolver import ModelResolver

logger = logging.getLogger(__name__)


class GenerationService:
    """
    Handles generation request lifecycle.

    Does NOT run inference directly — it creates jobs and delegates
    to the orchestrator. The API gets back a job_id immediately.
    """

    def __init__(
        self,
        job_repository: JobRepository,
        model_repository: ModelRepository | None = None,
    ) -> None:
        self._job_repo = job_repository
        self._model_repo = model_repository

    # ── Public API ────────────────────────────────

    async def submit_text_to_image(
        self,
        params: GenerationParams,
        model_id: str,
        correlation_id: str | None = None,
    ) -> UUID:
        """Submit a text-to-image generation job."""
        job_id = await self._job_creator.create_text_to_image_job(
            params, model_id, correlation_id
        )
        logger.info(
            "Created text-to-image job: %s (model_id=%s)",
            job_id,
            model_id,
        )
        return job_id

    async def submit_image_to_image(
        self,
        params: GenerationParams,
        model_id: str,
        source_image_path: str,
        strength: float = 0.75,
        correlation_id: str | None = None,
    ) -> UUID:
        """Submit an image-to-image generation job."""
        job_id = await self._job_creator.create_image_to_image_job(
            params, model_id, source_image_path, correlation_id, strength
        )
        logger.info(
            "Created image-to-image job: %s (model_id=%s)",
            job_id,
            model_id,
        )
        return job_id

    # ── Lazy property wiring ──────────────────────

    @property
    def _model_resolver(self) -> ModelResolver:
        if self._model_repo is None:
            raise RuntimeError(
                "ModelResolver requires a ModelRepository. "
                "Wire one via the constructor or use DirectExecutionService."
            )
        return ModelResolver(self._model_repo)

    @property
    def _job_creator(self) -> JobCreator:
        return JobCreator(self._job_repo, self._model_resolver)