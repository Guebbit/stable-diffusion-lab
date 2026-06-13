"""
Generation service — thin facade for submitting generation jobs.

Delegates model resolution to :class:`ModelResolver` and job creation to
:class:`JobCreator`. Keeps the API router code clean and testable.
"""

from __future__ import annotations

import logging
from uuid import UUID

from app.domain.value_objects import GenerationParams

from .job_creator import JobCreator

logger = logging.getLogger(__name__)


class GenerationService:
    """
    Handles generation request lifecycle.

    Does NOT run inference directly — it creates jobs and delegates
    to the orchestrator. The API gets back a job_id immediately.

    Dependencies are injected via the constructor (proper DI pattern).
    """

    def __init__(self, job_creator: JobCreator) -> None:
        self._job_creator = job_creator

    # ── Public API ────────────────────────────────────────

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

    async def submit_image_captioning(
        self,
        model_id: str,
        image_path: str,
        correlation_id: str | None = None,
    ) -> UUID:
        """Submit an image captioning (describe) job."""
        job_id = await self._job_creator.create_image_captioning_job(
            model_id, image_path, correlation_id
        )
        logger.info(
            "Created image-captioning job: %s (model_id=%s)",
            job_id,
            model_id,
        )
        return job_id