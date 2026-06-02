"""
Generation service — orchestrates image/video generation workflows.

This service decides how to handle generation requests:
- Validates that required models are available
- Creates job records
- Delegates to the job orchestrator for async execution
- Returns job references to the caller (API layer)
"""

from __future__ import annotations

import logging
from uuid import UUID

from app.domain.events import JobEvent
from app.domain.enums import JobStatus, JobType
from app.domain.value_objects import GenerationParams
from app.infrastructure.database.models import JobRecord
from app.infrastructure.database.repositories import JobRepository
from app.orchestrator.event_bus import event_bus

logger = logging.getLogger(__name__)


class GenerationService:
    """
    Handles generation request lifecycle.

    Does NOT run inference directly — it creates jobs and delegates
    to the orchestrator. The API gets back a job_id immediately.
    """

    def __init__(self, job_repository: JobRepository) -> None:
        self._job_repo = job_repository

    async def submit_text_to_image(
        self,
        params: GenerationParams,
        model_id: str,
        correlation_id: str | None = None,
    ) -> UUID:
        """
        Submit a text-to-image generation job.

        Creates a PENDING job record and returns its ID.
        The orchestrator will pick it up and execute it.
        """
        job = JobRecord(
            job_type=JobType.TEXT_TO_IMAGE,
            status=JobStatus.PENDING,
            params={
                "prompt": params.prompt,
                "negative_prompt": params.negative_prompt,
                "width": params.width,
                "height": params.height,
                "num_inference_steps": params.num_inference_steps,
                "guidance_scale": params.guidance_scale,
                "seed": params.seed,
                "num_images": params.num_images,
                "model_id": model_id,
                "correlation_id": correlation_id,
                **params.extra,
            },
        )
        job = await self._job_repo.create(job)
        await event_bus.publish_event(
            JobEvent(
                event_type="job.enqueued",
                correlation_id=correlation_id,
                job_id=str(job.id),
                message="Job enqueued",
                payload={"job_type": JobType.TEXT_TO_IMAGE, "model_id": model_id},
            )
        )
        logger.info("Created text-to-image job: %s", job.id)
        return job.id

    async def submit_image_to_image(
        self,
        params: GenerationParams,
        model_id: str,
        source_image_path: str,
        strength: float = 0.75,
        correlation_id: str | None = None,
    ) -> UUID:
        """Submit an image-to-image generation job."""
        job = JobRecord(
            job_type=JobType.IMAGE_TO_IMAGE,
            status=JobStatus.PENDING,
            params={
                "prompt": params.prompt,
                "negative_prompt": params.negative_prompt,
                "width": params.width,
                "height": params.height,
                "num_inference_steps": params.num_inference_steps,
                "guidance_scale": params.guidance_scale,
                "seed": params.seed,
                "num_images": params.num_images,
                "model_id": model_id,
                "source_image_path": source_image_path,
                "strength": strength,
                "correlation_id": correlation_id,
            },
        )
        job = await self._job_repo.create(job)
        await event_bus.publish_event(
            JobEvent(
                event_type="job.enqueued",
                correlation_id=correlation_id,
                job_id=str(job.id),
                message="Job enqueued",
                payload={"job_type": JobType.IMAGE_TO_IMAGE, "model_id": model_id},
            )
        )
        logger.info("Created image-to-image job: %s", job.id)
        return job.id

    async def get_job_status(self, job_id: UUID) -> JobRecord | None:
        """Get current status of a generation job."""
        return await self._job_repo.get_by_id(job_id)
