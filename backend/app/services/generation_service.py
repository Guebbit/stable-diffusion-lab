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
from app.infrastructure.database.repositories import JobRepository, ModelRepository
from app.orchestrator.event_bus import event_bus

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

    async def _resolve_model_identifier(self, model_id: str) -> str:
        """
        Resolve a user-facing model_id to the actual model identifier
        that the adapter should load.

        When model_repository is available, look up the catalog record
        and return the hf_repo_id (e.g. "runwayml/stable-diffusion-v1-5")
        so the adapter can download the correct weights.

        Falls back to the raw model_id when the repository is unavailable
        or the model is not found (backward compatibility).
        """
        if self._model_repo is None:
            return model_id

        model = await self._model_repo.get_by_model_id(model_id)
        if model is not None and model.model_id:
            return model.model_id

        return model_id

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
        resolved_id = await self._resolve_model_identifier(model_id)
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
                "model_id": resolved_id,
                "original_model_id": model_id,
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
        logger.info(
            "Created text-to-image job: %s (model_id=%s -> hf_repo=%s)",
            job.id,
            model_id,
            resolved_id,
        )
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
        resolved_id = await self._resolve_model_identifier(model_id)
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
                "model_id": resolved_id,
                "original_model_id": model_id,
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
        logger.info(
            "Created image-to-image job: %s (model_id=%s -> hf_repo=%s)",
            job.id,
            model_id,
            resolved_id,
        )
        return job.id

    async def get_job_status(self, job_id: UUID) -> JobRecord | None:
        """Get current status of a generation job."""
        return await self._job_repo.get_by_id(job_id)