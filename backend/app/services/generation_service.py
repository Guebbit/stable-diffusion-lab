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
        Resolve a user-facing model identifier to the actual model identifier
        that the adapter should load.

        Supports two input formats:
        1. Internal DB UUID (e.g. "a0000000-0000-0000-0000-000000000003") — looked up
           by primary key via get_by_id().
        2. External HuggingFace repo slug (e.g. "runwayml/stable-diffusion-v1-5") —
           looked up by model_id column via get_by_model_id().

        Returns the model_id column value (HuggingFace repo slug) so the adapter
        can download or load the correct weights.

        Falls back to the raw model_id when the repository is unavailable
        or the model is not found (backward compatibility).
        """
        if self._model_repo is None:
            return model_id

        # Try 1: Treat input as an internal DB primary key UUID
        try:
            from uuid import UUID as _UUID

            uuid_val = _UUID(model_id)
            model = await self._model_repo.get_by_id(uuid_val)
            if model is not None:
                return model.model_id
        except (ValueError, AttributeError):
            pass

        # Try 2: Treat input as an external model_id (HuggingFace repo slug)
        model = await self._model_repo.get_by_model_id(model_id)
        if model is not None and model.model_id:
            return model.model_id

        # Fallback: return as-is (backward compatibility)
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

