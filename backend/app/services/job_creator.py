"""
Job Creator — creates and persists job records.

Single responsibility: validate inputs, build job parameters,
persist a JobRecord to the database, and publish a job.enqueued event.

Covers both inference jobs (txt2img, img2img) and model lifecycle jobs
(download, load) to keep job creation in one place.
"""

from __future__ import annotations

import logging
from uuid import UUID

from app.domain.enums import JobStatus, JobType
from app.domain.events import JobEvent
from app.domain.value_objects import GenerationParams
from app.infrastructure.database.models import JobRecord
from app.infrastructure.database.repositories import JobRepository, ModelRepository
from app.orchestrator.event_bus import event_bus

from .model_resolver import ModelResolver

logger = logging.getLogger(__name__)


class JobCreator:
    """Create and persist job records for all job types."""

    def __init__(
        self,
        job_repository: JobRepository,
        model_resolver: ModelResolver,
        model_repository: ModelRepository | None = None,
    ) -> None:
        self._job_repo = job_repository
        self._model_resolver = model_resolver
        self._model_repo = model_repository

    async def create_text_to_image_job(
        self,
        params: GenerationParams,
        model_id: str,
        correlation_id: str | None = None,
    ) -> UUID:
        """Create a PENDING text-to-image job."""
        resolved_model = await self._model_resolver.resolve(model_id)

        job_params = self._build_params(params, resolved_model, correlation_id)
        job_params["original_model_id"] = model_id

        job = JobRecord(
            job_type=JobType.TEXT_TO_IMAGE,
            status=JobStatus.PENDING,
            params=job_params,
        )
        created_job = await self._job_repo.create(job)
        await self._publish_enqueued_event(created_job.id, JobType.TEXT_TO_IMAGE, correlation_id)
        return created_job.id

    async def create_image_to_image_job(
        self,
        params: GenerationParams,
        model_id: str,
        source_image_path: str,
        correlation_id: str | None = None,
        strength: float = 0.75,
    ) -> UUID:
        """Create a PENDING image-to-image job."""
        resolved_model = await self._model_resolver.resolve(model_id)

        job_params = self._build_params(params, resolved_model, correlation_id)
        job_params["original_model_id"] = model_id
        job_params["source_image_path"] = source_image_path
        job_params["strength"] = strength

        job = JobRecord(
            job_type=JobType.IMAGE_TO_IMAGE,
            status=JobStatus.PENDING,
            params=job_params,
        )
        created_job = await self._job_repo.create(job)
        await self._publish_enqueued_event(created_job.id, JobType.IMAGE_TO_IMAGE, correlation_id)
        return created_job.id

    # ── Private helpers ────────────────────────────────────────────────

    @staticmethod
    def _build_params(
        gen_params: GenerationParams,
        resolved_model_id: str,
        correlation_id: str | None,
    ) -> dict:
        """Serialize GenerationParams + metadata into a flat params dict."""
        params: dict = {
            "model_id": resolved_model_id,
            "prompt": gen_params.prompt,
            "negative_prompt": gen_params.negative_prompt or "",
            "width": gen_params.width,
            "height": gen_params.height,
            "num_inference_steps": gen_params.num_inference_steps,
            "guidance_scale": gen_params.guidance_scale,
            "seed": gen_params.seed,
            "num_images": gen_params.num_images,
        }

        if correlation_id is not None:
            params["correlation_id"] = correlation_id

        # Forward any extra/unknown parameters
        if gen_params.extra:
            params.update(gen_params.extra)

        return params

    async def create_model_download_job(
        self,
        model_id: str,
        source: str,
        correlation_id: str | None = None,
    ) -> UUID:
        """Create a PENDING model download job.

        Validates the model exists in the catalog, then queues a download job.
        This keeps ModelService focused on catalog CRUD only.
        """
        if self._model_repo is None:
            raise RuntimeError("ModelRepository not configured for download jobs")

        model = await self._model_repo.get_by_model_id(model_id)
        if model is None:
            raise ValueError(f"Model {model_id} not found in registry")

        job = JobRecord(
            job_type=JobType.MODEL_DOWNLOAD,
            status=JobStatus.PENDING,
            model_id=model.id,
            params={"model_id": model_id, "source": source},
        )

        if correlation_id is not None:
            job.params["correlation_id"] = correlation_id

        created_job = await self._job_repo.create(job)
        await self._publish_enqueued_event(
            created_job.id, JobType.MODEL_DOWNLOAD, correlation_id
        )
        return created_job.id

    async def create_image_analysis_job(
        self,
        model_id: str,
        image_path: str,
        correlation_id: str | None = None,
    ) -> UUID:
        """Create a PENDING image analysis job."""
        job_params: dict = {
            "model_id": model_id,
            "image_path": image_path,
        }

        if correlation_id is not None:
            job_params["correlation_id"] = correlation_id

        job = JobRecord(
            job_type=JobType.IMAGE_ANALYSIS,
            status=JobStatus.PENDING,
            params=job_params,
        )
        created_job = await self._job_repo.create(job)
        await self._publish_enqueued_event(
            created_job.id, JobType.IMAGE_ANALYSIS, correlation_id
        )
        return created_job.id

    async def create_upscale_job(
        self,
        model_id: str,
        image_path: str,
        scale_factor: float = 2.0,
        correlation_id: str | None = None,
    ) -> UUID:
        """Create a PENDING upscale job."""
        resolved_model = await self._model_resolver.resolve(model_id)

        job_params: dict = {
            "model_id": resolved_model,
            "original_model_id": model_id,
            "image_path": image_path,
            "scale_factor": scale_factor,
        }

        if correlation_id is not None:
            job_params["correlation_id"] = correlation_id

        job = JobRecord(
            job_type=JobType.UPSCALE,
            status=JobStatus.PENDING,
            params=job_params,
        )
        created_job = await self._job_repo.create(job)
        await self._publish_enqueued_event(
            created_job.id, JobType.UPSCALE, correlation_id
        )
        return created_job.id

    async def create_recolor_job(
        self,
        model_id: str,
        image_path: str,
        prompt: str,
        strength: float = 0.75,
        correlation_id: str | None = None,
    ) -> UUID:
        """Create a PENDING recolor job."""
        resolved_model = await self._model_resolver.resolve(model_id)

        job_params: dict = {
            "model_id": resolved_model,
            "original_model_id": model_id,
            "image_path": image_path,
            "prompt": prompt,
            "strength": strength,
        }

        if correlation_id is not None:
            job_params["correlation_id"] = correlation_id

        job = JobRecord(
            job_type=JobType.RECOLOR,
            status=JobStatus.PENDING,
            params=job_params,
        )
        created_job = await self._job_repo.create(job)
        await self._publish_enqueued_event(
            created_job.id, JobType.RECOLOR, correlation_id
        )
        return created_job.id

    async def create_sketch_to_ink_job(
        self,
        model_id: str,
        image_path: str,
        correlation_id: str | None = None,
    ) -> UUID:
        """Create a PENDING sketch-to-ink job."""
        resolved_model = await self._model_resolver.resolve(model_id)

        job_params: dict = {
            "model_id": resolved_model,
            "original_model_id": model_id,
            "image_path": image_path,
        }

        if correlation_id is not None:
            job_params["correlation_id"] = correlation_id

        job = JobRecord(
            job_type=JobType.SKETCH_TO_INK,
            status=JobStatus.PENDING,
            params=job_params,
        )
        created_job = await self._job_repo.create(job)
        await self._publish_enqueued_event(
            created_job.id, JobType.SKETCH_TO_INK, correlation_id
        )
        return created_job.id

    async def create_image_captioning_job(
        self,
        model_id: str,
        image_path: str,
        correlation_id: str | None = None,
    ) -> UUID:
        """Create a PENDING image captioning (describe) job."""
        job_params: dict = {
            "model_id": model_id,
            "image_path": image_path,
        }

        if correlation_id is not None:
            job_params["correlation_id"] = correlation_id

        job = JobRecord(
            job_type=JobType.IMAGE_ANALYSIS,
            status=JobStatus.PENDING,
            params=job_params,
        )
        created_job = await self._job_repo.create(job)
        await self._publish_enqueued_event(
            created_job.id, JobType.IMAGE_ANALYSIS, correlation_id
        )
        return created_job.id

    @staticmethod
    async def _publish_enqueued_event(
        job_id: UUID,
        job_type: JobType,
        correlation_id: str | None,
    ) -> None:
        """Publish a job.enqueued event so SSE clients can react."""
        await event_bus.publish_event(
            JobEvent(
                event_type="job.enqueued",
                correlation_id=correlation_id,
                job_id=str(job_id),
                message=f"Job {job_type.value} enqueued",
                payload={"job_type": job_type},
            )
        )
