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

    async def submit_image_analysis(
        self,
        model_id: str,
        image_path: str,
        correlation_id: str | None = None,
    ) -> UUID:
        """Submit an image analysis job."""
        job_id = await self._job_creator.create_image_analysis_job(
            model_id, image_path, correlation_id
        )
        logger.info(
            "Created image-analysis job: %s (model_id=%s)",
            job_id,
            model_id,
        )
        return job_id

    async def submit_upscale(
        self,
        model_id: str,
        image_path: str,
        scale_factor: float = 2.0,
        prompt: str = "",
        noise_level: int = 20,
        num_inference_steps: int = 20,
        enhance_model_id: str | None = None,
        enhance_strength: float = 0.4,
        face_restore_model_id: str | None = None,
        face_restore_fidelity: float = 0.5,
        correlation_id: str | None = None,
    ) -> UUID:
        """Submit a multi-step upscale pipeline job."""
        job_id = await self._job_creator.create_upscale_job(
            model_id, image_path,
            scale_factor=scale_factor,
            prompt=prompt,
            noise_level=noise_level,
            num_inference_steps=num_inference_steps,
            enhance_model_id=enhance_model_id,
            enhance_strength=enhance_strength,
            face_restore_model_id=face_restore_model_id,
            face_restore_fidelity=face_restore_fidelity,
            correlation_id=correlation_id,
        )
        logger.info(
            "Created upscale job: %s (model_id=%s)",
            job_id,
            model_id,
        )
        return job_id

    async def submit_recolor(
        self,
        model_id: str,
        image_path: str,
        prompt: str,
        strength: float = 0.75,
        correlation_id: str | None = None,
    ) -> UUID:
        """Submit a recolor job."""
        job_id = await self._job_creator.create_recolor_job(
            model_id, image_path, prompt, strength, correlation_id
        )
        logger.info(
            "Created recolor job: %s (model_id=%s)",
            job_id,
            model_id,
        )
        return job_id

    async def submit_sketch_to_ink(
        self,
        model_id: str,
        image_path: str,
        correlation_id: str | None = None,
    ) -> UUID:
        """Submit a sketch-to-ink job."""
        job_id = await self._job_creator.create_sketch_to_ink_job(
            model_id, image_path, correlation_id
        )
        logger.info(
            "Created sketch-to-ink job: %s (model_id=%s)",
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
