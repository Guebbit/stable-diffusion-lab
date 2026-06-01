"""
Generation router — endpoints for image and video generation.

All generation endpoints are asynchronous: they return a job_id (202 Accepted)
and the client polls or subscribes via WebSocket for completion.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    JobResponse,
    JobStatusResponse,
    TextToImageRequest,
)
from app.domain.value_objects import GenerationParams
from app.infrastructure.database.repositories import JobRepository
from app.infrastructure.database.session import get_async_session
from app.services.generation_service import GenerationService

router = APIRouter(prefix="/generation", tags=["generation"])


def _get_generation_service(session: AsyncSession = Depends(get_async_session)) -> GenerationService:
    """Dependency injection for GenerationService."""
    return GenerationService(job_repository=JobRepository(session))


@router.post("/text-to-image", response_model=JobResponse, status_code=202)
async def submit_text_to_image(
    request: TextToImageRequest,
    service: GenerationService = Depends(_get_generation_service),
) -> JobResponse:
    """
    Submit a text-to-image generation job.

    Returns immediately with a job_id. Poll GET /jobs/{job_id} or
    subscribe to WebSocket for progress updates.
    """
    params = GenerationParams(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        seed=request.seed,
        num_images=request.num_images,
    )
    job_id = await service.submit_text_to_image(params, request.model_id)
    return JobResponse(job_id=job_id, status="pending")


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: UUID,
    service: GenerationService = Depends(_get_generation_service),
) -> JobStatusResponse:
    """Get the current status of a generation job."""
    job = await service.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        id=job.id,
        job_type=job.job_type,
        status=job.status,
        progress_percent=job.progress_percent,
        current_step=job.current_step,
        total_steps=job.total_steps,
        message=job.message,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error=job.error,
        result=job.result,
    )
