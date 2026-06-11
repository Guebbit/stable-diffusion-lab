"""
Generation router — endpoints for image and video generation.

All generation endpoints are asynchronous: they return a job_id (202 Accepted)
and the client polls or subscribes via WebSocket for completion.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Response, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    DescribeRequest,
    JobResponse,
    TextToImageRequest,
)
from app.domain.errors import JobNotFoundError
from app.domain.value_objects import GenerationParams
from app.infrastructure.database.repositories import JobRepository, ModelRepository
from app.infrastructure.database.session import get_async_session
from app.services.artifact_service import ArtifactService
from app.services.generation_service import GenerationService
from app.services.job_service import JobService

router = APIRouter(prefix="/generation", tags=["generation"])


def _get_generation_service(
    session: AsyncSession = Depends(get_async_session),
) -> GenerationService:
    """Dependency injection for GenerationService."""
    return GenerationService(
        job_repository=JobRepository(session),
        model_repository=ModelRepository(session),
    )


def _get_job_service(
    session: AsyncSession = Depends(get_async_session),
) -> JobService:
    """Dependency injection for JobService."""
    from app.infrastructure.database.repositories import JobEventRepository
    return JobService(JobRepository(session), JobEventRepository(session))


@router.post("/text-to-image", response_model=JobResponse, status_code=202)
async def submit_text_to_image(
    request: TextToImageRequest,
    response: Response,
    service: GenerationService = Depends(_get_generation_service),
    correlation_id: str | None = Header(default=None, alias="X-Correlation-ID"),
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
    job_id = await service.submit_text_to_image(
        params,
        request.model_id,
        correlation_id=correlation_id,
    )
    if correlation_id:
        response.headers["X-Correlation-ID"] = correlation_id
    return JobResponse(job_id=job_id, status="pending", correlation_id=correlation_id)


@router.post("/describe", response_model=JobResponse, status_code=202)
async def submit_describe(
    model_id: str = Form(...),
    image: UploadFile = File(...),
    response: Response = Response(),
    service: GenerationService = Depends(_get_generation_service),
    correlation_id: str | None = Header(default=None, alias="X-Correlation-ID"),
) -> JobResponse:
    """
    Submit an image description (captioning) job.

    Accepts an image via multipart/form-data and returns a job_id.
    The vision model will analyze the image and produce a text description.
    """
    # Save uploaded image to a temporary path for the job processor
    import tempfile
    import os

    # Create a temp file with the same extension for proper handling
    suffix = os.path.splitext(image.filename)[1] if image.filename else ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir="/tmp") as tmp:
        content = await image.read()
        tmp.write(content)
        image_path = tmp.name

    job_id = await service.submit_image_captioning(
        model_id,
        image_path,
        correlation_id=correlation_id,
    )
    if correlation_id:
        response.headers["X-Correlation-ID"] = correlation_id
    return JobResponse(job_id=job_id, status="pending", correlation_id=correlation_id)


@router.get("/jobs/{job_id}")
async def get_job_status(
    job_id: UUID,
    service: JobService = Depends(_get_job_service),
) -> dict[str, Any]:
    """
    Poll the status of a generation job.

    Returns the current job state. When status is 'completed', the response
    includes artifact references.
    """
    try:
        job = await service.get_job(job_id)
    except JobNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found",
        )

    return {
        "id": str(job.id),
        "status": job.status,
        "job_type": job.job_type,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
        "progress_percent": job.progress_percent,
        "error": job.error,
        "model_id": str(job.model_id) if job.model_id else None,
        "params": job.params,
    }