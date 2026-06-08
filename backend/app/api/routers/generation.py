"""
Generation router — endpoints for image and video generation.

All generation endpoints are asynchronous: they return a job_id (202 Accepted)
and the client polls or subscribes via WebSocket for completion.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    JobResponse,
    TextToImageRequest,
)
from app.domain.value_objects import GenerationParams
from app.infrastructure.database.repositories import JobRepository, ModelRepository
from app.infrastructure.database.session import get_async_session
from app.services.generation_service import GenerationService

router = APIRouter(prefix="/generation", tags=["generation"])


def _get_generation_service(
    session: AsyncSession = Depends(get_async_session),
) -> GenerationService:
    """Dependency injection for GenerationService."""
    return GenerationService(
        job_repository=JobRepository(session),
        model_repository=ModelRepository(session),
    )


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
