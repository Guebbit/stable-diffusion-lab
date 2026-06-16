"""
Generation router — endpoints for image and video generation.

All generation endpoints are asynchronous: they return a job_id (202 Accepted)
and the client polls or subscribes via WebSocket for completion.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Response, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    JobResponse,
    RecolorRequest,
    SketchToInkRequest,
    TextToImageRequest,
)
from app.domain.errors import JobNotFoundError
from app.domain.value_objects import GenerationParams
from app.infrastructure.database.repositories import (
    JobEventRepository,
    JobRepository,
    ModelRepository,
)
from app.infrastructure.database.session import get_async_session
from app.services.artifact_service import ArtifactService
from app.services.generation_service import GenerationService
from app.services.job_creator import JobCreator
from app.services.job_service import JobService
from app.services.model_resolver import ModelResolver

router = APIRouter(prefix="/generation", tags=["generation"])


# ── Dependency factories ──────────────────────────────────────────────────

def _get_generation_service(
    session: AsyncSession = Depends(get_async_session),
) -> GenerationService:
    """Dependency injection for GenerationService."""
    model_repo = ModelRepository(session)
    job_repo = JobRepository(session)
    model_resolver = ModelResolver(model_repo)
    job_creator = JobCreator(job_repo, model_resolver, model_repo)
    return GenerationService(job_creator)


def _get_job_service(
    session: AsyncSession = Depends(get_async_session),
) -> JobService:
    """Dependency injection for JobService."""
    return JobService(JobRepository(session), JobEventRepository(session))


# ── Helper utilities ──────────────────────────────────────────────────────

def _set_correlation_header(response: Response, correlation_id: str | None) -> None:
    """Set X-Correlation-ID response header if a correlation ID was provided."""
    if correlation_id:
        response.headers["X-Correlation-ID"] = correlation_id


# ── Route handlers ────────────────────────────────────────────────────────

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
        lora_model_id=request.lora_model_id or None,
        lora_strength=request.lora_strength,
        correlation_id=correlation_id,
    )
    _set_correlation_header(response, correlation_id)
    return JobResponse(job_id=job_id, status="pending", correlation_id=correlation_id)


@router.post("/analyze", response_model=JobResponse, status_code=202)
async def submit_analyze(
    model_id: str = Form(...),
    image: UploadFile = File(...),
    response: Response = Response(),
    service: GenerationService = Depends(_get_generation_service),
    correlation_id: str | None = Header(default=None, alias="X-Correlation-ID"),
) -> JobResponse:
    """
    Submit an image analysis job.

    Accepts an image via multipart/form-data and returns a job_id.
    The vision model will analyze the image and produce a text description.
    """
    # Save uploaded image to a temporary path for the job processor
    suffix = os.path.splitext(image.filename)[1] if image.filename else ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir="/tmp") as tmp:
        content = await image.read()
        tmp.write(content)
        image_path = tmp.name

    job_id = await service.submit_image_analysis(
        model_id,
        image_path,
        correlation_id=correlation_id,
    )
    _set_correlation_header(response, correlation_id)
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
    Submit an image captioning (describe) job.

    Accepts an image via multipart/form-data and returns a job_id.
    The vision model will analyze the image and produce a text description.
    """
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
    _set_correlation_header(response, correlation_id)
    return JobResponse(job_id=job_id, status="pending", correlation_id=correlation_id)


@router.post("/recolor", response_model=JobResponse, status_code=202)
async def submit_recolor(
    model_id: str = Form(...),
    image: UploadFile = File(...),
    prompt: str = Form(...),
    strength: float = Form(0.75),
    response: Response = Response(),
    service: GenerationService = Depends(_get_generation_service),
    correlation_id: str | None = Header(default=None, alias="X-Correlation-ID"),
) -> JobResponse:
    """
    Submit an image recolor job.

    Accepts an image and a color guidance prompt via multipart/form-data.
    """
    suffix = os.path.splitext(image.filename)[1] if image.filename else ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir="/tmp") as tmp:
        content = await image.read()
        tmp.write(content)
        image_path = tmp.name

    job_id = await service.submit_recolor(
        model_id,
        image_path,
        prompt=prompt,
        strength=strength,
        correlation_id=correlation_id,
    )
    _set_correlation_header(response, correlation_id)
    return JobResponse(job_id=job_id, status="pending", correlation_id=correlation_id)


@router.post("/sketch-to-ink", response_model=JobResponse, status_code=202)
async def submit_sketch_to_ink(
    model_id: str = Form(...),
    image: UploadFile = File(...),
    prompt: str = Form("", description="Text prompt to guide the inking style and content"),
    negative_prompt: str = Form(""),
    num_inference_steps: int = Form(28, ge=1, le=150),
    guidance_scale: float = Form(8.0, ge=1.0, le=30.0),
    adapter_conditioning_scale: float = Form(
        0.9,
        ge=0.1,
        le=2.0,
        description="How closely the output follows the sketch structure (0.5–1.2 is the typical range)",
    ),
    base_model_id: str = Form("", description="Override the auto-resolved base model (leave empty to use the model's default)"),
    lora_model_id: str = Form("", description="Optional LoRA model ID to apply on top of the base model"),
    lora_strength: float = Form(0.8, ge=0.0, le=2.0, description="LoRA conditioning scale"),
    response: Response = Response(),
    service: GenerationService = Depends(_get_generation_service),
    correlation_id: str | None = Header(default=None, alias="X-Correlation-ID"),
) -> JobResponse:
    """
    Submit a sketch-to-ink job.

    Accepts a sketch image via multipart/form-data.
    The prompt guides style/content; adapter_conditioning_scale controls
    how faithfully the output traces the input sketch.
    Returns 202 Accepted — poll GET /api/v1/jobs/{job_id} for status.
    """
    suffix = os.path.splitext(image.filename)[1] if image.filename else ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir="/tmp") as tmp:
        content = await image.read()
        tmp.write(content)
        image_path = tmp.name

    job_id = await service.submit_sketch_to_ink(
        model_id,
        image_path,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        adapter_conditioning_scale=adapter_conditioning_scale,
        base_model_id_override=base_model_id or None,
        lora_model_id=lora_model_id or None,
        lora_strength=lora_strength,
        correlation_id=correlation_id,
    )
    _set_correlation_header(response, correlation_id)
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