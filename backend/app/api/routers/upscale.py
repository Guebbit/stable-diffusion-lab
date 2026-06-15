"""
Upscale router — dedicated pipeline for image upscaling and quality enhancement.

Separate from the general generation router so the upscale workflow can evolve
independently (multi-step, prompt guidance, noise control, etc.).
All endpoints are async job-based: they return 202 Accepted with a job_id.
"""

from __future__ import annotations

import os
import tempfile

from fastapi import APIRouter, Depends, File, Form, Header, Response, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import JobResponse
from app.infrastructure.database.repositories import JobEventRepository, JobRepository, ModelRepository
from app.infrastructure.database.session import get_async_session
from app.services.generation_service import GenerationService
from app.services.job_creator import JobCreator
from app.services.job_service import JobService
from app.services.model_resolver import ModelResolver

router = APIRouter(prefix="/upscale", tags=["upscale"])


def _get_generation_service(
    session: AsyncSession = Depends(get_async_session),
) -> GenerationService:
    model_repo = ModelRepository(session)
    job_repo = JobRepository(session)
    model_resolver = ModelResolver(model_repo)
    job_creator = JobCreator(job_repo, model_resolver, model_repo)
    return GenerationService(job_creator)


def _set_correlation_header(response: Response, correlation_id: str | None) -> None:
    if correlation_id:
        response.headers["X-Correlation-ID"] = correlation_id


@router.post("/run", response_model=JobResponse, status_code=202)
async def run_upscale_pipeline(
    # ── Step 2: upscale (required) ──────────────────────────────────────
    model_id: str = Form(..., description="ID of an upscale_image-capable model"),
    image: UploadFile = File(...),
    scale_factor: float = Form(2.0, ge=1.0, le=8.0),
    prompt: str = Form("", description="Optional prompt to guide detail synthesis"),
    noise_level: int = Form(20, ge=0, le=350),
    num_inference_steps: int = Form(20, ge=1, le=150),
    # ── Step 1: enhancement (optional) ──────────────────────────────────
    enhance_model_id: str | None = Form(None, description="img2img model for pre-upscale enhancement"),
    enhance_strength: float = Form(0.4, ge=0.05, le=0.95),
    # ── Step 3: face restore (optional) ─────────────────────────────────
    face_restore_model_id: str | None = Form(None, description="GFPGAN model for post-upscale face repair"),
    face_restore_fidelity: float = Form(0.5, ge=0.0, le=1.0),
    response: Response = Response(),
    service: GenerationService = Depends(_get_generation_service),
    correlation_id: str | None = Header(default=None, alias="X-Correlation-ID"),
) -> JobResponse:
    """
    Submit an upscale pipeline job.

    The pipeline runs up to three steps:
      1. Enhancement (optional) — img2img quality improvement before upscaling
      2. Upscale (required)     — SD x4 diffusion upscaler
      3. Face Restore (optional) — GFPGAN face sharpening after upscaling

    Accepts a source image and all parameters via multipart/form-data.
    Returns 202 Accepted immediately — poll GET /api/v1/jobs/{job_id} for status.
    """
    suffix = os.path.splitext(image.filename)[1] if image.filename else ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir="/tmp") as tmp:
        content = await image.read()
        tmp.write(content)
        image_path = tmp.name

    job_id = await service.submit_upscale(
        model_id,
        image_path,
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
    _set_correlation_header(response, correlation_id)
    return JobResponse(job_id=job_id, status="pending", correlation_id=correlation_id)
