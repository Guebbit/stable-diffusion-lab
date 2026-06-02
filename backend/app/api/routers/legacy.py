"""
Legacy compatibility router — bridges frontend expected paths to the v1 API.

The frontend calls /api/{path} while the backend exposes /api/v1/{router}/{path}.
This router maps the legacy paths so the frontend works without changes while
migration to the new contract is in progress.

All endpoints here delegate to the actual service layer directly.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    JobResponse,
    ModelRegisterRequest,
    ModelRegistryResponse,
    SystemStatusResponse,
)
from app.domain.enums import JobStatus, JobType
from app.domain.value_objects import GenerationParams
from app.infrastructure.config.settings import get_settings
from app.infrastructure.database.models import JobRecord
from app.infrastructure.database.repositories import (
    ArtifactRepository,
    JobRepository,
    ModelRepository,
)
from app.infrastructure.database.session import get_async_session
from app.infrastructure.storage.storage_manager import StorageManager
from app.services.generation_service import GenerationService
from app.services.model_service import ModelService

router = APIRouter(tags=["legacy"])


# ─── Dependencies ──


def _get_generation_service(session: AsyncSession = Depends(get_async_session)) -> GenerationService:
    """Dependency injection for GenerationService."""
    return GenerationService(job_repository=JobRepository(session))


def _get_model_service(session: AsyncSession = Depends(get_async_session)) -> ModelService:
    """Dependency injection for ModelService."""
    return ModelService(
        model_repository=ModelRepository(session),
        job_repository=JobRepository(session),
        storage_manager=StorageManager(),
    )


def _get_artifact_repository(
    session: AsyncSession = Depends(get_async_session),
) -> ArtifactRepository:
    """Dependency injection for ArtifactRepository."""
    return ArtifactRepository(session)


# ─── System ──


@router.get("/status")
async def legacy_status(request: Request) -> dict[str, Any]:
    """
    Legacy status endpoint.

    Maps to the frontend BackendStatus interface:
    {status, loaded_model, device, message}
    """
    import torch

    settings = get_settings()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Try to get currently loaded model from app state
    loaded_model: str | None = None
    if hasattr(request.app.state, "model_manager"):
        loaded_models = request.app.state.model_manager.get_loaded_models()
        loaded_model = loaded_models[0] if loaded_models else None

    return {
        "status": "ok",
        "loaded_model": loaded_model,
        "device": device,
        "message": f"v{settings.app_version}",
    }


# ─── Model Load ──


@router.post("/models/load")
async def legacy_load_model(request: Request) -> dict[str, Any]:
    """
    Legacy model load endpoint.

    Accepts {model_id, model_source, task?} and loads the model synchronously.
    Returns {success, model_id, message}.
    """
    body = await request.json()
    model_id = body.get("model_id")
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")

    model_manager = request.app.state.model_manager
    try:
        await model_manager.load_model(model_id)
        return {"success": True, "model_id": model_id, "message": "Model loaded successfully"}
    except Exception as exc:
        return {"success": False, "model_id": model_id, "message": str(exc)}


# ─── Generation ──


@router.post("/generate")
async def legacy_generate(
    request: Request,
    service: GenerationService = Depends(_get_generation_service),
) -> dict[str, Any]:
    """
    Legacy text-to-image generation endpoint.

    Submits a generation job and waits for completion (with timeout),
    returning the result in the format the frontend expects.
    """
    body = await request.json()
    params = GenerationParams(
        prompt=body["prompt"],
        negative_prompt=body.get("negative_prompt", ""),
        width=body.get("width", 512),
        height=body.get("height", 512),
        num_inference_steps=body.get("num_inference_steps", 20),
        guidance_scale=body.get("guidance_scale", 7.5),
        seed=body.get("seed"),
        num_images=body.get("num_images", 1),
    )
    model_id = body.get("model_id", "")

    job_id = await service.submit_text_to_image(params, model_id)

    # Poll for completion (the frontend expects a synchronous response)
    result = await _wait_for_job(request, job_id)
    return result


@router.post("/generate-from-image")
async def legacy_generate_from_image(
    request: Request,
    image: UploadFile = File(...),
    prompt: str = Form(...),
    model_id: str = Form(...),
    model_source: str = Form("huggingface"),
    negative_prompt: str = Form(""),
    num_inference_steps: int = Form(20),
    guidance_scale: float = Form(7.5),
    num_images: int = Form(1),
    width: int | None = Form(None),
    height: int | None = Form(None),
    seed: int | None = Form(None),
    workflow_preset: str | None = Form(None),
    strength: float = Form(0.75),
) -> dict[str, Any]:
    """Legacy img2img generation endpoint."""
    settings = get_settings()

    # Save uploaded image to temp
    temp_path = settings.temp_path / f"{uuid4()}{Path(image.filename or 'img').suffix}"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    content = await image.read()
    temp_path.write_bytes(content)

    # Create job directly
    from app.infrastructure.database.session import get_session_factory

    session_factory = get_session_factory()
    async with session_factory() as session:
        job_repo = JobRepository(session)
        job = JobRecord(
            job_type=JobType.IMAGE_TO_IMAGE,
            status=JobStatus.PENDING,
            params={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "model_id": model_id,
                "source_image_path": str(temp_path),
                "width": width or 512,
                "height": height or 512,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_images": num_images,
                "seed": seed,
                "strength": strength,
            },
        )
        job = await job_repo.create(job)
        await session.commit()
        job_id = job.id

    result = await _wait_for_job(request, job_id)
    return result


@router.post("/generate-sketch-to-ink")
async def legacy_generate_sketch_to_ink(
    request: Request,
    image: UploadFile = File(...),
    prompt: str = Form(...),
    model_id: str = Form(...),
    model_source: str = Form("huggingface"),
    negative_prompt: str = Form(""),
    num_inference_steps: int = Form(20),
    guidance_scale: float = Form(7.5),
    num_images: int = Form(1),
    width: int | None = Form(None),
    height: int | None = Form(None),
    seed: int | None = Form(None),
    controlnet_conditioning_scale: float = Form(1.0),
) -> dict[str, Any]:
    """Legacy sketch-to-ink ControlNet generation endpoint."""
    settings = get_settings()

    # Save uploaded image to temp
    temp_path = settings.temp_path / f"{uuid4()}{Path(image.filename or 'img').suffix}"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    content = await image.read()
    temp_path.write_bytes(content)

    from app.infrastructure.database.session import get_session_factory

    session_factory = get_session_factory()
    async with session_factory() as session:
        job_repo = JobRepository(session)
        job = JobRecord(
            job_type=JobType.IMAGE_TO_IMAGE,
            status=JobStatus.PENDING,
            params={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "model_id": model_id,
                "source_image_path": str(temp_path),
                "width": width or 512,
                "height": height or 512,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_images": num_images,
                "seed": seed,
                "strength": controlnet_conditioning_scale,
            },
        )
        job = await job_repo.create(job)
        await session.commit()
        job_id = job.id

    result = await _wait_for_job(request, job_id)
    return result


@router.post("/describe-image")
async def legacy_describe_image(
    request: Request,
    image: UploadFile = File(...),
    model_id: str = Form(...),
) -> dict[str, Any]:
    """Legacy image description/captioning endpoint."""
    settings = get_settings()

    # Save uploaded image to temp
    temp_path = settings.temp_path / f"{uuid4()}{Path(image.filename or 'img').suffix}"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    content = await image.read()
    temp_path.write_bytes(content)

    from app.infrastructure.database.session import get_session_factory

    session_factory = get_session_factory()
    async with session_factory() as session:
        job_repo = JobRepository(session)
        job = JobRecord(
            job_type=JobType.IMAGE_CAPTIONING,
            status=JobStatus.PENDING,
            params={
                "model_id": model_id,
                "image_path": str(temp_path),
            },
        )
        job = await job_repo.create(job)
        await session.commit()
        job_id = job.id

    result = await _wait_for_job(request, job_id)
    return {
        "description": result.get("message", ""),
        "model_id": model_id,
        "elapsed_seconds": 0,
    }


# ─── History (maps to artifacts) ──


@router.get("/history")
async def legacy_get_history(
    repo: ArtifactRepository = Depends(_get_artifact_repository),
) -> list[dict[str, Any]]:
    """
    Legacy history endpoint — returns all artifacts as GeneratedImage[].

    Maps the ArtifactRecord fields to the frontend GeneratedImage interface.
    """
    artifacts = await repo.list_recent(limit=200)
    return [
        {
            "id": str(a.id),
            "url": f"/artifacts/{a.id}/file",
            "prompt": a.prompt or "",
            "negative_prompt": a.negative_prompt or "",
            "model_id": a.model_id_ref or "",
            "width": a.width or 0,
            "height": a.height or 0,
            "seed": a.seed or 0,
            "created_at": a.created_at.isoformat() if a.created_at else "",
            "num_inference_steps": (a.generation_params or {}).get("num_inference_steps", 0),
            "guidance_scale": (a.generation_params or {}).get("guidance_scale", 0),
            "generation_time_seconds": (a.generation_params or {}).get("elapsed_seconds", 0),
            "device": (a.generation_params or {}).get("device", "unknown"),
            "scheduler": (a.generation_params or {}).get("scheduler", ""),
            "pipeline_class": (a.generation_params or {}).get("pipeline_class", ""),
        }
        for a in artifacts
    ]


@router.delete("/history/{image_id}")
async def legacy_delete_history_entry(
    image_id: UUID,
    repo: ArtifactRepository = Depends(_get_artifact_repository),
) -> None:
    """Delete a single history entry (artifact) by UUID."""
    artifact = await repo.get_by_id(image_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Not found")
    await repo.delete(image_id)


@router.delete("/history")
async def legacy_clear_history(
    repo: ArtifactRepository = Depends(_get_artifact_repository),
) -> None:
    """Delete all history entries (artifacts)."""
    await repo.delete_all()


# ─── Models (legacy paths) ──


@router.get("/models")
async def legacy_list_models(
    service: ModelService = Depends(_get_model_service),
) -> list[dict[str, Any]]:
    """Legacy model list — returns models matching frontend ModelRegistryEntry."""
    models = await service.list_models()
    return [
        {
            "id": m.model_id,
            "name": m.name,
            "source": m.source,
            "family": m.family,
            "description": m.description,
            "long_description": "",
            "tags": m.tags if isinstance(m.tags, list) else [],
            "source_url": m.source_url,
            "size": str(m.total_size_bytes or 0),
            "downloaded": m.status == "downloaded",
            "status": m.status,
        }
        for m in models
    ]


@router.get("/models/downloaded")
async def legacy_list_downloaded_models(
    service: ModelService = Depends(_get_model_service),
) -> list[dict[str, Any]]:
    """Legacy endpoint: only models that are downloaded and ready."""
    models = await service.list_models()
    return [
        {
            "id": m.model_id,
            "name": m.name,
            "source": m.source,
            "family": m.family,
            "description": m.description,
            "long_description": "",
            "tags": m.tags if isinstance(m.tags, list) else [],
            "source_url": m.source_url,
            "size": str(m.total_size_bytes or 0),
            "downloaded": True,
            "status": m.status,
        }
        for m in models
        if m.status == "downloaded"
    ]


@router.post("/models")
async def legacy_add_model(
    request: Request,
    service: ModelService = Depends(_get_model_service),
) -> dict[str, Any]:
    """Legacy register model endpoint."""
    body = await request.json()
    model = await service.register_model(
        model_id=body.get("id", ""),
        name=body.get("name", ""),
        source=body.get("source", "huggingface"),
        family=body.get("family", "custom"),
        description=body.get("description", ""),
        tags=body.get("tags", []),
        source_url=body.get("source_url", ""),
    )
    return {
        "id": model.model_id,
        "name": model.name,
        "source": model.source,
        "family": model.family,
        "description": model.description,
        "long_description": "",
        "tags": model.tags if isinstance(model.tags, list) else [],
        "source_url": model.source_url,
        "size": "0",
        "downloaded": False,
        "status": model.status,
    }


@router.delete("/models/{model_id}")
async def legacy_remove_model(
    model_id: str,
    service: ModelService = Depends(_get_model_service),
) -> None:
    """Legacy delete model endpoint."""
    try:
        await service.delete_model(model_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/models/{model_id}/download")
async def legacy_download_model(
    model_id: str,
    service: ModelService = Depends(_get_model_service),
) -> dict[str, Any]:
    """Legacy trigger model download."""
    try:
        job_id = await service.request_download(model_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"detail": "Download queued", "status": "pending"}


@router.get("/models/{model_id}/progress")
async def legacy_download_progress(
    model_id: str,
    service: ModelService = Depends(_get_model_service),
) -> dict[str, Any]:
    """Legacy download progress endpoint."""
    # Return a stub — real progress tracking requires model download events
    model = await service.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    if model.status == "downloaded":
        return {"downloaded_bytes": 1, "total_bytes": 1, "percentage": 100}
    elif model.status == "downloading":
        progress = model.download_progress or 0
        return {"downloaded_bytes": progress, "total_bytes": 100, "percentage": progress}
    return {"downloaded_bytes": 0, "total_bytes": 0, "percentage": 0}


@router.get("/models/download-events")
async def legacy_download_events() -> list[dict[str, Any]]:
    """Legacy download events — stub returning empty list."""
    return []


@router.delete("/models/download-events")
async def legacy_clear_download_events() -> dict[str, str]:
    """Legacy clear download events — stub."""
    return {"detail": "ok"}


# ─── Helpers ──


async def _wait_for_job(request: Request, job_id: UUID, timeout: float = 600.0) -> dict[str, Any]:
    """
    Poll-wait for a job to complete, then return its result.

    The legacy frontend expects synchronous responses, so we wait until
    the job worker finishes processing. Times out after `timeout` seconds.
    """
    from app.infrastructure.database.session import get_session_factory

    session_factory = get_session_factory()
    settings = get_settings()
    elapsed = 0.0
    poll_interval = 1.0

    while elapsed < timeout:
        async with session_factory() as session:
            job_repo = JobRepository(session)
            job = await job_repo.get_by_id(job_id)
            if job and job.status in ("completed", "failed"):
                if job.status == "failed":
                    raise HTTPException(status_code=500, detail=job.error or "Job failed")
                # Return job result (adapters store output details here)
                result = job.result or {}
                # Build response matching frontend GenerationResponse
                artifacts_path = settings.artifacts_path / str(job_id)
                images = []
                if artifacts_path.exists():
                    for img_file in sorted(artifacts_path.iterdir()):
                        if img_file.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                            images.append({
                                "id": str(uuid4()),
                                "url": f"/artifacts/{job_id}/{img_file.name}",
                                "prompt": (job.params or {}).get("prompt", ""),
                                "model_id": (job.params or {}).get("model_id", ""),
                                "width": (job.params or {}).get("width", 512),
                                "height": (job.params or {}).get("height", 512),
                                "seed": (job.params or {}).get("seed", 0),
                                "created_at": job.completed_at.isoformat() if job.completed_at else "",
                                "num_inference_steps": (job.params or {}).get("num_inference_steps", 20),
                                "guidance_scale": (job.params or {}).get("guidance_scale", 7.5),
                                "generation_time_seconds": 0,
                                "device": "cuda",
                                "scheduler": "",
                                "pipeline_class": "",
                            })
                return {
                    "images": images,
                    "model_id": (job.params or {}).get("model_id", ""),
                    "elapsed_seconds": 0,
                }

        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

    raise HTTPException(status_code=504, detail="Job timed out")
