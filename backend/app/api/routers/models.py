"""
Models router — endpoints for the model registry and lifecycle.

Handles model catalog CRUD and download triggering.
Job orchestration is delegated to JobCreator.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.error_handler import from_exception
from app.api.schemas import (
    JobResponse,
    ModelRegisterRequest,
    ModelRegistryResponse,
)
from app.infrastructure.database.repositories import JobRepository, ModelRepository
from app.infrastructure.database.session import get_async_session
from app.infrastructure.storage.storage_manager import StorageManager
from app.services.job_creator import JobCreator
from app.services.model_resolver import ModelResolver
from app.services.model_service import ModelService

router = APIRouter(prefix="/models", tags=["models"])


def _get_model_service(session: AsyncSession = Depends(get_async_session)) -> ModelService:
    """Dependency injection for ModelService (catalog CRUD only)."""
    return ModelService(
        model_repository=ModelRepository(session),
        storage_manager=StorageManager(),
    )


def _get_job_creator(session: AsyncSession = Depends(get_async_session)) -> JobCreator:
    """Dependency injection for JobCreator (model download jobs)."""
    return JobCreator(
        job_repository=JobRepository(session),
        model_resolver=ModelResolver(ModelRepository(session)),
        model_repository=ModelRepository(session),
    )


def _model_to_response(m) -> ModelRegistryResponse:
    """Convert a database model to a response object."""
    return ModelRegistryResponse(
        id=m.id,
        model_id=m.model_id,
        name=m.name,
        preferred_name=getattr(m, "preferred_name", None),
        source=m.source,
        family=m.family,
        variant=m.variant,
        description=m.description,
        tags=m.tags if isinstance(m.tags, list) else [],
        source_url=m.source_url,
        capabilities=m.capabilities if isinstance(m.capabilities, list) else [],
        status=m.status,
        total_size_bytes=m.total_size_bytes,
        disk_size_bytes=m.disk_size_bytes,
        download_size_bytes=m.download_size_bytes,
        download_progress=m.download_progress,
        recommended_vram_min_gb=m.recommended_vram_min_gb,
        recommended_vram_max_gb=m.recommended_vram_max_gb,
        license=m.license,
        base_model=m.base_model,
        precision=m.precision,
        requirements=m.requirements if isinstance(m.requirements, dict) else {},
        notes=m.notes,
        is_verified=m.is_verified,
        last_verified_at=m.last_verified_at,
        local_path=getattr(m, "local_path", None),
        file_count=getattr(m, "file_count", 0),
        created_at=m.created_at,
        updated_at=m.updated_at,
    )


@router.get("/", response_model=list[ModelRegistryResponse])
async def list_models(
    source: str | None = None,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    service: ModelService = Depends(_get_model_service),
) -> list[ModelRegistryResponse]:
    """List all registered models, optionally filtered by source."""
    models = await service.list_models(source=source)
    models = models[offset : offset + limit]
    return [_model_to_response(m) for m in models]


@router.get("/{model_id:path}", response_model=ModelRegistryResponse)
async def get_model(
    model_id: str,
    service: ModelService = Depends(_get_model_service),
) -> ModelRegistryResponse:
    """Get details for a specific model by external model_id."""
    model = await service.get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return _model_to_response(model)


@router.post("/", response_model=ModelRegistryResponse, status_code=201)
async def register_model(
    request: ModelRegisterRequest,
    service: ModelService = Depends(_get_model_service),
) -> ModelRegistryResponse:
    """Register a new model in the catalog."""
    model = await service.register_model(
        model_id=request.model_id,
        name=request.name,
        source=request.source,
        family=request.family,
        variant=request.variant,
        description=request.description,
        tags=request.tags,
        source_url=request.source_url,
        capabilities=request.capabilities,
        preferred_name=request.preferred_name,
    )
    return _model_to_response(model)


@router.post("/{model_id:path}/download", response_model=JobResponse, status_code=202)
async def download_model(
    model_id: str,
    model_service: ModelService = Depends(_get_model_service),
    job_creator: JobCreator = Depends(_get_job_creator),
) -> JobResponse:
    """Trigger download of a registered model."""
    try:
        # Look up model to get its source
        model = await model_service.get_model(model_id)
        if model is None:
            raise HTTPException(status_code=404, detail="Model not found")

        job_id = await job_creator.create_model_download_job(
            model_id=model_id,
            source=model.source,
        )
    except Exception as exc:
        raise from_exception(exc)
    return JobResponse(job_id=job_id, status="pending", message="Download queued")


@router.delete("/{model_id:path}", status_code=204, response_model=None)
async def delete_model(
    model_id: str,
    service: ModelService = Depends(_get_model_service),
) -> None:
    """Delete a model from catalog and disk."""
    try:
        await service.delete_model(model_id)
    except Exception as exc:
        raise from_exception(exc)