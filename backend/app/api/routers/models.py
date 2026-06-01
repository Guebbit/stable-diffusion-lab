"""
Models router — endpoints for the model registry and lifecycle.

Handles model catalog CRUD, download triggering, and status queries.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    JobResponse,
    ModelRegisterRequest,
    ModelRegistryResponse,
)
from app.infrastructure.database.repositories import JobRepository, ModelRepository
from app.infrastructure.database.session import get_async_session
from app.infrastructure.storage.storage_manager import StorageManager
from app.services.model_service import ModelService

router = APIRouter(prefix="/models", tags=["models"])


def _get_model_service(session: AsyncSession = Depends(get_async_session)) -> ModelService:
    """Dependency injection for ModelService."""
    return ModelService(
        model_repository=ModelRepository(session),
        job_repository=JobRepository(session),
        storage_manager=StorageManager(),
    )


@router.get("/", response_model=list[ModelRegistryResponse])
async def list_models(
    source: str | None = None,
    service: ModelService = Depends(_get_model_service),
) -> list[ModelRegistryResponse]:
    """List all registered models, optionally filtered by source."""
    models = await service.list_models(source=source)
    return [
        ModelRegistryResponse(
            id=m.id,
            model_id=m.model_id,
            name=m.name,
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
            download_progress=m.download_progress,
            is_verified=m.is_verified,
            last_verified_at=m.last_verified_at,
            created_at=m.created_at,
            updated_at=m.updated_at,
        )
        for m in models
    ]


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
    )
    return ModelRegistryResponse(
        id=model.id,
        model_id=model.model_id,
        name=model.name,
        source=model.source,
        family=model.family,
        variant=model.variant,
        description=model.description,
        tags=model.tags if isinstance(model.tags, list) else [],
        source_url=model.source_url,
        capabilities=model.capabilities if isinstance(model.capabilities, list) else [],
        status=model.status,
        total_size_bytes=model.total_size_bytes,
        disk_size_bytes=model.disk_size_bytes,
        download_progress=model.download_progress,
        is_verified=model.is_verified,
        last_verified_at=model.last_verified_at,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


@router.post("/{model_id}/download", response_model=JobResponse, status_code=202)
async def download_model(
    model_id: str,
    service: ModelService = Depends(_get_model_service),
) -> JobResponse:
    """Trigger download of a registered model."""
    try:
        job_id = await service.request_download(model_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return JobResponse(job_id=job_id, status="pending", message="Download queued")


@router.delete("/{model_id}", status_code=204)
async def delete_model(
    model_id: str,
    service: ModelService = Depends(_get_model_service),
) -> None:
    """Delete a model from catalog and disk."""
    try:
        await service.delete_model(model_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
