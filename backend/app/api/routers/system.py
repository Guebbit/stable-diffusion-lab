"""
System health and status API routes.

GET /api/system/health – health check
GET /api/system/status – detailed status
"""

from __future__ import annotations

import shutil

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.system import HealthResponse, StatusResponse
from app.infrastructure.config.settings import get_settings
from app.infrastructure.database.session import get_async_session
from app.infrastructure.database.repositories.model_repository import ModelRepository

router = APIRouter(prefix="/system", tags=["system"])


def _get_model_repo(
    session: AsyncSession = Depends(get_async_session),
) -> ModelRepository:
    """Dependency injection for ModelRepository."""
    return ModelRepository(session)


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy")


@router.get("/status", response_model=StatusResponse)
async def system_status(
    model_repo: ModelRepository = Depends(_get_model_repo),
):
    settings = get_settings()

    models = await model_repo.list_all()
    total_models = len(models)
    by_family: dict[str, int] = {}
    models_disk_bytes = 0
    for m in models:
        by_family[m.family] = by_family.get(m.family, 0) + 1
        if m.status == "downloaded":
            models_disk_bytes += m.total_size_bytes or 0

    disk_free_bytes = 0
    storage_root = settings.storage_root
    if storage_root.exists():
        disk_free_bytes = shutil.disk_usage(str(storage_root)).free

    device = settings.inference_device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    return StatusResponse(
        status="ok",
        device=device,
        total_models=total_models,
        models_by_family=by_family,
        models_disk_bytes=models_disk_bytes,
        disk_free_bytes=disk_free_bytes,
    )
