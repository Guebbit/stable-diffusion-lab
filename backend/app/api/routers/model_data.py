"""
Model data router — endpoints for managing model files without touching registry entries.

DELETE /models-data/{model_id}  — purge files for a single model (keeps DB record)
DELETE /models-data/            — purge files for all downloaded models (keeps DB records)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.error_handler import from_exception
from app.infrastructure.database.repositories import ModelRepository
from app.infrastructure.database.session import get_async_session
from app.infrastructure.storage.storage_manager import StorageManager
from app.services.model_service import ModelService

router = APIRouter(prefix="/models-data", tags=["models"])


def _get_model_service(session: AsyncSession = Depends(get_async_session)) -> ModelService:
    return ModelService(
        model_repository=ModelRepository(session),
        storage_manager=StorageManager(),
    )


@router.delete("/", status_code=204, response_model=None)
async def purge_all_model_files(
    service: ModelService = Depends(_get_model_service),
) -> None:
    """Delete downloaded files for ALL models and reset their status to not_downloaded."""
    try:
        await service.purge_all_model_files()
    except Exception as exc:
        raise from_exception(exc)


@router.delete("/{model_id:path}", status_code=204, response_model=None)
async def purge_model_files(
    model_id: str,
    service: ModelService = Depends(_get_model_service),
) -> None:
    """Delete downloaded files for a single model and reset its status to not_downloaded."""
    try:
        await service.delete_model_files(model_id)
    except Exception as exc:
        raise from_exception(exc)
