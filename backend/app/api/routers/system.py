"""
System health and status API routes.

GET /api/system/health – health check
GET /api/system/status – detailed status
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.system import HealthResponse, StatusResponse
from app.infrastructure.database.session import get_async_session
from app.infrastructure.database.repositories.model_repository import ModelRepository

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy")


@router.get("/status", response_model=StatusResponse)
async def system_status(
    session: AsyncSession = Depends(get_async_session),
):
    model_repo = ModelRepository(session)

    models = await model_repo.list_all()
    total_models = len(models)
    by_family: dict[str, int] = {}
    for m in models:
        by_family[m.family] = by_family.get(m.family, 0) + 1

    return StatusResponse(
        total_models=total_models,
        models_by_family=by_family,
    )
