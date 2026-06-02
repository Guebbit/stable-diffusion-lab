"""
System router — health check, status, and diagnostics.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import MetricsSnapshotResponse, SystemEventResponse, SystemStatusResponse
from app.infrastructure.database.session import get_async_session
from app.services.observability_service import ObservabilityService

router = APIRouter(prefix="/system", tags=["system"])


def _get_observability_service(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
) -> ObservabilityService:
    return ObservabilityService(app=request.app, session=session)


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status(
    service: ObservabilityService = Depends(_get_observability_service),
) -> SystemStatusResponse:
    """
    Return current system status.

    Used by the frontend health panel to show connection state,
    GPU availability, and active model info.
    """
    return await service.get_system_status()


@router.get("/events", response_model=list[SystemEventResponse])
async def get_recent_system_events(
    limit: int = Query(100, ge=1, le=500),
    service: ObservabilityService = Depends(_get_observability_service),
) -> list[SystemEventResponse]:
    return service.get_recent_events(limit=limit)


@router.get("/jobs/{job_id}/timeline", response_model=list[SystemEventResponse])
async def get_job_timeline(
    job_id: str,
    limit: int = Query(200, ge=1, le=500),
    service: ObservabilityService = Depends(_get_observability_service),
) -> list[SystemEventResponse]:
    return service.get_job_timeline(job_id=job_id, limit=limit)


@router.get("/metrics", response_model=MetricsSnapshotResponse)
async def get_metrics(
    service: ObservabilityService = Depends(_get_observability_service),
) -> MetricsSnapshotResponse:
    return service.get_metrics()
