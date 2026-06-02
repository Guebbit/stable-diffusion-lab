"""System router — observability status, events, health, and metrics."""

from __future__ import annotations

import uuid

import torch
from fastapi import APIRouter, Header, HTTPException, Query, Request, Response

from app.api.schemas import (
    HealthResponse,
    JobTimelineResponse,
    MetricsSnapshot,
    SystemStateSnapshot,
    SystemStatusResponse,
    TypedEventResponse,
)
from app.orchestrator.observability import ObservabilityService

router = APIRouter(prefix="/system", tags=["system"])


def _get_observability_service(request: Request) -> ObservabilityService:
    service = getattr(request.app.state, "observability", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Observability service not initialized")
    return service


def _resolve_correlation_id(correlation_id: str | None) -> str:
    """Reuse incoming correlation id or create a UUIDv4 for full request/event trace continuity."""
    return correlation_id or str(uuid.uuid4())


@router.get("/status", response_model=SystemStateSnapshot)
async def get_system_status(
    request: Request,
    response: Response,
    correlation_id: str | None = Header(default=None, alias="X-Correlation-ID"),
) -> SystemStateSnapshot:
    """Return comprehensive live system state snapshot."""
    resolved_id = _resolve_correlation_id(correlation_id)
    observability = _get_observability_service(request)
    snapshot = await observability.get_system_snapshot(correlation_id=resolved_id)
    response.headers["X-Correlation-ID"] = resolved_id
    return SystemStateSnapshot.model_validate(snapshot.to_dict())


@router.get("/status/legacy", response_model=SystemStatusResponse)
async def get_system_status_legacy(
    request: Request,
    response: Response,
    correlation_id: str | None = Header(default=None, alias="X-Correlation-ID"),
) -> SystemStatusResponse:
    """Backward-compatible status payload with legacy fields."""
    resolved_id = _resolve_correlation_id(correlation_id)
    observability = _get_observability_service(request)
    snapshot = await observability.get_system_snapshot(correlation_id=resolved_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    response.headers["X-Correlation-ID"] = resolved_id
    return SystemStatusResponse(
        status=snapshot.health.status,
        version=snapshot.runtime.version,
        device=device,
        gpu_busy=snapshot.gpu.busy,
        loaded_models=snapshot.models.loaded_models,
        pending_jobs=snapshot.jobs.pending,
        correlation_id=resolved_id,
    )


@router.get("/events", response_model=list[TypedEventResponse])
async def get_system_events(
    request: Request,
    limit: int = Query(200, ge=1, le=1000),
) -> list[TypedEventResponse]:
    """Return recent typed observability events from memory buffer."""
    observability = _get_observability_service(request)
    return [
        TypedEventResponse.model_validate(item) for item in observability.get_recent_events(limit)
    ]


@router.get("/health", response_model=HealthResponse)
async def get_system_health(
    request: Request,
    response: Response,
    correlation_id: str | None = Header(default=None, alias="X-Correlation-ID"),
) -> HealthResponse:
    """Return semantic health state (warnings + blockers)."""
    resolved_id = _resolve_correlation_id(correlation_id)
    observability = _get_observability_service(request)
    snapshot = await observability.get_system_snapshot(correlation_id=resolved_id)
    response.headers["X-Correlation-ID"] = resolved_id
    return HealthResponse(
        status=snapshot.health.status,
        warnings=snapshot.health.warnings,
        blockers=snapshot.health.blockers,
        recommendations=snapshot.health.recommendations,
        correlation_id=resolved_id,
    )


@router.get("/metrics", response_model=MetricsSnapshot)
async def get_system_metrics(request: Request) -> MetricsSnapshot:
    """Return in-memory counters, gauges, and histogram percentiles."""
    observability = _get_observability_service(request)
    return MetricsSnapshot.model_validate(observability.get_metrics_snapshot())


@router.get("/jobs/{job_id}/timeline", response_model=JobTimelineResponse)
async def get_job_timeline(
    request: Request,
    job_id: str,
    limit: int = Query(500, ge=1, le=2000),
) -> JobTimelineResponse:
    """Return timeline events for one job id."""
    observability = _get_observability_service(request)
    events = observability.get_job_timeline(job_id=job_id, limit=limit)
    return JobTimelineResponse(
        job_id=job_id,
        events=[TypedEventResponse.model_validate(event) for event in events],
    )
