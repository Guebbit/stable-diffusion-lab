"""
Jobs router — endpoints for job queue management.

Provides job listing, status checking, cancellation, retry, and
audit trail access. Jobs are the central async work unit in the system.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.error_handler import from_exception
from app.api.schemas.common import PaginatedResponse
from app.api.schemas.jobs import (
    JobCancelResponse,
    JobDetailResponse,
    JobEventResponse,
)
from app.api.schemas.system import JobTimelineResponse, TypedEventResponse
from app.domain.enums import JobStatus
from app.infrastructure.database.repositories import JobEventRepository, JobRepository
from app.infrastructure.database.session import get_async_session
from app.services.job_service import JobService

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _get_job_service(session: AsyncSession = Depends(get_async_session)) -> JobService:
    """Dependency injection for JobService."""
    return JobService(
        job_repository=JobRepository(session),
        job_event_repository=JobEventRepository(session),
    )


def _job_to_response(j) -> JobDetailResponse:
    """Convert a job domain/model object to JobDetailResponse."""
    return JobDetailResponse(
        id=j.id,
        job_type=j.job_type,
        status=j.status,
        progress_percent=j.progress_percent,
        current_step=j.current_step,
        total_steps=j.total_steps,
        priority=j.priority,
        attempt=j.attempt,
        max_attempts=j.max_attempts,
        params=j.params,
        error=j.error,
        created_at=j.created_at,
        started_at=j.started_at,
        completed_at=j.completed_at,
    )


@router.get("/", response_model=PaginatedResponse[JobDetailResponse])
async def list_jobs(
    status: str | None = Query(None, description="Filter by status"),
    job_type: str | None = Query(None, description="Filter by job type"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    service: JobService = Depends(_get_job_service),
) -> PaginatedResponse[JobDetailResponse]:
    """List jobs with optional filtering and pagination."""
    jobs, total = await service.list_jobs(
        status=status, job_type=job_type, limit=limit, offset=offset
    )
    items = [_job_to_response(j) for j in jobs]
    return PaginatedResponse(
        items=items, total=total, limit=limit, offset=offset, has_more=(offset + limit < total)
    )


@router.get("/{job_id}", response_model=JobDetailResponse)
async def get_job(
    job_id: UUID,
    service: JobService = Depends(_get_job_service),
) -> JobDetailResponse:
    """Get detailed status of a specific job."""
    try:
        job = await service.get_job(job_id)
    except Exception as exc:
        raise from_exception(exc)
    return _job_to_response(job)


@router.post("/{job_id}/cancel", response_model=JobCancelResponse)
async def cancel_job(
    job_id: UUID,
    request: Request,
    service: JobService = Depends(_get_job_service),
) -> JobCancelResponse:
    """Request cancellation of a pending or running job."""
    try:
        status = await service.request_cancellation(job_id)
    except Exception as exc:
        raise from_exception(exc)
    # Notify the worker so it fires the cancel_event mid-inference
    worker = getattr(request.app.state, "worker", None)
    if worker is not None:
        worker.request_cancellation(job_id)
    return JobCancelResponse(job_id=job_id, status=status, message="Cancellation requested")


@router.post("/{job_id}/retry", response_model=JobDetailResponse)
async def retry_job(
    job_id: UUID,
    service: JobService = Depends(_get_job_service),
) -> JobDetailResponse:
    """Retry a failed job (resets to pending with incremented attempt)."""
    try:
        job = await service.retry_job(job_id)
    except Exception as exc:
        raise from_exception(exc)
    return _job_to_response(job)


@router.get("/{job_id}/events", response_model=list[JobEventResponse])
async def get_job_events(
    job_id: UUID,
    service: JobService = Depends(_get_job_service),
) -> list[JobEventResponse]:
    """Get the audit trail (state transitions) for a job."""
    try:
        events = await service.get_job_events(job_id)
    except Exception as exc:
        raise from_exception(exc)
    return [
        JobEventResponse(
            id=e.id,
            job_id=e.job_id,
            from_status=e.from_status,
            to_status=e.to_status,
            message=e.message,
            metadata=e.event_metadata,
            created_at=e.created_at,
        )
        for e in events
    ]


@router.delete("/", status_code=204, response_model=None)
async def delete_all_finished_jobs(
    session: AsyncSession = Depends(get_async_session),
) -> None:
    """Delete all completed, failed, and cancelled job records.
    Associated artifacts are preserved — their job_id is set to NULL."""
    job_repo = JobRepository(session)
    for job in await job_repo.list_finished():
        await session.delete(job)


@router.delete("/{job_id}", status_code=204, response_model=None)
async def delete_job(
    job_id: UUID,
    session: AsyncSession = Depends(get_async_session),
) -> None:
    """Delete a single finished job record. Active jobs must be cancelled first.
    Associated artifacts are preserved — their job_id is set to NULL."""
    job_repo = JobRepository(session)
    job = await job_repo.get_by_id(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job.status in (JobStatus.PENDING, JobStatus.RUNNING):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot delete a job in '{job.status}' state — cancel it first",
        )
    await session.delete(job)


@router.get("/{job_id}/timeline", response_model=JobTimelineResponse)
async def get_job_timeline(
    job_id: UUID,
    request: Request,
    limit: int = Query(500, ge=1, le=2000),
) -> JobTimelineResponse:
    """Get typed observability timeline for a specific job."""
    observability = getattr(request.app.state, "observability", None)
    if observability is None:
        raise HTTPException(status_code=503, detail="Observability service not initialized")
    events = observability.get_job_timeline(str(job_id), limit=limit)
    return JobTimelineResponse(
        job_id=str(job_id),
        events=[TypedEventResponse.model_validate(event) for event in events],
    )