"""
Jobs router — endpoints for job queue management.

Provides job listing, status checking, cancellation, retry, and
audit trail access. Jobs are the central async work unit in the system.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.common import PaginatedResponse
from app.api.schemas.jobs import (
    JobCancelResponse,
    JobDetailResponse,
    JobEventResponse,
)
from app.domain.errors import JobNotFoundError, InvalidStateTransitionError, RetryLimitExceededError
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
    items = [
        JobDetailResponse(
            id=j.id,
            job_type=j.job_type,
            status=j.status,
            progress_percent=j.progress_percent,
            current_step=j.current_step,
            total_steps=j.total_steps,
            message=j.message,
            priority=j.priority,
            attempt=j.attempt,
            max_attempts=j.max_attempts,
            params=j.params,
            result=j.result,
            error=j.error,
            created_at=j.created_at,
            started_at=j.started_at,
            completed_at=j.completed_at,
            timeout_at=j.timeout_at,
        )
        for j in jobs
    ]
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
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return JobDetailResponse(
        id=job.id,
        job_type=job.job_type,
        status=job.status,
        progress_percent=job.progress_percent,
        current_step=job.current_step,
        total_steps=job.total_steps,
        message=job.message,
        priority=job.priority,
        attempt=job.attempt,
        max_attempts=job.max_attempts,
        params=job.params,
        result=job.result,
        error=job.error,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        timeout_at=job.timeout_at,
    )


@router.post("/{job_id}/cancel", response_model=JobCancelResponse)
async def cancel_job(
    job_id: UUID,
    service: JobService = Depends(_get_job_service),
) -> JobCancelResponse:
    """Request cancellation of a pending or running job."""
    try:
        status = await service.request_cancellation(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except InvalidStateTransitionError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return JobCancelResponse(job_id=job_id, status=status, message="Cancellation requested")


@router.post("/{job_id}/retry", response_model=JobDetailResponse)
async def retry_job(
    job_id: UUID,
    service: JobService = Depends(_get_job_service),
) -> JobDetailResponse:
    """Retry a failed job (resets to pending with incremented attempt)."""
    try:
        job = await service.retry_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (InvalidStateTransitionError, RetryLimitExceededError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return JobDetailResponse(
        id=job.id,
        job_type=job.job_type,
        status=job.status,
        progress_percent=job.progress_percent,
        current_step=job.current_step,
        total_steps=job.total_steps,
        message=job.message,
        priority=job.priority,
        attempt=job.attempt,
        max_attempts=job.max_attempts,
        params=job.params,
        result=job.result,
        error=job.error,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        timeout_at=job.timeout_at,
    )


@router.get("/{job_id}/events", response_model=list[JobEventResponse])
async def get_job_events(
    job_id: UUID,
    service: JobService = Depends(_get_job_service),
) -> list[JobEventResponse]:
    """Get the audit trail (state transitions) for a job."""
    try:
        events = await service.get_job_events(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
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
