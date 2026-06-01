"""
Job service — job queue management operations.

Handles job status queries, cancellation requests, retry logic,
and event history access. Does NOT execute jobs — that's the worker's role.
"""

from __future__ import annotations

import logging
from uuid import UUID

from app.domain.enums import JobStatus
from app.domain.errors import (
    InvalidStateTransitionError,
    JobNotFoundError,
    RetryLimitExceededError,
)
from app.infrastructure.database.models import JobEventRecord, JobRecord
from app.infrastructure.database.repositories import JobEventRepository, JobRepository

logger = logging.getLogger(__name__)


class JobService:
    """
    Manages job lifecycle queries and state transitions.

    Operations:
    - list: paginated, filtered job queries
    - get: single job detail
    - cancel: request job cancellation
    - retry: reset failed job to pending
    - events: access job audit trail
    """

    def __init__(
        self,
        job_repository: JobRepository,
        job_event_repository: JobEventRepository,
    ) -> None:
        self._job_repo = job_repository
        self._event_repo = job_event_repository

    async def get_job(self, job_id: UUID) -> JobRecord:
        """Get a single job by ID. Raises if not found."""
        job = await self._job_repo.get_by_id(job_id)
        if not job:
            raise JobNotFoundError(f"Job {job_id} not found")
        return job

    async def list_jobs(
        self,
        status: str | None = None,
        job_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[JobRecord], int]:
        """List jobs with filtering and pagination. Returns (items, total_count)."""
        return await self._job_repo.list_filtered(
            status=status, job_type=job_type, limit=limit, offset=offset
        )

    async def request_cancellation(self, job_id: UUID) -> str:
        """
        Request cancellation of a job.

        - PENDING jobs are immediately cancelled.
        - RUNNING jobs get a cancellation flag (checked by worker between steps).
        - COMPLETED/FAILED/CANCELLED jobs return 409.

        Returns the new status string.
        """
        job = await self._job_repo.get_by_id(job_id)
        if not job:
            raise JobNotFoundError(f"Job {job_id} not found")

        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            raise InvalidStateTransitionError(
                f"Cannot cancel job in '{job.status}' state"
            )

        if job.status == JobStatus.PENDING:
            # Immediately cancel pending jobs
            await self._job_repo.mark_cancelled(job_id)
            await self._event_repo.record_transition(
                job_id=job_id,
                from_status=JobStatus.PENDING,
                to_status=JobStatus.CANCELLED,
                message="Cancelled before worker pickup",
            )
            logger.info("Cancelled pending job: %s", job_id)
            return JobStatus.CANCELLED

        # RUNNING — set cancellation flag (worker checks between steps)
        # The worker's in-memory set handles this; we record the intent
        await self._event_repo.record_transition(
            job_id=job_id,
            from_status=job.status,
            to_status=job.status,  # Status hasn't changed yet
            message="Cancellation requested, awaiting worker acknowledgment",
        )
        logger.info("Cancellation requested for running job: %s", job_id)
        return job.status

    async def retry_job(self, job_id: UUID) -> JobRecord:
        """
        Retry a failed job by resetting it to PENDING.

        Increments the attempt counter. Raises if:
        - Job doesn't exist
        - Job isn't in FAILED state
        - Attempt counter has reached max_attempts
        """
        job = await self._job_repo.get_by_id(job_id)
        if not job:
            raise JobNotFoundError(f"Job {job_id} not found")

        if job.status != JobStatus.FAILED:
            raise InvalidStateTransitionError(
                f"Only failed jobs can be retried (current: '{job.status}')"
            )

        if job.attempt >= job.max_attempts:
            raise RetryLimitExceededError(
                f"Job has reached max attempts ({job.max_attempts})"
            )

        # Reset to pending with incremented attempt
        await self._job_repo.reset_to_pending(job_id, attempt=job.attempt + 1)
        await self._event_repo.record_transition(
            job_id=job_id,
            from_status=JobStatus.FAILED,
            to_status=JobStatus.PENDING,
            message=f"Retry attempt {job.attempt + 1}/{job.max_attempts}",
            metadata={"attempt": job.attempt + 1, "reason": "manual_retry"},
        )
        logger.info("Retrying job %s (attempt %d)", job_id, job.attempt + 1)

        # Return updated record
        return await self._job_repo.get_by_id(job_id)  # type: ignore[return-value]

    async def get_job_events(self, job_id: UUID) -> list[JobEventRecord]:
        """Get the full audit trail for a job."""
        job = await self._job_repo.get_by_id(job_id)
        if not job:
            raise JobNotFoundError(f"Job {job_id} not found")
        return await self._event_repo.get_by_job(job_id)
