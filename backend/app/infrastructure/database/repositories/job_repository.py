"""
Job repository — database access for the job queue.

The job orchestrator uses this to claim, update, and query jobs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import func as sa_func
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.database.models import JobRecord


class JobRepository:
    """Data access layer for job records."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, job_id: UUID) -> JobRecord | None:
        """Fetch a job by ID."""
        return await self._session.get(JobRecord, job_id)

    async def create(self, record: JobRecord) -> JobRecord:
        """Insert a new job into the queue."""
        self._session.add(record)
        await self._session.flush()
        return record

    async def claim_next_pending(self, job_type: str | None = None) -> JobRecord | None:
        """
        Atomically claim the oldest PENDING job by setting it to RUNNING.

        Uses SELECT ... FOR UPDATE SKIP LOCKED to prevent race conditions
        if multiple workers ever exist.
        """
        stmt = (
            select(JobRecord)
            .where(JobRecord.status == "pending")
            .order_by(JobRecord.created_at)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        if job_type:
            stmt = stmt.where(JobRecord.job_type == job_type)

        result = await self._session.execute(stmt)
        record = result.scalar_one_or_none()

        if record:
            record.status = "running"
            record.started_at = datetime.now(timezone.utc)
            await self._session.flush()

        return record

    async def update_progress(
        self,
        job_id: UUID,
        progress_percent: int,
        current_step: int = 0,
        total_steps: int = 0,
        message: str = "",
    ) -> None:
        """Update job progress fields."""
        stmt = (
            update(JobRecord)
            .where(JobRecord.id == job_id)
            .values(
                progress_percent=progress_percent,
                current_step=current_step,
                total_steps=total_steps,
                message=message,
            )
        )
        await self._session.execute(stmt)

    async def mark_completed(self, job_id: UUID, result: dict | None = None) -> None:
        """Mark a job as successfully completed."""
        stmt = (
            update(JobRecord)
            .where(JobRecord.id == job_id)
            .values(
                status="completed",
                progress_percent=100,
                completed_at=datetime.now(timezone.utc),
                result=result or {},
            )
        )
        await self._session.execute(stmt)

    async def mark_failed(self, job_id: UUID, error: str) -> None:
        """Mark a job as failed with an error message."""
        stmt = (
            update(JobRecord)
            .where(JobRecord.id == job_id)
            .values(
                status="failed",
                completed_at=datetime.now(timezone.utc),
                error=error,
            )
        )
        await self._session.execute(stmt)

    async def mark_cancelled(self, job_id: UUID) -> None:
        """Mark a job as cancelled."""
        stmt = (
            update(JobRecord)
            .where(JobRecord.id == job_id)
            .values(
                status="cancelled",
                completed_at=datetime.now(timezone.utc),
            )
        )
        await self._session.execute(stmt)

    async def list_recent(self, limit: int = 50) -> list[JobRecord]:
        """List recent jobs ordered by creation date (newest first)."""
        stmt = select(JobRecord).order_by(JobRecord.created_at.desc()).limit(limit)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def list_filtered(
        self,
        status: str | None = None,
        job_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[JobRecord], int]:
        """List jobs with optional filtering and pagination. Returns (items, total)."""
        base_stmt = select(JobRecord)
        if status:
            base_stmt = base_stmt.where(JobRecord.status == status)
        if job_type:
            base_stmt = base_stmt.where(JobRecord.job_type == job_type)

        # Count total
        count_stmt = select(sa_func.count()).select_from(base_stmt.subquery())
        count_result = await self._session.execute(count_stmt)
        total = count_result.scalar_one()

        # Paginated items
        items_stmt = base_stmt.order_by(JobRecord.created_at.desc()).limit(limit).offset(offset)
        items_result = await self._session.execute(items_stmt)
        items = list(items_result.scalars().all())

        return items, total

    async def reset_to_pending(self, job_id: UUID, attempt: int = 1) -> None:
        """Reset a failed job back to pending state for retry."""
        stmt = (
            update(JobRecord)
            .where(JobRecord.id == job_id)
            .values(
                status="pending",
                progress_percent=0,
                current_step=0,
                total_steps=0,
                message="",
                error="",
                attempt=attempt,
                started_at=None,
                completed_at=None,
            )
        )
        await self._session.execute(stmt)

    async def get_queue_counts(self) -> dict[str, int]:
        """Return queue/job counts grouped by status and total depth."""
        status_stmt = select(JobRecord.status, sa_func.count()).group_by(JobRecord.status)
        status_result = await self._session.execute(status_stmt)
        counts = {str(status): int(count) for status, count in status_result.all()}
        counts.setdefault("pending", 0)
        counts.setdefault("running", 0)
        counts.setdefault("completed", 0)
        counts.setdefault("failed", 0)
        counts.setdefault("cancelled", 0)
        counts["queue_depth"] = counts["pending"] + counts["running"]
        return counts

    async def get_oldest_pending_age_seconds(self) -> float | None:
        """Return age of the oldest pending job in seconds, if any exist."""
        stmt = select(sa_func.min(JobRecord.created_at)).where(JobRecord.status == "pending")
        result = await self._session.execute(stmt)
        oldest = result.scalar_one_or_none()
        if oldest is None:
            return None
        now = datetime.now(timezone.utc)
        return max((now - oldest).total_seconds(), 0.0)

    async def get_current_running_job_id(self) -> str | None:
        """Return the oldest currently running job id, if present."""
        stmt = (
            select(JobRecord.id)
            .where(JobRecord.status == "running")
            .order_by(JobRecord.started_at.asc().nulls_last(), JobRecord.created_at.asc())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        running = result.scalar_one_or_none()
        return str(running) if running else None
