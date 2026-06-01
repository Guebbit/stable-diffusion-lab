"""
Job repository — database access for the job queue.

The job orchestrator uses this to claim, update, and query jobs.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

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
            record.started_at = datetime.utcnow()
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
                completed_at=datetime.utcnow(),
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
                completed_at=datetime.utcnow(),
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
                completed_at=datetime.utcnow(),
            )
        )
        await self._session.execute(stmt)

    async def list_recent(self, limit: int = 50) -> list[JobRecord]:
        """List recent jobs ordered by creation date (newest first)."""
        stmt = (
            select(JobRecord)
            .order_by(JobRecord.created_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())
