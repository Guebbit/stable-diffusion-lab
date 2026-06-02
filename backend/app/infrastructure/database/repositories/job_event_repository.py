"""
Job event repository — database access for the immutable job audit trail.

Job events are append-only records tracking every state transition.
They provide full observability without polluting the main jobs table.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.database.models import JobEventRecord


class JobEventRepository:
    """Data access layer for job event (audit trail) records."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, record: JobEventRecord) -> JobEventRecord:
        """Append a new event to the audit trail."""
        self._session.add(record)
        await self._session.flush()
        return record

    async def record_transition(
        self,
        job_id: UUID,
        from_status: str,
        to_status: str,
        message: str = "",
        metadata: dict | None = None,
    ) -> JobEventRecord:
        """Record a job state transition as an immutable event."""
        event = JobEventRecord(
            job_id=job_id,
            from_status=from_status,
            to_status=to_status,
            message=message,
            event_metadata=metadata or {},
        )
        self._session.add(event)
        await self._session.flush()
        return event

    async def get_by_job(self, job_id: UUID) -> list[JobEventRecord]:
        """Get all events for a job, ordered chronologically."""
        stmt = (
            select(JobEventRecord)
            .where(JobEventRecord.job_id == job_id)
            .order_by(JobEventRecord.created_at.asc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_recent(self, limit: int = 100) -> list[JobEventRecord]:
        """Get most recent events across all jobs."""
        stmt = (
            select(JobEventRecord)
            .order_by(JobEventRecord.created_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())
