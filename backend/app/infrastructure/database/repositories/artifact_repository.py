"""
Artifact repository — database access for generated artifacts.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.database.models import ArtifactRecord


class ArtifactRepository:
    """Data access layer for artifact records."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, artifact_id: UUID) -> ArtifactRecord | None:
        """Fetch an artifact by ID."""
        return await self._session.get(ArtifactRecord, artifact_id)

    async def create(self, record: ArtifactRecord) -> ArtifactRecord:
        """Insert a new artifact record."""
        self._session.add(record)
        await self._session.flush()
        return record

    async def list_by_job(self, job_id: UUID) -> list[ArtifactRecord]:
        """List all artifacts produced by a given job."""
        stmt = (
            select(ArtifactRecord)
            .where(ArtifactRecord.job_id == job_id)
            .order_by(ArtifactRecord.created_at)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def list_recent(self, limit: int = 50) -> list[ArtifactRecord]:
        """List most recent artifacts (gallery view)."""
        stmt = (
            select(ArtifactRecord)
            .order_by(ArtifactRecord.created_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def delete(self, artifact_id: UUID) -> None:
        """Delete an artifact record."""
        record = await self.get_by_id(artifact_id)
        if record:
            await self._session.delete(record)
