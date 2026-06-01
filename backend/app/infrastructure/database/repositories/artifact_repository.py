"""
Artifact repository — database access for generated artifacts.

Handles CRUD, filtering, pagination, and gallery metadata updates
for all generated output files (images, videos, text).
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import select, update
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

    async def list_filtered(
        self,
        model_name: str | None = None,
        media_type: str | None = None,
        is_favorite: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[ArtifactRecord], int]:
        """List artifacts with filtering and pagination. Returns (items, total)."""
        base_stmt = select(ArtifactRecord)
        if model_name:
            base_stmt = base_stmt.where(ArtifactRecord.model_name == model_name)
        if media_type:
            base_stmt = base_stmt.where(ArtifactRecord.media_type == media_type)
        if is_favorite is not None:
            base_stmt = base_stmt.where(ArtifactRecord.is_favorite == is_favorite)

        # Count total
        from sqlalchemy import func as sa_func

        count_stmt = select(sa_func.count()).select_from(base_stmt.subquery())
        count_result = await self._session.execute(count_stmt)
        total = count_result.scalar_one()

        # Paginated items
        items_stmt = (
            base_stmt
            .order_by(ArtifactRecord.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        items_result = await self._session.execute(items_stmt)
        items = list(items_result.scalars().all())

        return items, total

    async def update(self, artifact_id: UUID, **kwargs: Any) -> None:
        """Update specific fields on an artifact record."""
        stmt = (
            update(ArtifactRecord)
            .where(ArtifactRecord.id == artifact_id)
            .values(**kwargs)
        )
        await self._session.execute(stmt)

    async def delete(self, artifact_id: UUID) -> None:
        """Delete an artifact record."""
        record = await self.get_by_id(artifact_id)
        if record:
            await self._session.delete(record)
