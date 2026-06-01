"""
Model file repository — database access for per-file download tracking.

Each model can have multiple files (weight shards, config, tokenizer).
This repository tracks individual file download state for resume capability.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.database.models import ModelFileRecord


class ModelFileRepository:
    """Data access layer for model file records."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, record: ModelFileRecord) -> ModelFileRecord:
        """Insert a new model file tracking record."""
        self._session.add(record)
        await self._session.flush()
        return record

    async def create_many(self, records: list[ModelFileRecord]) -> list[ModelFileRecord]:
        """Bulk insert model file records."""
        self._session.add_all(records)
        await self._session.flush()
        return records

    async def get_by_model(self, model_id: UUID) -> list[ModelFileRecord]:
        """Get all tracked files for a model."""
        stmt = (
            select(ModelFileRecord)
            .where(ModelFileRecord.model_id == model_id)
            .order_by(ModelFileRecord.relative_path)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_incomplete_files(self, model_id: UUID) -> list[ModelFileRecord]:
        """Get files that haven't been fully downloaded yet."""
        stmt = (
            select(ModelFileRecord)
            .where(
                ModelFileRecord.model_id == model_id,
                ModelFileRecord.status.in_(["pending", "downloading"]),
            )
            .order_by(ModelFileRecord.relative_path)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def update_progress(
        self, file_id: UUID, downloaded_bytes: int, status: str = "downloading"
    ) -> None:
        """Update download progress for a single file."""
        stmt = (
            update(ModelFileRecord)
            .where(ModelFileRecord.id == file_id)
            .values(downloaded_bytes=downloaded_bytes, status=status)
        )
        await self._session.execute(stmt)

    async def mark_complete(self, file_id: UUID) -> None:
        """Mark a file as fully downloaded."""
        stmt = (
            update(ModelFileRecord)
            .where(ModelFileRecord.id == file_id)
            .values(status="complete", is_verified=False)
        )
        await self._session.execute(stmt)

    async def mark_verified(self, file_id: UUID, is_verified: bool) -> None:
        """Update verification status for a file."""
        stmt = (
            update(ModelFileRecord)
            .where(ModelFileRecord.id == file_id)
            .values(is_verified=is_verified)
        )
        await self._session.execute(stmt)

    async def delete_by_model(self, model_id: UUID) -> None:
        """Delete all file records for a model (CASCADE handles this, but explicit is fine)."""
        stmt = select(ModelFileRecord).where(ModelFileRecord.model_id == model_id)
        result = await self._session.execute(stmt)
        for record in result.scalars().all():
            await self._session.delete(record)
