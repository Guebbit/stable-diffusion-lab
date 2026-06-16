"""
Model repository — database access for the model registry.

Encapsulates all SQL queries related to models. Services call these
methods instead of writing ORM queries directly.
"""

from __future__ import annotations

from uuid import UUID

import json

from sqlalchemy import cast, select, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.database.models import ModelRecord


class ModelRepository:
    """Data access layer for model records."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, record_id: UUID) -> ModelRecord | None:
        """Fetch a model by its internal UUID."""
        return await self._session.get(ModelRecord, record_id)

    async def get_by_model_id(self, model_id: str) -> ModelRecord | None:
        """Fetch a model by its external model_id (e.g., HuggingFace repo slug)."""
        stmt = select(ModelRecord).where(ModelRecord.model_id == model_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_all(
        self,
        source: str | None = None,
        capabilities: list[str] | None = None,
        model_type: str | None = None,
        compatible_base: str | None = None,
    ) -> list[ModelRecord]:
        """List all registered models with optional filters.

        compatible_base: return only models whose compatible_bases array contains
        this value (e.g. 'sdxl', 'sd1.5'). Multiple capabilities use OR logic.
        """
        stmt = select(ModelRecord).order_by(ModelRecord.name)
        if source:
            stmt = stmt.where(ModelRecord.source == source)
        if model_type:
            stmt = stmt.where(ModelRecord.model_type == model_type)
        if compatible_base:
            stmt = stmt.where(
                ModelRecord.compatible_bases.op("@>")(
                    cast(json.dumps([compatible_base]), JSONB)
                )
            )
        if capabilities:
            from sqlalchemy import or_
            stmt = stmt.where(
                or_(*[ModelRecord.capabilities.contains([c]) for c in capabilities])
            )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def create(self, record: ModelRecord) -> ModelRecord:
        """Insert a new model record."""
        self._session.add(record)
        await self._session.flush()
        return record

    async def update_status(self, model_id: str, status: str, **kwargs: object) -> None:
        """Update a model's status and optional extra fields atomically."""
        values: dict[str, object] = {"status": status, **kwargs}
        stmt = (
            update(ModelRecord)
            .where(ModelRecord.model_id == model_id)
            .values(**values)
        )
        await self._session.execute(stmt)

    async def delete(self, record_id: UUID) -> None:
        """Delete a model record by internal UUID."""
        record = await self.get_by_id(record_id)
        if record:
            await self._session.delete(record)
