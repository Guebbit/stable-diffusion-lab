"""
SQLAlchemy declarative base and common column mixins.

All ORM models inherit from Base. Common columns (id, timestamps) are
provided as mixins so every table gets them automatically without repetition.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """
    Declarative base for all ORM models.

    Convention: table names are plural snake_case (e.g., "models", "jobs").
    """

    pass


class TimestampMixin:
    """
    Mixin that adds created_at and updated_at columns to any model.

    created_at: set once at INSERT time (server-side default).
    updated_at: refreshed on every UPDATE (server-side onupdate).
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class UUIDPrimaryKeyMixin:
    """
    Mixin that provides a UUID primary key column named 'id'.

    Uses PostgreSQL's native UUID type for efficient storage and indexing.
    Generated client-side (uuid4) so we have the ID before INSERT.
    """

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
