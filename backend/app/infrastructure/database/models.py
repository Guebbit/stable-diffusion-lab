"""
ORM models — the database table definitions.

Each class here maps 1:1 to a PostgreSQL table. These are infrastructure
concerns — services and API never import from here directly. Repositories
translate between these ORM models and domain value objects.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Float, Integer, String, Text, Boolean, ForeignKey, Index
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.infrastructure.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class ModelRecord(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """
    A registered model in the catalog.

    Tracks metadata, download status, and filesystem location.
    One row per model version (a model can have multiple versions).
    """

    __tablename__ = "models"

    # --- Identity ---
    model_id: Mapped[str] = mapped_column(String(512), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    family: Mapped[str] = mapped_column(String(50), nullable=False, default="custom")

    # --- Metadata ---
    description: Mapped[str] = mapped_column(Text, default="")
    tags: Mapped[list] = mapped_column(JSONB, default=list)
    source_url: Mapped[str] = mapped_column(String(1024), default="")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)

    # --- Status ---
    status: Mapped[str] = mapped_column(String(50), default="not_downloaded")
    download_progress: Mapped[int] = mapped_column(Integer, default=0)
    file_path: Mapped[str] = mapped_column(String(1024), default="")
    checksum: Mapped[str] = mapped_column(String(128), default="")
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)

    # --- Relationships ---
    jobs: Mapped[list["JobRecord"]] = relationship(back_populates="model", lazy="selectin")

    __table_args__ = (
        Index("ix_models_source_status", "source", "status"),
        Index("ix_models_family", "family"),
    )


class JobRecord(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """
    A background job (generation, download, model load, etc.).

    The job orchestrator reads/writes these rows to manage the work queue.
    State machine: PENDING → RUNNING → COMPLETED | FAILED | CANCELLED
    """

    __tablename__ = "jobs"

    # --- Type and status ---
    job_type: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="pending", nullable=False)

    # --- Progress ---
    progress_percent: Mapped[int] = mapped_column(Integer, default=0)
    current_step: Mapped[int] = mapped_column(Integer, default=0)
    total_steps: Mapped[int] = mapped_column(Integer, default=0)
    message: Mapped[str] = mapped_column(Text, default="")

    # --- Timing ---
    started_at: Mapped[datetime | None] = mapped_column(default=None)
    completed_at: Mapped[datetime | None] = mapped_column(default=None)

    # --- Parameters (stored as JSON for flexibility across job types) ---
    params: Mapped[dict] = mapped_column(JSONB, default=dict)
    result: Mapped[dict] = mapped_column(JSONB, default=dict)
    error: Mapped[str] = mapped_column(Text, default="")

    # --- Retry tracking ---
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    max_attempts: Mapped[int] = mapped_column(Integer, default=1)

    # --- Relations ---
    model_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("models.id"), nullable=True
    )
    model: Mapped["ModelRecord | None"] = relationship(back_populates="jobs")
    artifacts: Mapped[list["ArtifactRecord"]] = relationship(back_populates="job", lazy="selectin")

    __table_args__ = (
        Index("ix_jobs_status", "status"),
        Index("ix_jobs_type_status", "job_type", "status"),
        Index("ix_jobs_created_at", "created_at"),
    )


class ArtifactRecord(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """
    A generated artifact (image, video, text output).

    Points to a file on disk. Metadata for gallery display and search.
    """

    __tablename__ = "artifacts"

    # --- File info ---
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    thumbnail_path: Mapped[str] = mapped_column(String(1024), default="")
    media_type: Mapped[str] = mapped_column(String(100), default="image/png")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)

    # --- Generation metadata (denormalized for fast gallery queries) ---
    prompt: Mapped[str] = mapped_column(Text, default="")
    negative_prompt: Mapped[str] = mapped_column(Text, default="")
    seed: Mapped[int] = mapped_column(Integer, default=0)
    model_name: Mapped[str] = mapped_column(String(255), default="")
    generation_params: Mapped[dict] = mapped_column(JSONB, default=dict)

    # --- Relations ---
    job_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("jobs.id"), nullable=False
    )
    job: Mapped["JobRecord"] = relationship(back_populates="artifacts")

    __table_args__ = (
        Index("ix_artifacts_job_id", "job_id"),
        Index("ix_artifacts_created_at", "created_at"),
    )
