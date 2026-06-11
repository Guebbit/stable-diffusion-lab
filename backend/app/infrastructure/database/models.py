"""
ORM models — the database table definitions.

Each class here maps 1:1 to a PostgreSQL table. These are infrastructure
concerns — services and API never import from here directly. Repositories
translate between these ORM models and domain value objects.

Tables:
- models: Model catalog (metadata + lifecycle)
- model_files: Per-file download tracking for multi-file models
- jobs: Background job queue (generation, download, load/unload)
- job_events: Immutable audit log of job state transitions
- artifacts: Generated output files (images, videos, text)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import BigInteger, Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy import ForeignKey, Index, UniqueConstraint
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
    preferred_name: Mapped[str] = mapped_column(String(255), default="")
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    family: Mapped[str] = mapped_column(String(50), nullable=False, default="custom")
    variant: Mapped[str] = mapped_column(String(100), default="")

    # --- Metadata ---
    description: Mapped[str] = mapped_column(Text, default="")
    tags: Mapped[list] = mapped_column(JSONB, default=list)
    source_url: Mapped[str] = mapped_column(String(1024), default="")
    version: Mapped[str] = mapped_column(String(100), default="")
    capabilities: Mapped[list] = mapped_column(JSONB, default=list)

    # --- Size and storage ---
    total_size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    disk_size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    file_path: Mapped[str] = mapped_column(String(1024), default="")

    # --- Lifecycle status ---
    status: Mapped[str] = mapped_column(String(50), default="not_downloaded")
    download_progress: Mapped[int] = mapped_column(Integer, default=0)

    # --- Integrity ---
    checksum: Mapped[str] = mapped_column(String(128), default="")
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    last_verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )

    # --- Requirements & compatibility ---
    download_size_bytes: Mapped[int | None] = mapped_column(BigInteger, default=None)
    recommended_vram_min_gb: Mapped[int | None] = mapped_column(Integer, default=None)
    recommended_vram_max_gb: Mapped[int | None] = mapped_column(Integer, default=None)

    # --- Metadata ---
    license: Mapped[str] = mapped_column(String(100), default="")
    base_model: Mapped[str] = mapped_column(String(255), default="")
    precision: Mapped[str] = mapped_column(String(50), default="")
    requirements: Mapped[dict] = mapped_column(JSONB, default=dict)
    notes: Mapped[str] = mapped_column(Text, default="")
    local_path: Mapped[str | None] = mapped_column(String(1024), default=None)
    file_count: Mapped[int] = mapped_column(Integer, default=0)

    # --- Relationships ---
    files: Mapped[list["ModelFileRecord"]] = relationship(
        back_populates="model", lazy="selectin", cascade="all, delete-orphan"
    )
    jobs: Mapped[list["JobRecord"]] = relationship(back_populates="model", lazy="selectin")

    __table_args__ = (
        Index("ix_models_source_status", "source", "status"),
        Index("ix_models_family", "family"),
        Index("ix_models_status", "status"),
    )


class ModelFileRecord(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """
    A tracked file belonging to a model on disk.

    Multi-file models (HuggingFace repos) have one row per weight shard,
    config file, tokenizer, etc. Enables per-file download tracking and
    resume capability after interruption.
    """

    __tablename__ = "model_files"

    # --- Parent model ---
    model_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("models.id", ondelete="CASCADE"), nullable=False
    )
    model: Mapped["ModelRecord"] = relationship(back_populates="files")

    # --- File info ---
    relative_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    downloaded_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    sha256: Mapped[str] = mapped_column(String(64), default="")
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (
        UniqueConstraint("model_id", "relative_path", name="uq_model_files_path"),
        Index("ix_model_files_model_id", "model_id"),
        Index("ix_model_files_status", "status"),
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

    # --- Priority and ordering ---
    priority: Mapped[int] = mapped_column(Integer, default=0)

    # --- Timing ---
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )
    timeout_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )

    # --- Parameters (stored as JSON for flexibility across job types) ---
    params: Mapped[dict] = mapped_column(JSONB, default=dict)
    result: Mapped[dict] = mapped_column(JSONB, default=dict)
    error: Mapped[str] = mapped_column(Text, default="")

    # --- Retry tracking ---
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    max_attempts: Mapped[int] = mapped_column(Integer, default=1)

    # --- Relations ---
    model_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("models.id", ondelete="SET NULL"), nullable=True
    )
    model: Mapped["ModelRecord | None"] = relationship(back_populates="jobs")
    artifacts: Mapped[list["ArtifactRecord"]] = relationship(
        back_populates="job", lazy="selectin", cascade="all, delete-orphan"
    )
    events: Mapped[list["JobEventRecord"]] = relationship(
        back_populates="job", lazy="noload", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_jobs_status", "status"),
        Index("ix_jobs_type_status", "job_type", "status"),
        Index("ix_jobs_created_at", "created_at"),
        Index("ix_jobs_priority_created", "status", "priority", "created_at"),
        Index("ix_jobs_model_id", "model_id"),
    )


class JobEventRecord(Base, UUIDPrimaryKeyMixin):
    """
    Immutable audit entry for job state transitions.

    Append-only table — rows are never updated or deleted (except via CASCADE).
    Provides full observability into job lifecycle without polluting the jobs table.
    """

    __tablename__ = "job_events"

    # --- Parent job ---
    job_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
    )
    job: Mapped["JobRecord"] = relationship(back_populates="events")

    # --- Event data ---
    from_status: Mapped[str] = mapped_column(String(50), nullable=False)
    to_status: Mapped[str] = mapped_column(String(50), nullable=False)
    message: Mapped[str] = mapped_column(Text, default="")
    event_metadata: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)

    # --- Timestamp (immutable — append-only) ---
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default="now()",
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_job_events_job_id", "job_id"),
        Index("ix_job_events_created_at", "created_at"),
    )


class ArtifactRecord(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """
    A generated artifact (image, video, text output).

    Points to a file on disk. Metadata for gallery display and search.
    Denormalized fields enable fast gallery queries without JOINs.
    """

    __tablename__ = "artifacts"

    # --- File info ---
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    thumbnail_path: Mapped[str] = mapped_column(String(1024), default="")
    media_type: Mapped[str] = mapped_column(String(100), default="image/png")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    duration_seconds: Mapped[float | None] = mapped_column(Float, default=None)

    # --- Generation metadata (denormalized for fast gallery queries) ---
    prompt: Mapped[str] = mapped_column(Text, default="")
    negative_prompt: Mapped[str] = mapped_column(Text, default="")
    seed: Mapped[int] = mapped_column(BigInteger, default=0)
    model_name: Mapped[str] = mapped_column(String(255), default="")
    model_id_ref: Mapped[str] = mapped_column(String(512), default="")
    generation_params: Mapped[dict] = mapped_column(JSONB, default=dict)

    # --- Relations ---
    job_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
    )
    job: Mapped["JobRecord"] = relationship(back_populates="artifacts")

    # --- Gallery organization ---
    is_favorite: Mapped[bool] = mapped_column(Boolean, default=False)
    rating: Mapped[int] = mapped_column(Integer, default=0)
    notes: Mapped[str] = mapped_column(Text, default="")

    __table_args__ = (
        Index("ix_artifacts_job_id", "job_id"),
        Index("ix_artifacts_created_at", "created_at"),
        Index("ix_artifacts_model_name", "model_name"),
    )
