"""Initial schema — models, model_files, jobs, job_events, artifacts.

Revision ID: 001_initial_schema
Revises: -
Create Date: 2026-06-02
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


# revision identifiers
revision = "001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create initial database tables."""
    # --- models ---
    op.create_table(
        "models",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("model_id", sa.String(512), unique=True, nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column("family", sa.String(50), nullable=False, server_default="custom"),
        sa.Column("variant", sa.String(100), server_default=""),
        sa.Column("description", sa.Text, server_default=""),
        sa.Column("tags", postgresql.JSONB, server_default="[]"),
        sa.Column("source_url", sa.String(1024), server_default=""),
        sa.Column("version", sa.String(100), server_default=""),
        sa.Column("capabilities", postgresql.JSONB, server_default="[]"),
        sa.Column("total_size_bytes", sa.BigInteger, server_default="0"),
        sa.Column("disk_size_bytes", sa.BigInteger, server_default="0"),
        sa.Column("file_path", sa.String(1024), server_default=""),
        sa.Column("status", sa.String(50), server_default="not_downloaded"),
        sa.Column("download_progress", sa.Integer, server_default="0"),
        sa.Column("checksum", sa.String(128), server_default=""),
        sa.Column("is_verified", sa.Boolean, server_default="false"),
        sa.Column("last_verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_models_source_status", "models", ["source", "status"])
    op.create_index("ix_models_family", "models", ["family"])
    op.create_index("ix_models_status", "models", ["status"])

    # --- model_files ---
    op.create_table(
        "model_files",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "model_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("models.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("relative_path", sa.String(1024), nullable=False),
        sa.Column("size_bytes", sa.BigInteger, server_default="0"),
        sa.Column("downloaded_bytes", sa.BigInteger, server_default="0"),
        sa.Column("status", sa.String(50), server_default="pending"),
        sa.Column("sha256", sa.String(64), server_default=""),
        sa.Column("is_verified", sa.Boolean, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("model_id", "relative_path", name="uq_model_files_path"),
    )
    op.create_index("ix_model_files_model_id", "model_files", ["model_id"])
    op.create_index("ix_model_files_status", "model_files", ["status"])

    # --- jobs ---
    op.create_table(
        "jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("job_type", sa.String(50), nullable=False),
        sa.Column("status", sa.String(50), nullable=False, server_default="pending"),
        sa.Column("progress_percent", sa.Integer, server_default="0"),
        sa.Column("current_step", sa.Integer, server_default="0"),
        sa.Column("total_steps", sa.Integer, server_default="0"),
        sa.Column("message", sa.Text, server_default=""),
        sa.Column("priority", sa.Integer, server_default="0"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("timeout_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("params", postgresql.JSONB, server_default="{}"),
        sa.Column("result", postgresql.JSONB, server_default="{}"),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column(
            "model_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("models.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_jobs_status", "jobs", ["status"])
    op.create_index("ix_jobs_job_type", "jobs", ["job_type"])
    op.create_index("ix_jobs_created_at", "jobs", ["created_at"])

    # --- job_events ---
    op.create_table(
        "job_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("payload", postgresql.JSONB, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_job_events_job_id", "job_events", ["job_id"])

    # --- artifacts ---
    op.create_table(
        "artifacts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("jobs.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("file_path", sa.String(1024), nullable=False),
        sa.Column("file_type", sa.String(50), server_default="image"),
        sa.Column("file_size_bytes", sa.BigInteger, server_default="0"),
        sa.Column("mime_type", sa.String(100), server_default=""),
        sa.Column("prompt", sa.Text, nullable=True),
        sa.Column("negative_prompt", sa.Text, nullable=True),
        sa.Column("model_id_ref", sa.String(512), nullable=True),
        sa.Column("width", sa.Integer, nullable=True),
        sa.Column("height", sa.Integer, nullable=True),
        sa.Column("seed", sa.Integer, nullable=True),
        sa.Column("generation_params", postgresql.JSONB, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_artifacts_job_id", "artifacts", ["job_id"])
    op.create_index("ix_artifacts_created_at", "artifacts", ["created_at"])


def downgrade() -> None:
    """Drop all tables created in this migration."""
    op.drop_table("artifacts")
    op.drop_table("job_events")
    op.drop_table("jobs")
    op.drop_table("model_files")
    op.drop_table("models")
