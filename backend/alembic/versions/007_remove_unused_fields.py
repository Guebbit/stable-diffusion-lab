"""Remove unused DB fields: jobs.message, jobs.result, jobs.timeout_at, artifacts.duration_seconds.

Revision ID: 007_remove_unused_fields
Revises: 006_seed_tools
Create Date: 2026-06-17
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers
revision = "007_remove_unused_fields"
down_revision = "006_seed_tools"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("jobs", "message")
    op.drop_column("jobs", "result")
    op.drop_column("jobs", "timeout_at")
    op.drop_column("artifacts", "duration_seconds")


def downgrade() -> None:
    op.add_column("jobs", sa.Column("message", sa.Text(), nullable=False, server_default=""))
    op.add_column("jobs", sa.Column("result", JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"))
    op.add_column("jobs", sa.Column("timeout_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("artifacts", sa.Column("duration_seconds", sa.Float(), nullable=True))
