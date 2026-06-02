"""Add attempt and max_attempts columns to jobs table.

Revision ID: 002_add_job_attempts
Revises: 001_initial_schema
Create Date: 2026-06-02
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


# revision identifiers
revision = "002_add_job_attempts"
down_revision = "001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add attempt and max_attempts columns to jobs."""
    op.add_column("jobs", sa.Column("attempt", sa.Integer, server_default="1", nullable=False))
    op.add_column("jobs", sa.Column("max_attempts", sa.Integer, server_default="1", nullable=False))


def downgrade() -> None:
    """Remove attempt and max_attempts columns from jobs."""
    op.drop_column("jobs", "max_attempts")
    op.drop_column("jobs", "attempt")
