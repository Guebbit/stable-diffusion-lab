"""
Alembic environment configuration for async PostgreSQL migrations.

This file is executed every time Alembic runs a migration command.
It sets up the async engine and imports all SQLAlchemy models so that
autogenerate can detect schema changes.
"""

from __future__ import annotations

import asyncio
import os
import sys
from logging.config import fileConfig

# Ensure the backend directory is on sys.path so that `app` is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Expose alembic/ itself so migration scripts can import seed_helpers
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

from app.infrastructure.config.settings import get_settings
from app.infrastructure.database.base import Base

# Import all models so Alembic can detect them
import app.infrastructure.database.models  # noqa: F401

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (generates SQL without connecting)."""
    settings = get_settings()
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    """Execute migrations against a live connection."""
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations in 'online' mode (connects to the database)."""
    settings = get_settings()
    engine = create_async_engine(settings.database_url)

    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
