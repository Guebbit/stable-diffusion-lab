import os
from typing import AsyncGenerator

import pytest_asyncio
from sqlalchemy import JSON, Text
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.infrastructure.database.models import Base


def _get_database_url() -> str:
    """Get database URL from environment or default to SQLite for tests."""
    return os.environ.get(
        "DATABASE_URL",
        "sqlite+aiosqlite:///:memory:",
    )


@pytest_asyncio.fixture
async def async_db() -> AsyncGenerator[AsyncSession, None]:
    """Provide an isolated async database session for integration tests.

    Uses SQLite in-memory by default. If DATABASE_URL env var is set to a
    postgresql+asyncpg URL, tests will run against a real PostgreSQL database.
    """
    database_url = _get_database_url()

    is_sqlite = database_url.startswith("sqlite")

    if is_sqlite:
        # SQLite: convert JSONB -> JSON for compatibility
        engine = create_async_engine(
            database_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )

        async with engine.begin() as conn:
            for table in Base.metadata.tables.values():
                for column in table.columns:
                    if isinstance(column.type, postgresql.JSONB):
                        column.type = JSON()
            await conn.run_sync(Base.metadata.create_all)

    else:
        # PostgreSQL: use native types
        engine = create_async_engine(database_url)

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        yield session

    # Tear down
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()