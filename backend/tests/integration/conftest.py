import asyncio
from typing import AsyncGenerator

import pytest_asyncio
from sqlalchemy import Text
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.infrastructure.database.models import Base


@pytest_asyncio.fixture
async def async_db() -> AsyncGenerator[AsyncSession, None]:
    """Provide an isolated async database session for integration tests.
    
    Uses SQLite with JSONB -> JSON type mapping for compatibility.
    """
    # Create a copy of metadata with SQLite-compatible types
    from sqlalchemy import MetaData
    from sqlalchemy.schema import CreateTable
    
    sqlite_metadata = MetaData()
    
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    async with engine.begin() as conn:
        # Rename JSONB columns to JSON for SQLite compatibility
        for table in Base.metadata.tables.values():
            for column in table.columns:
                if isinstance(column.type, postgresql.JSONB):
                    column.type = column.type._compiler_dispatch  # type: ignore
                    from sqlalchemy import JSON
                    column.type = JSON()
        
        await conn.run_sync(Base.metadata.create_all)

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()