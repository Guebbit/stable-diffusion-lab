"""
SQLAlchemy async engine and session factory.

Provides the database connection machinery. All repository classes receive
a session from the dependency injection layer — they never create engines
or sessions themselves.

Usage:
    from app.infrastructure.database.session import get_async_session

    async for session in get_async_session():
        # use session within this scope
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.infrastructure.config.settings import get_settings

# --- Engine (one per process, reused across all requests) ---
_engine = create_async_engine(
    get_settings().database_url,
    echo=get_settings().debug,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

# --- Session factory ---
_async_session_factory = async_sessionmaker(
    bind=_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """
    Return the session factory for use by background workers.

    Workers create their own sessions per unit of work rather than
    relying on FastAPI's dependency injection.
    """
    return _async_session_factory


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Yield an async database session for a single unit of work.

    Used as a FastAPI dependency — each request gets its own session
    that is committed/rolled back and closed automatically.
    """
    async with _async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
