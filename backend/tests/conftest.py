"""
Shared test fixtures and configuration.

Provides database sessions, test clients, and mock adapters
used across all test modules.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient


if "torch" not in sys.modules:
    torch_stub = ModuleType("torch")
    torch_stub.cuda = SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    sys.modules["torch"] = torch_stub

from app.main import create_app


@pytest.fixture
def app():
    """Create a fresh app instance for testing."""
    return create_app()


@pytest.fixture
async def client(app):
    """Async HTTP test client."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac
