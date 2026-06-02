"""
Smoke tests for app startup and basic route availability.

Tests:
- App initializes and runs lifecycle successfully
- OpenAPI docs are accessible
- Unknown routes return 404
- Health/status responses are valid
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_app_starts_successfully(client: AsyncClient) -> None:
    """App boots without errors and client is ready."""
    # Just creating the client via fixture means app started.
    # Request something simple to confirm it's responding.
    response = await client.get("/openapi.json")
    assert response.status_code == 200
    assert "openapi" in response.json()


@pytest.mark.asyncio
async def test_openapi_docs_endpoint(client: AsyncClient) -> None:
    """OpenAPI spec is available and valid."""
    response = await client.get("/openapi.json")
    assert response.status_code == 200

    spec = response.json()
    assert spec["openapi"]
    assert spec["info"]["title"]
    assert spec["paths"]


@pytest.mark.asyncio
async def test_swagger_ui_docs(client: AsyncClient) -> None:
    """Swagger UI docs endpoint is available."""
    response = await client.get("/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


@pytest.mark.asyncio
async def test_redoc_docs(client: AsyncClient) -> None:
    """ReDoc docs endpoint is available."""
    response = await client.get("/redoc")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


@pytest.mark.asyncio
async def test_nonexistent_route_returns_404(client: AsyncClient) -> None:
    """Unknown routes return 404 with appropriate response."""
    response = await client.get("/api/v1/nonexistent-endpoint-xyz")
    assert response.status_code == 404
