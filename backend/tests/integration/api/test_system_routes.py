"""
Integration tests for system health and status endpoints.

Tests:
- GET /api/v1/system/health  — liveness probe
- GET /api/v1/system/status  — model catalog statistics
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from httpx import AsyncClient

from app.api.routers import system


class _ModelRepoStub:
    """Stub ModelRepository for system status tests."""

    def __init__(self, models: list[SimpleNamespace] | None = None) -> None:
        self._models = models or []

    async def list_all(self, source: str | None = None) -> list[SimpleNamespace]:
        if source is None:
            return self._models
        return [m for m in self._models if m.source == source]


def _model(family: str = "sdxl", source: str = "huggingface") -> SimpleNamespace:
    return SimpleNamespace(family=family, source=source)


# ── Health ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_returns_200(client: AsyncClient) -> None:
    response = await client.get("/api/v1/system/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_response_contains_status_field(client: AsyncClient) -> None:
    body = (await client.get("/api/v1/system/health")).json()
    assert "status" in body


@pytest.mark.asyncio
async def test_health_status_is_healthy(client: AsyncClient) -> None:
    body = (await client.get("/api/v1/system/health")).json()
    assert body["status"] == "healthy"


@pytest.mark.asyncio
async def test_health_requires_no_auth(client: AsyncClient) -> None:
    """Health endpoint must be reachable without any credentials."""
    response = await client.get("/api/v1/system/health")
    assert response.status_code != 401
    assert response.status_code != 403


# ── Status ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_status_returns_200(client: AsyncClient, app) -> None:
    app.dependency_overrides[system._get_model_repo] = lambda: _ModelRepoStub()
    response = await client.get("/api/v1/system/status")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_status_response_shape(client: AsyncClient, app) -> None:
    app.dependency_overrides[system._get_model_repo] = lambda: _ModelRepoStub()
    body = (await client.get("/api/v1/system/status")).json()
    assert "total_models" in body
    assert "models_by_family" in body


@pytest.mark.asyncio
async def test_status_empty_catalog_returns_zero_totals(client: AsyncClient, app) -> None:
    app.dependency_overrides[system._get_model_repo] = lambda: _ModelRepoStub(models=[])
    body = (await client.get("/api/v1/system/status")).json()
    assert body["total_models"] == 0
    assert body["models_by_family"] == {}


@pytest.mark.asyncio
async def test_status_counts_total_models(client: AsyncClient, app) -> None:
    models = [_model("sdxl"), _model("sdxl"), _model("sd15")]
    app.dependency_overrides[system._get_model_repo] = lambda: _ModelRepoStub(models=models)
    body = (await client.get("/api/v1/system/status")).json()
    assert body["total_models"] == 3


@pytest.mark.asyncio
async def test_status_groups_models_by_family(client: AsyncClient, app) -> None:
    models = [_model("sdxl"), _model("sdxl"), _model("sd15")]
    app.dependency_overrides[system._get_model_repo] = lambda: _ModelRepoStub(models=models)
    body = (await client.get("/api/v1/system/status")).json()
    assert body["models_by_family"]["sdxl"] == 2
    assert body["models_by_family"]["sd15"] == 1


@pytest.mark.asyncio
async def test_status_single_model_single_family(client: AsyncClient, app) -> None:
    app.dependency_overrides[system._get_model_repo] = lambda: _ModelRepoStub(
        models=[_model("flux")]
    )
    body = (await client.get("/api/v1/system/status")).json()
    assert body["total_models"] == 1
    assert body["models_by_family"] == {"flux": 1}


@pytest.mark.asyncio
async def test_status_many_families(client: AsyncClient, app) -> None:
    models = [
        _model("sdxl"),
        _model("sd15"),
        _model("flux"),
        _model("sdxl"),
        _model("flux"),
        _model("flux"),
    ]
    app.dependency_overrides[system._get_model_repo] = lambda: _ModelRepoStub(models=models)
    body = (await client.get("/api/v1/system/status")).json()
    assert body["total_models"] == 6
    assert body["models_by_family"]["sdxl"] == 2
    assert body["models_by_family"]["sd15"] == 1
    assert body["models_by_family"]["flux"] == 3
