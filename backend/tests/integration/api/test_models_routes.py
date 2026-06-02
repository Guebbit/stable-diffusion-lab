from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.api.routers import models


def _model_record(model_id: str = "stabilityai-sdxl") -> SimpleNamespace:
    now = datetime.now(UTC)
    return SimpleNamespace(
        id=uuid4(),
        model_id=model_id,
        name="SDXL",
        source="huggingface",
        family="sdxl",
        variant="",
        description="",
        tags=["image"],
        source_url="https://huggingface.co/stabilityai/sdxl",
        capabilities=["text_to_image"],
        status="downloaded",
        total_size_bytes=1234,
        disk_size_bytes=1234,
        download_progress=100,
        is_verified=True,
        last_verified_at=now,
        created_at=now,
        updated_at=now,
    )


class _ModelServiceStub:
    def __init__(self) -> None:
        self.models: dict[str, SimpleNamespace] = {}
        self.list_source_calls: list[str | None] = []

    async def list_models(self, source: str | None = None) -> list[SimpleNamespace]:
        self.list_source_calls.append(source)
        values = list(self.models.values())
        if source is None:
            return values
        return [model for model in values if model.source == source]

    async def get_model(self, model_id: str) -> SimpleNamespace | None:
        return self.models.get(model_id)


@pytest.mark.asyncio
async def test_list_models_endpoint_availability(client, app) -> None:
    service = _ModelServiceStub()
    app.dependency_overrides[models._get_model_service] = lambda: service

    response = await client.get("/api/v1/models/")

    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_get_model_by_id_happy_path(client, app) -> None:
    service = _ModelServiceStub()
    model_record = _model_record()
    service.models[model_record.model_id] = model_record
    app.dependency_overrides[models._get_model_service] = lambda: service

    response = await client.get(f"/api/v1/models/{model_record.model_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["model_id"] == model_record.model_id
    assert body["name"] == "SDXL"


@pytest.mark.asyncio
async def test_get_model_by_id_returns_404_when_missing(client, app) -> None:
    service = _ModelServiceStub()
    app.dependency_overrides[models._get_model_service] = lambda: service

    response = await client.get("/api/v1/models/missing-model")

    assert response.status_code == 404
    assert response.json()["detail"] == "Model not found"


@pytest.mark.asyncio
async def test_models_list_supports_pagination(client, app) -> None:
    service = _ModelServiceStub()
    first = _model_record(model_id="model/a")
    second = _model_record(model_id="model/b")
    third = _model_record(model_id="model/c")
    service.models = {first.model_id: first, second.model_id: second, third.model_id: third}
    app.dependency_overrides[models._get_model_service] = lambda: service

    response = await client.get("/api/v1/models/", params={"limit": 2, "offset": 1})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert body[0]["model_id"] == "model/b"
    assert body[1]["model_id"] == "model/c"


@pytest.mark.asyncio
async def test_models_limit_validation(client, app) -> None:
    service = _ModelServiceStub()
    app.dependency_overrides[models._get_model_service] = lambda: service

    response = await client.get("/api/v1/models/", params={"limit": 101})
    assert response.status_code == 422
