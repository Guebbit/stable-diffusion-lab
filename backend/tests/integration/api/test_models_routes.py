from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from app.api.routers import models


def _model_record(
    model_id: str = "stabilityai/sdxl",
    family: str = "sdxl",
    source: str = "huggingface",
    capabilities: list[str] | None = None,
    preferred_name: str | None = None,
    status: str = "downloaded",
) -> SimpleNamespace:
    now = datetime.now(UTC)
    return SimpleNamespace(
        id=uuid4(),
        model_id=model_id,
        name="SDXL",
        preferred_name=preferred_name,
        source=source,
        family=family,
        variant="",
        description="",
        tags=["image"],
        source_url=f"https://huggingface.co/{model_id}",
        capabilities=capabilities or ["text_to_image"],
        status=status,
        total_size_bytes=1234,
        disk_size_bytes=1234,
        download_size_bytes=1234,
        download_progress=100,
        recommended_vram_min_gb=None,
        recommended_vram_max_gb=None,
        license="",
        base_model="",
        precision="",
        requirements={},
        notes="",
        is_verified=True,
        last_verified_at=now,
        created_at=now,
        updated_at=now,
    )


class _ModelServiceStub:
    def __init__(self) -> None:
        self.models: dict[str, SimpleNamespace] = {}
        self.list_calls: list[dict] = []
        self.register_calls: list[dict] = []
        self.delete_calls: list[str] = []

    async def list_models(
        self,
        source: str | None = None,
        capabilities: list[str] | None = None,
    ) -> list[SimpleNamespace]:
        self.list_calls.append({"source": source, "capabilities": capabilities})
        values = list(self.models.values())
        if source is not None:
            values = [m for m in values if m.source == source]
        if capabilities:
            values = [
                m for m in values
                if any(c in (m.capabilities or []) for c in capabilities)
            ]
        return values

    async def get_model(self, model_id: str) -> SimpleNamespace | None:
        return self.models.get(model_id)

    async def register_model(
        self,
        model_id: str,
        name: str,
        source: str,
        family: str = "custom",
        variant: str = "",
        description: str = "",
        tags: list[str] | None = None,
        source_url: str = "",
        capabilities: list[str] | None = None,
        preferred_name: str | None = None,
    ) -> SimpleNamespace:
        self.register_calls.append({
            "model_id": model_id,
            "name": name,
            "source": source,
            "family": family,
            "preferred_name": preferred_name,
        })
        record = _model_record(
            model_id=model_id,
            family=family,
            source=source,
            capabilities=capabilities or [],
            preferred_name=preferred_name,
            status="not_downloaded",
        )
        record.name = name
        self.models[model_id] = record
        return record

    async def delete_model(self, model_id: str) -> None:
        if model_id not in self.models:
            raise ValueError(f"Model '{model_id}' not found")
        self.delete_calls.append(model_id)
        del self.models[model_id]


class _JobCreatorStub:
    def __init__(self) -> None:
        self.download_calls: list[dict] = []

    async def create_model_download_job(self, model_id: str, source: str) -> UUID:
        self.download_calls.append({"model_id": model_id, "source": source})
        return uuid4()


# ── GET /models/ ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_models_empty_catalog(client, app) -> None:
    app.dependency_overrides[models._get_model_service] = lambda: _ModelServiceStub()
    response = await client.get("/api/v1/models/")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_list_models_returns_all_models(client, app) -> None:
    service = _ModelServiceStub()
    service.models = {
        "m/a": _model_record(model_id="m/a"),
        "m/b": _model_record(model_id="m/b"),
    }
    app.dependency_overrides[models._get_model_service] = lambda: service

    body = (await client.get("/api/v1/models/")).json()
    assert len(body) == 2


@pytest.mark.asyncio
async def test_list_models_source_filter_passed_to_service(client, app) -> None:
    service = _ModelServiceStub()
    service.models = {
        "hf/m": _model_record(model_id="hf/m", source="huggingface"),
        "cv/m": _model_record(model_id="cv/m", source="civitai"),
    }
    app.dependency_overrides[models._get_model_service] = lambda: service

    body = (await client.get("/api/v1/models/", params={"source": "huggingface"})).json()

    assert len(body) == 1
    assert body[0]["source"] == "huggingface"
    assert service.list_calls[0]["source"] == "huggingface"


@pytest.mark.asyncio
async def test_list_models_capabilities_filter(client, app) -> None:
    service = _ModelServiceStub()
    service.models = {
        "t2i": _model_record(model_id="t2i", capabilities=["text_to_image"]),
        "i2i": _model_record(model_id="i2i", capabilities=["image_to_image"]),
    }
    app.dependency_overrides[models._get_model_service] = lambda: service

    body = (await client.get("/api/v1/models/", params={"capabilities": "text_to_image"})).json()

    assert len(body) == 1
    assert body[0]["model_id"] == "t2i"
    assert "text_to_image" in service.list_calls[0]["capabilities"]


@pytest.mark.asyncio
async def test_list_models_pagination_limit_and_offset(client, app) -> None:
    service = _ModelServiceStub()
    service.models = {
        "model/a": _model_record(model_id="model/a"),
        "model/b": _model_record(model_id="model/b"),
        "model/c": _model_record(model_id="model/c"),
    }
    app.dependency_overrides[models._get_model_service] = lambda: service

    body = (await client.get("/api/v1/models/", params={"limit": 2, "offset": 1})).json()

    assert len(body) == 2
    assert body[0]["model_id"] == "model/b"
    assert body[1]["model_id"] == "model/c"


@pytest.mark.asyncio
async def test_list_models_limit_validation(client, app) -> None:
    app.dependency_overrides[models._get_model_service] = lambda: _ModelServiceStub()
    response = await client.get("/api/v1/models/", params={"limit": 101})
    assert response.status_code == 422


# ── GET /models/{model_id} ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_model_by_id_happy_path(client, app) -> None:
    service = _ModelServiceStub()
    model = _model_record(model_id="stabilityai-sdxl")
    service.models[model.model_id] = model
    app.dependency_overrides[models._get_model_service] = lambda: service

    response = await client.get(f"/api/v1/models/{model.model_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["model_id"] == model.model_id
    assert body["name"] == "SDXL"


@pytest.mark.asyncio
async def test_get_model_by_id_with_slashes_in_path(client, app) -> None:
    """Model IDs like 'stabilityai/sdxl-base' contain slashes — must be routed correctly."""
    service = _ModelServiceStub()
    model = _model_record(model_id="stabilityai/sdxl-base-1.0")
    service.models[model.model_id] = model
    app.dependency_overrides[models._get_model_service] = lambda: service

    response = await client.get("/api/v1/models/stabilityai/sdxl-base-1.0")

    assert response.status_code == 200
    assert response.json()["model_id"] == "stabilityai/sdxl-base-1.0"


@pytest.mark.asyncio
async def test_get_model_by_id_returns_404_when_missing(client, app) -> None:
    service = _ModelServiceStub()
    app.dependency_overrides[models._get_model_service] = lambda: service

    response = await client.get("/api/v1/models/missing-model")

    assert response.status_code == 404
    assert response.json()["detail"] == "Model not found"


@pytest.mark.asyncio
async def test_get_model_response_includes_all_fields(client, app) -> None:
    service = _ModelServiceStub()
    model = _model_record(
        model_id="test/full-model",
        capabilities=["text_to_image", "image_to_image"],
        preferred_name="Full Test Model",
    )
    service.models[model.model_id] = model
    app.dependency_overrides[models._get_model_service] = lambda: service

    body = (await client.get(f"/api/v1/models/{model.model_id}")).json()

    assert "capabilities" in body
    assert "text_to_image" in body["capabilities"]
    assert body["preferred_name"] == "Full Test Model"
    assert "created_at" in body
    assert "status" in body


# ── POST /models/ — register ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_register_model_returns_201(client, app) -> None:
    service = _ModelServiceStub()
    app.dependency_overrides[models._get_model_service] = lambda: service

    response = await client.post(
        "/api/v1/models/",
        json={"model_id": "test/new-model", "name": "New Model", "source": "huggingface"},
    )

    assert response.status_code == 201


@pytest.mark.asyncio
async def test_register_model_response_shape(client, app) -> None:
    service = _ModelServiceStub()
    app.dependency_overrides[models._get_model_service] = lambda: service

    body = (
        await client.post(
            "/api/v1/models/",
            json={"model_id": "test/new-model", "name": "New Model", "source": "huggingface"},
        )
    ).json()

    assert body["model_id"] == "test/new-model"
    assert body["name"] == "New Model"
    assert body["source"] == "huggingface"
    assert "id" in body
    assert "status" in body


@pytest.mark.asyncio
async def test_register_model_with_all_optional_fields(client, app) -> None:
    service = _ModelServiceStub()
    app.dependency_overrides[models._get_model_service] = lambda: service

    payload = {
        "model_id": "org/full-model",
        "name": "Full Model",
        "source": "civitai",
        "family": "sdxl",
        "variant": "fp16",
        "description": "A complete model",
        "tags": ["diffusion", "sdxl"],
        "source_url": "https://civitai.com/models/1",
        "preferred_name": "My SDXL",
        "capabilities": ["text_to_image", "image_to_image"],
    }
    body = (await client.post("/api/v1/models/", json=payload)).json()

    assert body["family"] == "sdxl"
    assert "text_to_image" in body["capabilities"]
    assert body["preferred_name"] == "My SDXL"


@pytest.mark.asyncio
async def test_register_model_calls_service_with_correct_args(client, app) -> None:
    service = _ModelServiceStub()
    app.dependency_overrides[models._get_model_service] = lambda: service

    await client.post(
        "/api/v1/models/",
        json={"model_id": "org/model", "name": "My Model", "source": "huggingface"},
    )

    assert len(service.register_calls) == 1
    call = service.register_calls[0]
    assert call["model_id"] == "org/model"
    assert call["name"] == "My Model"
    assert call["source"] == "huggingface"


@pytest.mark.asyncio
async def test_register_model_missing_model_id_returns_422(client, app) -> None:
    app.dependency_overrides[models._get_model_service] = lambda: _ModelServiceStub()
    response = await client.post(
        "/api/v1/models/",
        json={"name": "No ID Model", "source": "huggingface"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_register_model_missing_name_returns_422(client, app) -> None:
    app.dependency_overrides[models._get_model_service] = lambda: _ModelServiceStub()
    response = await client.post(
        "/api/v1/models/",
        json={"model_id": "test/x", "source": "huggingface"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_register_model_missing_source_returns_422(client, app) -> None:
    app.dependency_overrides[models._get_model_service] = lambda: _ModelServiceStub()
    response = await client.post(
        "/api/v1/models/",
        json={"model_id": "test/x", "name": "X"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_register_duplicate_model_returns_existing(client, app) -> None:
    """Registering the same model_id twice returns the existing record."""
    service = _ModelServiceStub()
    existing = _model_record(model_id="test/existing", status="downloaded")
    service.models["test/existing"] = existing
    app.dependency_overrides[models._get_model_service] = lambda: service

    body = (
        await client.post(
            "/api/v1/models/",
            json={"model_id": "test/existing", "name": "Duplicate", "source": "local"},
        )
    ).json()

    assert body["model_id"] == "test/existing"
    assert len(service.register_calls) == 1


# ── POST /models/{model_id}/download ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_download_model_returns_202_with_job_id(client, app) -> None:
    service = _ModelServiceStub()
    model = _model_record(model_id="org/downloadable")
    service.models[model.model_id] = model
    creator = _JobCreatorStub()
    app.dependency_overrides[models._get_model_service] = lambda: service
    app.dependency_overrides[models._get_job_creator] = lambda: creator

    response = await client.post("/api/v1/models/org/downloadable/download")

    assert response.status_code == 202
    body = response.json()
    assert "job_id" in body
    assert body["status"] == "pending"


@pytest.mark.asyncio
async def test_download_model_creates_job_with_correct_model_id(client, app) -> None:
    service = _ModelServiceStub()
    model = _model_record(model_id="org/model", source="huggingface")
    service.models[model.model_id] = model
    creator = _JobCreatorStub()
    app.dependency_overrides[models._get_model_service] = lambda: service
    app.dependency_overrides[models._get_job_creator] = lambda: creator

    await client.post("/api/v1/models/org/model/download")

    assert len(creator.download_calls) == 1
    assert creator.download_calls[0]["model_id"] == "org/model"
    assert creator.download_calls[0]["source"] == "huggingface"


@pytest.mark.asyncio
async def test_download_model_not_found_returns_404(client, app) -> None:
    service = _ModelServiceStub()
    creator = _JobCreatorStub()
    app.dependency_overrides[models._get_model_service] = lambda: service
    app.dependency_overrides[models._get_job_creator] = lambda: creator

    response = await client.post("/api/v1/models/nonexistent/model/download")

    assert response.status_code == 404


# ── DELETE /models/{model_id} ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_model_returns_204(client, app) -> None:
    service = _ModelServiceStub()
    model = _model_record(model_id="org/deletable")
    service.models[model.model_id] = model
    app.dependency_overrides[models._get_model_service] = lambda: service

    response = await client.delete("/api/v1/models/org/deletable")

    assert response.status_code == 204
    assert response.content == b""


@pytest.mark.asyncio
async def test_delete_model_removes_from_service(client, app) -> None:
    service = _ModelServiceStub()
    model = _model_record(model_id="org/to-delete")
    service.models[model.model_id] = model
    app.dependency_overrides[models._get_model_service] = lambda: service

    await client.delete("/api/v1/models/org/to-delete")

    assert "org/to-delete" in service.delete_calls


@pytest.mark.asyncio
async def test_delete_model_not_found_returns_404(client, app) -> None:
    service = _ModelServiceStub()
    app.dependency_overrides[models._get_model_service] = lambda: service

    response = await client.delete("/api/v1/models/nonexistent/model")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_model_with_slash_path(client, app) -> None:
    """DELETE supports model_id values that contain slashes."""
    service = _ModelServiceStub()
    model = _model_record(model_id="deep/nested/model-path")
    service.models[model.model_id] = model
    app.dependency_overrides[models._get_model_service] = lambda: service

    response = await client.delete("/api/v1/models/deep/nested/model-path")

    assert response.status_code == 204
