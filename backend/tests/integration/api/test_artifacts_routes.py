from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from app.api.routers import artifacts


def _artifact_record(artifact_id: UUID | None = None) -> SimpleNamespace:
    now = datetime.now(UTC)
    return SimpleNamespace(
        id=artifact_id or uuid4(),
        job_id=uuid4(),
        file_path="/artifact.png",
        thumbnail_path="",
        media_type="image/png",
        size_bytes=123,
        width=512,
        height=512,
        duration_seconds=None,
        prompt="a cat",
        negative_prompt="",
        seed=42,
        model_name="SDXL",
        model_id_ref="test/model",
        generation_params={},
        is_favorite=False,
        rating=0,
        notes="",
        created_at=now,
        updated_at=now,
    )


class _ArtifactServiceStub:
    def __init__(self) -> None:
        self.list_calls: list[dict[str, object]] = []
        self.artifacts: dict[UUID, SimpleNamespace] = {}

    async def list_artifacts(
        self,
        model_name: str | None,
        media_type: str | None,
        is_favorite: bool | None,
        limit: int,
        offset: int,
    ) -> tuple[list[SimpleNamespace], int]:
        self.list_calls.append(
            {
                "model_name": model_name,
                "media_type": media_type,
                "is_favorite": is_favorite,
                "limit": limit,
                "offset": offset,
            }
        )
        values = list(self.artifacts.values())
        return values[offset : offset + limit], len(values)

    async def get_artifact(self, artifact_id: UUID) -> SimpleNamespace | None:
        return self.artifacts.get(artifact_id)


@pytest.mark.asyncio
async def test_list_artifacts_supports_pagination_and_filters(client, app) -> None:
    service = _ArtifactServiceStub()
    service.artifacts[uuid4()] = _artifact_record()
    app.dependency_overrides[artifacts._get_artifact_service] = lambda: service

    response = await client.get(
        "/api/v1/artifacts/",
        params={"model_name": "SDXL", "media_type": "image/png", "is_favorite": False, "limit": 1},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["limit"] == 1
    assert len(body["items"]) == 1
    assert service.list_calls[0]["model_name"] == "SDXL"
    assert service.list_calls[0]["media_type"] == "image/png"
    assert service.list_calls[0]["is_favorite"] is False


@pytest.mark.asyncio
async def test_get_artifact_by_id_happy_path(client, app) -> None:
    service = _ArtifactServiceStub()
    artifact_record = _artifact_record()
    service.artifacts[artifact_record.id] = artifact_record
    app.dependency_overrides[artifacts._get_artifact_service] = lambda: service

    response = await client.get(f"/api/v1/artifacts/{artifact_record.id}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == str(artifact_record.id)
    assert body["media_type"] == "image/png"


@pytest.mark.asyncio
async def test_get_artifact_by_id_returns_404_when_missing(client, app) -> None:
    service = _ArtifactServiceStub()
    app.dependency_overrides[artifacts._get_artifact_service] = lambda: service

    response = await client.get(f"/api/v1/artifacts/{uuid4()}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Artifact not found"


@pytest.mark.asyncio
async def test_artifacts_limit_validation(client) -> None:
    response = await client.get("/api/v1/artifacts/", params={"limit": 101})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_artifacts_get_rejects_invalid_uuid(client) -> None:
    response = await client.get("/api/v1/artifacts/not-a-uuid")
    assert response.status_code == 422
