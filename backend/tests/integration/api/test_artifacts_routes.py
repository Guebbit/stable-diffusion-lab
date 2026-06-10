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
        self.update_calls: list[dict[str, object]] = []
        self.delete_calls: list[UUID] = []
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

    async def update_metadata(
        self,
        artifact_id: UUID,
        is_favorite: bool | None = None,
        rating: int | None = None,
        notes: str | None = None,
    ) -> SimpleNamespace | None:
        self.update_calls.append(
            {"artifact_id": artifact_id, "is_favorite": is_favorite, "rating": rating, "notes": notes}
        )
        art = self.artifacts.get(artifact_id)
        if not art:
            return None
        if is_favorite is not None:
            art.is_favorite = is_favorite
        if rating is not None:
            art.rating = rating
        if notes is not None:
            art.notes = notes
        return art

    async def delete_artifact(self, artifact_id: UUID) -> bool:
        self.delete_calls.append(artifact_id)
        if artifact_id not in self.artifacts:
            return False
        del self.artifacts[artifact_id]
        return True


# ── GET /artifacts/ ───────────────────────────────────────────────────────────


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
async def test_list_artifacts_is_favorite_true_filter(client, app) -> None:
    service = _ArtifactServiceStub()
    fav = _artifact_record()
    fav.is_favorite = True
    service.artifacts[fav.id] = fav
    app.dependency_overrides[artifacts._get_artifact_service] = lambda: service

    body = (await client.get("/api/v1/artifacts/", params={"is_favorite": True})).json()

    assert service.list_calls[0]["is_favorite"] is True


@pytest.mark.asyncio
async def test_list_artifacts_no_filters_returns_all(client, app) -> None:
    service = _ArtifactServiceStub()
    for _ in range(3):
        a = _artifact_record()
        service.artifacts[a.id] = a
    app.dependency_overrides[artifacts._get_artifact_service] = lambda: service

    body = (await client.get("/api/v1/artifacts/")).json()

    assert body["total"] == 3
    assert len(body["items"]) == 3


@pytest.mark.asyncio
async def test_list_artifacts_pagination_has_more(client, app) -> None:
    service = _ArtifactServiceStub()
    for _ in range(5):
        a = _artifact_record()
        service.artifacts[a.id] = a
    app.dependency_overrides[artifacts._get_artifact_service] = lambda: service

    body = (await client.get("/api/v1/artifacts/", params={"limit": 2})).json()

    assert body["has_more"] is True
    assert body["total"] == 5


@pytest.mark.asyncio
async def test_artifacts_limit_validation(client) -> None:
    response = await client.get("/api/v1/artifacts/", params={"limit": 101})
    assert response.status_code == 422


# ── GET /artifacts/{artifact_id} ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_artifact_by_id_happy_path(client, app) -> None:
    service = _ArtifactServiceStub()
    artifact = _artifact_record()
    service.artifacts[artifact.id] = artifact
    app.dependency_overrides[artifacts._get_artifact_service] = lambda: service

    response = await client.get(f"/api/v1/artifacts/{artifact.id}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == str(artifact.id)
    assert body["media_type"] == "image/png"


@pytest.mark.asyncio
async def test_get_artifact_response_includes_all_fields(client, app) -> None:
    service = _ArtifactServiceStub()
    artifact = _artifact_record()
    service.artifacts[artifact.id] = artifact
    app.dependency_overrides[artifacts._get_artifact_service] = lambda: service

    body = (await client.get(f"/api/v1/artifacts/{artifact.id}")).json()

    for field in ("id", "job_id", "file_path", "media_type", "prompt", "seed",
                  "model_name", "is_favorite", "rating", "notes", "created_at"):
        assert field in body


@pytest.mark.asyncio
async def test_get_artifact_by_id_returns_404_when_missing(client, app) -> None:
    service = _ArtifactServiceStub()
    app.dependency_overrides[artifacts._get_artifact_service] = lambda: service

    response = await client.get(f"/api/v1/artifacts/{uuid4()}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Artifact not found"


@pytest.mark.asyncio
async def test_artifacts_get_rejects_invalid_uuid(client) -> None:
    response = await client.get("/api/v1/artifacts/not-a-uuid")
    assert response.status_code == 422


# ── PATCH /artifacts/{artifact_id} ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_patch_artifact_marks_as_favorite(client, app) -> None:
    service = _ArtifactServiceStub()
    artifact = _artifact_record()
    service.artifacts[artifact.id] = artifact
    app.dependency_overrides[artifacts._get_artifact_service] = lambda: service

    response = await client.patch(
        f"/api/v1/artifacts/{artifact.id}",
        json={"is_favorite": True},
    )

    assert response.status_code == 200
    assert response.json()["is_favorite"] is True


@pytest.mark.asyncio
async def test_patch_artifact_updates_rating(client, app) -> None:
    service = _ArtifactServiceStub()
    artifact = _artifact_record()
    service.artifacts[artifact.id] = artifact
    app.dependency_overrides[artifacts._get_artifact_service] = lambda: service

    response = await client.patch(
        f"/api/v1/artifacts/{artifact.id}",
        json={"rating": 4},
    )

    assert response.status_code == 200
    assert response.json()["rating"] == 4


@pytest.mark.asyncio
async def test_patch_artifact_updates_notes(client, app) -> None:
    service = _ArtifactServiceStub()
    artifact = _artifact_record()
    service.artifacts[artifact.id] = artifact
    app.dependency_overrides[artifacts._get_artifact_service] = lambda: service

    response = await client.patch(
        f"/api/v1/artifacts/{artifact.id}",
        json={"notes": "Beautiful sunset"},
    )

    assert response.status_code == 200
    assert response.json()["notes"] == "Beautiful sunset"


@pytest.mark.asyncio
async def test_patch_artifact_updates_multiple_fields(client, app) -> None:
    service = _ArtifactServiceStub()
    artifact = _artifact_record()
    service.artifacts[artifact.id] = artifact
    app.dependency_overrides[artifacts._get_artifact_service] = lambda: service

    body = (
        await client.patch(
            f"/api/v1/artifacts/{artifact.id}",
            json={"is_favorite": True, "rating": 5, "notes": "Perfect!"},
        )
    ).json()

    assert body["is_favorite"] is True
    assert body["rating"] == 5
    assert body["notes"] == "Perfect!"


@pytest.mark.asyncio
async def test_patch_artifact_partial_update_preserves_other_fields(client, app) -> None:
    service = _ArtifactServiceStub()
    artifact = _artifact_record()
    artifact.rating = 3
    service.artifacts[artifact.id] = artifact
    app.dependency_overrides[artifacts._get_artifact_service] = lambda: service

    body = (
        await client.patch(
            f"/api/v1/artifacts/{artifact.id}",
            json={"notes": "Just updating notes"},
        )
    ).json()

    assert body["notes"] == "Just updating notes"
    assert body["rating"] == 3


@pytest.mark.asyncio
async def test_patch_artifact_passes_fields_to_service(client, app) -> None:
    service = _ArtifactServiceStub()
    artifact = _artifact_record()
    service.artifacts[artifact.id] = artifact
    app.dependency_overrides[artifacts._get_artifact_service] = lambda: service

    await client.patch(
        f"/api/v1/artifacts/{artifact.id}",
        json={"is_favorite": True, "rating": 5},
    )

    call = service.update_calls[0]
    assert call["is_favorite"] is True
    assert call["rating"] == 5


@pytest.mark.asyncio
async def test_patch_artifact_not_found_returns_404(client, app) -> None:
    service = _ArtifactServiceStub()
    app.dependency_overrides[artifacts._get_artifact_service] = lambda: service

    response = await client.patch(
        f"/api/v1/artifacts/{uuid4()}",
        json={"is_favorite": True},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Artifact not found"


@pytest.mark.asyncio
async def test_patch_artifact_rating_too_high_returns_422(client, app) -> None:
    service = _ArtifactServiceStub()
    artifact = _artifact_record()
    service.artifacts[artifact.id] = artifact
    app.dependency_overrides[artifacts._get_artifact_service] = lambda: service

    response = await client.patch(
        f"/api/v1/artifacts/{artifact.id}",
        json={"rating": 6},
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_patch_artifact_rating_negative_returns_422(client, app) -> None:
    service = _ArtifactServiceStub()
    artifact = _artifact_record()
    service.artifacts[artifact.id] = artifact
    app.dependency_overrides[artifacts._get_artifact_service] = lambda: service

    response = await client.patch(
        f"/api/v1/artifacts/{artifact.id}",
        json={"rating": -1},
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_patch_artifact_invalid_uuid_returns_422(client) -> None:
    response = await client.patch("/api/v1/artifacts/not-a-uuid", json={"rating": 1})
    assert response.status_code == 422


# ── DELETE /artifacts/{artifact_id} ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_artifact_returns_204(client, app) -> None:
    service = _ArtifactServiceStub()
    artifact = _artifact_record()
    service.artifacts[artifact.id] = artifact
    app.dependency_overrides[artifacts._get_artifact_service] = lambda: service

    response = await client.delete(f"/api/v1/artifacts/{artifact.id}")

    assert response.status_code == 204
    assert response.content == b""


@pytest.mark.asyncio
async def test_delete_artifact_removes_from_service(client, app) -> None:
    service = _ArtifactServiceStub()
    artifact = _artifact_record()
    service.artifacts[artifact.id] = artifact
    app.dependency_overrides[artifacts._get_artifact_service] = lambda: service

    await client.delete(f"/api/v1/artifacts/{artifact.id}")

    assert artifact.id not in service.artifacts


@pytest.mark.asyncio
async def test_delete_artifact_not_found_returns_404(client, app) -> None:
    service = _ArtifactServiceStub()
    app.dependency_overrides[artifacts._get_artifact_service] = lambda: service

    response = await client.delete(f"/api/v1/artifacts/{uuid4()}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Artifact not found"


@pytest.mark.asyncio
async def test_delete_artifact_invalid_uuid_returns_422(client) -> None:
    response = await client.delete("/api/v1/artifacts/not-a-uuid")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_delete_artifact_not_accessible_after_deletion(client, app) -> None:
    """After deleting, a GET on the same ID returns 404."""
    service = _ArtifactServiceStub()
    artifact = _artifact_record()
    service.artifacts[artifact.id] = artifact
    app.dependency_overrides[artifacts._get_artifact_service] = lambda: service

    await client.delete(f"/api/v1/artifacts/{artifact.id}")
    response = await client.get(f"/api/v1/artifacts/{artifact.id}")

    assert response.status_code == 404
