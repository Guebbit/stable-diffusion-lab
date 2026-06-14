"""
End-to-end workflow tests.

Each test exercises a realistic multi-step sequence across different
API endpoints, verifying that the full request → response lifecycle
is consistent and coherent (job IDs, resource IDs, status transitions).

All external dependencies (services) are replaced with stateful in-process
stubs so tests run without a database, GPU, or network.
"""

from __future__ import annotations

import io
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from app.api.routers import artifacts, generation, jobs, models
from app.domain.errors import (
    InvalidStateTransitionError,
    JobNotFoundError,
)


# ─── Stateful stubs ──────────────────────────────────────────────────────────


def _now() -> datetime:
    return datetime.now(UTC)


def _artifact_ns(artifact_id: UUID | None = None) -> SimpleNamespace:
    now = _now()
    return SimpleNamespace(
        id=artifact_id or uuid4(),
        job_id=uuid4(),
        file_path="/output/test.png",
        thumbnail_path="",
        media_type="image/png",
        size_bytes=512,
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


def _job_ns(job_id: UUID | None = None, status: str = "pending") -> SimpleNamespace:
    now = _now()
    return SimpleNamespace(
        id=job_id or uuid4(),
        job_type="text_to_image",
        status=status,
        progress_percent=0,
        current_step=0,
        total_steps=0,
        message="",
        priority=0,
        attempt=1,
        max_attempts=3,
        params={},
        result={},
        error="",
        created_at=now,
        started_at=None,
        completed_at=None,
        timeout_at=None,
    )


def _model_ns(model_id: str = "org/model") -> SimpleNamespace:
    now = _now()
    return SimpleNamespace(
        id=uuid4(),
        model_id=model_id,
        name="Test Model",
        preferred_name=None,
        source="huggingface",
        family="sdxl",
        variant="",
        description="",
        tags=[],
        source_url="",
        capabilities=["text_to_image"],
        status="not_downloaded",
        total_size_bytes=0,
        disk_size_bytes=0,
        download_size_bytes=None,
        recommended_vram_min_gb=None,
        recommended_vram_max_gb=None,
        license="",
        base_model="",
        precision="",
        requirements={},
        notes="",
        is_verified=False,
        last_verified_at=None,
        local_path=None,
        file_count=0,
        created_at=now,
        updated_at=now,
    )


class _StatefulArtifactService:
    def __init__(self) -> None:
        self._store: dict[UUID, SimpleNamespace] = {}

    async def list_artifacts(
        self, model_name=None, media_type=None, is_favorite=None, limit=50, offset=0
    ) -> tuple[list[SimpleNamespace], int]:
        items = list(self._store.values())
        return items[offset : offset + limit], len(items)

    async def get_artifact(self, artifact_id: UUID) -> SimpleNamespace | None:
        return self._store.get(artifact_id)

    async def update_metadata(
        self, artifact_id: UUID, is_favorite=None, rating=None, notes=None
    ) -> SimpleNamespace | None:
        art = self._store.get(artifact_id)
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
        if artifact_id not in self._store:
            return False
        del self._store[artifact_id]
        return True

    def seed(self, art: SimpleNamespace) -> None:
        self._store[art.id] = art


class _StatefulModelService:
    def __init__(self) -> None:
        self._store: dict[str, SimpleNamespace] = {}

    async def list_models(self, source=None, capabilities=None) -> list[SimpleNamespace]:
        items = list(self._store.values())
        if source:
            items = [m for m in items if m.source == source]
        return items

    async def get_model(self, model_id: str) -> SimpleNamespace | None:
        return self._store.get(model_id)

    async def register_model(self, model_id, name, source, **kwargs) -> SimpleNamespace:
        rec = _model_ns(model_id=model_id)
        rec.name = name
        rec.source = source
        self._store[model_id] = rec
        return rec

    async def delete_model(self, model_id: str) -> None:
        if model_id not in self._store:
            raise ValueError(f"Model '{model_id}' not found")
        del self._store[model_id]


class _StatefulJobCreatorStub:
    def __init__(self) -> None:
        self._jobs: dict[str, UUID] = {}

    async def create_model_download_job(self, model_id: str, source: str) -> UUID:
        job_id = uuid4()
        self._jobs[model_id] = job_id
        return job_id


class _StatefulJobService:
    def __init__(self) -> None:
        self._store: dict[UUID, SimpleNamespace] = {}
        self._events: dict[UUID, list[SimpleNamespace]] = {}

    async def list_jobs(self, status=None, job_type=None, limit=50, offset=0):
        items = list(self._store.values())
        if status:
            items = [j for j in items if j.status == status]
        return items[offset : offset + limit], len(items)

    async def get_job(self, job_id: UUID) -> SimpleNamespace:
        job = self._store.get(job_id)
        if not job:
            raise JobNotFoundError(f"Job {job_id} not found")
        return job

    async def request_cancellation(self, job_id: UUID) -> str:
        job = self._store.get(job_id)
        if not job:
            raise JobNotFoundError(f"Job {job_id} not found")
        if job.status in ("completed", "failed", "cancelled"):
            raise InvalidStateTransitionError(f"Cannot cancel '{job.status}' job")
        if job.status == "pending":
            job.status = "cancelled"
            return "cancelled"
        return job.status

    async def retry_job(self, job_id: UUID) -> SimpleNamespace:
        job = self._store.get(job_id)
        if not job:
            raise JobNotFoundError(f"Job {job_id} not found")
        if job.status != "failed":
            raise InvalidStateTransitionError(f"Only failed jobs can be retried")
        job.status = "pending"
        job.attempt += 1
        return job

    async def get_job_events(self, job_id: UUID) -> list:
        job = self._store.get(job_id)
        if not job:
            raise JobNotFoundError(f"Job {job_id} not found")
        return self._events.get(job_id, [])

    def seed(self, job: SimpleNamespace) -> None:
        self._store[job.id] = job


class _StatefulGenerationService:
    def __init__(self, job_service: _StatefulJobService) -> None:
        self._job_service = job_service

    async def submit_text_to_image(self, params, model_id, correlation_id=None) -> UUID:
        job_id = uuid4()
        job = _job_ns(job_id=job_id, status="pending")
        self._job_service.seed(job)
        return job_id


# ─── Workflow: Artifact gallery lifecycle ────────────────────────────────────


@pytest.mark.asyncio
async def test_artifact_gallery_lifecycle(client, app) -> None:
    """
    Full artifact CRUD lifecycle:
    list (empty) → seed → list → get → patch favorite → verify → delete → 404
    """
    svc = _StatefulArtifactService()
    app.dependency_overrides[artifacts._get_artifact_service] = lambda: svc

    # Empty gallery
    body = (await client.get("/api/v1/artifacts/")).json()
    assert body["total"] == 0

    # Seed an artifact
    art = _artifact_ns()
    svc.seed(art)

    # Appears in list
    body = (await client.get("/api/v1/artifacts/")).json()
    assert body["total"] == 1
    assert body["items"][0]["id"] == str(art.id)

    # Get by ID
    r = await client.get(f"/api/v1/artifacts/{art.id}")
    assert r.status_code == 200
    assert r.json()["is_favorite"] is False

    # Mark as favorite
    r = await client.patch(f"/api/v1/artifacts/{art.id}", json={"is_favorite": True, "rating": 5})
    assert r.status_code == 200
    assert r.json()["is_favorite"] is True
    assert r.json()["rating"] == 5

    # Verify change persisted (GET again)
    r = await client.get(f"/api/v1/artifacts/{art.id}")
    assert r.json()["is_favorite"] is True

    # Delete
    r = await client.delete(f"/api/v1/artifacts/{art.id}")
    assert r.status_code == 204

    # No longer findable
    r = await client.get(f"/api/v1/artifacts/{art.id}")
    assert r.status_code == 404

    # Gone from list too
    body = (await client.get("/api/v1/artifacts/")).json()
    assert body["total"] == 0


# ─── Workflow: Model catalog lifecycle ───────────────────────────────────────


@pytest.mark.asyncio
async def test_model_catalog_lifecycle(client, app) -> None:
    """
    Full model CRUD lifecycle:
    register → list → get → download job → delete → 404
    """
    model_svc = _StatefulModelService()
    job_creator = _StatefulJobCreatorStub()
    app.dependency_overrides[models._get_model_service] = lambda: model_svc
    app.dependency_overrides[models._get_job_creator] = lambda: job_creator

    # Empty catalog
    assert (await client.get("/api/v1/models/")).json() == []

    # Register a model
    r = await client.post(
        "/api/v1/models/",
        json={"model_id": "org/flow-model", "name": "Flow Model", "source": "huggingface"},
    )
    assert r.status_code == 201
    assert r.json()["model_id"] == "org/flow-model"

    # Appears in list
    body = (await client.get("/api/v1/models/")).json()
    assert len(body) == 1
    assert body[0]["model_id"] == "org/flow-model"

    # Get by ID
    r = await client.get("/api/v1/models/org/flow-model")
    assert r.status_code == 200

    # Trigger download
    r = await client.post("/api/v1/models/org/flow-model/download")
    assert r.status_code == 202
    assert "job_id" in r.json()

    # Delete
    r = await client.delete("/api/v1/models/org/flow-model")
    assert r.status_code == 204

    # No longer findable
    r = await client.get("/api/v1/models/org/flow-model")
    assert r.status_code == 404


# ─── Workflow: Generation submission and job tracking ────────────────────────


@pytest.mark.asyncio
async def test_generation_submission_and_job_tracking(client, app) -> None:
    """
    Submit a text-to-image job then query it via the /jobs/ API.
    The job_id returned from /generation must be resolvable via /jobs/{id}.
    """
    job_svc = _StatefulJobService()
    gen_svc = _StatefulGenerationService(job_svc)
    app.dependency_overrides[generation._get_generation_service] = lambda: gen_svc
    app.dependency_overrides[jobs._get_job_service] = lambda: job_svc

    # Submit generation
    r = await client.post(
        "/api/v1/generation/text-to-image",
        json={"prompt": "a dragon", "model_id": "sdxl"},
    )
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    # Job visible in list
    body = (await client.get("/api/v1/jobs/")).json()
    assert body["total"] == 1
    assert body["items"][0]["id"] == job_id

    # Job findable by ID
    r = await client.get(f"/api/v1/jobs/{job_id}")
    assert r.status_code == 200
    assert r.json()["id"] == job_id
    assert r.json()["status"] == "pending"


# ─── Workflow: Job cancel lifecycle ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_job_cancel_lifecycle(client, app) -> None:
    """
    Seed a pending job → cancel → verify cancelled → second cancel returns 409.
    """
    svc = _StatefulJobService()
    job = _job_ns(status="pending")
    svc.seed(job)
    app.dependency_overrides[jobs._get_job_service] = lambda: svc

    # Job is pending
    r = await client.get(f"/api/v1/jobs/{job.id}")
    assert r.json()["status"] == "pending"

    # Cancel it
    r = await client.post(f"/api/v1/jobs/{job.id}/cancel")
    assert r.status_code == 200
    assert r.json()["status"] == "cancelled"

    # Job now shows cancelled
    r = await client.get(f"/api/v1/jobs/{job.id}")
    assert r.json()["status"] == "cancelled"

    # Trying to cancel again returns 409
    r = await client.post(f"/api/v1/jobs/{job.id}/cancel")
    assert r.status_code == 409


# ─── Workflow: Job fail-then-retry lifecycle ─────────────────────────────────


@pytest.mark.asyncio
async def test_job_fail_retry_lifecycle(client, app) -> None:
    """
    Seed a failed job → retry → verify pending → events endpoint returns list.
    """
    svc = _StatefulJobService()
    job = _job_ns(status="failed")
    svc.seed(job)
    app.dependency_overrides[jobs._get_job_service] = lambda: svc

    # Job is failed
    r = await client.get(f"/api/v1/jobs/{job.id}")
    assert r.json()["status"] == "failed"

    # Retry
    r = await client.post(f"/api/v1/jobs/{job.id}/retry")
    assert r.status_code == 200
    assert r.json()["status"] == "pending"

    # Now it's pending again
    r = await client.get(f"/api/v1/jobs/{job.id}")
    assert r.json()["status"] == "pending"

    # Events endpoint doesn't 500 — returns a list
    r = await client.get(f"/api/v1/jobs/{job.id}/events")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
