from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from app.api.routers import jobs
from app.domain.errors import JobNotFoundError


def _job_record(job_id: UUID | None = None, status: str = "pending") -> SimpleNamespace:
    now = datetime.now(UTC)
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
        max_attempts=1,
        params={},
        result={},
        error="",
        created_at=now,
        started_at=None,
        completed_at=None,
        timeout_at=None,
    )


class _JobServiceStub:
    def __init__(self) -> None:
        self.list_calls: list[dict[str, object]] = []
        self.jobs: dict[UUID, SimpleNamespace] = {}

    async def list_jobs(
        self, status: str | None, job_type: str | None, limit: int, offset: int
    ) -> tuple[list[SimpleNamespace], int]:
        self.list_calls.append(
            {"status": status, "job_type": job_type, "limit": limit, "offset": offset}
        )
        values = list(self.jobs.values())
        return values[offset : offset + limit], len(values)

    async def get_job(self, job_id: UUID) -> SimpleNamespace:
        job = self.jobs.get(job_id)
        if not job:
            raise JobNotFoundError(f"Job {job_id} not found")
        return job


@pytest.mark.asyncio
async def test_list_jobs_supports_pagination_and_filters(client, app) -> None:
    service = _JobServiceStub()
    service.jobs[uuid4()] = _job_record()
    app.dependency_overrides[jobs._get_job_service] = lambda: service

    response = await client.get(
        "/api/v1/jobs/",
        params={"status": "pending", "job_type": "text_to_image", "limit": 1, "offset": 0},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["limit"] == 1
    assert body["offset"] == 0
    assert len(body["items"]) == 1
    assert service.list_calls == [
        {"status": "pending", "job_type": "text_to_image", "limit": 1, "offset": 0}
    ]


@pytest.mark.asyncio
async def test_get_job_by_id_returns_404_when_missing(client, app) -> None:
    service = _JobServiceStub()
    app.dependency_overrides[jobs._get_job_service] = lambda: service

    response = await client.get(f"/api/v1/jobs/{uuid4()}")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_get_job_by_id_happy_path(client, app) -> None:
    service = _JobServiceStub()
    job = _job_record(status="running")
    service.jobs[job.id] = job
    app.dependency_overrides[jobs._get_job_service] = lambda: service

    response = await client.get(f"/api/v1/jobs/{job.id}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == str(job.id)
    assert body["status"] == "running"


@pytest.mark.asyncio
async def test_jobs_limit_validation(client) -> None:
    response = await client.get("/api/v1/jobs/", params={"limit": 101})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_jobs_get_rejects_invalid_uuid(client) -> None:
    response = await client.get("/api/v1/jobs/not-a-uuid")
    assert response.status_code == 422
