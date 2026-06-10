from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from app.api.routers import jobs
from app.domain.errors import (
    InvalidStateTransitionError,
    JobNotFoundError,
    RetryLimitExceededError,
)


def _job_record(
    job_id: UUID | None = None,
    status: str = "pending",
    attempt: int = 1,
    max_attempts: int = 3,
) -> SimpleNamespace:
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
        attempt=attempt,
        max_attempts=max_attempts,
        params={},
        result={},
        error="",
        created_at=now,
        started_at=None,
        completed_at=None,
        timeout_at=None,
    )


def _job_event(
    job_id: UUID,
    from_status: str = "pending",
    to_status: str = "running",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        job_id=job_id,
        from_status=from_status,
        to_status=to_status,
        message="State transition",
        event_metadata={},
        created_at=datetime.now(UTC),
    )


def _typed_event_dict(job_id: str) -> dict:
    return {
        "event_id": str(uuid4()),
        "event_type": "job.progress",
        "timestamp": datetime.now(UTC).isoformat(),
        "component": "worker",
        "level": "info",
        "job_id": job_id,
        "message": "Processing",
        "payload": {},
        "correlation_id": None,
    }


class _JobServiceStub:
    def __init__(self) -> None:
        self.list_calls: list[dict[str, object]] = []
        self.jobs: dict[UUID, SimpleNamespace] = {}
        self.events: dict[UUID, list[SimpleNamespace]] = {}

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

    async def request_cancellation(self, job_id: UUID) -> str:
        job = self.jobs.get(job_id)
        if not job:
            raise JobNotFoundError(f"Job {job_id} not found")
        if job.status in ("completed", "failed", "cancelled"):
            raise InvalidStateTransitionError(f"Cannot cancel job in '{job.status}' state")
        if job.status == "pending":
            job.status = "cancelled"
            return "cancelled"
        # running — flag requested, status unchanged
        return job.status

    async def retry_job(self, job_id: UUID) -> SimpleNamespace:
        job = self.jobs.get(job_id)
        if not job:
            raise JobNotFoundError(f"Job {job_id} not found")
        if job.status != "failed":
            raise InvalidStateTransitionError(
                f"Only failed jobs can be retried (current: '{job.status}')"
            )
        if job.attempt >= job.max_attempts:
            raise RetryLimitExceededError(
                f"Job has reached max attempts ({job.max_attempts})"
            )
        job.status = "pending"
        job.attempt += 1
        return job

    async def get_job_events(self, job_id: UUID) -> list[SimpleNamespace]:
        job = self.jobs.get(job_id)
        if not job:
            raise JobNotFoundError(f"Job {job_id} not found")
        return self.events.get(job_id, [])


# ── GET /jobs/ ────────────────────────────────────────────────────────────────


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
async def test_list_jobs_default_params(client, app) -> None:
    service = _JobServiceStub()
    app.dependency_overrides[jobs._get_job_service] = lambda: service

    response = await client.get("/api/v1/jobs/")

    assert response.status_code == 200
    assert service.list_calls[0]["status"] is None
    assert service.list_calls[0]["job_type"] is None
    assert service.list_calls[0]["limit"] == 50
    assert service.list_calls[0]["offset"] == 0


@pytest.mark.asyncio
async def test_list_jobs_pagination_has_more_flag(client, app) -> None:
    service = _JobServiceStub()
    for _ in range(3):
        job = _job_record()
        service.jobs[job.id] = job
    app.dependency_overrides[jobs._get_job_service] = lambda: service

    body = (await client.get("/api/v1/jobs/", params={"limit": 2})).json()
    assert body["total"] == 3
    assert body["has_more"] is True


@pytest.mark.asyncio
async def test_list_jobs_has_more_false_when_all_returned(client, app) -> None:
    service = _JobServiceStub()
    job = _job_record()
    service.jobs[job.id] = job
    app.dependency_overrides[jobs._get_job_service] = lambda: service

    body = (await client.get("/api/v1/jobs/")).json()
    assert body["has_more"] is False


# ── GET /jobs/{job_id} ────────────────────────────────────────────────────────


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


# ── POST /jobs/{job_id}/cancel ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_pending_job_returns_200_cancelled(client, app) -> None:
    service = _JobServiceStub()
    job = _job_record(status="pending")
    service.jobs[job.id] = job
    app.dependency_overrides[jobs._get_job_service] = lambda: service

    response = await client.post(f"/api/v1/jobs/{job.id}/cancel")

    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] == str(job.id)
    assert body["status"] == "cancelled"
    assert "message" in body


@pytest.mark.asyncio
async def test_cancel_running_job_returns_200_still_running(client, app) -> None:
    """Running jobs cannot be immediately cancelled; status stays 'running'."""
    service = _JobServiceStub()
    job = _job_record(status="running")
    service.jobs[job.id] = job
    app.dependency_overrides[jobs._get_job_service] = lambda: service

    response = await client.post(f"/api/v1/jobs/{job.id}/cancel")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "running"


@pytest.mark.asyncio
async def test_cancel_unknown_job_returns_404(client, app) -> None:
    service = _JobServiceStub()
    app.dependency_overrides[jobs._get_job_service] = lambda: service

    response = await client.post(f"/api/v1/jobs/{uuid4()}/cancel")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_cancel_completed_job_returns_409(client, app) -> None:
    service = _JobServiceStub()
    job = _job_record(status="completed")
    service.jobs[job.id] = job
    app.dependency_overrides[jobs._get_job_service] = lambda: service

    response = await client.post(f"/api/v1/jobs/{job.id}/cancel")

    assert response.status_code == 409


@pytest.mark.asyncio
async def test_cancel_failed_job_returns_409(client, app) -> None:
    service = _JobServiceStub()
    job = _job_record(status="failed")
    service.jobs[job.id] = job
    app.dependency_overrides[jobs._get_job_service] = lambda: service

    response = await client.post(f"/api/v1/jobs/{job.id}/cancel")

    assert response.status_code == 409


@pytest.mark.asyncio
async def test_cancel_already_cancelled_job_returns_409(client, app) -> None:
    service = _JobServiceStub()
    job = _job_record(status="cancelled")
    service.jobs[job.id] = job
    app.dependency_overrides[jobs._get_job_service] = lambda: service

    response = await client.post(f"/api/v1/jobs/{job.id}/cancel")

    assert response.status_code == 409


@pytest.mark.asyncio
async def test_cancel_invalid_uuid_returns_422(client) -> None:
    response = await client.post("/api/v1/jobs/not-a-uuid/cancel")
    assert response.status_code == 422


# ── POST /jobs/{job_id}/retry ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retry_failed_job_returns_200_pending(client, app) -> None:
    service = _JobServiceStub()
    job = _job_record(status="failed", attempt=1, max_attempts=3)
    service.jobs[job.id] = job
    app.dependency_overrides[jobs._get_job_service] = lambda: service

    response = await client.post(f"/api/v1/jobs/{job.id}/retry")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "pending"
    assert body["id"] == str(job.id)


@pytest.mark.asyncio
async def test_retry_increments_attempt_counter(client, app) -> None:
    service = _JobServiceStub()
    job = _job_record(status="failed", attempt=1, max_attempts=3)
    service.jobs[job.id] = job
    app.dependency_overrides[jobs._get_job_service] = lambda: service

    await client.post(f"/api/v1/jobs/{job.id}/retry")

    assert service.jobs[job.id].attempt == 2


@pytest.mark.asyncio
async def test_retry_unknown_job_returns_404(client, app) -> None:
    service = _JobServiceStub()
    app.dependency_overrides[jobs._get_job_service] = lambda: service

    response = await client.post(f"/api/v1/jobs/{uuid4()}/retry")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_retry_pending_job_returns_409(client, app) -> None:
    service = _JobServiceStub()
    job = _job_record(status="pending")
    service.jobs[job.id] = job
    app.dependency_overrides[jobs._get_job_service] = lambda: service

    response = await client.post(f"/api/v1/jobs/{job.id}/retry")

    assert response.status_code == 409


@pytest.mark.asyncio
async def test_retry_completed_job_returns_409(client, app) -> None:
    service = _JobServiceStub()
    job = _job_record(status="completed")
    service.jobs[job.id] = job
    app.dependency_overrides[jobs._get_job_service] = lambda: service

    response = await client.post(f"/api/v1/jobs/{job.id}/retry")

    assert response.status_code == 409


@pytest.mark.asyncio
async def test_retry_exhausted_job_returns_409(client, app) -> None:
    """Job that has reached max_attempts cannot be retried."""
    service = _JobServiceStub()
    job = _job_record(status="failed", attempt=3, max_attempts=3)
    service.jobs[job.id] = job
    app.dependency_overrides[jobs._get_job_service] = lambda: service

    response = await client.post(f"/api/v1/jobs/{job.id}/retry")

    assert response.status_code == 409


@pytest.mark.asyncio
async def test_retry_invalid_uuid_returns_422(client) -> None:
    response = await client.post("/api/v1/jobs/not-a-uuid/retry")
    assert response.status_code == 422


# ── GET /jobs/{job_id}/events ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_job_events_returns_list(client, app) -> None:
    service = _JobServiceStub()
    job = _job_record(status="running")
    service.jobs[job.id] = job
    service.events[job.id] = [
        _job_event(job.id, "pending", "running"),
    ]
    app.dependency_overrides[jobs._get_job_service] = lambda: service

    response = await client.get(f"/api/v1/jobs/{job.id}/events")

    assert response.status_code == 200
    events = response.json()
    assert isinstance(events, list)
    assert len(events) == 1


@pytest.mark.asyncio
async def test_get_job_events_response_shape(client, app) -> None:
    service = _JobServiceStub()
    job = _job_record(status="completed")
    service.jobs[job.id] = job
    event = _job_event(job.id, "running", "completed")
    service.events[job.id] = [event]
    app.dependency_overrides[jobs._get_job_service] = lambda: service

    body = (await client.get(f"/api/v1/jobs/{job.id}/events")).json()

    e = body[0]
    assert e["job_id"] == str(job.id)
    assert e["from_status"] == "running"
    assert e["to_status"] == "completed"
    assert "id" in e
    assert "created_at" in e


@pytest.mark.asyncio
async def test_get_job_events_empty_when_no_transitions(client, app) -> None:
    service = _JobServiceStub()
    job = _job_record(status="pending")
    service.jobs[job.id] = job
    app.dependency_overrides[jobs._get_job_service] = lambda: service

    body = (await client.get(f"/api/v1/jobs/{job.id}/events")).json()

    assert body == []


@pytest.mark.asyncio
async def test_get_job_events_multiple_transitions(client, app) -> None:
    service = _JobServiceStub()
    job = _job_record(status="completed")
    service.jobs[job.id] = job
    service.events[job.id] = [
        _job_event(job.id, "pending", "running"),
        _job_event(job.id, "running", "completed"),
    ]
    app.dependency_overrides[jobs._get_job_service] = lambda: service

    body = (await client.get(f"/api/v1/jobs/{job.id}/events")).json()

    assert len(body) == 2


@pytest.mark.asyncio
async def test_get_job_events_unknown_job_returns_404(client, app) -> None:
    service = _JobServiceStub()
    app.dependency_overrides[jobs._get_job_service] = lambda: service

    response = await client.get(f"/api/v1/jobs/{uuid4()}/events")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_get_job_events_invalid_uuid_returns_422(client) -> None:
    response = await client.get("/api/v1/jobs/not-a-uuid/events")
    assert response.status_code == 422


# ── GET /jobs/{job_id}/timeline ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_job_timeline_503_when_observability_missing(client, app) -> None:
    """503 when app.state.observability is not initialised."""
    # Ensure observability is absent
    if hasattr(app.state, "observability"):
        del app.state.observability

    response = await client.get(f"/api/v1/jobs/{uuid4()}/timeline")

    assert response.status_code == 503
    assert "observability" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_get_job_timeline_returns_event_list(client, app) -> None:
    job_id = uuid4()
    events = [_typed_event_dict(str(job_id))]

    observability = MagicMock()
    observability.get_job_timeline.return_value = events
    app.state.observability = observability

    response = await client.get(f"/api/v1/jobs/{job_id}/timeline")

    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] == str(job_id)
    assert isinstance(body["events"], list)
    assert len(body["events"]) == 1


@pytest.mark.asyncio
async def test_get_job_timeline_empty_list_when_no_events(client, app) -> None:
    job_id = uuid4()
    observability = MagicMock()
    observability.get_job_timeline.return_value = []
    app.state.observability = observability

    body = (await client.get(f"/api/v1/jobs/{job_id}/timeline")).json()

    assert body["events"] == []


@pytest.mark.asyncio
async def test_get_job_timeline_passes_job_id_and_limit_to_observability(client, app) -> None:
    job_id = uuid4()
    observability = MagicMock()
    observability.get_job_timeline.return_value = []
    app.state.observability = observability

    await client.get(f"/api/v1/jobs/{job_id}/timeline", params={"limit": 100})

    observability.get_job_timeline.assert_called_once_with(str(job_id), limit=100)


@pytest.mark.asyncio
async def test_get_job_timeline_invalid_uuid_returns_422(client) -> None:
    response = await client.get("/api/v1/jobs/not-a-uuid/timeline")
    assert response.status_code == 422
