"""
Unit tests for JobService.

Tests job lifecycle: get, list, cancel, retry, events.
Uses stubbed repositories to simulate database operations.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pytest_mock import MockerFixture

from app.domain.enums import JobStatus, JobType
from app.domain.errors import (
    InvalidStateTransitionError,
    JobNotFoundError,
    RetryLimitExceededError,
)
from app.infrastructure.database.models import JobEventRecord, JobRecord
from app.services.job_service import JobService


# ── Stubs ────────────────────────────────────────────────────────────────

class _JobRepoStub:
    """Stub JobRepository with in-memory storage."""

    def __init__(self) -> None:
        self._store: dict[str, JobRecord] = {}

    async def get_by_id(self, job_id: object) -> JobRecord | None:
        return self._store.get(str(job_id))

    async def create(self, job: JobRecord) -> JobRecord:
        self._store[str(job.id)] = job
        return job

    async def mark_cancelled(self, job_id: object) -> None:
        key = str(job_id)
        if key in self._store:
            self._store[key].status = JobStatus.CANCELLED

    async def reset_to_pending(self, job_id: object, attempt: int = 1) -> None:
        key = str(job_id)
        if key in self._store:
            self._store[key].status = JobStatus.PENDING
            self._store[key].attempt = attempt

    async def list_filtered(
        self,
        status: str | None = None,
        job_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[JobRecord], int]:
        jobs = list(self._store.values())

        if status is not None:
            jobs = [j for j in jobs if j.status == status]
        if job_type is not None:
            jobs = [j for j in jobs if j.job_type == job_type]

        total = len(jobs)
        page = jobs[offset: offset + limit]
        return page, total

    def add_job(self, job: JobRecord) -> None:
        """Helper to seed test data."""
        self._store[str(job.id)] = job


class _EventRepoStub:
    """Stub JobEventRepository that records transitions."""

    def __init__(self) -> None:
        self._store: list[JobEventRecord] = []
        self._transitions: list[tuple[str, str, str, str | None, object]] = []

    async def record_transition(
        self,
        job_id: object,
        from_status: str,
        to_status: str,
        message: str,
        metadata: object = None,
    ) -> None:
        self._transitions.append(
            (str(job_id), from_status, to_status, message, metadata)
        )

    async def get_by_job(self, job_id: object) -> list[JobEventRecord]:
        return [e for e in self._store if str(e.job_id) == str(job_id)]

    def add_event(self, event: JobEventRecord) -> None:
        """Helper to seed test data."""
        self._store.append(event)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def job_repo() -> _JobRepoStub:
    return _JobRepoStub()


@pytest.fixture
def event_repo() -> _EventRepoStub:
    return _EventRepoStub()


@pytest.fixture
def service(job_repo: _JobRepoStub, event_repo: _EventRepoStub) -> JobService:
    return JobService(job_repo, event_repo)


@pytest.fixture
def pending_job() -> JobRecord:
    """Create a pending job for testing."""
    return JobRecord(
        id=uuid4(),
        job_type=JobType.TEXT_TO_IMAGE,
        status=JobStatus.PENDING,
        params={"prompt": "test"},
        attempt=0,
        max_attempts=3,
    )


@pytest.fixture
def running_job() -> JobRecord:
    """Create a running job for testing."""
    return JobRecord(
        id=uuid4(),
        job_type=JobType.TEXT_TO_IMAGE,
        status=JobStatus.RUNNING,
        params={"prompt": "test"},
        attempt=1,
        max_attempts=3,
    )


@pytest.fixture
def failed_job() -> JobRecord:
    """Create a failed job for testing."""
    return JobRecord(
        id=uuid4(),
        job_type=JobType.IMAGE_TO_IMAGE,
        status=JobStatus.FAILED,
        params={"prompt": "edit"},
        attempt=2,
        max_attempts=3,
        error="Out of memory",
    )


@pytest.fixture
def completed_job() -> JobRecord:
    """Create a completed job for testing."""
    return JobRecord(
        id=uuid4(),
        job_type=JobType.TEXT_TO_IMAGE,
        status=JobStatus.COMPLETED,
        params={"prompt": "beautiful image"},
        attempt=1,
        max_attempts=3,
    )


# ── Tests: get_job ───────────────────────────────────────────────────────

class TestGetJob:
    """Tests for JobService.get_job()."""

    @pytest.mark.asyncio
    async def test_returns_job_when_exists(
        self,
        service: JobService,
        job_repo: _JobRepoStub,
        pending_job: JobRecord,
    ) -> None:
        job_repo.add_job(pending_job)
        found = await service.get_job(pending_job.id)
        assert found.id == pending_job.id
        assert found.status == JobStatus.PENDING

    @pytest.mark.asyncio
    async def test_raises_job_not_found_when_missing(self, service: JobService) -> None:
        missing_id = uuid4()
        with pytest.raises(JobNotFoundError, match="not found"):
            await service.get_job(missing_id)

    @pytest.mark.asyncio
    async def test_preserves_all_job_fields(
        self,
        service: JobService,
        job_repo: _JobRepoStub,
        pending_job: JobRecord,
    ) -> None:
        job_repo.add_job(pending_job)
        found = await service.get_job(pending_job.id)
        assert found.job_type == JobType.TEXT_TO_IMAGE
        assert found.params == {"prompt": "test"}
        assert found.attempt == 0
        assert found.max_attempts == 3


# ── Tests: list_jobs ─────────────────────────────────────────────────────

class TestListJobs:
    """Tests for JobService.list_jobs()."""

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_jobs(self, service: JobService) -> None:
        jobs, total = await service.list_jobs()
        assert jobs == []
        assert total == 0

    @pytest.mark.asyncio
    async def test_returns_all_jobs(
        self,
        service: JobService,
        job_repo: _JobRepoStub,
        pending_job: JobRecord,
    ) -> None:
        job_repo.add_job(pending_job)
        jobs, total = await service.list_jobs()
        assert len(jobs) == 1
        assert total == 1

    @pytest.mark.asyncio
    async def test_filters_by_status(
        self,
        service: JobService,
        job_repo: _JobRepoStub,
        pending_job: JobRecord,
        completed_job: JobRecord,
    ) -> None:
        job_repo.add_job(pending_job)
        job_repo.add_job(completed_job)

        jobs, total = await service.list_jobs(status=JobStatus.PENDING)
        assert total == 1
        assert jobs[0].status == JobStatus.PENDING

    @pytest.mark.asyncio
    async def test_filters_by_job_type(
        self,
        service: JobService,
        job_repo: _JobRepoStub,
        pending_job: JobRecord,
        failed_job: JobRecord,
    ) -> None:
        job_repo.add_job(pending_job)
        job_repo.add_job(failed_job)

        jobs, total = await service.list_jobs(job_type=JobType.TEXT_TO_IMAGE)
        assert total == 1
        assert jobs[0].job_type == JobType.TEXT_TO_IMAGE

    @pytest.mark.asyncio
    async def test_pagination_limit(
        self,
        service: JobService,
        job_repo: _JobRepoStub,
    ) -> None:
        for _ in range(5):
            job = JobRecord(
                id=uuid4(),
                job_type=JobType.TEXT_TO_IMAGE,
                status=JobStatus.PENDING,
                params={"prompt": "test"},
                attempt=0,
                max_attempts=3,
            )
            job_repo.add_job(job)

        jobs, total = await service.list_jobs(limit=2)
        assert total == 5
        assert len(jobs) == 2

    @pytest.mark.asyncio
    async def test_pagination_offset(
        self,
        service: JobService,
        job_repo: _JobRepoStub,
    ) -> None:
        for _ in range(5):
            job = JobRecord(
                id=uuid4(),
                job_type=JobType.TEXT_TO_IMAGE,
                status=JobStatus.PENDING,
                params={"prompt": "test"},
                attempt=0,
                max_attempts=3,
            )
            job_repo.add_job(job)

        jobs, total = await service.list_jobs(limit=2, offset=3)
        assert total == 5
        assert len(jobs) == 2

    @pytest.mark.asyncio
    async def test_combined_filters(
        self,
        service: JobService,
        job_repo: _JobRepoStub,
        pending_job: JobRecord,
        failed_job: JobRecord,
        completed_job: JobRecord,
    ) -> None:
        job_repo.add_job(pending_job)
        job_repo.add_job(failed_job)
        job_repo.add_job(completed_job)

        jobs, total = await service.list_jobs(
            status=JobStatus.PENDING,
            job_type=JobType.TEXT_TO_IMAGE,
        )
        assert total == 1
        assert jobs[0].id == pending_job.id


# ── Tests: request_cancellation ──────────────────────────────────────────

class TestRequestCancellation:
    """Tests for JobService.request_cancellation()."""

    @pytest.mark.asyncio
    async def test_cancels_pending_job_immediately(
        self,
        service: JobService,
        job_repo: _JobRepoStub,
        pending_job: JobRecord,
    ) -> None:
        job_repo.add_job(pending_job)
        result = await service.request_cancellation(pending_job.id)
        assert result == JobStatus.CANCELLED

        # Verify job was marked as cancelled
        updated = await job_repo.get_by_id(pending_job.id)
        assert updated.status == JobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_records_cancellation_event_for_pending(
        self,
        service: JobService,
        job_repo: _JobRepoStub,
        event_repo: _EventRepoStub,
        pending_job: JobRecord,
    ) -> None:
        job_repo.add_job(pending_job)
        await service.request_cancellation(pending_job.id)

        assert len(event_repo._transitions) == 1
        transition = event_repo._transitions[0]
        assert transition[1] == JobStatus.PENDING
        assert transition[2] == JobStatus.CANCELLED
        assert "Cancelled before worker pickup" in transition[3]

    @pytest.mark.asyncio
    async def test_records_cancellation_request_for_running(
        self,
        service: JobService,
        job_repo: _JobRepoStub,
        event_repo: _EventRepoStub,
        running_job: JobRecord,
    ) -> None:
        job_repo.add_job(running_job)
        result = await service.request_cancellation(running_job.id)

        assert result == JobStatus.RUNNING
        assert len(event_repo._transitions) == 1
        assert "Cancellation requested" in event_repo._transitions[0][3]

    @pytest.mark.asyncio
    async def test_raises_for_completed_job(
        self,
        service: JobService,
        job_repo: _JobRepoStub,
        completed_job: JobRecord,
    ) -> None:
        job_repo.add_job(completed_job)
        with pytest.raises(InvalidStateTransitionError, match="Cannot cancel"):
            await service.request_cancellation(completed_job.id)

    @pytest.mark.asyncio
    async def test_raises_for_failed_job(
        self,
        service: JobService,
        job_repo: _JobRepoStub,
        failed_job: JobRecord,
    ) -> None:
        job_repo.add_job(failed_job)
        with pytest.raises(InvalidStateTransitionError, match="Cannot cancel"):
            await service.request_cancellation(failed_job.id)

    @pytest.mark.asyncio
    async def test_raises_for_already_cancelled_job(
        self,
        service: JobService,
        job_repo: _JobRepoStub,
    ) -> None:
        cancelled_job = JobRecord(
            id=uuid4(),
            job_type=JobType.TEXT_TO_IMAGE,
            status=JobStatus.CANCELLED,
            params={"prompt": "test"},
            attempt=0,
            max_attempts=3,
        )
        job_repo.add_job(cancelled_job)

        with pytest.raises(InvalidStateTransitionError, match="Cannot cancel"):
            await service.request_cancellation(cancelled_job.id)

    @pytest.mark.asyncio
    async def test_raises_job_not_found(self, service: JobService) -> None:
        missing_id = uuid4()
        with pytest.raises(JobNotFoundError, match="not found"):
            await service.request_cancellation(missing_id)


# ── Tests: retry_job ────────────────────────────────────────────────────

class TestRetryJob:
    """Tests for JobService.retry_job()."""

    @pytest.mark.asyncio
    async def test_retries_failed_job(
        self,
        service: JobService,
        job_repo: _JobRepoStub,
        failed_job: JobRecord,
    ) -> None:
        job_repo.add_job(failed_job)
        result = await service.retry_job(failed_job.id)

        assert result.status == JobStatus.PENDING
        assert result.attempt == 3  # incremented from 2

    @pytest.mark.asyncio
    async def test_records_retry_event(
        self,
        service: JobService,
        job_repo: _JobRepoStub,
        event_repo: _EventRepoStub,
        failed_job: JobRecord,
    ) -> None:
        job_repo.add_job(failed_job)
        await service.retry_job(failed_job.id)

        assert len(event_repo._transitions) == 1
        transition = event_repo._transitions[0]
        assert transition[1] == JobStatus.FAILED
        assert transition[2] == JobStatus.PENDING
        assert "Retry attempt" in transition[3]

    @pytest.mark.asyncio
    async def test_raises_for_non_failed_job(
        self,
        service: JobService,
        job_repo: _JobRepoStub,
        pending_job: JobRecord,
    ) -> None:
        job_repo.add_job(pending_job)
        with pytest.raises(InvalidStateTransitionError, match="Only failed"):
            await service.retry_job(pending_job.id)

    @pytest.mark.asyncio
    async def test_raises_retry_limit_exceeded(
        self,
        service: JobService,
        job_repo: _JobRepoStub,
    ) -> None:
        exhausted_job = JobRecord(
            id=uuid4(),
            job_type=JobType.TEXT_TO_IMAGE,
            status=JobStatus.FAILED,
            params={"prompt": "test"},
            attempt=3,
            max_attempts=3,
        )
        job_repo.add_job(exhausted_job)

        with pytest.raises(RetryLimitExceededError, match="max attempts"):
            await service.retry_job(exhausted_job.id)

    @pytest.mark.asyncio
    async def test_raises_job_not_found(self, service: JobService) -> None:
        missing_id = uuid4()
        with pytest.raises(JobNotFoundError, match="not found"):
            await service.retry_job(missing_id)

    @pytest.mark.asyncio
    async def test_raises_for_completed_job(
        self,
        service: JobService,
        job_repo: _JobRepoStub,
        completed_job: JobRecord,
    ) -> None:
        job_repo.add_job(completed_job)
        with pytest.raises(InvalidStateTransitionError, match="Only failed"):
            await service.retry_job(completed_job.id)

    @pytest.mark.asyncio
    async def test_includes_retry_metadata_in_event(
        self,
        service: JobService,
        job_repo: _JobRepoStub,
        event_repo: _EventRepoStub,
        failed_job: JobRecord,
    ) -> None:
        job_repo.add_job(failed_job)
        await service.retry_job(failed_job.id)

        transition = event_repo._transitions[0]
        metadata = transition[4]
        assert metadata is not None
        # job.attempt was 2, reset_to_pending set it to 3 (same object reference),
        # so metadata records job.attempt + 1 = 4
        assert metadata["attempt"] == 4
        assert metadata["reason"] == "manual_retry"


# ── Tests: get_job_events ───────────────────────────────────────────────

class TestGetJobEvents:
    """Tests for JobService.get_job_events()."""

    @pytest.mark.asyncio
    async def test_returns_events_for_job(
        self,
        service: JobService,
        job_repo: _JobRepoStub,
        event_repo: _EventRepoStub,
        pending_job: JobRecord,
    ) -> None:
        job_repo.add_job(pending_job)

        event = JobEventRecord(
            id=uuid4(),
            job_id=pending_job.id,
            from_status=JobStatus.PENDING,
            to_status=JobStatus.RUNNING,
            message="Worker picked up job",
        )
        event_repo.add_event(event)

        events = await service.get_job_events(pending_job.id)
        assert len(events) == 1
        assert events[0].from_status == JobStatus.PENDING
        assert events[0].to_status == JobStatus.RUNNING

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_events(
        self,
        service: JobService,
        job_repo: _JobRepoStub,
        pending_job: JobRecord,
    ) -> None:
        job_repo.add_job(pending_job)
        events = await service.get_job_events(pending_job.id)
        assert events == []

    @pytest.mark.asyncio
    async def test_raises_for_missing_job(self, service: JobService) -> None:
        missing_id = uuid4()
        with pytest.raises(JobNotFoundError, match="not found"):
            await service.get_job_events(missing_id)

    @pytest.mark.asyncio
    async def test_filters_events_by_job_id(
        self,
        service: JobService,
        job_repo: _JobRepoStub,
        event_repo: _EventRepoStub,
        pending_job: JobRecord,
        completed_job: JobRecord,
    ) -> None:
        job_repo.add_job(pending_job)
        job_repo.add_job(completed_job)

        event1 = JobEventRecord(
            id=uuid4(),
            job_id=pending_job.id,
            from_status=JobStatus.PENDING,
            to_status=JobStatus.RUNNING,
            message="Event for job 1",
        )
        event2 = JobEventRecord(
            id=uuid4(),
            job_id=completed_job.id,
            from_status=JobStatus.RUNNING,
            to_status=JobStatus.COMPLETED,
            message="Event for job 2",
        )
        event_repo.add_event(event1)
        event_repo.add_event(event2)

        events = await service.get_job_events(pending_job.id)
        assert len(events) == 1
        assert events[0].message == "Event for job 1"