"""
Integration tests for database repositories.

Tests direct CRUD operations on all repository classes against the
SQLite in-memory test database to verify data access layer correctness.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.database.models import (
    ArtifactRecord,
    JobEventRecord,
    JobRecord,
    ModelFileRecord,
    ModelRecord,
)
from app.infrastructure.database.repositories.artifact_repository import ArtifactRepository
from app.infrastructure.database.repositories.job_event_repository import JobEventRepository
from app.infrastructure.database.repositories.job_repository import JobRepository
from app.infrastructure.database.repositories.model_file_repository import ModelFileRepository
from app.infrastructure.database.repositories.model_repository import ModelRepository

pytestmark = pytest.mark.integration


# ─── ModelRepository Tests ───────────────────────────────────────────────────


class TestModelRepository:
    """Tests for ModelRepository CRUD operations."""

    @pytest.fixture
    def repo(self, async_db: AsyncSession) -> ModelRepository:
        return ModelRepository(async_db)

    @pytest.fixture
    def sample_model(self) -> ModelRecord:
        return ModelRecord(
            model_id="test-model-001",
            name="Test Model",
            source="huggingface",
            family="stable-diffusion",
            variant="v1-5",
            description="A test model",
            tags=["test", "sd"],
            status="available",
            capabilities=["text-to-image"],
        )

    async def test_create_and_get_by_id(
        self, repo: ModelRepository, sample_model: ModelRecord
    ) -> None:
        """Create a model and retrieve it by internal UUID."""
        created = await repo.create(sample_model)
        fetched = await repo.get_by_id(created.id)
        assert fetched is not None
        assert fetched.model_id == "test-model-001"
        assert fetched.name == "Test Model"
        assert fetched.status == "available"

    async def test_get_by_model_id(
        self, repo: ModelRepository, sample_model: ModelRecord
    ) -> None:
        """Retrieve a model by external model_id."""
        created = await repo.create(sample_model)
        fetched = await repo.get_by_model_id("test-model-001")
        assert fetched is not None
        assert fetched.id == created.id

    async def test_get_by_model_id_not_found(self, repo: ModelRepository) -> None:
        """Querying a non-existent model_id returns None."""
        result = await repo.get_by_model_id("nonexistent")
        assert result is None

    async def test_get_by_id_not_found(self, repo: ModelRepository) -> None:
        """Querying a non-existent UUID returns None."""
        import uuid
        result = await repo.get_by_id(uuid.uuid4())
        assert result is None

    async def test_list_all(self, repo: ModelRepository) -> None:
        """List all models."""
        m1 = ModelRecord(model_id="m1", name="Model 1", source="hf", family="sd")
        m2 = ModelRecord(model_id="m2", name="Model 2", source="civitai", family="sdxl")
        await repo.create(m1)
        await repo.create(m2)
        all_models = await repo.list_all()
        assert len(all_models) == 2

    async def test_list_all_filtered_by_source(self, repo: ModelRepository) -> None:
        """List models filtered by source."""
        m1 = ModelRecord(model_id="m1", name="M1", source="hf", family="sd")
        m2 = ModelRecord(model_id="m2", name="M2", source="civitai", family="sd")
        m3 = ModelRecord(model_id="m3", name="M3", source="hf", family="sdxl")
        await repo.create(m1)
        await repo.create(m2)
        await repo.create(m3)
        hf_models = await repo.list_all(source="hf")
        assert len(hf_models) == 2

    async def test_list_all_empty_when_no_models(self, repo: ModelRepository) -> None:
        result = await repo.list_all()
        assert result == []

    async def test_update_status(
        self, repo: ModelRepository, sample_model: ModelRecord
    ) -> None:
        """Update model status atomically."""
        await repo.create(sample_model)
        await repo.update_status("test-model-001", "downloading")
        fetched = await repo.get_by_model_id("test-model-001")
        assert fetched is not None
        assert fetched.status == "downloading"

    async def test_delete(self, repo: ModelRepository, sample_model: ModelRecord) -> None:
        """Delete a model by internal UUID."""
        created = await repo.create(sample_model)
        await repo.delete(created.id)
        await repo._session.commit()
        deleted = await repo.get_by_id(created.id)
        assert deleted is None


# ─── JobRepository Tests ────────────────────────────────────────────────────


class TestJobRepository:
    """Tests for JobRepository CRUD operations."""

    @pytest.fixture
    def repo(self, async_db: AsyncSession) -> JobRepository:
        return JobRepository(async_db)

    @pytest.fixture
    def sample_job(self) -> JobRecord:
        return JobRecord(
            job_type="text-to-image",
            status="pending",
            progress_percent=0,
            params={"prompt": "a test", "model_id": "test-model"},
        )

    async def test_create_job(self, repo: JobRepository, sample_job: JobRecord) -> None:
        """Create a new job record."""
        created = await repo.create(sample_job)
        assert created.id is not None
        assert created.status == "pending"

    async def test_get_job_by_id(self, repo: JobRepository, sample_job: JobRecord) -> None:
        """Retrieve a job by UUID."""
        created = await repo.create(sample_job)
        fetched = await repo.get_by_id(created.id)
        assert fetched is not None
        assert fetched.job_type == "text-to-image"

    async def test_get_job_by_id_not_found(self, repo: JobRepository) -> None:
        import uuid
        result = await repo.get_by_id(uuid.uuid4())
        assert result is None

    async def test_list_filtered_pending(self, repo: JobRepository) -> None:
        """List pending jobs with filtering."""
        j1 = JobRecord(job_type="text-to-image", status="pending", priority=10)
        j2 = JobRecord(job_type="image-to-image", status="pending", priority=5)
        j3 = JobRecord(job_type="text-to-image", status="running")
        await repo.create(j1)
        await repo.create(j2)
        await repo.create(j3)
        pending, total = await repo.list_filtered(status="pending")
        assert len(pending) == 2
        assert total == 2

    async def test_list_filtered_by_job_type(self, repo: JobRepository) -> None:
        """Filter jobs by job_type."""
        j1 = JobRecord(job_type="text-to-image", status="pending")
        j2 = JobRecord(job_type="model-download", status="pending")
        await repo.create(j1)
        await repo.create(j2)
        results, total = await repo.list_filtered(job_type="text-to-image")
        assert total == 1
        assert results[0].job_type == "text-to-image"

    async def test_list_filtered_pagination(self, repo: JobRepository) -> None:
        """Pagination with limit and offset."""
        for _ in range(5):
            await repo.create(JobRecord(job_type="text-to-image", status="pending"))
        page1, total = await repo.list_filtered(limit=2, offset=0)
        assert total == 5
        assert len(page1) == 2

    async def test_update_progress(self, repo: JobRepository, sample_job: JobRecord) -> None:
        """Update job progress."""
        created = await repo.create(sample_job)
        await repo.update_progress(created.id, progress_percent=50, message="Processing...")
        fetched = await repo.get_by_id(created.id)
        assert fetched.progress_percent == 50
        assert fetched.message == "Processing..."

    async def test_mark_completed(self, repo: JobRepository, sample_job: JobRecord) -> None:
        """Mark a job as completed."""
        created = await repo.create(sample_job)
        await repo.mark_completed(created.id, result={"artifact_ids": ["abc-123"]})
        fetched = await repo.get_by_id(created.id)
        assert fetched.status == "completed"
        assert fetched.result == {"artifact_ids": ["abc-123"]}

    async def test_mark_failed(self, repo: JobRepository, sample_job: JobRecord) -> None:
        """Mark a job as failed."""
        created = await repo.create(sample_job)
        await repo.mark_failed(created.id, error="Something went wrong")
        fetched = await repo.get_by_id(created.id)
        assert fetched.status == "failed"
        assert fetched.error == "Something went wrong"

    async def test_mark_cancelled(self, repo: JobRepository, sample_job: JobRecord) -> None:
        """Mark a job as cancelled."""
        created = await repo.create(sample_job)
        await repo.mark_cancelled(created.id)
        fetched = await repo.get_by_id(created.id)
        assert fetched.status == "cancelled"

    async def test_list_recent(self, repo: JobRepository, sample_job: JobRecord) -> None:
        """List recent jobs."""
        await repo.create(sample_job)
        recent = await repo.list_recent(limit=50)
        assert len(recent) >= 1

    async def test_queue_counts(self, repo: JobRepository) -> None:
        """Get queue status counts."""
        j1 = JobRecord(job_type="text-to-image", status="pending")
        j2 = JobRecord(job_type="text-to-image", status="running")
        j3 = JobRecord(job_type="text-to-image", status="completed")
        await repo.create(j1)
        await repo.create(j2)
        await repo.create(j3)
        counts = await repo.get_queue_counts()
        assert counts["pending"] == 1
        assert counts["running"] == 1
        assert counts["completed"] == 1
        assert counts["queue_depth"] == 2

    async def test_reset_to_pending(self, repo: JobRepository, sample_job: JobRecord) -> None:
        """Reset a failed job to pending."""
        created = await repo.create(sample_job)
        await repo.mark_failed(created.id, error="Failed")
        await repo.reset_to_pending(created.id, attempt=2)
        fetched = await repo.get_by_id(created.id)
        assert fetched.status == "pending"
        assert fetched.attempt == 2


# ─── ArtifactRepository Tests ────────────────────────────────────────────────


class TestArtifactRepository:
    """Tests for ArtifactRepository CRUD operations."""

    @pytest.fixture
    def repo(self, async_db: AsyncSession) -> ArtifactRepository:
        return ArtifactRepository(async_db)

    @pytest.fixture
    async def sample_job(self, async_db: AsyncSession) -> JobRecord:
        """Create a job to associate artifacts with."""
        job = JobRecord(job_type="text-to-image", status="completed")
        async_db.add(job)
        await async_db.flush()
        return job

    async def test_create_artifact(
        self, repo: ArtifactRepository, sample_job: JobRecord
    ) -> None:
        """Create a new artifact record."""
        art = ArtifactRecord(
            job_id=sample_job.id,
            file_path="/output/test.png",
            media_type="image/png",
            size_bytes=12345,
            width=512,
            height=512,
            prompt="a test image",
            seed=42,
        )
        created = await repo.create(art)
        assert created.id is not None
        assert created.file_path == "/output/test.png"

    async def test_get_by_id(
        self, repo: ArtifactRepository, sample_job: JobRecord
    ) -> None:
        """Retrieve artifact by UUID."""
        art = ArtifactRecord(job_id=sample_job.id, file_path="/out/get.png")
        created = await repo.create(art)
        fetched = await repo.get_by_id(created.id)
        assert fetched is not None
        assert fetched.id == created.id

    async def test_get_by_id_not_found(self, repo: ArtifactRepository) -> None:
        import uuid
        result = await repo.get_by_id(uuid.uuid4())
        assert result is None

    async def test_list_by_job(
        self, repo: ArtifactRepository, sample_job: JobRecord
    ) -> None:
        """Get all artifacts for a specific job."""
        art1 = ArtifactRecord(job_id=sample_job.id, file_path="/out/1.png", seed=1)
        art2 = ArtifactRecord(job_id=sample_job.id, file_path="/out/2.png", seed=2)
        await repo.create(art1)
        await repo.create(art2)
        artifacts = await repo.list_by_job(sample_job.id)
        assert len(artifacts) == 2

    async def test_list_recent(
        self, repo: ArtifactRepository, sample_job: JobRecord
    ) -> None:
        """List most recent artifacts."""
        art1 = ArtifactRecord(job_id=sample_job.id, file_path="/out/recent1.png")
        art2 = ArtifactRecord(job_id=sample_job.id, file_path="/out/recent2.png")
        await repo.create(art1)
        await repo.create(art2)
        recent = await repo.list_recent(limit=10)
        assert len(recent) == 2

    async def test_list_filtered_by_favorite(
        self, repo: ArtifactRepository, sample_job: JobRecord
    ) -> None:
        """Filter artifacts by favorite status."""
        art1 = ArtifactRecord(
            job_id=sample_job.id, file_path="/out/fav.png",
            is_favorite=True, media_type="image/png",
        )
        art2 = ArtifactRecord(
            job_id=sample_job.id, file_path="/out/normal.png",
            is_favorite=False, media_type="image/png",
        )
        await repo.create(art1)
        await repo.create(art2)
        favorites, total = await repo.list_filtered(is_favorite=True)
        assert len(favorites) == 1
        assert total == 1

    async def test_list_filtered_by_media_type(
        self, repo: ArtifactRepository, sample_job: JobRecord
    ) -> None:
        """Filter artifacts by media type."""
        art1 = ArtifactRecord(
            job_id=sample_job.id, file_path="/out/img.png", media_type="image/png"
        )
        art2 = ArtifactRecord(
            job_id=sample_job.id, file_path="/out/vid.mp4", media_type="video/mp4"
        )
        await repo.create(art1)
        await repo.create(art2)
        png_artifacts, png_total = await repo.list_filtered(media_type="image/png")
        assert len(png_artifacts) == 1
        assert png_total == 1

    async def test_update_artifact_fields(
        self, repo: ArtifactRepository, sample_job: JobRecord
    ) -> None:
        """Update specific fields on an artifact."""
        art = ArtifactRecord(
            job_id=sample_job.id, file_path="/out/update.png", rating=0, is_favorite=False
        )
        created = await repo.create(art)
        await repo.update(created.id, rating=5, is_favorite=True)
        fetched = await repo.get_by_id(created.id)
        assert fetched.rating == 5
        assert fetched.is_favorite is True

    async def test_delete_artifact(
        self, repo: ArtifactRepository, sample_job: JobRecord
    ) -> None:
        """Delete an artifact record."""
        art = ArtifactRecord(job_id=sample_job.id, file_path="/out/delete.png")
        created = await repo.create(art)
        await repo.delete(created.id)
        await repo._session.commit()
        deleted = await repo.get_by_id(created.id)
        assert deleted is None

    async def test_delete_all_artifacts(
        self, repo: ArtifactRepository, sample_job: JobRecord
    ) -> None:
        """Delete all artifact records."""
        art1 = ArtifactRecord(job_id=sample_job.id, file_path="/out/all1.png")
        art2 = ArtifactRecord(job_id=sample_job.id, file_path="/out/all2.png")
        await repo.create(art1)
        await repo.create(art2)
        await repo.delete_all()
        await repo._session.commit()
        remaining = await repo.list_recent()
        assert len(remaining) == 0


# ─── JobEventRepository Tests ────────────────────────────────────────────────


class TestJobEventRepository:
    """Tests for JobEventRepository (append-only audit log)."""

    @pytest.fixture
    def repo(self, async_db: AsyncSession) -> JobEventRepository:
        return JobEventRepository(async_db)

    @pytest.fixture
    async def sample_job(self, async_db: AsyncSession) -> JobRecord:
        job = JobRecord(job_type="text-to-image", status="pending")
        async_db.add(job)
        await async_db.flush()
        return job

    async def test_create_event(
        self, repo: JobEventRepository, sample_job: JobRecord
    ) -> None:
        """Create a job event entry."""
        event = JobEventRecord(
            job_id=sample_job.id,
            from_status="pending",
            to_status="running",
            message="Worker picked up job",
            created_at=datetime.now(timezone.utc),
        )
        created = await repo.create(event)
        assert created.id is not None
        assert created.to_status == "running"

    async def test_get_by_job(
        self, repo: JobEventRepository, sample_job: JobRecord
    ) -> None:
        """Get all events for a job in chronological order."""
        e1 = JobEventRecord(
            job_id=sample_job.id,
            from_status="pending",
            to_status="running",
            created_at=datetime.now(timezone.utc),
        )
        e2 = JobEventRecord(
            job_id=sample_job.id,
            from_status="running",
            to_status="completed",
            created_at=datetime.now(timezone.utc),
        )
        await repo.create(e1)
        await repo.create(e2)
        events = await repo.get_by_job(sample_job.id)
        assert len(events) == 2
        assert events[0].to_status == "running"
        assert events[1].to_status == "completed"

    async def test_get_by_job_empty_when_no_events(
        self, repo: JobEventRepository, sample_job: JobRecord
    ) -> None:
        events = await repo.get_by_job(sample_job.id)
        assert events == []

    async def test_record_transition(
        self, repo: JobEventRepository, sample_job: JobRecord
    ) -> None:
        """Record a job state transition using helper method."""
        event = await repo.record_transition(
            job_id=sample_job.id,
            from_status="pending",
            to_status="running",
            message="Worker started",
            metadata={"worker_id": "test-worker"},
        )
        assert event.id is not None
        assert event.to_status == "running"
        assert event.event_metadata["worker_id"] == "test-worker"

    async def test_get_recent_events(
        self, repo: JobEventRepository, sample_job: JobRecord
    ) -> None:
        """Get most recent events across all jobs."""
        e1 = JobEventRecord(
            job_id=sample_job.id,
            from_status="pending",
            to_status="running",
            created_at=datetime.now(timezone.utc),
        )
        e2 = JobEventRecord(
            job_id=sample_job.id,
            from_status="running",
            to_status="completed",
            created_at=datetime.now(timezone.utc),
        )
        await repo.create(e1)
        await repo.create(e2)
        recent = await repo.get_recent(limit=10)
        assert len(recent) == 2

    async def test_events_store_structured_metadata(
        self, repo: JobEventRepository, sample_job: JobRecord
    ) -> None:
        """Job events can store structured metadata."""
        event = JobEventRecord(
            job_id=sample_job.id,
            from_status="running",
            to_status="completed",
            event_metadata={"duration_ms": 1234, "images_generated": 4},
            created_at=datetime.now(timezone.utc),
        )
        created = await repo.create(event)
        assert created.event_metadata["duration_ms"] == 1234
        assert created.event_metadata["images_generated"] == 4


# ─── ModelFileRecord Tests ──────────────────────────────────────────────────


class TestModelFileRecord:
    """Tests for model file tracking (ORM-level, no dedicated repository)."""

    async def test_create_model_file(self, async_db: AsyncSession) -> None:
        """Create a model file record linked to a model."""
        model = ModelRecord(model_id="m1", name="M", source="hf", family="sd")
        async_db.add(model)
        await async_db.flush()

        mf = ModelFileRecord(
            model_id=model.id,
            relative_path="model.safetensors",
            size_bytes=2_000_000_000,
            status="downloaded",
        )
        async_db.add(mf)
        await async_db.flush()

        assert mf.id is not None
        assert mf.status == "downloaded"

    async def test_cascade_delete_model_files(self, async_db: AsyncSession) -> None:
        """Deleting a model cascades to its files."""
        model = ModelRecord(model_id="m1", name="M", source="hf", family="sd")
        async_db.add(model)
        await async_db.flush()

        mf = ModelFileRecord(
            model_id=model.id,
            relative_path="model.safetensors",
            status="downloaded",
        )
        async_db.add(mf)
        await async_db.flush()
        file_id = mf.id

        await async_db.delete(model)
        await async_db.commit()

        remaining = await async_db.get(ModelFileRecord, file_id)
        assert remaining is None

    async def test_model_can_have_multiple_files(self, async_db: AsyncSession) -> None:
        """A single model record can have multiple sharded files."""
        model = ModelRecord(model_id="sharded/model", name="Sharded", source="hf", family="flux")
        async_db.add(model)
        await async_db.flush()

        shards = [
            ModelFileRecord(
                model_id=model.id,
                relative_path=f"model-{i:05d}-of-00003.safetensors",
                size_bytes=500_000_000,
                status="downloaded",
            )
            for i in range(1, 4)
        ]
        for shard in shards:
            async_db.add(shard)
        await async_db.flush()

        for shard in shards:
            assert shard.id is not None
