"""
Unit tests for ModelService.

Tests model registration, download queuing, listing, retrieval, and deletion.
Uses mock repositories and storage manager to isolate service logic.
"""

from __future__ import annotations

from typing import Any

import logging
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.domain.enums import JobStatus, JobType, ModelStatus
from app.services.model_service import ModelService


def _make_model_record(
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
    name: str = "SDXL Base 1.0",
    source: str = "huggingface",
    status: str = "not_downloaded",
    family: str = "sdxl",
    variant: str = "",
    description: str = "Test model",
    tags: list[str] | None = None,
    capabilities: list[str] | None = None,
) -> SimpleNamespace:
    """Create a mock ModelRecord."""
    now = datetime.now(UTC)
    return SimpleNamespace(
        id=uuid4(),
        model_id=model_id,
        name=name,
        source=source,
        family=family,
        variant=variant,
        description=description,
        tags=tags or ["diffusion"],
        source_url="https://huggingface.co/test",
        capabilities=capabilities or ["text_to_image"],
        status=status,
        total_size_bytes=1000,
        disk_size_bytes=1000,
        download_size_bytes=1000,
        download_progress=0 if status == "not_downloaded" else 100,
        recommended_vram_min_gb=None,
        recommended_vram_max_gb=None,
        license="",
        base_model="",
        precision="",
        requirements={},
        notes="",
        is_verified=False,
        last_verified_at=None,
        created_at=now,
        updated_at=now,
    )


class _ModelRepoStub:
    """Stub ModelRepository for unit testing."""

    def __init__(self) -> None:
        self._records: dict[str, SimpleNamespace] = {}
        self.created_records: list[SimpleNamespace] = []
        self.deleted_ids: list[str] = []
        self.status_updates: list[tuple[str, str]] = []

    async def get_by_model_id(self, model_id: str) -> SimpleNamespace | None:
        return self._records.get(model_id)

    async def create(self, record: SimpleNamespace) -> SimpleNamespace:
        self._records[record.model_id] = record
        self.created_records.append(record)
        return record

    async def delete(self, record_id: str) -> None:
        self.deleted_ids.append(str(record_id))

    async def update_status(self, model_id: str, status: ModelStatus) -> None:
        self.status_updates.append((model_id, status))
        if model_id in self._records:
            object.__setattr__(self._records[model_id], "status", status)

    async def list_all(self, source: str | None = None) -> list[SimpleNamespace]:
        records = list(self._records.values())
        if source is not None:
            records = [r for r in records if r.source == source]
        return records


class _JobRepoStub:
    """Stub JobRepository for unit testing."""

    def __init__(self) -> None:
        self.created_jobs: list[SimpleNamespace] = []
        self._next_id: int = 0

    async def create(self, job: Any) -> Any:
        # SQLAlchemy models have id=None initially, so check for None
        if getattr(job, "id", None) is None:
            job.id = uuid4()
        self.created_jobs.append(job)
        return job


class _StorageManagerStub:
    """Stub StorageManager for unit testing."""

    def __init__(self) -> None:
        self.deleted_files: list[tuple[str, str]] = []

    def delete_model_files(self, source: str, model_id: str) -> None:
        self.deleted_files.append((source, model_id))


class TestRegisterModel:
    """Tests for ModelService.register_model()."""

    @pytest.fixture
    def model_repo(self) -> _ModelRepoStub:
        return _ModelRepoStub()

    @pytest.fixture
    def job_repo(self) -> _JobRepoStub:
        return _JobRepoStub()

    @pytest.fixture
    def storage(self) -> _StorageManagerStub:
        return _StorageManagerStub()

    @pytest.fixture
    def service(
        self, model_repo: _ModelRepoStub, job_repo: _JobRepoStub, storage: _StorageManagerStub
    ) -> ModelService:
        return ModelService(model_repo, job_repo, storage)

    @pytest.mark.asyncio
    async def test_register_model_creates_new_record(
        self, service: ModelService, model_repo: _ModelRepoStub
    ) -> None:
        record = await service.register_model(
            model_id="test/model",
            name="Test Model",
            source="huggingface",
        )
        assert record.model_id == "test/model"
        assert record.name == "Test Model"
        assert record.source == "huggingface"
        assert record.status == ModelStatus.NOT_DOWNLOADED
        assert len(model_repo.created_records) == 1

    @pytest.mark.asyncio
    async def test_register_model_with_all_fields(
        self, service: ModelService, model_repo: _ModelRepoStub
    ) -> None:
        record = await service.register_model(
            model_id="test/full-model",
            name="Full Model",
            source="civitai",
            family="sdxl",
            variant="fp16",
            description="A full test model",
            tags=["diffusion", "sdxl"],
            source_url="https://civitai.com/models/test",
            capabilities=["text_to_image", "image_to_image"],
        )
        assert record.family == "sdxl"
        assert record.variant == "fp16"
        assert record.description == "A full test model"
        assert record.tags == ["diffusion", "sdxl"]
        assert record.capabilities == ["text_to_image", "image_to_image"]

    @pytest.mark.asyncio
    async def test_register_model_defaults_for_optional_fields(
        self, service: ModelService
    ) -> None:
        record = await service.register_model(
            model_id="test/minimal",
            name="Minimal Model",
            source="local",
        )
        assert record.family == "custom"
        assert record.variant == ""
        assert record.description == ""
        assert record.tags == []
        assert record.capabilities == []

    @pytest.mark.asyncio
    async def test_register_model_returns_existing_when_duplicate(
        self, service: ModelService, model_repo: _ModelRepoStub
    ) -> None:
        existing = _make_model_record(model_id="test/duplicate", status="downloaded")
        model_repo._records["test/duplicate"] = existing

        record = await service.register_model(
            model_id="test/duplicate",
            name="Different Name",
            source="different",
        )
        assert record.model_id == "test/duplicate"
        assert record.name == existing.name
        assert len(model_repo.created_records) == 0

    @pytest.mark.asyncio
    async def test_register_model_sets_status_not_downloaded(
        self, service: ModelService, model_repo: _ModelRepoStub
    ) -> None:
        record = await service.register_model(
            model_id="test/status",
            name="Status Model",
            source="huggingface",
        )
        assert record.status == ModelStatus.NOT_DOWNLOADED

    @pytest.mark.asyncio
    async def test_register_model_empty_tags_becomes_empty_list(
        self, service: ModelService
    ) -> None:
        record = await service.register_model(
            model_id="test/empty-tags",
            name="Empty Tags",
            source="huggingface",
            tags=[],
        )
        assert record.tags == []

    @pytest.mark.asyncio
    async def test_register_model_empty_capabilities_becomes_empty_list(
        self, service: ModelService
    ) -> None:
        record = await service.register_model(
            model_id="test/empty-cap",
            name="Empty Cap",
            source="huggingface",
            capabilities=[],
        )
        assert record.capabilities == []


class TestRequestDownload:
    """Tests for ModelService.request_download()."""

    @pytest.fixture
    def model_repo(self) -> _ModelRepoStub:
        return _ModelRepoStub()

    @pytest.fixture
    def job_repo(self) -> _JobRepoStub:
        return _JobRepoStub()

    @pytest.fixture
    def storage(self) -> _StorageManagerStub:
        return _StorageManagerStub()

    @pytest.fixture
    def service(
        self, model_repo: _ModelRepoStub, job_repo: _JobRepoStub, storage: _StorageManagerStub
    ) -> ModelService:
        model = _make_model_record(model_id="test/model", status="not_downloaded")
        model_repo._records["test/model"] = model
        return ModelService(model_repo, job_repo, storage)

    @pytest.mark.asyncio
    async def test_request_download_creates_job(
        self, service: ModelService, job_repo: _JobRepoStub
    ) -> None:
        job_id = await service.request_download("test/model")
        assert job_id is not None
        assert len(job_repo.created_jobs) == 1
        job = job_repo.created_jobs[0]
        assert job.job_type == JobType.MODEL_DOWNLOAD
        assert job.status == JobStatus.PENDING

    @pytest.mark.asyncio
    async def test_request_download_updates_model_status(
        self, service: ModelService, model_repo: _ModelRepoStub
    ) -> None:
        await service.request_download("test/model")
        assert len(model_repo.status_updates) == 1
        model_id, status = model_repo.status_updates[0]
        assert model_id == "test/model"
        assert status == ModelStatus.DOWNLOADING

    @pytest.mark.asyncio
    async def test_request_download_sets_job_params(
        self, service: ModelService, job_repo: _JobRepoStub
    ) -> None:
        await service.request_download("test/model")
        job = job_repo.created_jobs[0]
        assert job.params == {"model_id": "test/model", "source": "huggingface"}

    @pytest.mark.asyncio
    async def test_request_download_sets_model_id_ref(
        self, service: ModelService, job_repo: _JobRepoStub, model_repo: _ModelRepoStub
    ) -> None:
        await service.request_download("test/model")
        job = job_repo.created_jobs[0]
        assert job.model_id == model_repo._records["test/model"].id

    @pytest.mark.asyncio
    async def test_request_download_raises_when_model_not_found(
        self, service: ModelService
    ) -> None:
        with pytest.raises(ValueError, match="not found in registry"):
            await service.request_download("nonexistent/model")

    @pytest.mark.asyncio
    async def test_request_download_error_message_includes_model_id(
        self, service: ModelService
    ) -> None:
        with pytest.raises(ValueError) as exc_info:
            await service.request_download("missing/id")
        assert "missing/id" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_request_download_returns_uuid(
        self, service: ModelService
    ) -> None:
        job_id = await service.request_download("test/model")
        # Verify it's a valid UUID
        str(job_id)  # Should not raise


class TestListModels:
    """Tests for ModelService.list_models()."""

    @pytest.fixture
    def model_repo(self) -> _ModelRepoStub:
        repo = _ModelRepoStub()
        repo._records["hf/model"] = _make_model_record(
            model_id="hf/model", source="huggingface"
        )
        repo._records["cv/model"] = _make_model_record(
            model_id="cv/model", source="civitai"
        )
        repo._records["local/model"] = _make_model_record(
            model_id="local/model", source="local"
        )
        return repo

    @pytest.fixture
    def service(self, model_repo: _ModelRepoStub) -> ModelService:
        return ModelService(model_repo, _JobRepoStub(), _StorageManagerStub())

    @pytest.mark.asyncio
    async def test_list_models_returns_all_when_no_filter(
        self, service: ModelService
    ) -> None:
        models = await service.list_models()
        assert len(models) == 3

    @pytest.mark.asyncio
    async def test_list_models_filters_by_source_huggingface(
        self, service: ModelService
    ) -> None:
        models = await service.list_models(source="huggingface")
        assert len(models) == 1
        assert models[0].source == "huggingface"

    @pytest.mark.asyncio
    async def test_list_models_filters_by_source_civitai(
        self, service: ModelService
    ) -> None:
        models = await service.list_models(source="civitai")
        assert len(models) == 1
        assert models[0].model_id == "cv/model"

    @pytest.mark.asyncio
    async def test_list_models_returns_empty_when_no_match(
        self, service: ModelService
    ) -> None:
        models = await service.list_models(source="nonexistent")
        assert models == []

    @pytest.mark.asyncio
    async def test_list_models_returns_empty_when_no_models(
        self, model_repo: _ModelRepoStub
    ) -> None:
        model_repo._records.clear()
        service = ModelService(model_repo, _JobRepoStub(), _StorageManagerStub())
        models = await service.list_models()
        assert models == []

    @pytest.mark.asyncio
    async def test_list_models_filters_by_source_local(
        self, service: ModelService
    ) -> None:
        models = await service.list_models(source="local")
        assert len(models) == 1
        assert models[0].source == "local"


class TestGetModel:
    """Tests for ModelService.get_model()."""

    @pytest.fixture
    def model_repo(self) -> _ModelRepoStub:
        repo = _ModelRepoStub()
        repo._records["test/existing"] = _make_model_record(
            model_id="test/existing", name="Existing Model", status="downloaded"
        )
        return repo

    @pytest.fixture
    def service(self, model_repo: _ModelRepoStub) -> ModelService:
        return ModelService(model_repo, _JobRepoStub(), _StorageManagerStub())

    @pytest.mark.asyncio
    async def test_get_model_returns_existing_model(self, service: ModelService) -> None:
        model = await service.get_model("test/existing")
        assert model is not None
        assert model.model_id == "test/existing"
        assert model.name == "Existing Model"
        assert model.status == "downloaded"

    @pytest.mark.asyncio
    async def test_get_model_returns_none_for_missing(self, service: ModelService) -> None:
        model = await service.get_model("nonexistent/model")
        assert model is None

    @pytest.mark.asyncio
    async def test_get_model_preserves_all_fields(self, service: ModelService) -> None:
        model = await service.get_model("test/existing")
        assert model.source == "huggingface"
        assert model.family == "sdxl"
        assert model.capabilities == ["text_to_image"]


class TestDeleteModel:
    """Tests for ModelService.delete_model()."""

    @pytest.fixture
    def model_repo(self) -> _ModelRepoStub:
        repo = _ModelRepoStub()
        repo._records["test/deletable"] = _make_model_record(
            model_id="test/deletable", source="huggingface", status="downloaded"
        )
        return repo

    @pytest.fixture
    def storage(self) -> _StorageManagerStub:
        return _StorageManagerStub()

    @pytest.fixture
    def service(self, model_repo: _ModelRepoStub, storage: _StorageManagerStub) -> ModelService:
        return ModelService(model_repo, _JobRepoStub(), storage)

    @pytest.mark.asyncio
    async def test_delete_model_removes_from_storage(
        self, service: ModelService, storage: _StorageManagerStub
    ) -> None:
        await service.delete_model("test/deletable")
        assert ("huggingface", "test/deletable") in storage.deleted_files

    @pytest.mark.asyncio
    async def test_delete_model_removes_from_database(
        self, service: ModelService, model_repo: _ModelRepoStub
    ) -> None:
        await service.delete_model("test/deletable")
        assert len(model_repo.deleted_ids) == 1

    @pytest.mark.asyncio
    async def test_delete_model_deletes_files_then_db(
        self, service: ModelService, storage: _StorageManagerStub, model_repo: _ModelRepoStub
    ) -> None:
        await service.delete_model("test/deletable")
        # Both should have been called
        assert len(storage.deleted_files) == 1
        assert len(model_repo.deleted_ids) == 1

    @pytest.mark.asyncio
    async def test_delete_model_raises_when_not_found(self, service: ModelService) -> None:
        with pytest.raises(ValueError, match="not found"):
            await service.delete_model("nonexistent/model")

    @pytest.mark.asyncio
    async def test_delete_model_error_includes_model_id(
        self, service: ModelService
    ) -> None:
        with pytest.raises(ValueError) as exc_info:
            await service.delete_model("missing/model")
        assert "missing/model" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_delete_model_passes_correct_source_to_storage(
        self, service: ModelService, storage: _StorageManagerStub
    ) -> None:
        await service.delete_model("test/deletable")
        source, model_id = storage.deleted_files[0]
        assert source == "huggingface"
        assert model_id == "test/deletable"

    @pytest.mark.asyncio
    async def test_delete_model_with_different_source(
        self, model_repo: _ModelRepoStub, storage: _StorageManagerStub
    ) -> None:
        model_repo._records["cv/model"] = _make_model_record(
            model_id="cv/model", source="civitai", status="downloaded"
        )
        service = ModelService(model_repo, _JobRepoStub(), storage)
        await service.delete_model("cv/model")
        assert ("civitai", "cv/model") in storage.deleted_files