"""
Unit tests for GenerationService.

Tests job submission (text-to-image, image-to-image) using the refactored
service that delegates to JobCreator and ModelResolver.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from pytest_mock import MockerFixture
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import JobStatus, JobType
from app.domain.value_objects import GenerationParams
from app.services.generation_service import GenerationService


# ── Stubs ────────────────────────────────────────────────────────────────

class _Model:
    """Minimal model with model_id attribute."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id


class _AsyncModelRepository:
    """
    AsyncModelRepository stub that works with ModelResolver.
    ModelResolver calls self._repo.get_by_model_id(str(identifier)) which
    eventually hits the real ModelRepository using SQLAlchemy session.execute().
    
    For unit tests we patch ModelResolver directly.
    """

    def __init__(self, models: dict[str, _Model] | None = None) -> None:
        self._models = dict(models) if models else {}

    async def get_by_model_id(self, model_id: str) -> _Model | None:
        return self._models.get(model_id)


class _JobRepositoryStub:
    """Stub JobRepository that records created jobs."""

    def __init__(self) -> None:
        self.created_jobs: list[Any] = []
        self._by_id: dict[str, Any] = {}

    async def create(self, job: Any) -> Any:
        job_id = uuid4()
        object.__setattr__(job, "id", job_id)
        self.created_jobs.append(job)
        self._by_id[str(job_id)] = job
        return job

    async def get_by_id(self, job_id: object) -> Any | None:
        return self._by_id.get(str(job_id))


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def job_repo() -> _JobRepositoryStub:
    return _JobRepositoryStub()


@pytest.fixture
def model_repo() -> _AsyncModelRepository:
    return _AsyncModelRepository(
        models={
            "runwayml/stable-diffusion-v1-5": _Model("runwayml/stable-diffusion-v1-5"),
            "stabilityai/sdxl": _Model("stabilityai/sdxl"),
        }
    )


@pytest.fixture
def service(job_repo: _JobRepositoryStub, model_repo: _AsyncModelRepository) -> GenerationService:
    return GenerationService(job_repo, model_repo)


@pytest.fixture
def params() -> GenerationParams:
    return GenerationParams(
        prompt="a beautiful sunset",
        negative_prompt="blurry, low quality",
        width=768,
        height=768,
        num_inference_steps=30,
        guidance_scale=8.0,
        seed=42,
        num_images=2,
        extra={"custom_key": "custom_value"},
    )


# ── Tests: submit_text_to_image ─────────────────────────────────────────

class TestSubmitTextToImage:
    """Tests for GenerationService.submit_text_to_image()."""

    @pytest.mark.asyncio
    async def test_creates_pending_job(
        self, service: GenerationService, job_repo: _JobRepositoryStub, params: GenerationParams, mocker: MockerFixture
    ) -> None:
        """submit_text_to_image creates a PENDING TEXT_TO_IMAGE job."""
        await service.submit_text_to_image(params, model_id="runwayml/stable-diffusion-v1-5")
        assert len(job_repo.created_jobs) == 1
        job = job_repo.created_jobs[0]
        assert job.status == JobStatus.PENDING
        assert job.job_type == JobType.TEXT_TO_IMAGE

    @pytest.mark.asyncio
    async def test_sets_correct_params(
        self, service: GenerationService, job_repo: _JobRepositoryStub, params: GenerationParams, mocker: MockerFixture
    ) -> None:
        """Job params contain all generation parameters."""
        await service.submit_text_to_image(params, model_id="runwayml/stable-diffusion-v1-5")
        job = job_repo.created_jobs[0]
        assert job.params["prompt"] == "a beautiful sunset"
        assert job.params["negative_prompt"] == "blurry, low quality"
        assert job.params["width"] == 768
        assert job.params["height"] == 768
        assert job.params["num_inference_steps"] == 30
        assert job.params["guidance_scale"] == 8.0
        assert job.params["seed"] == 42
        assert job.params["num_images"] == 2

    @pytest.mark.asyncio
    async def test_resolves_model_id(
        self, service: GenerationService, job_repo: _JobRepositoryStub, params: GenerationParams, mocker: MockerFixture
    ) -> None:
        """Model ID is resolved and stored in params."""
        await service.submit_text_to_image(params, model_id="runwayml/stable-diffusion-v1-5")
        job = job_repo.created_jobs[0]
        assert job.params["model_id"] == "runwayml/stable-diffusion-v1-5"
        assert job.params["original_model_id"] == "runwayml/stable-diffusion-v1-5"

    @pytest.mark.asyncio
    async def test_stores_correlation_id(
        self, service: GenerationService, job_repo: _JobRepositoryStub, params: GenerationParams, mocker: MockerFixture
    ) -> None:
        """Correlation ID is stored in job params."""
        await service.submit_text_to_image(
            params, model_id="runwayml/stable-diffusion-v1-5", correlation_id="corr-123"
        )
        job = job_repo.created_jobs[0]
        assert job.params["correlation_id"] == "corr-123"

    @pytest.mark.asyncio
    async def test_stores_extra_params(
        self, service: GenerationService, job_repo: _JobRepositoryStub, params: GenerationParams, mocker: MockerFixture
    ) -> None:
        """Extra params from GenerationParams are merged into job params."""
        await service.submit_text_to_image(params, model_id="runwayml/stable-diffusion-v1-5")
        job = job_repo.created_jobs[0]
        assert job.params["custom_key"] == "custom_value"

    @pytest.mark.asyncio
    async def test_returns_valid_uuid(
        self, service: GenerationService, params: GenerationParams, mocker: MockerFixture
    ) -> None:
        """submit_text_to_image returns a valid UUID string."""
        job_id = await service.submit_text_to_image(params, model_id="runwayml/stable-diffusion-v1-5")
        from uuid import UUID
        UUID(str(job_id))  # raises if invalid

    @pytest.mark.asyncio
    async def test_fallback_model_id_when_not_in_repo(
        self, service: GenerationService, job_repo: _JobRepositoryStub, params: GenerationParams, mocker: MockerFixture
    ) -> None:
        """When model not found in repo, fallback to the provided model_id."""
        await service.submit_text_to_image(params, model_id="unknown/model")
        job = job_repo.created_jobs[0]
        assert job.params["model_id"] == "unknown/model"
        assert job.params["original_model_id"] == "unknown/model"

    @pytest.mark.asyncio
    async def test_default_params_used_when_minimal(
        self, service: GenerationService, job_repo: _JobRepositoryStub, mocker: MockerFixture
    ) -> None:
        """Minimal GenerationParams use sensible defaults."""
        minimal = GenerationParams(prompt="test")
        await service.submit_text_to_image(minimal, model_id="runwayml/stable-diffusion-v1-5")
        job = job_repo.created_jobs[0]
        assert job.params["width"] == 512
        assert job.params["height"] == 512
        assert job.params["num_inference_steps"] == 20
        assert job.params["guidance_scale"] == 7.5
        assert job.params["seed"] is None
        assert job.params["num_images"] == 1

    @pytest.mark.asyncio
    async def test_allows_empty_prompt(
        self, service: GenerationService, job_repo: _JobRepositoryStub, mocker: MockerFixture
    ) -> None:
        """Empty prompt string is allowed."""
        empty_prompt = GenerationParams(prompt="")
        await service.submit_text_to_image(empty_prompt, model_id="runwayml/stable-diffusion-v1-5")
        job = job_repo.created_jobs[0]
        assert job.params["prompt"] == ""

    @pytest.mark.asyncio
    async def test_multiple_jobs_independent(
        self, service: GenerationService, job_repo: _JobRepositoryStub, params: GenerationParams, mocker: MockerFixture
    ) -> None:
        """Multiple submissions create independent jobs with unique IDs."""
        id1 = await service.submit_text_to_image(params, model_id="runwayml/stable-diffusion-v1-5")
        id2 = await service.submit_text_to_image(params, model_id="stabilityai/sdxl")
        assert id1 != id2
        assert len(job_repo.created_jobs) == 2

    @pytest.mark.asyncio
    async def test_publishes_job_enqueued_event(
        self, mocker: MockerFixture, service: GenerationService, params: GenerationParams
    ) -> None:
        """A job.enqueued event is published via the event bus after job creation."""
        mock_event_bus = mocker.MagicMock()
        mock_event_bus.publish_event = AsyncMock()
        mocker.patch("app.services.job_creator.event_bus", mock_event_bus)
        await service.submit_text_to_image(
            params, model_id="runwayml/stable-diffusion-v1-5", correlation_id="corr-evt"
        )
        mock_event_bus.publish_event.assert_awaited_once()
        event_arg = mock_event_bus.publish_event.call_args[0][0]
        assert event_arg.event_type == "job.enqueued"
        assert event_arg.correlation_id == "corr-evt"
        assert event_arg.payload["job_type"] == JobType.TEXT_TO_IMAGE


# ── Tests: submit_image_to_image ────────────────────────────────────────

class TestSubmitImageToImage:
    """Tests for GenerationService.submit_image_to_image()."""

    @pytest.mark.asyncio
    async def test_creates_image_to_image_job(
        self, service: GenerationService, job_repo: _JobRepositoryStub, params: GenerationParams, mocker: MockerFixture
    ) -> None:
        """Creates an IMAGE_TO_IMAGE job with PENDING status."""
        await service.submit_image_to_image(
            params, model_id="runwayml/stable-diffusion-v1-5", source_image_path="/tmp/source.png"
        )
        assert len(job_repo.created_jobs) == 1
        job = job_repo.created_jobs[0]
        assert job.job_type == JobType.IMAGE_TO_IMAGE
        assert job.status == JobStatus.PENDING

    @pytest.mark.asyncio
    async def test_sets_source_image_path(
        self, service: GenerationService, job_repo: _JobRepositoryStub, params: GenerationParams, mocker: MockerFixture
    ) -> None:
        """source_image_path is stored in job params."""
        await service.submit_image_to_image(
            params, model_id="runwayml/stable-diffusion-v1-5", source_image_path="/tmp/img.png"
        )
        job = job_repo.created_jobs[0]
        assert job.params["source_image_path"] == "/tmp/img.png"

    @pytest.mark.asyncio
    async def test_sets_strength(
        self, service: GenerationService, job_repo: _JobRepositoryStub, params: GenerationParams, mocker: MockerFixture
    ) -> None:
        """Custom strength is stored in job params."""
        await service.submit_image_to_image(
            params,
            model_id="runwayml/stable-diffusion-v1-5",
            source_image_path="/tmp/img.png",
            strength=0.5,
        )
        job = job_repo.created_jobs[0]
        assert job.params["strength"] == 0.5

    @pytest.mark.asyncio
    async def test_default_strength_is_0_75(
        self, service: GenerationService, job_repo: _JobRepositoryStub, params: GenerationParams, mocker: MockerFixture
    ) -> None:
        """Default strength is 0.75."""
        await service.submit_image_to_image(
            params, model_id="runwayml/stable-diffusion-v1-5", source_image_path="/tmp/img.png"
        )
        job = job_repo.created_jobs[0]
        assert job.params["strength"] == 0.75

    @pytest.mark.asyncio
    async def test_resolves_model_id(
        self, service: GenerationService, job_repo: _JobRepositoryStub, params: GenerationParams, mocker: MockerFixture
    ) -> None:
        """Model ID is resolved and stored in params."""
        await service.submit_image_to_image(
            params, model_id="stabilityai/sdxl", source_image_path="/tmp/img.png"
        )
        job = job_repo.created_jobs[0]
        assert job.params["model_id"] == "stabilityai/sdxl"

    @pytest.mark.asyncio
    async def test_returns_valid_uuid(
        self, service: GenerationService, params: GenerationParams, mocker: MockerFixture
    ) -> None:
        """Returns a valid UUID string."""
        job_id = await service.submit_image_to_image(
            params, model_id="runwayml/stable-diffusion-v1-5", source_image_path="/tmp/img.png"
        )
        from uuid import UUID
        UUID(str(job_id))

    @pytest.mark.asyncio
    async def test_stores_correlation_id(
        self, service: GenerationService, job_repo: _JobRepositoryStub, params: GenerationParams, mocker: MockerFixture
    ) -> None:
        """Correlation ID is stored in job params."""
        await service.submit_image_to_image(
            params,
            model_id="runwayml/stable-diffusion-v1-5",
            source_image_path="/tmp/img.png",
            correlation_id="corr-img2img",
        )
        job = job_repo.created_jobs[0]
        assert job.params["correlation_id"] == "corr-img2img"

    @pytest.mark.asyncio
    async def test_publishes_enqueued_event(
        self, mocker: MockerFixture, service: GenerationService, params: GenerationParams
    ) -> None:
        """A job.enqueued event is published via the event bus."""
        mock_event_bus = mocker.MagicMock()
        mock_event_bus.publish_event = AsyncMock()
        mocker.patch("app.services.job_creator.event_bus", mock_event_bus)
        await service.submit_image_to_image(
            params, model_id="runwayml/stable-diffusion-v1-5", source_image_path="/tmp/img.png"
        )
        mock_event_bus.publish_event.assert_awaited_once()
        event_arg = mock_event_bus.publish_event.call_args[0][0]
        assert event_arg.event_type == "job.enqueued"
        assert event_arg.payload["job_type"] == JobType.IMAGE_TO_IMAGE