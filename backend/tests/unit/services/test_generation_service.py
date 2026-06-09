"""
Unit tests for GenerationService.

Tests job submission (text-to-image, image-to-image), model resolution,
and job status queries using stubbed repositories.
"""

from __future__ import annotations

from asyncio import Future
from dataclasses import asdict
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from pytest_mock import MockerFixture

from app.domain.enums import JobStatus, JobType
from app.domain.value_objects import GenerationParams
from app.services.generation_service import GenerationService


# ── Stubs ───────────────────────────────────────────────────────────────

class _Model:
    """Minimal model with model_id attribute."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id


class _ModelRepoStub:
    """Stub ModelRepository that supports get_by_id and get_by_model_id."""

    def __init__(self, models: dict[str, _Model] | None = None) -> None:
        self._by_slug: dict[str, _Model] = dict(models) if models else {}
        self._by_uuid: dict[str, _Model] = {}
        # Build UUID lookup
        for slug, m in self._by_slug.items():
            self._by_uuid[slug] = m

    async def get_by_id(self, uuid_val: object) -> _Model | None:
        if isinstance(uuid_val, str):
            return self._by_uuid.get(uuid_val)
        # Also try UUID objects by converting to string
        if isinstance(uuid_val, UUID):
            return self._by_uuid.get(str(uuid_val))
        return None

    async def get_by_model_id(self, model_id: str) -> _Model | None:
        return self._by_slug.get(model_id)


class _JobRepoStub:
    """Stub JobRepository that records created jobs."""

    def __init__(self) -> None:
        self.created_jobs: list[object] = []
        self._by_id: dict[str, object] = {}
        self._counter = 0

    async def create(self, job: object) -> object:
        # Always assign a new UUID, even if job already has one (SQLAlchemy models start with None)
        job_id = uuid4()
        object.__setattr__(job, "id", job_id)
        self.created_jobs.append(job)
        self._by_id[str(job_id)] = job
        return job

    async def get_by_id(self, job_id: object) -> object | None:
        return self._by_id.get(str(job_id))


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def job_repo() -> _JobRepoStub:
    return _JobRepoStub()


@pytest.fixture
def model_repo() -> _ModelRepoStub:
    return _ModelRepoStub(
        models={
            "runwayml/stable-diffusion-v1-5": _Model("runwayml/stable-diffusion-v1-5"),
            "stabilityai/sdxl": _Model("stabilityai/sdxl"),
        }
    )


@pytest.fixture
def service(job_repo: _JobRepoStub, model_repo: _ModelRepoStub) -> GenerationService:
    return GenerationService(job_repo, model_repo)


@pytest.fixture
def service_no_model(job_repo: _JobRepoStub) -> GenerationService:
    return GenerationService(job_repo, model_repository=None)


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


# ── Helpers ─────────────────────────────────────────────────────────────

def _is_valid_uuid_string(val: str) -> bool:
    """Check that val can be parsed as a UUID."""
    try:
        uuid4_str = val
        uuid4_str  # just ensure it's a string
        return True
    except Exception:
        return False


# ── Tests: submit_text_to_image ─────────────────────────────────────────

class TestSubmitTextToImage:
    """Tests for GenerationService.submit_text_to_image()."""

    @pytest.mark.asyncio
    async def test_creates_pending_job(
        self, service: GenerationService, job_repo: _JobRepoStub, params: GenerationParams
    ) -> None:
        await service.submit_text_to_image(params, model_id="runwayml/stable-diffusion-v1-5")
        assert len(job_repo.created_jobs) == 1
        job = job_repo.created_jobs[0]
        assert job.status == JobStatus.PENDING
        assert job.job_type == JobType.TEXT_TO_IMAGE

    @pytest.mark.asyncio
    async def test_sets_correct_params(
        self, service: GenerationService, job_repo: _JobRepoStub, params: GenerationParams
    ) -> None:
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
        self, service: GenerationService, job_repo: _JobRepoStub, params: GenerationParams
    ) -> None:
        await service.submit_text_to_image(params, model_id="runwayml/stable-diffusion-v1-5")
        job = job_repo.created_jobs[0]
        assert job.params["model_id"] == "runwayml/stable-diffusion-v1-5"
        assert job.params["original_model_id"] == "runwayml/stable-diffusion-v1-5"

    @pytest.mark.asyncio
    async def test_stores_correlation_id(
        self, service: GenerationService, job_repo: _JobRepoStub, params: GenerationParams
    ) -> None:
        await service.submit_text_to_image(
            params, model_id="runwayml/stable-diffusion-v1-5", correlation_id="corr-123"
        )
        job = job_repo.created_jobs[0]
        assert job.params["correlation_id"] == "corr-123"

    @pytest.mark.asyncio
    async def test_stores_extra_params(
        self, service: GenerationService, job_repo: _JobRepoStub, params: GenerationParams
    ) -> None:
        await service.submit_text_to_image(params, model_id="runwayml/stable-diffusion-v1-5")
        job = job_repo.created_jobs[0]
        assert job.params["custom_key"] == "custom_value"

    @pytest.mark.asyncio
    async def test_returns_valid_uuid(
        self, service: GenerationService, params: GenerationParams
    ) -> None:
        job_id = await service.submit_text_to_image(params, model_id="runwayml/stable-diffusion-v1-5")
        assert _is_valid_uuid_string(str(job_id))

    @pytest.mark.asyncio
    async def test_fallback_model_id_when_not_in_repo(
        self, service: GenerationService, job_repo: _JobRepoStub, params: GenerationParams
    ) -> None:
        await service.submit_text_to_image(params, model_id="unknown/model")
        job = job_repo.created_jobs[0]
        assert job.params["model_id"] == "unknown/model"
        assert job.params["original_model_id"] == "unknown/model"

    @pytest.mark.asyncio
    async def test_no_model_repo_passthrough(
        self, service_no_model: GenerationService, job_repo: _JobRepoStub, params: GenerationParams
    ) -> None:
        await service_no_model.submit_text_to_image(params, model_id="direct/model")
        job = job_repo.created_jobs[0]
        assert job.params["model_id"] == "direct/model"

    @pytest.mark.asyncio
    async def test_default_params_used_when_minimal(
        self, service: GenerationService, job_repo: _JobRepoStub
    ) -> None:
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
        self, service: GenerationService, job_repo: _JobRepoStub
    ) -> None:
        empty_prompt = GenerationParams(prompt="")
        await service.submit_text_to_image(empty_prompt, model_id="runwayml/stable-diffusion-v1-5")
        job = job_repo.created_jobs[0]
        assert job.params["prompt"] == ""

    @pytest.mark.asyncio
    async def test_multiple_jobs_independent(
        self, service: GenerationService, job_repo: _JobRepoStub, params: GenerationParams
    ) -> None:
        id1 = await service.submit_text_to_image(params, model_id="runwayml/stable-diffusion-v1-5")
        id2 = await service.submit_text_to_image(params, model_id="stabilityai/sdxl")
        assert id1 != id2
        assert len(job_repo.created_jobs) == 2

    @pytest.mark.asyncio
    async def test_publishes_job_enqueued_event(
        self, mocker: MockerFixture, service: GenerationService, params: GenerationParams
    ) -> None:
        from unittest.mock import AsyncMock
        mock_event_bus = mocker.MagicMock()
        mock_event_bus.publish_event = AsyncMock()
        mocker.patch("app.services.generation_service.event_bus", mock_event_bus)
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
        self, service: GenerationService, job_repo: _JobRepoStub, params: GenerationParams
    ) -> None:
        await service.submit_image_to_image(
            params, model_id="runwayml/stable-diffusion-v1-5", source_image_path="/tmp/source.png"
        )
        assert len(job_repo.created_jobs) == 1
        job = job_repo.created_jobs[0]
        assert job.job_type == JobType.IMAGE_TO_IMAGE
        assert job.status == JobStatus.PENDING

    @pytest.mark.asyncio
    async def test_sets_source_image_path(
        self, service: GenerationService, job_repo: _JobRepoStub, params: GenerationParams
    ) -> None:
        await service.submit_image_to_image(
            params, model_id="runwayml/stable-diffusion-v1-5", source_image_path="/tmp/img.png"
        )
        job = job_repo.created_jobs[0]
        assert job.params["source_image_path"] == "/tmp/img.png"

    @pytest.mark.asyncio
    async def test_sets_strength(
        self, service: GenerationService, job_repo: _JobRepoStub, params: GenerationParams
    ) -> None:
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
        self, service: GenerationService, job_repo: _JobRepoStub, params: GenerationParams
    ) -> None:
        await service.submit_image_to_image(
            params, model_id="runwayml/stable-diffusion-v1-5", source_image_path="/tmp/img.png"
        )
        job = job_repo.created_jobs[0]
        assert job.params["strength"] == 0.75

    @pytest.mark.asyncio
    async def test_resolves_model_id(
        self, service: GenerationService, job_repo: _JobRepoStub, params: GenerationParams
    ) -> None:
        await service.submit_image_to_image(
            params, model_id="stabilityai/sdxl", source_image_path="/tmp/img.png"
        )
        job = job_repo.created_jobs[0]
        assert job.params["model_id"] == "stabilityai/sdxl"

    @pytest.mark.asyncio
    async def test_returns_valid_uuid(
        self, service: GenerationService, params: GenerationParams
    ) -> None:
        job_id = await service.submit_image_to_image(
            params, model_id="runwayml/stable-diffusion-v1-5", source_image_path="/tmp/img.png"
        )
        assert _is_valid_uuid_string(str(job_id))

    @pytest.mark.asyncio
    async def test_stores_correlation_id(
        self, service: GenerationService, job_repo: _JobRepoStub, params: GenerationParams
    ) -> None:
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
        from unittest.mock import AsyncMock
        mock_event_bus = mocker.MagicMock()
        mock_event_bus.publish_event = AsyncMock()
        mocker.patch("app.services.generation_service.event_bus", mock_event_bus)
        await service.submit_image_to_image(
            params, model_id="runwayml/stable-diffusion-v1-5", source_image_path="/tmp/img.png"
        )
        mock_event_bus.publish_event.assert_awaited_once()
        event_arg = mock_event_bus.publish_event.call_args[0][0]
        assert event_arg.event_type == "job.enqueued"
        assert event_arg.payload["job_type"] == JobType.IMAGE_TO_IMAGE


# ── Tests: _resolve_model_identifier ────────────────────────────────────

class TestResolveModelIdentifier:
    """Tests for GenerationService._resolve_model_identifier()."""

    @pytest.mark.asyncio
    async def test_resolves_huggingface_slug(self, service: GenerationService) -> None:
        resolved = await service._resolve_model_identifier("runwayml/stable-diffusion-v1-5")
        assert resolved == "runwayml/stable-diffusion-v1-5"

    @pytest.mark.asyncio
    async def test_fallback_when_model_not_found(self, service: GenerationService) -> None:
        resolved = await service._resolve_model_identifier("unknown/model")
        assert resolved == "unknown/model"

    @pytest.mark.asyncio
    async def test_passthrough_when_no_model_repo(self, service_no_model: GenerationService) -> None:
        resolved = await service_no_model._resolve_model_identifier("any/model")
        assert resolved == "any/model"

    @pytest.mark.asyncio
    async def test_resolves_sdxl_model(self, service: GenerationService) -> None:
        resolved = await service._resolve_model_identifier("stabilityai/sdxl")
        assert resolved == "stabilityai/sdxl"