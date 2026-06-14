"""Unit tests for JobCreator service.

Covers all public methods:
  - create_text_to_image_job
  - create_image_to_image_job
  - create_model_download_job
  - create_image_captioning_job
"""

from __future__ import annotations

from uuid import UUID, uuid4
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.enums import JobStatus, JobType
from app.services.job_creator import JobCreator


class _FakeResolvedModel:
    model_id = "resolved/stabilityai-sdxl"


class _FakeModelRecord:
    def __init__(self):
        self.id = uuid4()
        self.model_id = "stabilityai/sdxl"
        self.source = "huggingface"


class _FakeJobRecord:
    def __init__(self):
        self.id = uuid4()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_job_creator(
    job_repo=None,
    model_resolver=None,
    model_repo=None,
) -> JobCreator:
    if job_repo is None:
        job_repo = AsyncMock()
    if model_resolver is None:
        model_resolver = AsyncMock()
    return JobCreator(
        job_repository=job_repo,
        model_resolver=model_resolver,
        model_repository=model_repo,
    )


# ── create_text_to_image_job ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_text_to_image_job_returns_uuid() -> None:
    """A text-to-image job is persisted and its UUID is returned."""
    job_repo = AsyncMock()
    created = _FakeJobRecord()
    job_repo.create = AsyncMock(return_value=created)

    model_resolver = AsyncMock()
    model_resolver.resolve = AsyncMock(return_value="resolved/model")

    gen_params = MagicMock()
    gen_params.prompt = "a photo"
    gen_params.negative_prompt = None
    gen_params.width = 512
    gen_params.height = 512
    gen_params.num_inference_steps = 20
    gen_params.guidance_scale = 7.5
    gen_params.seed = 42
    gen_params.num_images = 1
    gen_params.extra = {}

    creator = _make_job_creator(job_repo, model_resolver)
    job_id = await creator.create_text_to_image_job(gen_params, "original/model")

    assert isinstance(job_id, UUID)  # UUID
    job_repo.create.assert_awaited_once()
    call_args = job_repo.create.call_args[0][0]
    assert call_args.job_type == JobType.TEXT_TO_IMAGE
    assert call_args.status == JobStatus.PENDING
    assert call_args.params["model_id"] == "resolved/model"
    assert call_args.params["original_model_id"] == "original/model"
    assert call_args.params["prompt"] == "a photo"


@pytest.mark.asyncio
async def test_create_text_to_image_job_forwards_correlation_id() -> None:
    job_repo = AsyncMock()
    job_repo.create = AsyncMock(return_value=_FakeJobRecord())
    model_resolver = AsyncMock()
    model_resolver.resolve = AsyncMock(return_value="resolved/model")

    gen_params = MagicMock()
    gen_params.prompt = "a photo"
    gen_params.negative_prompt = None
    gen_params.width = 512
    gen_params.height = 512
    gen_params.num_inference_steps = 20
    gen_params.guidance_scale = 7.5
    gen_params.seed = 42
    gen_params.num_images = 1
    gen_params.extra = {}

    creator = _make_job_creator(job_repo, model_resolver)
    await creator.create_text_to_image_job(gen_params, "model", correlation_id="corr-1")

    call_args = job_repo.create.call_args[0][0]
    assert call_args.params["correlation_id"] == "corr-1"


@pytest.mark.asyncio
async def test_create_text_to_image_job_extras_merged() -> None:
    """Extra params in GenerationParams are forwarded to job params."""
    job_repo = AsyncMock()
    job_repo.create = AsyncMock(return_value=_FakeJobRecord())
    model_resolver = AsyncMock()
    model_resolver.resolve = AsyncMock(return_value="resolved/model")

    gen_params = MagicMock()
    gen_params.prompt = "a photo"
    gen_params.negative_prompt = None
    gen_params.width = 512
    gen_params.height = 512
    gen_params.num_inference_steps = 20
    gen_params.guidance_scale = 7.5
    gen_params.seed = 42
    gen_params.num_images = 1
    gen_params.extra = {"refiner_model": "some-refiner"}

    creator = _make_job_creator(job_repo, model_resolver)
    await creator.create_text_to_image_job(gen_params, "model")

    call_args = job_repo.create.call_args[0][0]
    assert call_args.params["refiner_model"] == "some-refiner"


# ── create_image_to_image_job ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_image_to_image_job_sets_correct_type() -> None:
    job_repo = AsyncMock()
    job_repo.create = AsyncMock(return_value=_FakeJobRecord())
    model_resolver = AsyncMock()
    model_resolver.resolve = AsyncMock(return_value="resolved/model")

    gen_params = MagicMock()
    gen_params.prompt = "a photo"
    gen_params.negative_prompt = None
    gen_params.width = 512
    gen_params.height = 512
    gen_params.num_inference_steps = 20
    gen_params.guidance_scale = 7.5
    gen_params.seed = 42
    gen_params.num_images = 1
    gen_params.extra = {}

    creator = _make_job_creator(job_repo, model_resolver)
    await creator.create_image_to_image_job(
        gen_params, "model", source_image_path="/tmp/img.png", strength=0.8
    )

    call_args = job_repo.create.call_args[0][0]
    assert call_args.job_type == JobType.IMAGE_TO_IMAGE
    assert call_args.params["source_image_path"] == "/tmp/img.png"
    assert call_args.params["strength"] == 0.8


# ── create_model_download_job ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_model_download_job_success() -> None:
    job_repo = AsyncMock()
    job_repo.create = AsyncMock(return_value=_FakeJobRecord())
    model_resolver = AsyncMock()
    model_repo = AsyncMock()
    model_repo.get_by_model_id = AsyncMock(return_value=_FakeModelRecord())

    creator = _make_job_creator(job_repo, model_resolver, model_repo)
    job_id = await creator.create_model_download_job("stabilityai/sdxl", "huggingface")

    assert isinstance(job_id, UUID)
    call_args = job_repo.create.call_args[0][0]
    assert call_args.job_type == JobType.MODEL_DOWNLOAD
    assert call_args.params["model_id"] == "stabilityai/sdxl"
    assert call_args.params["source"] == "huggingface"


@pytest.mark.asyncio
async def test_create_model_download_job_raises_when_no_model_repo() -> None:
    """If model_repo is None, download jobs raise RuntimeError."""
    creator = _make_job_creator(model_repo=None)
    with pytest.raises(RuntimeError, match="ModelRepository not configured"):
        await creator.create_model_download_job("some/model", "huggingface")


@pytest.mark.asyncio
async def test_create_model_download_job_raises_when_model_not_found() -> None:
    job_repo = AsyncMock()
    model_resolver = AsyncMock()
    model_repo = AsyncMock()
    model_repo.get_by_model_id = AsyncMock(return_value=None)

    creator = _make_job_creator(job_repo, model_resolver, model_repo)
    with pytest.raises(ValueError, match="not found in registry"):
        await creator.create_model_download_job("missing/model", "huggingface")


# ── create_image_analysis_job ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_image_analysis_job_success() -> None:
    job_repo = AsyncMock()
    created = _FakeJobRecord()
    job_repo.create = AsyncMock(return_value=created)
    model_resolver = AsyncMock()

    creator = _make_job_creator(job_repo, model_resolver)
    job_id = await creator.create_image_analysis_job(
        model_id="vision-model", image_path="/tmp/photo.jpg"
    )

    assert isinstance(job_id, UUID)
    call_args = job_repo.create.call_args[0][0]
    assert call_args.job_type == JobType.IMAGE_ANALYSIS
    assert call_args.params["image_path"] == "/tmp/photo.jpg"


@pytest.mark.asyncio
async def test_create_image_analysis_job_with_correlation_id() -> None:
    job_repo = AsyncMock()
    job_repo.create = AsyncMock(return_value=_FakeJobRecord())
    model_resolver = AsyncMock()

    creator = _make_job_creator(job_repo, model_resolver)
    await creator.create_image_analysis_job(
        model_id="vision-model",
        image_path="/tmp/photo.jpg",
        correlation_id="corr-describe",
    )

    call_args = job_repo.create.call_args[0][0]
    assert call_args.params["correlation_id"] == "corr-describe"


# ── _build_params static method ──────────────────────────────────────────────

def test_build_params_without_correlation_id() -> None:
    gen_params = MagicMock()
    gen_params.prompt = "a test"
    gen_params.negative_prompt = None
    gen_params.width = 512
    gen_params.height = 768
    gen_params.num_inference_steps = 30
    gen_params.guidance_scale = 8.0
    gen_params.seed = 123
    gen_params.num_images = 2
    gen_params.extra = {}

    result = JobCreator._build_params(gen_params, "resolved-model", None)

    assert result["prompt"] == "a test"
    assert result["negative_prompt"] == ""
    assert result["width"] == 512
    assert result["height"] == 768
    assert result["num_inference_steps"] == 30
    assert result["guidance_scale"] == 8.0
    assert result["seed"] == 123
    assert result["num_images"] == 2
    assert "correlation_id" not in result


def test_build_params_with_correlation_id() -> None:
    gen_params = MagicMock()
    gen_params.prompt = "a test"
    gen_params.negative_prompt = "blurry"
    gen_params.width = 512
    gen_params.height = 512
    gen_params.num_inference_steps = 20
    gen_params.guidance_scale = 7.5
    gen_params.seed = 0
    gen_params.num_images = 1
    gen_params.extra = {}

    result = JobCreator._build_params(gen_params, "resolved-model", "corr-42")

    assert result["correlation_id"] == "corr-42"
    assert result["negative_prompt"] == "blurry"