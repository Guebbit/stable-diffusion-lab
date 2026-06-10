"""
Unit tests for GenerationService.

Tests job submission (text-to-image, image-to-image) using the refactored
service that delegates to JobCreator and ModelResolver.

Since GenerationService is a thin facade, we mock the internal JobCreator
to test the service contract: correct delegation, logging, and return value.
The actual job creation logic is tested in test_job_creator.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from pytest_mock import MockerFixture

from app.domain.enums import JobStatus, JobType
from app.domain.events import JobEvent
from app.domain.value_objects import GenerationParams
from app.services.generation_service import GenerationService


# ── Fixtures ─────────────────────────────────────────────────────────────

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


@pytest.fixture
def minimal_params() -> GenerationParams:
    """Minimal GenerationParams relying on defaults."""
    return GenerationParams(prompt="test")


# ── Tests: submit_text_to_image ─────────────────────────────────────────

class TestSubmitTextToImage:
    """Tests for GenerationService.submit_text_to_image()."""

    @pytest.fixture
    def mock_job_creator(self, mocker: MockerFixture) -> AsyncMock:
        """Create a mock JobCreator with patched property access."""
        mock = AsyncMock()
        mock.create_text_to_image_job = AsyncMock(return_value=uuid4())
        mocker.patch.object(GenerationService, "_job_creator", property(lambda self: mock))
        return mock

    @pytest.mark.asyncio
    async def test_delegates_to_job_creator(self, params: GenerationParams, mock_job_creator: AsyncMock) -> None:
        """submit_text_to_image delegates to JobCreator.create_text_to_image_job."""
        service = GenerationService(job_repository=MagicMock(), model_repository=MagicMock())

        await service.submit_text_to_image(params, model_id="runwayml/stable-diffusion-v1-5")

        mock_job_creator.create_text_to_image_job.assert_awaited_once_with(
            params, "runwayml/stable-diffusion-v1-5", None
        )

    @pytest.mark.asyncio
    async def test_returns_job_id_from_creator(self, params: GenerationParams, mock_job_creator: AsyncMock) -> None:
        """submit_text_to_image returns the job_id from JobCreator."""
        expected_id = uuid4()
        mock_job_creator.create_text_to_image_job = AsyncMock(return_value=expected_id)

        service = GenerationService(job_repository=MagicMock(), model_repository=MagicMock())
        result = await service.submit_text_to_image(params, model_id="runwayml/stable-diffusion-v1-5")
        assert result == expected_id

    @pytest.mark.asyncio
    async def test_passes_correlation_id(self, params: GenerationParams, mock_job_creator: AsyncMock) -> None:
        """Correlation ID is passed through to JobCreator."""
        service = GenerationService(job_repository=MagicMock(), model_repository=MagicMock())

        await service.submit_text_to_image(
            params, model_id="runwayml/stable-diffusion-v1-5", correlation_id="corr-123"
        )

        mock_job_creator.create_text_to_image_job.assert_awaited_once_with(
            params, "runwayml/stable-diffusion-v1-5", "corr-123"
        )

    @pytest.mark.asyncio
    async def test_passes_params_correctly(self, params: GenerationParams, mock_job_creator: AsyncMock) -> None:
        """GenerationParams are passed correctly to JobCreator."""
        service = GenerationService(job_repository=MagicMock(), model_repository=MagicMock())

        await service.submit_text_to_image(params, model_id="some-model")

        call_args = mock_job_creator.create_text_to_image_job.call_args[0]
        called_params, called_model_id, called_corr_id = call_args
        assert called_params.prompt == "a beautiful sunset"
        assert called_params.negative_prompt == "blurry, low quality"
        assert called_params.width == 768
        assert called_params.height == 768
        assert called_params.num_inference_steps == 30
        assert called_params.guidance_scale == 8.0
        assert called_params.seed == 42
        assert called_params.num_images == 2
        assert called_model_id == "some-model"
        assert called_corr_id is None

    @pytest.mark.asyncio
    async def test_passes_minimal_params_with_defaults(
        self, minimal_params: GenerationParams, mock_job_creator: AsyncMock
    ) -> None:
        """Minimal GenerationParams with defaults are passed correctly."""
        service = GenerationService(job_repository=MagicMock(), model_repository=MagicMock())

        await service.submit_text_to_image(minimal_params, model_id="runwayml/stable-diffusion-v1-5")

        call_args = mock_job_creator.create_text_to_image_job.call_args[0]
        called_params = call_args[0]
        assert called_params.prompt == "test"
        assert called_params.width == 512
        assert called_params.height == 512
        assert called_params.num_inference_steps == 20
        assert called_params.guidance_scale == 7.5
        assert called_params.seed is None
        assert called_params.num_images == 1

    @pytest.mark.asyncio
    async def test_allows_empty_prompt(self, mock_job_creator: AsyncMock) -> None:
        """Empty prompt string is allowed and passed through."""
        service = GenerationService(job_repository=MagicMock(), model_repository=MagicMock())

        empty_params = GenerationParams(prompt="")
        await service.submit_text_to_image(empty_params, model_id="runwayml/stable-diffusion-v1-5")

        call_args = mock_job_creator.create_text_to_image_job.call_args[0]
        assert call_args[0].prompt == ""

    @pytest.mark.asyncio
    async def test_multiple_calls_create_multiple_jobs(
        self, params: GenerationParams, mock_job_creator: AsyncMock
    ) -> None:
        """Multiple submissions result in multiple independent job creator calls."""
        id1, id2 = uuid4(), uuid4()
        mock_job_creator.create_text_to_image_job = AsyncMock(side_effect=[id1, id2])

        service = GenerationService(job_repository=MagicMock(), model_repository=MagicMock())

        result1 = await service.submit_text_to_image(params, model_id="runwayml/stable-diffusion-v1-5")
        result2 = await service.submit_text_to_image(params, model_id="stabilityai/sdxl")

        assert result1 == id1
        assert result2 == id2
        assert mock_job_creator.create_text_to_image_job.await_count == 2


# ── Tests: submit_image_to_image ────────────────────────────────────────

class TestSubmitImageToImage:
    """Tests for GenerationService.submit_image_to_image()."""

    @pytest.fixture
    def mock_job_creator(self, mocker: MockerFixture) -> AsyncMock:
        """Create a mock JobCreator with patched property access."""
        mock = AsyncMock()
        mock.create_image_to_image_job = AsyncMock(return_value=uuid4())
        mocker.patch.object(GenerationService, "_job_creator", property(lambda self: mock))
        return mock

    @pytest.mark.asyncio
    async def test_delegates_to_job_creator(self, params: GenerationParams, mock_job_creator: AsyncMock) -> None:
        """submit_image_to_image delegates to JobCreator.create_image_to_image_job."""
        service = GenerationService(job_repository=MagicMock(), model_repository=MagicMock())

        await service.submit_image_to_image(
            params, model_id="runwayml/stable-diffusion-v1-5", source_image_path="/tmp/source.png"
        )

        mock_job_creator.create_image_to_image_job.assert_awaited_once_with(
            params, "runwayml/stable-diffusion-v1-5", "/tmp/source.png", None, 0.75
        )

    @pytest.mark.asyncio
    async def test_returns_job_id_from_creator(self, params: GenerationParams, mock_job_creator: AsyncMock) -> None:
        """Returns the job_id from JobCreator."""
        expected_id = uuid4()
        mock_job_creator.create_image_to_image_job = AsyncMock(return_value=expected_id)

        service = GenerationService(job_repository=MagicMock(), model_repository=MagicMock())
        result = await service.submit_image_to_image(
            params, model_id="runwayml/stable-diffusion-v1-5", source_image_path="/tmp/img.png"
        )
        assert result == expected_id

    @pytest.mark.asyncio
    async def test_passes_source_image_path(self, params: GenerationParams, mock_job_creator: AsyncMock) -> None:
        """source_image_path is passed through to JobCreator."""
        service = GenerationService(job_repository=MagicMock(), model_repository=MagicMock())

        await service.submit_image_to_image(
            params, model_id="runwayml/stable-diffusion-v1-5", source_image_path="/tmp/img.png"
        )

        call_args = mock_job_creator.create_image_to_image_job.call_args[0]
        assert call_args[2] == "/tmp/img.png"

    @pytest.mark.asyncio
    async def test_passes_custom_strength(self, params: GenerationParams, mock_job_creator: AsyncMock) -> None:
        """Custom strength is passed through to JobCreator."""
        service = GenerationService(job_repository=MagicMock(), model_repository=MagicMock())

        await service.submit_image_to_image(
            params,
            model_id="runwayml/stable-diffusion-v1-5",
            source_image_path="/tmp/img.png",
            strength=0.5,
        )

        call_args = mock_job_creator.create_image_to_image_job.call_args[0]
        assert call_args[4] == 0.5

    @pytest.mark.asyncio
    async def test_default_strength_is_0_75(self, params: GenerationParams, mock_job_creator: AsyncMock) -> None:
        """Default strength is 0.75."""
        service = GenerationService(job_repository=MagicMock(), model_repository=MagicMock())

        await service.submit_image_to_image(
            params, model_id="runwayml/stable-diffusion-v1-5", source_image_path="/tmp/img.png"
        )

        call_args = mock_job_creator.create_image_to_image_job.call_args[0]
        assert call_args[4] == 0.75

    @pytest.mark.asyncio
    async def test_passes_correlation_id(self, params: GenerationParams, mock_job_creator: AsyncMock) -> None:
        """Correlation ID is passed through to JobCreator."""
        service = GenerationService(job_repository=MagicMock(), model_repository=MagicMock())

        await service.submit_image_to_image(
            params,
            model_id="runwayml/stable-diffusion-v1-5",
            source_image_path="/tmp/img.png",
            correlation_id="corr-img2img",
        )

        call_args = mock_job_creator.create_image_to_image_job.call_args[0]
        assert call_args[3] == "corr-img2img"


# ── Tests: ModelResolver wiring ────────────────────────────────────────

class TestModelResolverWiring:
    """Verify GenerationService correctly wires ModelResolver."""

    @pytest.mark.asyncio
    async def test_raises_when_model_repo_none(self, mocker: MockerFixture) -> None:
        """_model_resolver raises RuntimeError when model_repository is None."""
        # GenerationService created without model_repository
        service = GenerationService(job_repository=MagicMock(), model_repository=None)

        with pytest.raises(RuntimeError, match="ModelResolver requires a ModelRepository"):
            _ = service._model_resolver

    @pytest.mark.asyncio
    async def test_model_resolver_created_with_repo(self, mocker: MockerFixture) -> None:
        """_model_resolver returns a ModelResolver instance when repo is provided."""
        from app.infrastructure.database.repositories import ModelRepository
        from app.services.model_resolver import ModelResolver

        mock_repo = MagicMock(spec=ModelRepository)
        service = GenerationService(job_repository=MagicMock(), model_repository=mock_repo)

        resolver = service._model_resolver
        assert isinstance(resolver, ModelResolver)