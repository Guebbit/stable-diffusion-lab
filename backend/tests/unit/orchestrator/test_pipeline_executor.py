"""
Unit tests for PipelineExecutor.

Tests the executor's routing logic with mocked adapters and registry.
Verifies that:
  - Correct adapter is selected per job type
  - GenerationParams are built correctly from raw params
  - Progress callback is invoked
  - Unknown job types raise ValueError
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.adapters.adapter_registry import AdapterRegistry
from app.domain.enums import InferenceBackend, JobType
from app.domain.value_objects import GenerationParams, JobProgress
from app.orchestrator.pipeline_executor import PipelineExecutor, _build_gen_params
from tests.fixtures.fake_data import (
    TEST_CORRELATION_ID,
    TEST_JOB_ID,
    TEST_RESOLVED_MODEL_ID,
)


@pytest.fixture
def mock_registry():
    """Adapter registry pre-configured with mock adapters for every job type."""
    registry = MagicMock(spec=AdapterRegistry)

    registry._adapters = {
        JobType.TEXT_TO_IMAGE: {
            InferenceBackend.DIRECT_PYTHON: AsyncMock(),
        },
        JobType.IMAGE_TO_IMAGE: {
            InferenceBackend.DIRECT_PYTHON: AsyncMock(),
        },
        JobType.IMAGE_CAPTIONING: {
            InferenceBackend.DIRECT_PYTHON: AsyncMock(),
        },
        JobType.VIDEO_GENERATION: {
            InferenceBackend.DIRECT_PYTHON: AsyncMock(),
        },
        JobType.LLM_INFERENCE: {
            InferenceBackend.DIRECT_PYTHON: AsyncMock(),
        },
    }

    def get_provider(job_type, backend=None):
        return registry._adapters[job_type][backend or InferenceBackend.DIRECT_PYTHON]

    registry.get_provider = get_provider

    return registry


@pytest.fixture
def executor(mock_registry):
    return PipelineExecutor(mock_registry)


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.artifacts_path = Path("/tmp/test-artifacts")
    return settings


# ────────────────────────── _build_gen_params ─────────────────────────

class TestBuildGenParams:
    def test_default_values(self):
        params = _build_gen_params({"prompt": "hello"})
        assert params.prompt == "hello"
        assert params.width == 512
        assert params.height == 512
        assert params.num_inference_steps == 20
        assert params.guidance_scale == 7.5
        assert params.seed is None
        assert params.num_images == 1

    def test_custom_values(self):
        params = _build_gen_params(
            {
                "prompt": "a cat",
                "negative_prompt": "ugly",
                "width": 768,
                "height": 1024,
                "num_inference_steps": 50,
                "guidance_scale": 9.0,
                "seed": 999,
                "num_images": 4,
            }
        )
        assert params == GenerationParams(
            prompt="a cat",
            negative_prompt="ugly",
            width=768,
            height=1024,
            num_inference_steps=50,
            guidance_scale=9.0,
            seed=999,
            num_images=4,
        )

    def test_default_steps_override(self):
        params = _build_gen_params({}, default_steps=25)
        assert params.num_inference_steps == 25


# ──────────────────────── PipelineExecutor.execute ───────────────────

class TestPipelineExecutorTextToImage:
    @pytest.mark.asyncio
    async def test_routes_to_correct_adapter(self, executor, mock_registry, mock_settings):
        """Executor calls the correct adapter's generate method."""
        params = {
            "job_type": JobType.TEXT_TO_IMAGE,
            "model_id": TEST_RESOLVED_MODEL_ID,
            "backend": InferenceBackend.DIRECT_PYTHON,
            "correlation_id": TEST_CORRELATION_ID,
            "prompt": "a sunset",
        }

        with patch("app.orchestrator.pipeline_executor.get_settings", return_value=mock_settings):
            await executor.execute(TEST_JOB_ID, JobType.TEXT_TO_IMAGE, params)

        mock_registry._adapters[JobType.TEXT_TO_IMAGE][
            InferenceBackend.DIRECT_PYTHON
        ].generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_invokes_progress_callback(self, executor, mock_registry, mock_settings):
        """Progress updates from the adapter are forwarded to the callback."""
        progress_log: list[JobProgress] = []

        def on_progress(p: JobProgress):
            progress_log.append(p)

        params = {
            "model_id": TEST_RESOLVED_MODEL_ID,
            "backend": InferenceBackend.DIRECT_PYTHON,
            "correlation_id": TEST_CORRELATION_ID,
            "prompt": "test",
        }

        async def fake_generate(*a, on_progress=None, **kw):
            if on_progress:
                on_progress(
                    JobProgress(
                        job_id=TEST_JOB_ID,
                        status="running",
                        current_step=1,
                        total_steps=20,
                        message="Step 1",
                    )
                )

        mock_registry._adapters[JobType.TEXT_TO_IMAGE][
            InferenceBackend.DIRECT_PYTHON
        ].generate = fake_generate

        with patch("app.orchestrator.pipeline_executor.get_settings", return_value=mock_settings):
            await executor.execute(
                TEST_JOB_ID, JobType.TEXT_TO_IMAGE, params, on_progress=on_progress
            )

        assert len(progress_log) == 1
        assert progress_log[0].current_step == 1


# ──────────────────────── Unknown job type ──────────────────────────────

class TestPipelineExecutorUnknownJobType:
    @pytest.mark.asyncio
    async def test_raises_value_error_for_invalid_enum(self, executor, mock_settings):
        """An invalid job type string raises ValueError at enum construction."""
        with patch("app.orchestrator.pipeline_executor.get_settings", return_value=mock_settings):
            with pytest.raises(ValueError):
                await executor.execute(TEST_JOB_ID, "nonexistent_job", {})


# ────────────────────────── _run_video defaults ──────────────────────

class TestPipelineExecutorVideo:
    @pytest.mark.asyncio
    async def test_video_uses_25_default_steps(self, executor, mock_registry, mock_settings):
        """Video generation defaults to 25 inference steps."""
        params = {
            "model_id": TEST_RESOLVED_MODEL_ID,
            "backend": InferenceBackend.DIRECT_PYTHON,
            "correlation_id": TEST_CORRELATION_ID,
        }

        captured_gen_params = None

        async def fake_generate(gen_params, *a, **kw):
            nonlocal captured_gen_params
            captured_gen_params = gen_params

        mock_registry._adapters[JobType.VIDEO_GENERATION][
            InferenceBackend.DIRECT_PYTHON
        ].generate = fake_generate

        with patch("app.orchestrator.pipeline_executor.get_settings", return_value=mock_settings):
            await executor.execute(TEST_JOB_ID, JobType.VIDEO_GENERATION, params)

        assert captured_gen_params.num_inference_steps == 25