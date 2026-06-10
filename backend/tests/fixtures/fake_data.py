"""
Test fixtures - Fake data generators for unit tests.

Provides consistent test data that can be shared across all test modules.
"""
from __future__ import annotations

from uuid import UUID, uuid4

from app.domain.enums import (
    GenerationTask,
    InferenceBackend,
    JobStatus,
    JobType,
    ModelFamily,
    ModelSource,
    ModelStatus,
)
from app.domain.value_objects import GenerationParams, JobProgress


def make_generation_params(**overrides) -> GenerationParams:
    """Create a default GenerationParams with sensible defaults."""
    defaults = {
        "prompt": "a test image",
        "negative_prompt": "blurry",
        "width": 512,
        "height": 512,
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "seed": 42,
        "num_images": 1,
    }
    defaults.update(overrides)
    return GenerationParams(**defaults)


def make_job_params(**overrides) -> dict:
    """Create a default job params dict matching what _build_gen_params produces."""
    params = {
        "model_id": "stable-diffusion-v1-5",
        "prompt": "a test image",
        "negative_prompt": "blurry",
        "width": 512,
        "height": 512,
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "seed": 42,
        "num_images": 1,
        "correlation_id": "test-correlation-001",
        "backend": InferenceBackend.DIRECT_PYTHON,
    }
    params.update(overrides)
    return params


def make_mock_adapter(job_type: JobType | None = None):
    """Create a mock adapter with async generate/caption methods."""
    from unittest.mock import AsyncMock

    adapter = AsyncMock()
    adapter.job_type = job_type
    return adapter


def make_job_progress(
    job_id=None,
    status: str = "running",
    progress_percent: int = 5,
    current_step: int = 1,
    total_steps: int = 20,
    message: str = "Generating",
) -> JobProgress:
    """Create a JobProgress instance."""
    from uuid import UUID

    return JobProgress(
        job_id=job_id or UUID("00000000-0000-0000-0000-000000000001"),
        status=status,
        progress_percent=progress_percent,
        current_step=current_step,
        total_steps=total_steps,
        message=message,
    )


def make_uuid() -> UUID:
    """Return a deterministic test UUID."""
    return UUID("00000000-0000-0000-0000-000000000001")


# ── Quick constants ──────────────────────────────────────────────────────

TEST_MODEL_ID = "stable-diffusion-v1-5"
TEST_RESOLVED_MODEL_ID = "/models/sd-v1-5.ckpt"
TEST_CORRELATION_ID = "test-correlation-001"
TEST_JOB_ID = make_uuid()

FULL_JOB_PARAMS_TXT2IMG = {
    "model_id": TEST_RESOLVED_MODEL_ID,
    "prompt": "a beautiful landscape",
    "negative_prompt": "blurry, low quality",
    "width": 512,
    "height": 768,
    "num_inference_steps": 30,
    "guidance_scale": 8.0,
    "seed": 12345,
    "num_images": 2,
    "correlation_id": TEST_CORRELATION_ID,
    "backend": InferenceBackend.DIRECT_PYTHON,
    "original_model_id": TEST_MODEL_ID,
}

FULL_GEN_PARAMS = GenerationParams(
    prompt="a beautiful landscape",
    negative_prompt="blurry, low quality",
    width=512,
    height=768,
    num_inference_steps=30,
    guidance_scale=8.0,
    seed=12345,
    num_images=2,
)