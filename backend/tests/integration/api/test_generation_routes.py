"""
Integration tests for generation API routes.

Tests cover:
- Text-to-image submission (happy path, validation, correlation id)
- Job status polling (found, not found)
- Request validation (limits, required fields)
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.api.routers import generation


def _job_record() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        job_type="text_to_image",
        status="pending",
        progress_percent=0,
        current_step=0,
        total_steps=50,
        message="",
        created_at=datetime.now(timezone.utc),
        started_at=None,
        completed_at=None,
        error="",
        result={},
    )


class _GenerationServiceStub:
    """Stub generation service that records calls."""

    def __init__(self) -> None:
        self.submit_calls: list[dict[str, object]] = []
        self.status_calls: list[str] = []
        self._jobs: dict[str, SimpleNamespace] = {}

    def set_job(self, job_id, record: SimpleNamespace) -> None:
        # Store keyed by string representation so UUID lookups work
        self._jobs[str(job_id)] = record

    async def submit_text_to_image(
        self,
        params,
        model_id: str,
        correlation_id: str | None = None,
    ) -> str:
        job_id = str(uuid4())
        self.submit_calls.append(
            {
                "params": params,
                "model_id": model_id,
                "correlation_id": correlation_id,
            }
        )
        return job_id

    async def get_job_status(self, job_id) -> SimpleNamespace | None:
        self.status_calls.append(str(job_id))
        return self._jobs.get(str(job_id))


@pytest.mark.asyncio
async def test_submit_text_to_image_returns_202(client, app) -> None:
    """Submitting a valid text-to-image request returns 202 with a job_id."""
    service = _GenerationServiceStub()
    app.dependency_overrides[generation._get_generation_service] = lambda: service

    response = await client.post(
        "/api/v1/generation/text-to-image",
        json={
            "prompt": "a beautiful landscape",
            "model_id": "sd-xlarge",
        },
    )

    assert response.status_code == 202
    body = response.json()
    assert "job_id" in body
    assert body["status"] == "pending"
    assert len(service.submit_calls) == 1
    assert service.submit_calls[0]["model_id"] == "sd-xlarge"


@pytest.mark.asyncio
async def test_submit_text_to_image_with_correlation_id(client, app) -> None:
    """Correlation ID in header is echoed back in response."""
    service = _GenerationServiceStub()
    app.dependency_overrides[generation._get_generation_service] = lambda: service

    response = await client.post(
        "/api/v1/generation/text-to-image",
        json={
            "prompt": "a cat",
            "model_id": "sd-xlarge",
        },
        headers={"X-Correlation-ID": "test-123"},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["correlation_id"] == "test-123"
    assert response.headers.get("x-correlation-id") == "test-123"


@pytest.mark.asyncio
async def test_submit_text_to_image_with_all_params(client, app) -> None:
    """All optional parameters are forwarded to the service."""
    service = _GenerationServiceStub()
    app.dependency_overrides[generation._get_generation_service] = lambda: service

    payload = {
        "prompt": "a sunset",
        "model_id": "sd-xlarge",
        "negative_prompt": "blurry",
        "width": 768,
        "height": 768,
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "seed": 42,
        "num_images": 2,
    }
    response = await client.post(
        "/api/v1/generation/text-to-image",
        json=payload,
    )

    assert response.status_code == 202
    assert len(service.submit_calls) == 1
    call = service.submit_calls[0]
    assert call["params"].prompt == "a sunset"
    assert call["params"].negative_prompt == "blurry"
    assert call["params"].width == 768
    assert call["params"].height == 768
    assert call["params"].num_inference_steps == 30
    assert call["params"].guidance_scale == 7.5
    assert call["params"].seed == 42
    assert call["params"].num_images == 2


@pytest.mark.asyncio
async def test_submit_text_to_image_validates_prompt_required(client) -> None:
    """Missing prompt returns 422."""
    response = await client.post(
        "/api/v1/generation/text-to-image",
        json={"model_id": "sd-xlarge"},
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_submit_text_to_image_validates_model_id_required(client) -> None:
    """Missing model_id returns 422."""
    response = await client.post(
        "/api/v1/generation/text-to-image",
        json={"prompt": "a cat"},
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_job_status_returns_job(client, app) -> None:
    """Polling a known job returns its status."""
    service = _GenerationServiceStub()
    job = _job_record()
    service.set_job(str(job.id), job)
    app.dependency_overrides[generation._get_generation_service] = lambda: service

    response = await client.get(f"/api/v1/generation/jobs/{job.id}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == str(job.id)
    assert body["status"] == "pending"
    assert body["job_type"] == "text_to_image"
    assert str(job.id) in service.status_calls


@pytest.mark.asyncio
async def test_get_job_status_returns_404_when_missing(client, app) -> None:
    """Polling an unknown job id returns 404."""
    service = _GenerationServiceStub()
    app.dependency_overrides[generation._get_generation_service] = lambda: service

    response = await client.get(f"/api/v1/generation/jobs/{uuid4()}")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_get_job_status_rejects_invalid_uuid(client) -> None:
    """Invalid UUID in path returns 422."""
    response = await client.get("/api/v1/generation/jobs/not-a-uuid")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_submit_text_to_image_validates_width_range(client) -> None:
    """Width outside acceptable range returns 422."""
    response = await client.post(
        "/api/v1/generation/text-to-image",
        json={
            "prompt": "a cat",
            "model_id": "sd-xlarge",
            "width": 10,
        },
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_submit_text_to_image_validates_height_range(client) -> None:
    """Height outside acceptable range returns 422."""
    response = await client.post(
        "/api/v1/generation/text-to-image",
        json={
            "prompt": "a cat",
            "model_id": "sd-xlarge",
            "height": 10000,
        },
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_submit_text_to_image_validates_num_images_range(client) -> None:
    """num_images outside acceptable range returns 422."""
    response = await client.post(
        "/api/v1/generation/text-to-image",
        json={
            "prompt": "a cat",
            "model_id": "sd-xlarge",
            "num_images": 10,
        },
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_submit_text_to_image_validates_guidance_scale_range(client) -> None:
    """guidance_scale outside acceptable range returns 422."""
    response = await client.post(
        "/api/v1/generation/text-to-image",
        json={
            "prompt": "a cat",
            "model_id": "sd-xlarge",
            "guidance_scale": 0,
        },
    )

    assert response.status_code == 422