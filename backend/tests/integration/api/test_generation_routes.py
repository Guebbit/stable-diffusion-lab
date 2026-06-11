"""
Integration tests for generation API routes.

Tests cover:
- Text-to-image submission (happy path, validation, correlation id)
- Image captioning / describe (multipart upload, validation)
- Job status polling (found, not found)
- Request validation (limits, required fields)
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from app.api.routers import generation
from app.domain.errors import JobNotFoundError


def _job_record() -> SimpleNamespace:
    now = datetime.now(timezone.utc)
    return SimpleNamespace(
        id=uuid4(),
        job_type="text_to_image",
        status="pending",
        progress_percent=0,
        error="",
        model_id=None,
        params={},
        created_at=now,
        updated_at=now,
    )


class _GenerationServiceStub:
    """Stub generation service that records calls."""

    def __init__(self) -> None:
        self.submit_calls: list[dict[str, object]] = []
        self.captioning_calls: list[dict[str, object]] = []

    async def submit_text_to_image(
        self,
        params,
        model_id: str,
        correlation_id: str | None = None,
    ) -> UUID:
        job_id = uuid4()
        self.submit_calls.append(
            {
                "params": params,
                "model_id": model_id,
                "correlation_id": correlation_id,
            }
        )
        return job_id

    async def submit_image_captioning(
        self,
        model_id: str,
        image_path: str,
        correlation_id: str | None = None,
    ) -> UUID:
        job_id = uuid4()
        self.captioning_calls.append(
            {
                "model_id": model_id,
                "image_path": image_path,
                "correlation_id": correlation_id,
            }
        )
        return job_id


class _JobServiceStub:
    """Stub job service for status queries — mirrors the real JobService interface."""

    def __init__(self) -> None:
        self.status_calls: list[str] = []
        self._jobs: dict[UUID, SimpleNamespace] = {}

    def set_job(self, job_id: UUID, record: SimpleNamespace) -> None:
        self._jobs[job_id] = record

    async def get_job(self, job_id: UUID) -> SimpleNamespace:
        self.status_calls.append(str(job_id))
        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(f"Job {job_id} not found")
        return job


# ── POST /generation/text-to-image ────────────────────────────────────────────


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
    """Correlation ID in header is echoed back in response and response header."""
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
async def test_submit_text_to_image_without_correlation_id(client, app) -> None:
    """When no correlation ID is provided the field is null."""
    service = _GenerationServiceStub()
    app.dependency_overrides[generation._get_generation_service] = lambda: service

    body = (
        await client.post(
            "/api/v1/generation/text-to-image",
            json={"prompt": "sunset", "model_id": "sd"},
        )
    ).json()

    assert body.get("correlation_id") is None


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
    response = await client.post(
        "/api/v1/generation/text-to-image",
        json={"model_id": "sd-xlarge"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_submit_text_to_image_validates_model_id_required(client) -> None:
    response = await client.post(
        "/api/v1/generation/text-to-image",
        json={"prompt": "a cat"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_submit_text_to_image_validates_width_range(client) -> None:
    response = await client.post(
        "/api/v1/generation/text-to-image",
        json={"prompt": "a cat", "model_id": "sd-xlarge", "width": 10},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_submit_text_to_image_validates_height_range(client) -> None:
    response = await client.post(
        "/api/v1/generation/text-to-image",
        json={"prompt": "a cat", "model_id": "sd-xlarge", "height": 10000},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_submit_text_to_image_validates_num_images_range(client) -> None:
    response = await client.post(
        "/api/v1/generation/text-to-image",
        json={"prompt": "a cat", "model_id": "sd-xlarge", "num_images": 10},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_submit_text_to_image_validates_guidance_scale_range(client) -> None:
    response = await client.post(
        "/api/v1/generation/text-to-image",
        json={"prompt": "a cat", "model_id": "sd-xlarge", "guidance_scale": 0},
    )
    assert response.status_code == 422


# ── POST /generation/describe ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_submit_describe_returns_202(client, app) -> None:
    """Multipart describe request with image returns 202 with job_id."""
    service = _GenerationServiceStub()
    app.dependency_overrides[generation._get_generation_service] = lambda: service

    response = await client.post(
        "/api/v1/generation/describe",
        data={"model_id": "vision-model"},
        files={"image": ("test.png", io.BytesIO(b"\x89PNG\r\n\x1a\n"), "image/png")},
    )

    assert response.status_code == 202
    body = response.json()
    assert "job_id" in body
    assert body["status"] == "pending"


@pytest.mark.asyncio
async def test_submit_describe_calls_captioning_service(client, app) -> None:
    """The describe endpoint delegates to submit_image_captioning, not text_to_image."""
    service = _GenerationServiceStub()
    app.dependency_overrides[generation._get_generation_service] = lambda: service

    await client.post(
        "/api/v1/generation/describe",
        data={"model_id": "vision-model-id"},
        files={"image": ("photo.jpg", io.BytesIO(b"JPEG"), "image/jpeg")},
    )

    assert len(service.captioning_calls) == 1
    assert len(service.submit_calls) == 0
    assert service.captioning_calls[0]["model_id"] == "vision-model-id"


@pytest.mark.asyncio
async def test_submit_describe_saves_image_to_temp_path(client, app) -> None:
    """Service receives a non-empty string path to the uploaded image."""
    service = _GenerationServiceStub()
    app.dependency_overrides[generation._get_generation_service] = lambda: service

    await client.post(
        "/api/v1/generation/describe",
        data={"model_id": "vision-model"},
        files={"image": ("photo.png", io.BytesIO(b"PNG"), "image/png")},
    )

    image_path = service.captioning_calls[0]["image_path"]
    assert isinstance(image_path, str)
    assert len(image_path) > 0


@pytest.mark.asyncio
async def test_submit_describe_with_correlation_id(client, app) -> None:
    """Correlation ID is echoed in response body and X-Correlation-ID header."""
    service = _GenerationServiceStub()
    app.dependency_overrides[generation._get_generation_service] = lambda: service

    response = await client.post(
        "/api/v1/generation/describe",
        data={"model_id": "vision-model"},
        files={"image": ("img.png", io.BytesIO(b"PNG"), "image/png")},
        headers={"X-Correlation-ID": "describe-corr-001"},
    )

    assert response.status_code == 202
    assert response.json()["correlation_id"] == "describe-corr-001"
    assert response.headers.get("x-correlation-id") == "describe-corr-001"


@pytest.mark.asyncio
async def test_submit_describe_passes_correlation_id_to_service(client, app) -> None:
    service = _GenerationServiceStub()
    app.dependency_overrides[generation._get_generation_service] = lambda: service

    await client.post(
        "/api/v1/generation/describe",
        data={"model_id": "vision-model"},
        files={"image": ("img.png", io.BytesIO(b"PNG"), "image/png")},
        headers={"X-Correlation-ID": "corr-xyz"},
    )

    assert service.captioning_calls[0]["correlation_id"] == "corr-xyz"


@pytest.mark.asyncio
async def test_submit_describe_missing_model_id_returns_422(client, app) -> None:
    service = _GenerationServiceStub()
    app.dependency_overrides[generation._get_generation_service] = lambda: service

    response = await client.post(
        "/api/v1/generation/describe",
        files={"image": ("test.png", io.BytesIO(b"PNG"), "image/png")},
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_submit_describe_missing_image_returns_422(client, app) -> None:
    service = _GenerationServiceStub()
    app.dependency_overrides[generation._get_generation_service] = lambda: service

    response = await client.post(
        "/api/v1/generation/describe",
        data={"model_id": "vision-model"},
    )

    assert response.status_code == 422


# ── GET /generation/jobs/{job_id} ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_job_status_returns_job(client, app) -> None:
    """Polling a known job returns its status."""
    job_service = _JobServiceStub()
    job = _job_record()
    job_service.set_job(job.id, job)
    app.dependency_overrides[generation._get_job_service] = lambda: job_service

    response = await client.get(f"/api/v1/generation/jobs/{job.id}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == str(job.id)
    assert body["status"] == "pending"
    assert body["job_type"] == "text_to_image"
    assert body["progress_percent"] == 0
    assert body["error"] == ""
    assert str(job.id) in job_service.status_calls


@pytest.mark.asyncio
async def test_get_job_status_returns_404_when_missing(client, app) -> None:
    """Polling an unknown job id returns 404."""
    job_service = _JobServiceStub()
    app.dependency_overrides[generation._get_job_service] = lambda: job_service

    response = await client.get(f"/api/v1/generation/jobs/{uuid4()}")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_get_job_status_rejects_invalid_uuid(client, app) -> None:
    """Invalid UUID in path returns 422."""
    job_service = _JobServiceStub()
    app.dependency_overrides[generation._get_job_service] = lambda: job_service

    response = await client.get("/api/v1/generation/jobs/not-a-uuid")
    assert response.status_code == 422
