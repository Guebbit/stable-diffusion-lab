"""
Unit tests for ModelOperationHandler SSE event emission.

Verifies that the handler publishes the correct typed events to the event bus
so the frontend can update download progress via SSE instead of polling.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.domain.events import JobEvent, ModelEvent
from app.services.model_operation_handler import ModelOperationHandler


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_handler() -> tuple[ModelOperationHandler, MagicMock]:
    """Return a handler wired to a mock session factory."""
    session_factory = MagicMock()
    session_ctx = AsyncMock()
    session_factory.return_value.__aenter__ = AsyncMock(return_value=session_ctx)
    session_factory.return_value.__aexit__ = AsyncMock(return_value=False)
    return ModelOperationHandler(session_factory=session_factory), session_factory


def _fake_provider(manifest: list[dict] | None = None):
    """Return a mock source provider that 'downloads' without hitting disk."""
    provider = AsyncMock()
    provider.resolve_model_info.return_value = {
        "files": manifest or [{"relative_path": "model.safetensors", "size_bytes": 100}],
        "total_size_bytes": 100,
        "model_id": "org/model",
        "source": "huggingface",
    }
    provider.download_file.return_value = {"sha256": "aabb", "size_bytes": 100}
    return provider


# ── job.progress events carry model_id in payload ────────────────────────────


@pytest.mark.asyncio
async def test_progress_event_contains_model_id() -> None:
    """
    Each job.progress event published during download must include model_id
    in its payload so the FE can update the correct model without a separate mapping.
    """
    handler, _ = _make_handler()
    job_id = uuid4()
    model_id = "runwayml/stable-diffusion-v1-5"
    published: list[JobEvent] = []

    async def capture(event):
        if isinstance(event, JobEvent) and event.event_type == "job.progress":
            published.append(event)

    # Wire up a mock model repo that returns a dummy record
    mock_model = MagicMock(id=uuid4(), model_id=model_id)
    mock_job_repo = AsyncMock()
    mock_model_repo = AsyncMock()
    mock_model_repo.get_by_model_id.return_value = mock_model
    mock_model_repo.update_status = AsyncMock()
    mock_job_repo.update_progress = AsyncMock()

    with (
        patch("app.services.model_operation_handler.ModelRepository", return_value=mock_model_repo),
        patch("app.services.model_operation_handler.JobRepository", return_value=mock_job_repo),
        patch("app.services.model_operation_handler.get_settings") as mock_settings,
        patch("app.services.model_operation_handler.HuggingFaceSourceProvider") as MockHF,
        patch("app.services.model_operation_handler.event_bus") as mock_bus,
    ):
        mock_settings.return_value = MagicMock(
            models_path=Path("/tmp/models"),
            huggingface_token=None,
            civitai_token=None,
        )
        MockHF.return_value = _fake_provider()
        mock_bus.publish_event = capture

        # Inject a progress callback that fires immediately
        async def download_with_progress(model_id, file_path, destination, on_progress=None, **kwargs):
            if on_progress:
                await on_progress({"percent": 50.0, "bytes_downloaded": 50, "total_size": 100})
            return {"sha256": "aa", "size_bytes": 100}

        MockHF.return_value.download_file = download_with_progress

        await handler.handle(
            job_id=job_id,
            job_type="model_download",
            params={"model_id": model_id, "source": "huggingface"},
        )

    assert len(published) >= 1, "at least one job.progress event must be published"
    for event in published:
        assert event.payload.get("model_id") == model_id, (
            f"job.progress payload missing model_id: {event.payload}"
        )
        assert "progress_percent" in event.payload


# ── model.download_failed event on error ─────────────────────────────────────


@pytest.mark.asyncio
async def test_download_failed_event_published_on_error() -> None:
    """
    When a download raises an exception, ModelOperationHandler must publish
    model.download_failed before re-raising, so the FE receives the failure
    notification via SSE without needing to poll.
    """
    handler, _ = _make_handler()
    job_id = uuid4()
    model_id = "org/broken-model"
    published: list[ModelEvent] = []

    async def capture(event):
        if isinstance(event, ModelEvent):
            published.append(event)

    mock_model = MagicMock(id=uuid4(), model_id=model_id)
    mock_model_repo = AsyncMock()
    mock_model_repo.get_by_model_id.return_value = mock_model
    mock_model_repo.update_status = AsyncMock()
    mock_job_repo = AsyncMock()

    with (
        patch("app.services.model_operation_handler.ModelRepository", return_value=mock_model_repo),
        patch("app.services.model_operation_handler.JobRepository", return_value=mock_job_repo),
        patch("app.services.model_operation_handler.get_settings") as mock_settings,
        patch("app.services.model_operation_handler.HuggingFaceSourceProvider") as MockHF,
        patch("app.services.model_operation_handler.event_bus") as mock_bus,
    ):
        mock_settings.return_value = MagicMock(
            models_path=Path("/tmp/models"),
            huggingface_token=None,
            civitai_token=None,
        )
        provider = _fake_provider()
        provider.download_file.side_effect = RuntimeError("network error")
        MockHF.return_value = provider
        mock_bus.publish_event = capture

        with pytest.raises(RuntimeError, match="network error"):
            await handler.handle(
                job_id=job_id,
                job_type="model_download",
                params={"model_id": model_id, "source": "huggingface"},
            )

    failed_events = [e for e in published if e.event_type == "model.download_failed"]
    assert len(failed_events) == 1, "exactly one model.download_failed event must be published"
    assert failed_events[0].payload["model_id"] == model_id
    assert failed_events[0].level == "error"
    assert failed_events[0].job_id == str(job_id)


# ── model.downloaded event on success ────────────────────────────────────────


@pytest.mark.asyncio
async def test_downloaded_event_published_on_success() -> None:
    """
    On successful download, ModelOperationHandler publishes model.downloaded
    so the FE knows to refresh the model state.
    """
    handler, _ = _make_handler()
    job_id = uuid4()
    model_id = "org/good-model"
    published: list[ModelEvent] = []

    async def capture(event):
        if isinstance(event, ModelEvent):
            published.append(event)

    mock_model = MagicMock(id=uuid4(), model_id=model_id)
    mock_model_repo = AsyncMock()
    mock_model_repo.get_by_model_id.return_value = mock_model
    mock_model_repo.update_status = AsyncMock()
    mock_job_repo = AsyncMock()
    mock_job_repo.update_progress = AsyncMock()

    with (
        patch("app.services.model_operation_handler.ModelRepository", return_value=mock_model_repo),
        patch("app.services.model_operation_handler.JobRepository", return_value=mock_job_repo),
        patch("app.services.model_operation_handler.get_settings") as mock_settings,
        patch("app.services.model_operation_handler.HuggingFaceSourceProvider") as MockHF,
        patch("app.services.model_operation_handler.event_bus") as mock_bus,
    ):
        mock_settings.return_value = MagicMock(
            models_path=Path("/tmp/models"),
            huggingface_token=None,
            civitai_token=None,
        )
        MockHF.return_value = _fake_provider()
        mock_bus.publish_event = capture

        await handler.handle(
            job_id=job_id,
            job_type="model_download",
            params={"model_id": model_id, "source": "huggingface"},
        )

    downloaded_events = [e for e in published if e.event_type == "model.downloaded"]
    assert len(downloaded_events) == 1
    assert downloaded_events[0].payload["model_id"] == model_id
    assert downloaded_events[0].job_id == str(job_id)
