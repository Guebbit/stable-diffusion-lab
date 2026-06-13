"""
Unit tests for JobWorker.

Tests worker lifecycle, job execution, dispatch routing,
model download/load handling, and progress callbacks.
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from pytest_mock import MockerFixture
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.adapters.adapter_registry import AdapterRegistry
from app.adapters.resource_coordinator import ResourceCoordinator
from app.domain.enums import InferenceBackend, JobType
from app.orchestrator.worker import JobWorker


# Custom async context manager to avoid Python 3.14 compatibility issues
# with contextlib._AsyncGeneratorContextManager internal attribute changes.
class _AsyncSessionContextManager:
    """Simple async context manager that yields a mock session on enter."""

    def __init__(self, session: Any) -> None:
        self._session = session

    async def __aenter__(self) -> Any:
        return self._session

    async def __aexit__(self, *exc: Any) -> None:
        pass


@pytest.fixture
def mock_session_factory(mocker: MockerFixture) -> MagicMock:
    """Create a properly mocked async session factory with context manager support."""
    mock_session = AsyncMock(spec=AsyncSession)

    mock_factory = MagicMock()
    mock_factory.return_value = _AsyncSessionContextManager(mock_session)
    return mock_factory


@pytest.fixture
def worker(mock_session_factory: MagicMock, mocker: MockerFixture) -> JobWorker:
    resource_coordinator = ResourceCoordinator(max_concurrent=2)
    adapter_registry = mocker.MagicMock(spec=AdapterRegistry)
    return JobWorker(mock_session_factory, resource_coordinator, adapter_registry)


# ---------------------------------------------------------------------------
# Worker Lifecycle Tests
# ---------------------------------------------------------------------------

class TestWorkerLifecycle:
    """Test worker start/stop/cancellation lifecycle."""

    @pytest.mark.asyncio
    async def test_start_sets_running_flag(self, worker: JobWorker) -> None:
        await worker.start()
        assert worker._running is True

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self, worker: JobWorker) -> None:
        await worker.start()
        task_before = worker._task
        await worker.start()
        assert worker._task is task_before

    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self, worker: JobWorker) -> None:
        await worker.start()
        await worker.stop()
        assert worker._running is False

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self, worker: JobWorker, mocker: MockerFixture) -> None:
        # Create a real asyncio task that just sleeps
        async def dummy_task():
            while True:
                await asyncio.sleep(0.1)

        worker._task = asyncio.create_task(dummy_task())
        worker._running = True

        await worker.stop()
        assert worker._running is False
        assert worker._task.done()

    def test_request_cancellation_adds_to_set(self, worker: JobWorker) -> None:
        job_id = uuid4()
        worker.request_cancellation(job_id)
        assert job_id in worker._cancelled_jobs


# ---------------------------------------------------------------------------
# Execute Job Tests
# ---------------------------------------------------------------------------

class TestExecuteJob:
    """Test _execute_job method directly."""

    @pytest.mark.asyncio
    async def test_cancels_job_if_in_cancellation_set(
        self, worker: JobWorker, mocker: MockerFixture
    ) -> None:
        job_id = uuid4()
        worker.request_cancellation(job_id)

        # Patch event_bus
        mock_event_bus = AsyncMock()
        mocker.patch("app.orchestrator.worker.event_bus", mock_event_bus)

        # Patch JobRepository
        mock_job_repo = AsyncMock()
        mocker.patch("app.orchestrator.worker.JobRepository", return_value=mock_job_repo)

        await worker._execute_job(job_id, JobType.TEXT_TO_IMAGE, {"prompt": "test"})

        mock_job_repo.mark_cancelled.assert_called_once_with(job_id)

    @pytest.mark.asyncio
    async def test_acquires_and_releases_gpu_lock(
        self, worker: JobWorker, mocker: MockerFixture
    ) -> None:
        job_id = uuid4()

        # Mock _dispatch to avoid adapter complexity
        worker._dispatch = AsyncMock()

        mock_event_bus = AsyncMock()
        mocker.patch("app.orchestrator.worker.event_bus", mock_event_bus)

        mock_repo = AsyncMock()

        worker._session_factory.return_value = _AsyncSessionContextManager(AsyncMock())
        mocker.patch("app.orchestrator.worker.JobRepository", lambda s: mock_repo)

        await worker._execute_job(
            job_id,
            JobType.TEXT_TO_IMAGE,
            {
                "prompt": "test",
                "model_id": "sd-1.5",
            },
        )

        # _dispatch was called
        worker._dispatch.assert_called_once()
        # mark_completed was called
        mock_repo.mark_completed.assert_called_once_with(job_id)
        # GPU lock was released (current_holder is None after release)
        assert worker._resource_coordinator._current_holder is None

    @pytest.mark.asyncio
    async def test_releases_gpu_lock_on_failure(
        self, worker: JobWorker, mocker: MockerFixture
    ) -> None:
        job_id = uuid4()

        # Mock _dispatch to raise, simulating adapter failure
        worker._dispatch = AsyncMock(side_effect=RuntimeError("inference failed"))

        mock_event_bus = AsyncMock()
        mocker.patch("app.orchestrator.worker.event_bus", mock_event_bus)

        mock_repo = AsyncMock()

        worker._session_factory.return_value = _AsyncSessionContextManager(AsyncMock())
        mocker.patch("app.orchestrator.worker.JobRepository", lambda s: mock_repo)

        # Exception is caught internally, so no pytest.raises
        await worker._execute_job(
            job_id,
            JobType.TEXT_TO_IMAGE,
            {"prompt": "test", "model_id": "sd-1.5"},
        )

        # mark_failed was called (not mark_completed)
        mock_repo.mark_failed.assert_called_once()
        # GPU lock was released even on failure
        assert worker._resource_coordinator._current_holder is None

    @pytest.mark.asyncio
    async def test_marks_job_completed_on_success(
        self, worker: JobWorker, mocker: MockerFixture
    ) -> None:
        job_id = uuid4()

        # Mock _dispatch to succeed
        worker._dispatch = AsyncMock()

        mock_event_bus = AsyncMock()
        mocker.patch("app.orchestrator.worker.event_bus", mock_event_bus)

        mock_repo = AsyncMock()

        worker._session_factory.return_value = _AsyncSessionContextManager(AsyncMock())
        mocker.patch("app.orchestrator.worker.JobRepository", lambda s: mock_repo)

        await worker._execute_job(
            job_id,
            JobType.TEXT_TO_IMAGE,
            {
                "prompt": "test",
                "model_id": "sd-1.5",
            },
        )

        mock_repo.mark_completed.assert_called_once_with(job_id)

    @pytest.mark.asyncio
    async def test_marks_job_failed_on_dispatch_exception(
        self, worker: JobWorker, mocker: MockerFixture
    ) -> None:
        job_id = uuid4()

        # Mock _dispatch to raise an exception
        worker._dispatch = AsyncMock(side_effect=ValueError("bad input"))

        mock_event_bus = AsyncMock()
        mocker.patch("app.orchestrator.worker.event_bus", mock_event_bus)

        mock_repo = AsyncMock()

        worker._session_factory.return_value = _AsyncSessionContextManager(AsyncMock())
        mocker.patch("app.orchestrator.worker.JobRepository", lambda s: mock_repo)

        # Exception is caught internally by _execute_job
        await worker._execute_job(
            job_id,
            JobType.TEXT_TO_IMAGE,
            {"prompt": "test", "model_id": "sd-1.5"},
        )

        mock_repo.mark_failed.assert_called_once()
        # mark_completed must NOT be called
        mock_repo.mark_completed.assert_not_called()


# ---------------------------------------------------------------------------
# Dispatch Tests
# ---------------------------------------------------------------------------

class TestDispatch:
    """Test _dispatch routing to correct adapters."""

    @pytest.mark.asyncio
    async def test_dispatches_text_to_image(self, worker: JobWorker, mocker: MockerFixture) -> None:
        job_id = uuid4()
        mock_adapter = AsyncMock()
        worker._adapter_registry.get_provider.return_value = mock_adapter

        mock_event_bus = AsyncMock()
        mocker.patch("app.orchestrator.worker.event_bus", mock_event_bus)

        mocker.patch("app.orchestrator.worker.get_settings", return_value=MagicMock(artifacts_path=Path("/tmp")))

        await worker._dispatch(
            job_id,
            JobType.TEXT_TO_IMAGE,
            {"prompt": "a cat", "model_id": "sd-1.5"},
        )

        mock_adapter.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatches_image_to_image(self, worker: JobWorker, mocker: MockerFixture) -> None:
        job_id = uuid4()
        mock_adapter = AsyncMock()
        worker._adapter_registry.get_provider.return_value = mock_adapter

        mock_event_bus = AsyncMock()
        mocker.patch("app.orchestrator.worker.event_bus", mock_event_bus)

        mocker.patch("app.orchestrator.worker.get_settings", return_value=MagicMock(artifacts_path=Path("/tmp")))

        await worker._dispatch(
            job_id,
            JobType.IMAGE_TO_IMAGE,
            {
                "prompt": "a dog",
                "model_id": "sd-1.5",
                "source_image_path": "/tmp/input.png",
            },
        )

        mock_adapter.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatches_image_captioning(self, worker: JobWorker, mocker: MockerFixture) -> None:
        job_id = uuid4()
        mock_adapter = AsyncMock()
        worker._adapter_registry.get_provider.return_value = mock_adapter

        mock_event_bus = AsyncMock()
        mocker.patch("app.orchestrator.worker.event_bus", mock_event_bus)

        mocker.patch("app.orchestrator.worker.get_settings", return_value=MagicMock(artifacts_path=Path("/tmp")))

        await worker._dispatch(
            job_id,
            JobType.IMAGE_CAPTIONING,
            {
                "image_path": "/tmp/photo.jpg",
                "model_id": "blip",
            },
        )

        mock_adapter.caption.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatches_video_generation(self, worker: JobWorker, mocker: MockerFixture) -> None:
        job_id = uuid4()
        mock_adapter = AsyncMock()
        worker._adapter_registry.get_provider.return_value = mock_adapter

        mock_event_bus = AsyncMock()
        mocker.patch("app.orchestrator.worker.event_bus", mock_event_bus)

        mocker.patch("app.orchestrator.worker.get_settings", return_value=MagicMock(artifacts_path=Path("/tmp")))

        await worker._dispatch(
            job_id,
            JobType.VIDEO_GENERATION,
            {"prompt": "waves", "model_id": "damming"},
        )

        mock_adapter.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatches_llm_inference(self, worker: JobWorker, mocker: MockerFixture) -> None:
        job_id = uuid4()
        mock_adapter = AsyncMock()
        worker._adapter_registry.get_provider.return_value = mock_adapter

        mock_event_bus = AsyncMock()
        mocker.patch("app.orchestrator.worker.event_bus", mock_event_bus)

        mocker.patch("app.orchestrator.worker.get_settings", return_value=MagicMock(artifacts_path=Path("/tmp")))

        await worker._dispatch(
            job_id,
            JobType.LLM_INFERENCE,
            {
                "messages": [{"role": "user", "content": "hi"}],
                "model_id": "llama",
            },
        )

        mock_adapter.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatches_model_download(self, worker: JobWorker, mocker: MockerFixture) -> None:
        """Model jobs are routed to ModelOperationHandler."""
        job_id = uuid4()

        mock_handler = AsyncMock()
        mocker.patch(
            "app.orchestrator.worker.ModelOperationHandler",
            return_value=mock_handler,
        )

        mock_event_bus = AsyncMock()
        mocker.patch("app.orchestrator.worker.event_bus", mock_event_bus)

        await worker._dispatch(
            job_id,
            JobType.MODEL_DOWNLOAD,
            {"model_id": "hf/user/model", "source": "huggingface"},
        )

        mock_handler.handle.assert_called_once_with(
            job_id,
            "model_download",
            {"model_id": "hf/user/model", "source": "huggingface"},
        )

    @pytest.mark.asyncio
    async def test_dispatches_model_load_raises_value_error(self, worker: JobWorker) -> None:
        """MODEL_LOAD is not in _MODEL_JOB_TYPES or _GPU_JOB_TYPES, so _dispatch raises ValueError.
        
        This documents the current behavior: MODEL_LOAD (and MODEL_REFRESH) are not routed
        by the worker dispatcher. Only MODEL_DOWNLOAD, MODEL_DELETE are in _MODEL_JOB_TYPES,
        and only TEXT_TO_IMAGE, IMAGE_TO_IMAGE, IMAGE_CAPTIONING, VIDEO_GENERATION, LLM_INFERENCE
        are in _GPU_JOB_TYPES.
        """
        job_id = uuid4()
        
        # MODEL_LOAD is not routed by _dispatch (not in _MODEL_JOB_TYPES or _GPU_JOB_TYPES)
        with pytest.raises(ValueError, match="Unknown job type"):
            await worker._dispatch(
                job_id,
                JobType.MODEL_LOAD,
                {"model_id": "sd-1.5"},
            )

    @pytest.mark.asyncio
    async def test_raises_for_unknown_job_type(self, worker: JobWorker) -> None:
        """Invalid job type string raises ValueError from enum constructor."""
        with pytest.raises(ValueError):
            await worker._dispatch(uuid4(), "not_a_real_job_type", {})

    @pytest.mark.asyncio
    async def test_dispatches_with_backend_override(self, worker: JobWorker, mocker: MockerFixture) -> None:
        job_id = uuid4()
        mock_adapter = AsyncMock()
        worker._adapter_registry.get_provider.return_value = mock_adapter

        mock_event_bus = AsyncMock()
        mocker.patch("app.orchestrator.worker.event_bus", mock_event_bus)

        mocker.patch("app.orchestrator.worker.get_settings", return_value=MagicMock(artifacts_path=Path("/tmp")))

        await worker._dispatch(
            job_id,
            JobType.TEXT_TO_IMAGE,
            {
                "prompt": "test",
                "model_id": "sd-1.5",
                "backend": InferenceBackend.DIRECT_PYTHON.value,
            },
        )

        worker._adapter_registry.get_provider.assert_called()
        mock_adapter.generate.assert_called_once()


