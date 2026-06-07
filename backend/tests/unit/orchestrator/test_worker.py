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


@pytest.fixture
def mock_session_factory(mocker: MockerFixture) -> MagicMock:
    """Create a properly mocked async session factory with context manager support."""
    mock_session = AsyncMock(spec=AsyncSession)

    @asynccontextmanager
    async def session_cm():
        yield mock_session

    mock_factory = MagicMock()
    mock_factory.return_value = session_cm()
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

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_cm():
            yield AsyncMock()

        worker._session_factory.return_value = mock_cm()
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

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_cm():
            yield AsyncMock()

        worker._session_factory.return_value = mock_cm()
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

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_cm():
            yield AsyncMock()

        worker._session_factory.return_value = mock_cm()
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

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_cm():
            yield AsyncMock()

        worker._session_factory.return_value = mock_cm()
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
        job_id = uuid4()
        mock_handle = AsyncMock()
        worker._handle_model_download = mock_handle

        mock_event_bus = AsyncMock()
        mocker.patch("app.orchestrator.worker.event_bus", mock_event_bus)

        await worker._dispatch(
            job_id,
            JobType.MODEL_DOWNLOAD,
            {"model_id": "hf/user/model", "source": "huggingface"},
        )

        mock_handle.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatches_model_load(self, worker: JobWorker, mocker: MockerFixture) -> None:
        job_id = uuid4()
        mock_handle = AsyncMock()
        worker._handle_model_load = mock_handle

        mock_event_bus = AsyncMock()
        mocker.patch("app.orchestrator.worker.event_bus", mock_event_bus)

        await worker._dispatch(
            job_id,
            JobType.MODEL_LOAD,
            {"model_id": "sd-1.5"},
        )

        mock_handle.assert_called_once()

    @pytest.mark.asyncio
    async def test_raises_for_unknown_job_type(self, worker: JobWorker) -> None:
        with pytest.raises(ValueError, match="Unknown job type"):
            await worker._dispatch(uuid4(), "UNKNOWN_TYPE", {})

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


# ---------------------------------------------------------------------------
# Handle Model Download Tests
# ---------------------------------------------------------------------------

class TestHandleModelDownload:
    """Test _handle_model_download method."""

    @pytest.mark.asyncio
    async def test_raises_for_unsupported_source(self, worker: JobWorker) -> None:
        with pytest.raises(NotImplementedError, match="not yet supported"):
            await worker._handle_model_download(
                uuid4(),
                {"model_id": "some-model", "source": "unknown"},
            )

    @pytest.mark.asyncio
    async def test_calls_snapshot_download_for_huggingface(
        self, worker: JobWorker, mocker: MockerFixture
    ) -> None:
        job_id = uuid4()
        model_id = "hf/user/model"

        # Inject fake huggingface_hub module so the import doesn't fail
        mock_hf_module = MagicMock()
        mock_hf_module.snapshot_download = MagicMock(return_value="/models/hf-user-model")
        mocker.patch.dict("sys.modules", {"huggingface_hub": mock_hf_module})

        # Patch event_bus
        mock_event_bus = AsyncMock()
        mocker.patch("app.orchestrator.worker.event_bus", mock_event_bus)

        # Patch get_settings
        mocker.patch(
            "app.orchestrator.worker.get_settings",
            return_value=MagicMock(models_path=Path("/models")),
        )

        # Patch ModelRepository
        mock_model_repo = AsyncMock()
        mocker.patch("app.infrastructure.database.repositories.ModelRepository", return_value=mock_model_repo)

        await worker._handle_model_download(
            job_id,
            {"model_id": model_id, "source": "huggingface"},
        )

        # snapshot_download was called
        mock_hf_module.snapshot_download.assert_called_once()
        # At least start + complete events published
        assert mock_event_bus.publish_event.call_count >= 2
        # Model status updated
        mock_model_repo.update_status.assert_called()

    @pytest.mark.asyncio
    async def test_raises_runtime_error_when_snapshot_download_fails(
        self, worker: JobWorker, mocker: MockerFixture
    ) -> None:
        """
        _handle_model_download does NOT wrap snapshot_download in try/except,
        so exceptions propagate to _execute_job which catches them at the
        job level and calls mark_failed + publishes job.failed event.
        """
        job_id = uuid4()

        # Inject fake huggingface_hub module that raises on download
        mock_hf_module = MagicMock()
        mock_hf_module.snapshot_download = MagicMock(side_effect=RuntimeError("network error"))
        mocker.patch.dict("sys.modules", {"huggingface_hub": mock_hf_module})

        mock_event_bus = AsyncMock()
        mocker.patch("app.orchestrator.worker.event_bus", mock_event_bus)

        mocker.patch(
            "app.orchestrator.worker.get_settings",
            return_value=MagicMock(models_path=Path("/models")),
        )
        mock_model_repo = AsyncMock()
        mocker.patch("app.infrastructure.database.repositories.ModelRepository", return_value=mock_model_repo)

        # Exception propagates up (caught by _execute_job at the job level)
        with pytest.raises(RuntimeError, match="network error"):
            await worker._handle_model_download(
                job_id,
                {"model_id": "test-model", "source": "huggingface"},
            )

        # At least the "Starting download" progress event was published
        calls = mock_event_bus.publish_event.call_args_list
        assert len(calls) >= 1
        assert any("Starting download" in str(c) for c in calls)


# ---------------------------------------------------------------------------
# Handle Model Load Tests
# ---------------------------------------------------------------------------

class TestHandleModelLoad:
    """Test _handle_model_load method."""

    @pytest.mark.asyncio
    async def test_raises_when_no_pipeline_cache(self, worker: JobWorker, mocker: MockerFixture) -> None:
        job_id = uuid4()

        mock_event_bus = AsyncMock()
        mocker.patch("app.orchestrator.worker.event_bus", mock_event_bus)

        # Adapter without _cache attribute
        mock_adapter = MagicMock(spec=[])
        worker._adapter_registry.get_provider.return_value = mock_adapter

        with pytest.raises(RuntimeError, match="No pipeline cache"):
            await worker._handle_model_load(job_id, {"model_id": "sd-1.5"})

    @pytest.mark.asyncio
    async def test_loads_model_via_direct_manager(
        self, worker: JobWorker, mocker: MockerFixture
    ) -> None:
        job_id = uuid4()
        model_id = "sd-1.5"

        mock_event_bus = AsyncMock()
        mocker.patch("app.orchestrator.worker.event_bus", mock_event_bus)

        # Mock adapter with _cache
        mock_cache = MagicMock()
        mock_adapter = MagicMock()
        mock_adapter._cache = mock_cache
        worker._adapter_registry.get_provider.return_value = mock_adapter

        # Mock DirectModelManager
        mock_manager_class = MagicMock()
        mock_manager = AsyncMock()
        mock_manager_class.return_value = mock_manager
        mocker.patch(
            "app.adapters.direct.model_manager.DirectModelManager",
            mock_manager_class,
        )

        await worker._handle_model_load(job_id, {"model_id": model_id, "device": "cpu"})

        mock_manager.load_model.assert_called_once_with(model_id, device="cpu")


# ---------------------------------------------------------------------------
# Progress Callback Tests
# ---------------------------------------------------------------------------

class TestProgressCallback:
    """Test that progress callbacks from adapters get routed through event bus."""

    @pytest.mark.asyncio
    async def test_progress_callback_receives_updates(
        self, worker: JobWorker, mocker: MockerFixture
    ) -> None:
        job_id = uuid4()

        # Track events published
        published_events: list[Any] = []
        mock_event_bus = AsyncMock()

        async def capture_event(event):
            published_events.append(event)

        mock_event_bus.publish_event = capture_event
        mocker.patch("app.orchestrator.worker.event_bus", mock_event_bus)

        # Mock adapter that calls on_progress
        async def adapter_generate(gen_params, model_id, output_dir, on_progress=None):
            if on_progress:
                # Simulate progress updates - JobProgress requires job_id
                from app.domain.value_objects import JobProgress
                for i in range(3):
                    on_progress(
                        JobProgress(
                            job_id=job_id,
                            status="running",
                            message=f"Step {i+1}",
                            progress_percent=(i + 1) * 33,
                            current_step=i + 1,
                            total_steps=3,
                        )
                    )
                await asyncio.sleep(0)

        mock_adapter = MagicMock()
        mock_adapter.generate = adapter_generate
        worker._adapter_registry.get_provider.return_value = mock_adapter

        # Patch JobRepository
        repo_instance = AsyncMock()
        mocker.patch("app.orchestrator.worker.JobRepository", return_value=repo_instance)
        mocker.patch(
            "app.orchestrator.worker.get_settings",
            return_value=MagicMock(
                artifacts_path=Path("/tmp"),
                inference_backend=InferenceBackend.DIRECT_PYTHON.value,
            ),
        )

        await worker._execute_job(
            job_id,
            JobType.TEXT_TO_IMAGE,
            {
                "prompt": "test",
                "model_id": "sd-1.5",
            },
        )

        # Allow time for thread-safe coroutines to complete
        await asyncio.sleep(0.2)

        # Progress events should have been published
        progress_events = [e for e in published_events if hasattr(e, "event_type") and e.event_type == "job.progress"]
        assert len(progress_events) >= 3