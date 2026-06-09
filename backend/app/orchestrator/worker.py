"""
Job worker — executes jobs picked from the queue.

The worker loop runs as a background asyncio task. It polls the database
for PENDING jobs, acquires the GPU lock, dispatches to the appropriate
adapter via the AdapterRegistry, and updates job status on completion or failure.

State machine enforced by the worker:
    PENDING → RUNNING → COMPLETED
                     → FAILED
                     → CANCELLED (if cancellation was requested)
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.adapters.adapter_registry import AdapterRegistry
from app.adapters.resource_coordinator import ResourceCoordinator
from app.domain.events import ArtifactEvent, JobEvent, ModelEvent, ResourceEvent
from app.domain.enums import InferenceBackend, JobType
from app.domain.value_objects import GenerationParams
from app.infrastructure.config.settings import get_settings
from app.infrastructure.database.repositories import JobRepository
from app.orchestrator.event_bus import event_bus

if TYPE_CHECKING:
    from app.domain.value_objects import JobProgress


logger = logging.getLogger(__name__)


class JobWorker:
    """
    Background worker that processes jobs from the queue.

    Lifecycle:
    1. Polls DB for PENDING jobs at a configurable interval
    2. Claims a job (atomic status update to RUNNING)
    3. Acquires GPU lock from ResourceCoordinator
    4. Dispatches to the correct adapter via AdapterRegistry
    5. Updates job to COMPLETED or FAILED
    6. Releases GPU lock
    7. Broadcasts progress/completion event via EventBus
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        resource_coordinator: ResourceCoordinator,
        adapter_registry: AdapterRegistry,
    ) -> None:
        self._session_factory = session_factory
        self._resource_coordinator = resource_coordinator
        self._adapter_registry = adapter_registry
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._cancelled_jobs: set[UUID] = set()

    async def start(self) -> None:
        """Start the worker loop as a background task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info("Job worker started")

    async def stop(self) -> None:
        """Gracefully stop the worker loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Job worker stopped")

    def request_cancellation(self, job_id: UUID) -> None:
        """Request cancellation of a running or pending job."""
        self._cancelled_jobs.add(job_id)

    async def _poll_loop(self) -> None:
        """Main loop: poll for jobs and dispatch them."""
        settings = get_settings()
        while self._running:
            try:
                async with self._session_factory() as session:
                    job_repo = JobRepository(session)
                    job = await job_repo.claim_next_pending()
                    if job:
                        await session.commit()
                        await self._execute_job(job.id, job.job_type, job.params)
                    else:
                        await asyncio.sleep(settings.job_poll_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in job worker poll loop")
                await asyncio.sleep(settings.job_poll_interval)

    async def _execute_job(self, job_id: UUID, job_type: str, params: dict) -> None:
        """Execute a single job with resource locking and error handling."""
        job_id_str = str(job_id)
        correlation_id = params.get("correlation_id")
        started_at = time.monotonic()

        # Check for early cancellation
        if job_id in self._cancelled_jobs:
            self._cancelled_jobs.discard(job_id)
            async with self._session_factory() as session:
                job_repo = JobRepository(session)
                await job_repo.mark_cancelled(job_id)
                await session.commit()
            await event_bus.publish_event(
                JobEvent(
                    event_type="job.cancelled",
                    correlation_id=correlation_id,
                    job_id=job_id_str,
                    message="Job cancelled before execution",
                )
            )
            return

        # Determine if this job needs GPU before entering try block
        job_type_enum = JobType(job_type)
        needs_gpu = job_type_enum in {
            JobType.TEXT_TO_IMAGE,
            JobType.IMAGE_TO_IMAGE,
            JobType.IMAGE_CAPTIONING,
            JobType.VIDEO_GENERATION,
            JobType.LLM_INFERENCE,
        }

        try:
            await event_bus.publish_event(
                JobEvent(
                    event_type="job.started",
                    correlation_id=correlation_id,
                    job_id=job_id_str,
                    message="Job started",
                    payload={"job_type": job_type},
                )
            )
            if needs_gpu:
                lock_wait_start = time.monotonic()
                await self._resource_coordinator.acquire(job_id_str)
                await event_bus.publish_event(
                    ResourceEvent(
                        event_type="resource.lock_acquired",
                        correlation_id=correlation_id,
                        job_id=job_id_str,
                        message="GPU lock acquired",
                        payload={
                            "acquired_at": datetime.now(timezone.utc).isoformat(),
                            "wait_seconds": round(time.monotonic() - lock_wait_start, 6),
                        },
                    )
                )

            # Dispatch to appropriate handler via adapter registry
            await self._dispatch(job_id, job_type, params)

            # Mark completed
            async with self._session_factory() as session:
                job_repo = JobRepository(session)
                await job_repo.mark_completed(job_id)
                await session.commit()
            await event_bus.publish_event(
                JobEvent(
                    event_type="job.completed",
                    correlation_id=correlation_id,
                    job_id=job_id_str,
                    message="Job completed successfully",
                    payload={
                        "job_type": job_type,
                        "duration_seconds": round(time.monotonic() - started_at, 6),
                    },
                )
            )
            await event_bus.publish_event(
                ArtifactEvent(
                    event_type="job.artifact_saved",
                    correlation_id=correlation_id,
                    job_id=job_id_str,
                    message="Job output persisted",
                    payload={"output_path": str(get_settings().artifacts_path / job_id_str)},
                )
            )

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            async with self._session_factory() as session:
                job_repo = JobRepository(session)
                await job_repo.mark_failed(job_id, error_msg)
                await session.commit()
            await event_bus.publish_event(
                JobEvent(
                    event_type="job.failed",
                    correlation_id=correlation_id,
                    job_id=job_id_str,
                    level="error",
                    message=str(exc),
                    payload={"job_type": job_type},
                )
            )
            if "out of memory" in str(exc).lower():
                await event_bus.publish_event(
                    ResourceEvent(
                        event_type="resource.oom",
                        correlation_id=correlation_id,
                        job_id=job_id_str,
                        level="error",
                        message="CUDA out-of-memory detected",
                    )
                )
            logger.exception("Job %s failed", job_id)

        finally:
            if needs_gpu:
                self._resource_coordinator.release(job_id_str)
                await event_bus.publish_event(
                    ResourceEvent(
                        event_type="resource.lock_released",
                        correlation_id=correlation_id,
                        job_id=job_id_str,
                        message="GPU lock released",
                    )
                )

    async def _dispatch(self, job_id: UUID, job_type: str, params: dict) -> None:
        """
        Route a job to the correct adapter via the AdapterRegistry.

        Extracts parameters from the job's params dict and calls the
        appropriate adapter method based on job_type.
        """
        settings = get_settings()
        correlation_id = params.get("correlation_id")

        # Determine backend override (if specified in job params) — use .get() to avoid mutating
        backend_str = params.get("backend")
        backend = InferenceBackend(backend_str) if backend_str else None

        # Capture the event loop BEFORE adapters spawn threads so the
        # progress callback can safely schedule coroutines from worker threads.
        _loop = asyncio.get_running_loop()

        # Build progress callback that publishes typed events to the event bus
        def on_progress(progress: "JobProgress") -> None:
            """Convert adapter progress callback into a typed observability event."""

            async def _publish_progress() -> None:
                await event_bus.publish_event(
                    JobEvent(
                        event_type="job.progress",
                        correlation_id=correlation_id,
                        job_id=str(job_id),
                        message=progress.message,
                        payload={
                            "status": progress.status,
                            "progress_percent": progress.progress_percent,
                            "current_step": progress.current_step,
                            "total_steps": progress.total_steps,
                            "model_id": params.get("model_id"),
                            "pipeline": str(
                                backend.value if backend else settings.inference_backend
                            ),
                            "adapter": "registry_dispatch",
                        },
                    )
                )

            asyncio.run_coroutine_threadsafe(_publish_progress(), _loop)

        # Get output directory for artifacts
        output_dir = settings.artifacts_path / str(job_id)

        if job_type == JobType.TEXT_TO_IMAGE:
            adapter = self._adapter_registry.get_provider(JobType.TEXT_TO_IMAGE, backend)
            gen_params = GenerationParams(
                prompt=params["prompt"],
                negative_prompt=params.get("negative_prompt", ""),
                width=params.get("width", 512),
                height=params.get("height", 512),
                num_inference_steps=params.get("num_inference_steps", 20),
                guidance_scale=params.get("guidance_scale", 7.5),
                seed=params.get("seed"),
                num_images=params.get("num_images", 1),
            )
            await adapter.generate(
                gen_params, params["model_id"], output_dir, on_progress=on_progress
            )

        elif job_type == JobType.IMAGE_TO_IMAGE:
            adapter = self._adapter_registry.get_provider(JobType.IMAGE_TO_IMAGE, backend)
            gen_params = GenerationParams(
                prompt=params["prompt"],
                negative_prompt=params.get("negative_prompt", ""),
                width=params.get("width", 512),
                height=params.get("height", 512),
                num_inference_steps=params.get("num_inference_steps", 20),
                guidance_scale=params.get("guidance_scale", 7.5),
                seed=params.get("seed"),
                num_images=params.get("num_images", 1),
            )
            await adapter.generate(
                gen_params,
                params["model_id"],
                Path(params["source_image_path"]),
                output_dir,
                strength=params.get("strength", 0.75),
                on_progress=on_progress,
            )

        elif job_type == JobType.IMAGE_CAPTIONING:
            adapter = self._adapter_registry.get_provider(JobType.IMAGE_CAPTIONING, backend)
            await adapter.caption(
                Path(params["image_path"]),
                params["model_id"],
                prompt=params.get("prompt", ""),
            )

        elif job_type == JobType.VIDEO_GENERATION:
            adapter = self._adapter_registry.get_provider(JobType.VIDEO_GENERATION, backend)
            gen_params = GenerationParams(
                prompt=params.get("prompt", ""),
                negative_prompt=params.get("negative_prompt", ""),
                width=params.get("width", 512),
                height=params.get("height", 512),
                num_inference_steps=params.get("num_inference_steps", 25),
                guidance_scale=params.get("guidance_scale", 7.5),
                seed=params.get("seed"),
            )
            source_path = (
                Path(params["source_image_path"]) if params.get("source_image_path") else None
            )
            await adapter.generate(
                gen_params,
                params["model_id"],
                output_dir,
                source_image_path=source_path,
                on_progress=on_progress,
            )

        elif job_type == JobType.LLM_INFERENCE:
            adapter = self._adapter_registry.get_provider(JobType.LLM_INFERENCE, backend)
            await adapter.generate(
                params["messages"],
                params["model_id"],
                max_tokens=params.get("max_tokens", 512),
                temperature=params.get("temperature", 0.7),
            )

        elif job_type == JobType.MODEL_DOWNLOAD:
            logger.info("Dispatching model download job %s", job_id)
            await self._handle_model_download(job_id, params)

        elif job_type == JobType.MODEL_LOAD:
            logger.info("Dispatching model load job %s", job_id)
            await self._handle_model_load(job_id, params)
            await event_bus.publish_event(
                ModelEvent(
                    event_type="job.model_loaded",
                    correlation_id=params.get("correlation_id"),
                    job_id=str(job_id),
                    message=f"Model loaded: {params['model_id']}",
                    payload={"model_id": params["model_id"]},
                )
            )

        else:
            raise ValueError(f"Unknown job type: {job_type}")

    async def _handle_model_download(self, job_id: UUID, params: dict) -> None:
        """
        Download model weights from the configured source.

        Uses huggingface_hub for HuggingFace models. Publishes progress
        events so the frontend can show download percentage. Updates
        model.download_progress in the DB during download.
        """
        model_id = params["model_id"]
        source = params.get("source", "huggingface")

        await event_bus.publish_event(
            JobEvent(
                event_type="job.progress",
                job_id=str(job_id),
                message=f"Starting download: {model_id}",
                payload={"status": "running", "progress_percent": 0},
            )
        )

        if source == "huggingface":
            from huggingface_hub import snapshot_download

            settings = get_settings()
            local_dir = settings.models_path / "huggingface" / model_id.replace("/", "--")

            # --- Progress callback that updates DB from the download thread ---
            import threading
            from tqdm import tqdm

            from app.domain.enums import ModelStatus
            from app.infrastructure.database.repositories import ModelRepository

            # Capture worker reference for closure
            worker = self

            # Capture the event loop before entering the thread
            _download_loop = asyncio.get_event_loop()

            # Shared aggregate progress tracker across ALL file downloads
            _shared_progress = {
                "lock": threading.Lock(),
                "total_bytes": 0,
                "downloaded_bytes": 0,
                "previous_percent": -1,
            }

            class _HfTqdmProgress(tqdm):
                 """
                 A tqdm subclass compatible with huggingface_hub's tqdm_class parameter.

                 Tracks aggregate progress across all files by accumulating bytes
                 in a shared dict, so percentage is total downloaded / total size.
                 """

                 _lock = threading.Lock()

                 @classmethod
                 def get_lock(cls) -> threading.Lock:
                     """Return a thread lock, as required by huggingface_hub."""
                     return cls._lock

                 def __init__(self, *args, **kwargs) -> None:
                     file_total = kwargs.get("total", 0)
                     file_desc = kwargs.get("desc", "")
                     self._model_id = model_id
                     self._job_id_str = str(job_id)

                     # Register this file's size in the shared total
                     with _shared_progress["lock"]:
                         _shared_progress["total_bytes"] += file_total

                     super().__init__(*args, **kwargs)

                 def update(self, n: int = 1) -> None:
                     """
                     Accumulate bytes in shared state, compute aggregate %,
                     update DB + log to console every ~5%.
                     """
                     self._downloaded_n = n
                     super().update(n)

                     with _shared_progress["lock"]:
                         _shared_progress["downloaded_bytes"] += n
                         total = _shared_progress["total_bytes"]
                         downloaded = _shared_progress["downloaded_bytes"]
                         prev = _shared_progress["previous_percent"]

                     if total > 0:
                         progress = round((downloaded / total) * 100, 1)
                     else:
                         progress = 0.0

                     # Clamp to 100
                     if progress > 100.0:
                         progress = 100.0

                     # Log to console every ~5% so docker logs show progress
                     if prev < 0 or abs(progress - prev) >= 5.0:
                         with _shared_progress["lock"]:
                             _shared_progress["previous_percent"] = progress

                         logger.info(
                             "Download progress: %.1f%%  (%.1f MB / %.1f MB)",
                             progress,
                             downloaded / (1024 * 1024),
                             total / (1024 * 1024),
                         )

                         # Schedule DB update from the download thread
                         async def _update_progress() -> None:
                             async with worker._session_factory() as session:
                                 model_repo = ModelRepository(session)
                                 await model_repo.update_status(
                                     self._model_id, ModelStatus.DOWNLOADING, download_progress=progress
                                 )
                                 await session.commit()

                         asyncio.run_coroutine_threadsafe(_update_progress(), _download_loop)

                         # Also publish a progress event
                         async def _publish_progress() -> None:
                             await event_bus.publish_event(
                                 JobEvent(
                                     event_type="job.progress",
                                     job_id=self._job_id_str,
                                     message=f"Downloading {self._model_id}: {progress}%",
                                     payload={
                                         "status": "downloading",
                                         "progress_percent": progress,
                                     },
                                 )
                             )

                         asyncio.run_coroutine_threadsafe(_publish_progress(), _download_loop)

            await asyncio.to_thread(
                snapshot_download,
                repo_id=model_id,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                tqdm_class=_HfTqdmProgress,
            )
        elif source == "civitai":
            logger.info("Downloading Civitai model: %s", model_id)

            from app.domain.enums import ModelStatus
            from app.infrastructure.database.repositories import ModelRepository
            from app.services.sources.civitai import CivitaiSourceProvider

            settings = get_settings()
            local_dir = settings.models_path / "civitai" / model_id.replace("/", "--")
            civitai_provider = CivitaiSourceProvider(api_key=settings.civitai_token)

            _civitai_loop = asyncio.get_event_loop()

            async def _civitai_progress(snapshot: dict) -> None:
                """Handle Civitai download progress updates."""
                percent = snapshot.get("percent", 0)
                if percent > 100.0:
                    percent = 100.0

                logger.info("Civitai download progress: %.1f%%", percent)

                async with self._session_factory() as session:
                    model_repo = ModelRepository(session)
                    await model_repo.update_status(
                        model_id, ModelStatus.DOWNLOADING, download_progress=percent
                    )
                    await session.commit()

                await event_bus.publish_event(
                    JobEvent(
                        event_type="job.progress",
                        job_id=str(job_id),
                        message=f"Downloading {model_id}: {percent}%",
                        payload={
                            "status": "downloading",
                            "progress_percent": percent,
                        },
                    )
                )

            # Resolve model info
            model_info = await civitai_provider.resolve_model_info(model_id)
            files = model_info.get("files", [])

            for file_info in files:
                rel_path = file_info.get("relative_path", "model.safetensors")
                dest_file = local_dir / rel_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)

                await civitai_provider.download_file(
                    model_id=model_id,
                    file_path=rel_path,
                    destination=dest_file,
                    on_progress=_civitai_progress,
                )

        elif source == "local":
            logger.info("Importing local model: %s", model_id)

            from app.domain.enums import ModelStatus
            from app.infrastructure.database.repositories import ModelRepository
            from app.services.sources.local import LocalSourceProvider

            settings = get_settings()
            local_dir = settings.models_path / "local" / model_id.replace("/", "--")
            local_provider = LocalSourceProvider()

            # Resolve model info
            model_info = await local_provider.resolve_model_info(model_id)

            # Import/copy files
            result = await local_provider.import_model(
                source_path=Path(model_id),
                destination=local_dir,
                copy=True,
            )

            # Publish import progress
            await event_bus.publish_event(
                JobEvent(
                    event_type="job.progress",
                    job_id=str(job_id),
                    message=f"Imported {len(result.get('files_imported', []))} files from local source",
                    payload={
                        "status": "running",
                        "progress_percent": 100,
                        "files_imported": result.get("files_imported", []),
                        "total_bytes": result.get("total_bytes", 0),
                    },
                )
            )

        else:
            raise NotImplementedError(f"Download source '{source}' not yet supported")

        # Update model status in DB
        async with self._session_factory() as session:
            from app.infrastructure.database.repositories import ModelRepository

            model_repo = ModelRepository(session)
            from app.domain.enums import ModelStatus

            await model_repo.update_status(model_id, ModelStatus.DOWNLOADED, download_progress=100.0)
            await session.commit()

        await event_bus.publish_event(
            JobEvent(
                event_type="job.progress",
                job_id=str(job_id),
                message=f"Download complete: {model_id}",
                payload={"status": "running", "progress_percent": 100},
            )
        )

    async def _handle_model_load(self, job_id: UUID, params: dict) -> None:
        """
        Load model weights into GPU/CPU memory via the ModelManager.

        Retrieves the model manager from the adapter registry's pipeline cache.
        """
        model_id = params["model_id"]
        device = params.get("device", "cuda")

        await event_bus.publish_event(
            JobEvent(
                event_type="job.progress",
                job_id=str(job_id),
                message=f"Loading model: {model_id}",
                payload={"status": "running", "progress_percent": 0},
            )
        )

        # Use DirectModelManager via the pipeline cache in the adapter registry
        from app.adapters.direct.model_manager import DirectModelManager
        from app.adapters.direct.pipeline_cache import PipelineCache

        # Get the pipeline cache from any registered direct adapter
        adapter = self._adapter_registry.get_provider(
            JobType.TEXT_TO_IMAGE, InferenceBackend.DIRECT_PYTHON
        )
        if hasattr(adapter, "_cache"):
            cache: PipelineCache = adapter._cache
            model_manager = DirectModelManager(cache)
            await model_manager.load_model(model_id, device=device)
        else:
            raise RuntimeError("No pipeline cache available for model loading")

        await event_bus.publish_event(
            JobEvent(
                event_type="job.progress",
                job_id=str(job_id),
                message=f"Model loaded: {model_id}",
                payload={"status": "running", "progress_percent": 100},
            )
        )
