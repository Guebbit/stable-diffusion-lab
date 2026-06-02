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
import traceback
from pathlib import Path
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.adapters.adapter_registry import AdapterRegistry
from app.adapters.resource_coordinator import ResourceCoordinator
from app.domain.enums import InferenceBackend, JobType
from app.domain.value_objects import GenerationParams, JobProgress
from app.infrastructure.config.settings import get_settings
from app.infrastructure.database.repositories import JobRepository
from app.orchestrator.event_bus import event_bus


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

        # Check for early cancellation
        if job_id in self._cancelled_jobs:
            self._cancelled_jobs.discard(job_id)
            async with self._session_factory() as session:
                job_repo = JobRepository(session)
                await job_repo.mark_cancelled(job_id)
                await session.commit()
            return

        # Determine if this job needs GPU before entering try block
        needs_gpu = job_type in {
            JobType.TEXT_TO_IMAGE,
            JobType.IMAGE_TO_IMAGE,
            JobType.IMAGE_CAPTIONING,
            JobType.VIDEO_GENERATION,
            JobType.LLM_INFERENCE,
        }

        try:
            if needs_gpu:
                await self._resource_coordinator.acquire(job_id_str)

            # Dispatch to appropriate handler via adapter registry
            await self._dispatch(job_id, job_type, params)

            # Mark completed
            async with self._session_factory() as session:
                job_repo = JobRepository(session)
                await job_repo.mark_completed(job_id)
                await session.commit()
            await event_bus.publish(
                JobProgress(
                    job_id=job_id,
                    status="completed",
                    progress_percent=100,
                    message="Job completed successfully",
                )
            )

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            async with self._session_factory() as session:
                job_repo = JobRepository(session)
                await job_repo.mark_failed(job_id, error_msg)
                await session.commit()
            await event_bus.publish(
                JobProgress(
                    job_id=job_id,
                    status="failed",
                    message=str(exc),
                )
            )
            logger.exception("Job %s failed", job_id)

        finally:
            if needs_gpu:
                self._resource_coordinator.release(job_id_str)

    async def _dispatch(self, job_id: UUID, job_type: str, params: dict) -> None:
        """
        Route a job to the correct adapter via the AdapterRegistry.

        Extracts parameters from the job's params dict and calls the
        appropriate adapter method based on job_type.
        """
        settings = get_settings()

        # Determine backend override (if specified in job params) — use .get() to avoid mutating
        backend_str = params.get("backend")
        backend = InferenceBackend(backend_str) if backend_str else None

        # Build progress callback that publishes to the event bus
        def on_progress(progress: JobProgress) -> None:
            # Replace placeholder job_id with the real one
            real_progress = JobProgress(
                job_id=job_id,
                status=progress.status,
                progress_percent=progress.progress_percent,
                current_step=progress.current_step,
                total_steps=progress.total_steps,
                message=progress.message,
            )
            asyncio.get_event_loop().create_task(event_bus.publish(real_progress))

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

        else:
            raise ValueError(f"Unknown job type: {job_type}")

    async def _handle_model_download(self, job_id: UUID, params: dict) -> None:
        """
        Download model weights from the configured source.

        Uses huggingface_hub for HuggingFace models. Publishes progress
        events so the frontend can show download percentage.
        """
        model_id = params["model_id"]
        source = params.get("source", "huggingface")

        await event_bus.publish(
            JobProgress(
                job_id=job_id,
                status="running",
                progress_percent=0,
                message=f"Starting download: {model_id}",
            )
        )

        if source == "huggingface":
            from huggingface_hub import snapshot_download

            settings = get_settings()
            local_dir = settings.models_path / "huggingface" / model_id.replace("/", "--")

            await asyncio.to_thread(
                snapshot_download,
                repo_id=model_id,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )
        else:
            raise NotImplementedError(f"Download source '{source}' not yet supported")

        # Update model status in DB
        async with self._session_factory() as session:
            from app.infrastructure.database.repositories import ModelRepository

            model_repo = ModelRepository(session)
            from app.domain.enums import ModelStatus

            await model_repo.update_status(model_id, ModelStatus.DOWNLOADED)
            await session.commit()

        await event_bus.publish(
            JobProgress(
                job_id=job_id,
                status="running",
                progress_percent=100,
                message=f"Download complete: {model_id}",
            )
        )

    async def _handle_model_load(self, job_id: UUID, params: dict) -> None:
        """
        Load model weights into GPU/CPU memory via the ModelManager.

        Retrieves the model manager from the adapter registry's pipeline cache.
        """
        model_id = params["model_id"]
        device = params.get("device", "cuda")

        await event_bus.publish(
            JobProgress(
                job_id=job_id,
                status="running",
                progress_percent=0,
                message=f"Loading model: {model_id}",
            )
        )

        # Use DirectModelManager via the pipeline cache in the adapter registry
        from app.adapters.direct.model_manager import DirectModelManager
        from app.adapters.direct.pipeline_cache import PipelineCache

        # Get the pipeline cache from any registered direct adapter
        adapter = self._adapter_registry.get_provider(JobType.TEXT_TO_IMAGE, InferenceBackend.DIRECT_PYTHON)
        if hasattr(adapter, "_cache"):
            cache: PipelineCache = adapter._cache
            model_manager = DirectModelManager(cache)
            await model_manager.load_model(model_id, device=device)
        else:
            raise RuntimeError("No pipeline cache available for model loading")

        await event_bus.publish(
            JobProgress(
                job_id=job_id,
                status="running",
                progress_percent=100,
                message=f"Model loaded: {model_id}",
            )
        )
