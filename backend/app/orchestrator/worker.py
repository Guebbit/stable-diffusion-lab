"""
Job worker — polls the queue and orchestrates job execution.

The worker loop runs as a background asyncio task. It polls the database
for PENDING jobs and delegates actual execution to specialized handlers:

- :class:`PipelineExecutor` — GPU-bound inference pipelines (txt2img, img2img, etc.)
- :class:`ModelOperationHandler` — model lifecycle jobs (download, load, etc.)

State machine enforced by the worker:
    PENDING → RUNNING → COMPLETED
                      → FAILED
                      → CANCELLED (if cancellation was requested)
"""

from __future__ import annotations

import asyncio
import gc
import logging
import threading
import time
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.adapters.adapter_registry import AdapterRegistry
from app.adapters.base import GenerationCancelledError
from app.adapters.direct.pipeline_cache import PipelineCache
from app.adapters.resource_coordinator import ResourceCoordinator
from app.domain.events import ArtifactEvent, JobEvent, ResourceEvent
from app.domain.enums import JobType
from app.domain.value_objects import ArtifactReference
from app.infrastructure.config.settings import get_settings
from app.infrastructure.database.models import ArtifactRecord
from app.infrastructure.database.repositories import ArtifactRepository, JobRepository
from app.orchestrator.event_bus import event_bus
from app.orchestrator.pipeline_executor import PipelineExecutor
from app.services.model_operation_handler import ModelOperationHandler

logger = logging.getLogger(__name__)

# Jobs that require the GPU resource lock.
_GPU_JOB_TYPES: frozenset[JobType] = frozenset(
    {
        JobType.TEXT_TO_IMAGE,
        JobType.IMAGE_TO_IMAGE,
        JobType.IMAGE_ANALYSIS,
        JobType.UPSCALE,
        JobType.RECOLOR,
        JobType.SKETCH_TO_INK,
        JobType.VIDEO_GENERATION,
        JobType.LLM_INFERENCE,
    }
)

# Jobs that are model lifecycle operations (download, delete, refresh).
_MODEL_JOB_TYPES: frozenset[JobType] = frozenset(
    {
        JobType.MODEL_DOWNLOAD,
        JobType.MODEL_DELETE,
        JobType.MODEL_REFRESH,
    }
)


class JobWorker:
    """
    Background worker that processes jobs from the queue.

    Lifecycle:
    1. Polls DB for PENDING jobs at a configurable interval
    2. Claims a job (atomic status update to RUNNING)
    3. Delegates execution to the appropriate handler
    4. Updates job to COMPLETED or FAILED
    5. Broadcasts events via EventBus
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        resource_coordinator: ResourceCoordinator,
        adapter_registry: AdapterRegistry,
        pipeline_cache: PipelineCache | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._resource_coordinator = resource_coordinator
        self._adapter_registry = adapter_registry
        self._pipeline_cache = pipeline_cache
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._cancelled_jobs: set[UUID] = set()
        # Per-job threading.Event for cooperative mid-inference cancellation.
        # Set by request_cancellation(); checked by the diffusers step callback.
        self._cancel_events: dict[UUID, threading.Event] = {}

    # ── Lifecycle ────────────────────────────────────────

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
        """
        Request cancellation of a running or pending job.

        Sets _cancelled_jobs so the next pre-execution check catches it,
        and also fires the per-job threading.Event so the diffusers step
        callback can raise GenerationCancelledError mid-inference.
        """
        self._cancelled_jobs.add(job_id)
        event = self._cancel_events.get(job_id)
        if event is not None:
            event.set()

    # ── Poll loop ────────────────────────────────────────

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

    # ── Job execution with resource locking ──────────────

    async def _execute_job(self, job_id: UUID, job_type: str, params: dict) -> None:
        """Execute a single job with resource locking, timeout, and cleanup."""
        job_id_str = str(job_id)
        correlation_id = params.get("correlation_id")
        started_at = time.monotonic()

        # Check for early cancellation before doing any work
        if job_id in self._cancelled_jobs:
            self._cancelled_jobs.discard(job_id)
            await self._mark_cancelled(job_id)
            await event_bus.publish_event(
                JobEvent(
                    event_type="job.cancelled",
                    correlation_id=correlation_id,
                    job_id=job_id_str,
                    message="Job cancelled before execution",
                )
            )
            return

        job_type_enum = JobType(job_type)
        needs_gpu = job_type_enum in _GPU_JOB_TYPES
        gpu_lock_acquired = False
        _oom_failure = False

        # Per-job cancel event for cooperative mid-inference stop
        cancel_event = threading.Event()
        self._cancel_events[job_id] = cancel_event

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
                await self._acquire_gpu_lock(job_id_str, correlation_id)
                gpu_lock_acquired = True

            # Dispatch with optional timeout
            settings = get_settings()
            timeout = settings.job_timeout_seconds if settings.job_timeout_seconds > 0 else None
            try:
                if timeout:
                    artifacts = await asyncio.wait_for(
                        self._dispatch(job_id, job_type, params, cancel_event),
                        timeout=float(timeout),
                    )
                else:
                    artifacts = await self._dispatch(job_id, job_type, params, cancel_event)
            except asyncio.TimeoutError:
                # Signal the inference thread to stop at the next step callback
                cancel_event.set()
                raise TimeoutError(
                    f"Job timed out after {timeout}s — "
                    "increase JOB_TIMEOUT_SECONDS or reduce generation steps"
                )

            # Persist any produced artifacts to the DB
            if artifacts:
                await self._persist_artifacts(job_id, artifacts, params)

            # Mark completed
            await self._mark_completed(job_id)
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
                    payload={
                        "output_path": str(
                            get_settings().artifacts_path / job_id_str
                        )
                    },
                )
            )

        except GenerationCancelledError:
            # Mid-inference cooperative cancellation from the step callback
            self._cancelled_jobs.discard(job_id)
            await self._mark_cancelled(job_id)
            await event_bus.publish_event(
                JobEvent(
                    event_type="job.cancelled",
                    correlation_id=correlation_id,
                    job_id=job_id_str,
                    message="Job cancelled mid-inference",
                )
            )
            logger.info("Job %s cancelled mid-inference", job_id)

        except Exception as exc:
            # If cancellation was requested at the same time as an exception, treat as cancel
            if job_id in self._cancelled_jobs:
                self._cancelled_jobs.discard(job_id)
                await self._mark_cancelled(job_id)
                await event_bus.publish_event(
                    JobEvent(
                        event_type="job.cancelled",
                        correlation_id=correlation_id,
                        job_id=job_id_str,
                        message="Job cancelled",
                    )
                )
                logger.info("Job %s cancelled (exception during cancellation: %s)", job_id, exc)
            else:
                _oom_failure = "out of memory" in str(exc).lower()
                await self._mark_failed(job_id, exc)
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
                if _oom_failure:
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
            self._cancel_events.pop(job_id, None)
            if gpu_lock_acquired:
                await self._release_gpu_lock(job_id_str, correlation_id)
                await self._gpu_cleanup(_oom_failure)

    # ── GPU cleanup ─────────────────────────────────────

    async def _gpu_cleanup(self, evict_pipelines: bool) -> None:
        """
        Release GPU memory after any GPU job (success, failure, or cancel).

        On OOM, also evicts all cached pipelines — the loaded model is likely
        the cause of memory exhaustion and must be cleared before the next job.
        On all paths, runs gc.collect() + torch.cuda.empty_cache() to free
        temporary tensors produced during inference.
        """
        if evict_pipelines and self._pipeline_cache is not None:
            logger.warning("OOM detected — evicting all cached pipelines to free VRAM")
            await self._pipeline_cache.evict_all()
        try:
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ── Dispatch routing ────────────────────────────────

    async def _dispatch(
        self,
        job_id: UUID,
        job_type: str,
        params: dict,
        cancel_event: threading.Event | None = None,
    ) -> list[ArtifactReference]:
        """Route a job to the correct handler based on job type."""
        job_type_enum = JobType(job_type)

        if job_type_enum in _MODEL_JOB_TYPES:
            await self._model_operation_handler.handle(job_id, job_type, params)
            return []
        elif job_type_enum in _GPU_JOB_TYPES:
            return await self._pipeline_executor.execute(
                job_id, job_type_enum, params, cancel_event=cancel_event
            )
        else:
            raise ValueError(f"Unknown job type: {job_type}")

    # ── GPU lock helpers ────────────────────────────────

    async def _acquire_gpu_lock(
        self, job_id: str, correlation_id: str | None
    ) -> None:
        lock_wait_start = time.monotonic()
        await self._resource_coordinator.acquire(job_id)
        await event_bus.publish_event(
            ResourceEvent(
                event_type="resource.lock_acquired",
                correlation_id=correlation_id,
                job_id=job_id,
                message="GPU lock acquired",
                payload={
                    "acquired_at": datetime.now(timezone.utc).isoformat(),
                    "wait_seconds": round(time.monotonic() - lock_wait_start, 6),
                },
            )
        )

    async def _release_gpu_lock(
        self, job_id: str, correlation_id: str | None
    ) -> None:
        self._resource_coordinator.release(job_id)
        await event_bus.publish_event(
            ResourceEvent(
                event_type="resource.lock_released",
                correlation_id=correlation_id,
                job_id=job_id,
                message="GPU lock released",
            )
        )

    # ── Job status helpers ─────────────────────────────

    async def _mark_completed(self, job_id: UUID) -> None:
        async with self._session_factory() as session:
            job_repo = JobRepository(session)
            await job_repo.mark_completed(job_id)
            await session.commit()

    async def _mark_failed(self, job_id: UUID, exc: Exception) -> None:
        error_msg = f"{type(exc).__name__}: {exc}"
        async with self._session_factory() as session:
            job_repo = JobRepository(session)
            await job_repo.mark_failed(job_id, error_msg)
            await session.commit()

    async def _mark_cancelled(self, job_id: UUID) -> None:
        async with self._session_factory() as session:
            job_repo = JobRepository(session)
            await job_repo.mark_cancelled(job_id)
            await session.commit()

    async def _persist_artifacts(
        self,
        job_id: UUID,
        artifacts: list[ArtifactReference],
        params: dict,
    ) -> None:
        """Save artifact references produced by a generation job to the DB."""
        async with self._session_factory() as session:
            repo = ArtifactRepository(session)
            for ref in artifacts:
                record = ArtifactRecord(
                    id=ref.artifact_id,
                    job_id=job_id,
                    file_path=ref.file_path,
                    thumbnail_path=ref.thumbnail_path,
                    media_type=ref.media_type,
                    size_bytes=ref.size_bytes,
                    width=ref.width,
                    height=ref.height,
                    prompt=params.get("prompt", ""),
                    negative_prompt=params.get("negative_prompt", ""),
                    seed=params.get("seed") or 0,
                    model_name=params.get("model_id", ""),
                    model_id_ref=params.get("original_model_id", params.get("model_id", "")),
                    generation_params=params,
                )
                await repo.create(record)
            await session.commit()
        logger.info("Persisted %d artifact(s) for job %s", len(artifacts), job_id)

    # ── Lazy handler wiring ─────────────────────────────

    @property
    def _pipeline_executor(self) -> PipelineExecutor:
        return PipelineExecutor(
            adapter_registry=self._adapter_registry,
        )

    @property
    def _model_operation_handler(self) -> ModelOperationHandler:
        return ModelOperationHandler(
            session_factory=self._session_factory,
        )
