"""
Resource coordinator — manages GPU mutex, VRAM budgeting, and device status.

Ensures only one heavy inference operation uses the GPU at a time.
This prevents CUDA OOM crashes that happen when multiple models or
large batch operations compete for limited GPU memory.

Extended capabilities:
- VRAM usage tracking (estimated, from adapter reports)
- Device status reporting (for system/status API endpoint)
- Configurable concurrency (default: 1 = exclusive GPU access)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from app.domain.observability import ObservabilityEvent
from app.orchestrator.observability_bus import observability_bus


logger = logging.getLogger(__name__)


@dataclass
class DeviceStatus:
    """Snapshot of GPU/device state for the system status API."""

    is_busy: bool = False
    current_job_id: str | None = None
    estimated_vram_used_mb: int = 0
    vram_budget_mb: int = 0
    total_jobs_processed: int = 0
    device_name: str = "unknown"
    last_activity: datetime | None = None


class ResourceCoordinator:
    """
    Coordinates access to the GPU as a shared resource.

    Uses an asyncio Semaphore to serialize GPU-intensive operations.
    Only one inference job can hold the GPU at a time — others wait in queue.

    VRAM tracking:
    - Jobs report their estimated VRAM usage on acquire
    - The coordinator tracks total estimated usage
    - Future: allow concurrent jobs if they fit within budget

    Device status:
    - Exposes get_device_status() for the system/status API
    - Tracks total jobs processed for monitoring
    """

    def __init__(
        self,
        max_concurrent: int = 1,
        vram_budget_mb: int = 0,
    ) -> None:
        """
        Initialize the resource coordinator.

        Args:
            max_concurrent: Maximum concurrent GPU jobs (default 1 = mutex).
            vram_budget_mb: VRAM budget in MB (0 = no budget, rely on semaphore only).
        """
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._vram_budget_mb = vram_budget_mb
        self._current_holder: str | None = None
        self._estimated_vram_used_mb: int = 0
        self._total_jobs_processed: int = 0
        self._last_activity: datetime | None = None
        self._acquired_at: datetime | None = None

    async def acquire(self, job_id: str, estimated_vram_mb: int = 0) -> None:
        """
        Acquire the GPU lock for a job.

        Blocks until the GPU is available. Jobs are served FIFO
        by asyncio's semaphore implementation.

        Args:
            job_id: Unique identifier for the requesting job.
            estimated_vram_mb: Expected VRAM usage (for tracking, not gating yet).
        """
        logger.debug("Job %s waiting for GPU lock (est. %d MB)", job_id, estimated_vram_mb)
        observability_bus.publish_sync(
            ObservabilityEvent(
                event_type="resource.lock.waiting",
                component="resource_coordinator",
                message=f"Job {job_id} waiting for GPU lock",
                job_id=job_id,
                correlation_id=job_id,
                payload={"estimated_vram_mb": estimated_vram_mb},
            )
        )
        await self._semaphore.acquire()
        self._current_holder = job_id
        self._estimated_vram_used_mb = estimated_vram_mb
        self._last_activity = datetime.now(timezone.utc)
        self._acquired_at = self._last_activity
        observability_bus.metrics.set_gauge("gpu.lock_held", 1.0)
        observability_bus.publish_sync(
            ObservabilityEvent(
                event_type="resource.lock.acquired",
                component="resource_coordinator",
                message=f"Job {job_id} acquired GPU lock",
                job_id=job_id,
                correlation_id=job_id,
                payload={"estimated_vram_mb": estimated_vram_mb},
            )
        )
        logger.info("Job %s acquired GPU lock (est. %d MB VRAM)", job_id, estimated_vram_mb)

    def release(self, job_id: str) -> None:
        """Release the GPU lock after a job completes or fails."""
        self._semaphore.release()
        self._current_holder = None
        self._estimated_vram_used_mb = 0
        self._total_jobs_processed += 1
        self._last_activity = datetime.now(timezone.utc)
        # Guard against unexpected release-without-acquire edge cases.
        if self._acquired_at is None:
            logger.warning("Release called without a recorded acquire for job %s", job_id)
        held_seconds = (
            (self._last_activity - self._acquired_at).total_seconds() if self._acquired_at else 0.0
        )
        if held_seconds > 0:
            observability_bus.metrics.observe("resource.lock_held_seconds", held_seconds)
        observability_bus.metrics.set_gauge("gpu.lock_held", 0.0)
        observability_bus.publish_sync(
            ObservabilityEvent(
                event_type="resource.lock.released",
                component="resource_coordinator",
                message=f"Job {job_id} released GPU lock",
                job_id=job_id,
                correlation_id=job_id,
                payload={"held_seconds": held_seconds},
            )
        )
        self._acquired_at = None
        logger.info("Job %s released GPU lock", job_id)

    @property
    def is_busy(self) -> bool:
        """Check if the GPU is currently in use."""
        return self._current_holder is not None

    @property
    def current_holder(self) -> str | None:
        """Return the job_id currently holding the GPU lock."""
        return self._current_holder

    def can_fit(self, estimated_vram_mb: int) -> bool:
        """
        Check if a job with the given VRAM estimate could fit.

        Currently simple: if GPU is free, it can fit.
        Future: check against budget for concurrent scheduling.
        """
        if self._vram_budget_mb == 0:
            return not self.is_busy
        return (
            not self.is_busy
            or (self._estimated_vram_used_mb + estimated_vram_mb) <= self._vram_budget_mb
        )

    def get_device_status(self) -> DeviceStatus:
        """
        Get current device status snapshot for the system API.

        Returns comprehensive state info for monitoring and display.
        """
        device_name = "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
        except ImportError:
            pass

        return DeviceStatus(
            is_busy=self.is_busy,
            current_job_id=self._current_holder,
            estimated_vram_used_mb=self._estimated_vram_used_mb,
            vram_budget_mb=self._vram_budget_mb,
            total_jobs_processed=self._total_jobs_processed,
            device_name=device_name,
            last_activity=self._last_activity,
        )
