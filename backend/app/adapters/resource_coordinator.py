"""
Resource coordinator — manages GPU mutex and VRAM budgeting.

Ensures only one heavy inference operation uses the GPU at a time.
This prevents CUDA OOM crashes that happen when multiple models or
large batch operations compete for limited GPU memory.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


class ResourceCoordinator:
    """
    Coordinates access to the GPU as a shared resource.

    Uses an asyncio Lock to serialize GPU-intensive operations.
    Only one inference job can hold the GPU at a time — others wait in queue.

    Future enhancement: VRAM budget tracking to allow concurrent small jobs
    if total VRAM usage stays within budget.
    """

    def __init__(self, max_concurrent: int = 1) -> None:
        # Semaphore allows configurable concurrency (default 1 = mutex)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._current_holder: str | None = None

    async def acquire(self, job_id: str) -> None:
        """
        Acquire the GPU lock for a job.

        Blocks until the GPU is available. Jobs are served FIFO
        by asyncio's semaphore implementation.
        """
        logger.debug("Job %s waiting for GPU lock", job_id)
        await self._semaphore.acquire()
        self._current_holder = job_id
        logger.info("Job %s acquired GPU lock", job_id)

    def release(self, job_id: str) -> None:
        """Release the GPU lock after a job completes or fails."""
        self._semaphore.release()
        self._current_holder = None
        logger.info("Job %s released GPU lock", job_id)

    @property
    def is_busy(self) -> bool:
        """Check if the GPU is currently in use."""
        return self._current_holder is not None

    @property
    def current_holder(self) -> str | None:
        """Return the job_id currently holding the GPU lock."""
        return self._current_holder
