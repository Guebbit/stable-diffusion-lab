"""
Event bus — in-process publish/subscribe for progress events.

The orchestrator publishes job progress events here. The WebSocket hub
subscribes and pushes them to connected clients. This decouples the
worker execution from the real-time delivery mechanism.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any

from app.domain.events import EventRingBuffer, JobEvent, TypedEvent
from app.domain.value_objects import JobProgress

logger = logging.getLogger(__name__)

# Type alias for event subscribers
ProgressSubscriber = Callable[[JobProgress], Any]
TypedSubscriber = Callable[[TypedEvent], Any]


class EventBus:
    """
    Simple in-process event bus for job progress notifications.

    Subscribers register callbacks. When a progress event is published,
    all subscribers are notified asynchronously. Failed subscribers are
    logged but don't block other subscribers.
    """

    def __init__(self) -> None:
        self._subscribers: list[ProgressSubscriber] = []
        self._typed_subscribers: list[TypedSubscriber] = []
        self._history = EventRingBuffer(size=1000)

    def subscribe(self, callback: ProgressSubscriber) -> None:
        """Register a callback to receive progress events."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: ProgressSubscriber) -> None:
        """Remove a previously registered callback."""
        self._subscribers = [s for s in self._subscribers if s is not callback]

    async def publish(self, event: JobProgress) -> None:
        """
        Broadcast a progress event to all subscribers.

        Each subscriber is called independently. If one fails, others
        still receive the event.
        """
        for subscriber in self._subscribers:
            try:
                result = subscriber(event)
                # Support both sync and async subscribers
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("Event subscriber failed for job %s", event.job_id)

    def subscribe_typed(self, callback: TypedSubscriber) -> None:
        """Register a callback for typed observability events."""
        self._typed_subscribers.append(callback)

    def unsubscribe_typed(self, callback: TypedSubscriber) -> None:
        """Unregister a typed event callback."""
        self._typed_subscribers = [s for s in self._typed_subscribers if s is not callback]

    async def publish_event(self, event: TypedEvent) -> None:
        """Broadcast typed events and store them in memory history."""
        self._history.append(event)
        logger.info(json.dumps(event.to_dict(), ensure_ascii=False, default=str))
        for subscriber in self._typed_subscribers:
            try:
                result = subscriber(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception(
                    "Typed event subscriber failed for event %s (%s)",
                    event.event_id,
                    event.event_type,
                )

    async def publish_progress_as_typed(
        self,
        event: JobProgress,
        correlation_id: str | None = None,
    ) -> None:
        """Bridge legacy progress updates into the typed event stream."""
        typed = JobEvent(
            event_type="job.progress",
            correlation_id=correlation_id,
            job_id=str(event.job_id),
            message=event.message,
            payload={
                "status": event.status,
                "progress_percent": event.progress_percent,
                "current_step": event.current_step,
                "total_steps": event.total_steps,
            },
        )
        await self.publish_event(typed)

    def get_recent_events(self, limit: int = 200) -> list[TypedEvent]:
        """Return most recent typed events, newest first."""
        return self._history.get_recent(limit=limit)

    def get_job_events(self, job_id: str, limit: int = 500) -> list[TypedEvent]:
        """Return typed events for one job, chronological."""
        return self._history.get_by_job(job_id=job_id, limit=limit)


# Singleton event bus instance (shared across the application)
event_bus = EventBus()
