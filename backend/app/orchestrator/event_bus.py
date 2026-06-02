"""
Event bus — in-process publish/subscribe for typed observability events.

The orchestrator publishes typed events here. The WebSocket hub
subscribes and pushes them to connected clients. This decouples the
worker execution from the real-time delivery mechanism.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any

from app.domain.events import EventRingBuffer, TypedEvent

logger = logging.getLogger(__name__)

# Type alias for event subscribers
TypedSubscriber = Callable[[TypedEvent], Any]


class EventBus:
    """
    Simple in-process event bus for typed observability events.

    Subscribers register callbacks. When an event is published,
    all subscribers are notified asynchronously. Failed subscribers are
    logged but don't block other subscribers.
    """

    def __init__(self) -> None:
        self._typed_subscribers: list[TypedSubscriber] = []
        self._history = EventRingBuffer(size=1000)

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

    def get_recent_events(self, limit: int = 200) -> list[TypedEvent]:
        """Return most recent typed events, newest first."""
        return self._history.get_recent(limit=limit)

    def get_job_events(self, job_id: str, limit: int = 500) -> list[TypedEvent]:
        """Return typed events for one job, chronological."""
        return self._history.get_by_job(job_id=job_id, limit=limit)


# Singleton event bus instance (shared across the application)
event_bus = EventBus()
