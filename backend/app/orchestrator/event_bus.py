"""
Event bus — in-process publish/subscribe for progress events.

The orchestrator publishes job progress events here. The WebSocket hub
subscribes and pushes them to connected clients. This decouples the
worker execution from the real-time delivery mechanism.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from app.domain.value_objects import JobProgress

logger = logging.getLogger(__name__)

# Type alias for event subscribers
ProgressSubscriber = Callable[[JobProgress], Any]


class EventBus:
    """
    Simple in-process event bus for job progress notifications.

    Subscribers register callbacks. When a progress event is published,
    all subscribers are notified asynchronously. Failed subscribers are
    logged but don't block other subscribers.
    """

    def __init__(self) -> None:
        self._subscribers: list[ProgressSubscriber] = []

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


# Singleton event bus instance (shared across the application)
event_bus = EventBus()
