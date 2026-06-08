"""
SSE hub — manages Server-Sent Event connections and broadcasts typed observability events.

Clients connect to /sse/observability and receive a unidirectional stream of
JSON-encoded events. SSE provides native reconnection and is simpler than
WebSockets for server-to-client-only communication.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict

from fastapi import Request
from fastapi.responses import StreamingResponse

from app.domain.events import TypedEvent

logger = logging.getLogger(__name__)


class SSEHub:
    """
    Manages SSE client streams and broadcasts observability events.

    Each client gets an async generator that yields properly formatted
    SSE frames. The hub subscribes to the EventBus and relays events
    to all connected SSE streams.
    """

    def __init__(self) -> None:
        # Map category → list of asyncio.Queue
        self._streams: dict[str, list[asyncio.Queue]] = defaultdict(list)
        # Global stream (no filters)
        self._global_streams: list[asyncio.Queue] = []

    @property
    def connection_count(self) -> int:
        """Number of active SSE connections."""
        return len(self._global_streams) + sum(len(q_list) for q_list in self._streams.values())

    async def broadcast_typed(self, event: TypedEvent) -> None:
        """Broadcast a typed event to all connected SSE streams."""
        payload = event.to_dict()
        payload["type"] = event.event_type
        sse_data = json.dumps(payload)
        category = event.event_type.split(".")[0]

        # Send to global streams (no filters)
        dead_global: list[asyncio.Queue] = []
        for queue in self._global_streams:
            try:
                queue.put_nowait(f"data: {sse_data}\n\n")
            except asyncio.QueueFull:
                dead_global.append(queue)

        for queue in dead_global:
            if queue in self._global_streams:
                self._global_streams.remove(queue)

        # Send to filtered streams
        if category in self._streams:
            dead_filtered: list[asyncio.Queue] = []
            for queue in self._streams[category]:
                try:
                    queue.put_nowait(f"data: {sse_data}\n\n")
                except asyncio.QueueFull:
                    dead_filtered.append(queue)

            for queue in dead_filtered:
                if queue in self._streams[category]:
                    self._streams[category].remove(queue)

    async def create_stream(
        self, request: Request, subscribe: set[str] | None = None
    ) -> StreamingResponse:
        """
        Create a new SSE stream for a client.

        Returns a StreamingResponse that yields SSE-formatted frames.
        The stream automatically closes when the client disconnects.
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)

        if subscribe:
            for category in subscribe:
                self._streams[category].append(queue)
        else:
            self._global_streams.append(queue)

        logger.info("SSE client connected (%d total)", self.connection_count)

        async def event_generator() -> None:
            try:
                while True:
                    # Check if client disconnected
                    if await request.is_disconnected():
                        break

                    try:
                        data = await asyncio.wait_for(queue.get(), timeout=1.0)
                        yield data
                    except asyncio.TimeoutError:
                        # Keep-alive: yield empty comment to prevent proxy timeout
                        yield ": keepalive\n\n"
            finally:
                # Clean up on disconnect
                if queue in self._global_streams:
                    self._global_streams.remove(queue)
                for category in list(self._streams.keys()):
                    if queue in self._streams[category]:
                        self._streams[category].remove(queue)
                logger.info("SSE client disconnected (%d remaining)", self.connection_count)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )


# Singleton hub instance
sse_hub = SSEHub()