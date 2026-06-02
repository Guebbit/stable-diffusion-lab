"""
WebSocket hub — manages connections and broadcasts typed observability events.

Clients connect to /ws/observability and receive real-time typed events
(job progress, resource changes, model events, artifacts).
"""

from __future__ import annotations

import json
import logging

from fastapi import WebSocket

from app.domain.events import TypedEvent

logger = logging.getLogger(__name__)


class WebSocketHub:
    """
    Manages typed WebSocket connections and broadcasts observability events.

    Connected clients receive JSON messages filtered by event category.
    The hub subscribes to the EventBus and relays events to all clients.
    """

    def __init__(self) -> None:
        self._typed_connections: dict[WebSocket, set[str]] = {}

    async def connect_typed(self, websocket: WebSocket, subscribe: set[str] | None = None) -> None:
        """Accept a typed event stream connection with optional filters."""
        await websocket.accept()
        self._typed_connections[websocket] = subscribe or set()
        logger.info("WebSocket client connected (%d total)", len(self._typed_connections))

    def disconnect_typed(self, websocket: WebSocket) -> None:
        """Remove a disconnected typed event client."""
        if websocket in self._typed_connections:
            del self._typed_connections[websocket]
        logger.info(
            "WebSocket client disconnected (%d remaining)", len(self._typed_connections)
        )

    async def broadcast_typed(self, event: TypedEvent) -> None:
        """Broadcast typed events with optional per-connection filtering."""
        if not self._typed_connections:
            return
        payload = event.to_dict()
        payload["type"] = event.event_type
        message = json.dumps(payload)
        category = event.event_type.split(".")[0]

        dead_connections: list[WebSocket] = []
        for ws, filters in self._typed_connections.items():
            if filters and category not in filters:
                continue
            try:
                await ws.send_text(message)
            except Exception:
                dead_connections.append(ws)

        for ws in dead_connections:
            self.disconnect_typed(ws)

    @property
    def connection_count(self) -> int:
        """Number of active WebSocket connections."""
        return len(self._typed_connections)


# Singleton hub instance
ws_hub = WebSocketHub()
