"""
WebSocket hub — manages connections and broadcasts job progress events.

Clients connect to /ws/progress and receive real-time updates about
running jobs (progress percentage, step count, completion, errors).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict

from fastapi import WebSocket

from app.domain.events import TypedEvent
from app.domain.value_objects import JobProgress

logger = logging.getLogger(__name__)


class WebSocketHub:
    """
    Manages WebSocket connections and broadcasts progress events.

    Connected clients receive JSON messages for every job progress update.
    The hub subscribes to the EventBus and relays events to all clients.
    """

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []
        self._typed_connections: dict[WebSocket, set[str]] = {}

    async def connect(self, websocket: WebSocket) -> None:
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self._connections.append(websocket)
        logger.info("WebSocket client connected (%d total)", len(self._connections))

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a disconnected client."""
        self._connections = [ws for ws in self._connections if ws is not websocket]
        logger.info("WebSocket client disconnected (%d remaining)", len(self._connections))

    async def broadcast(self, event: JobProgress) -> None:
        """
        Send a progress event to all connected clients.

        Serializes the event to JSON and sends to each connection.
        Silently removes dead connections.
        """
        if not self._connections:
            return

        # Serialize with UUID/datetime handling
        data = asdict(event)
        data["job_id"] = str(data["job_id"])
        data["timestamp"] = data["timestamp"].isoformat()
        message = json.dumps(data)

        dead_connections: list[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_text(message)
            except Exception:
                dead_connections.append(ws)

        # Clean up dead connections
        for ws in dead_connections:
            self.disconnect(ws)

    async def connect_typed(self, websocket: WebSocket, subscribe: set[str] | None = None) -> None:
        """Accept a typed event stream connection with optional filters."""
        await websocket.accept()
        self._typed_connections[websocket] = subscribe or set()
        logger.info("Typed WebSocket client connected (%d total)", len(self._typed_connections))

    def disconnect_typed(self, websocket: WebSocket) -> None:
        """Remove a disconnected typed event client."""
        if websocket in self._typed_connections:
            del self._typed_connections[websocket]
        logger.info(
            "Typed WebSocket client disconnected (%d remaining)", len(self._typed_connections)
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
        return len(self._connections)


# Singleton hub instance
ws_hub = WebSocketHub()
