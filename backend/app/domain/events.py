"""Typed observability domain events and in-memory history buffer."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class TypedEvent:
    """Base event for all observability messages."""

    event_id: str = field(default_factory=lambda: str(uuid4()))
    event_type: str = "system.event"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: str | None = None
    component: str = "system"
    level: str = "info"
    job_id: str | None = None
    message: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize event for APIs/logs/websockets."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


@dataclass(frozen=True, slots=True)
class SystemEvent(TypedEvent):
    """System-level event."""

    event_type: str = "system.event"
    component: str = "system"


@dataclass(frozen=True, slots=True)
class JobEvent(TypedEvent):
    """Job lifecycle event."""

    event_type: str = "job.event"
    component: str = "job"


@dataclass(frozen=True, slots=True)
class ModelEvent(TypedEvent):
    """Model lifecycle event."""

    event_type: str = "model.event"
    component: str = "model"


@dataclass(frozen=True, slots=True)
class ResourceEvent(TypedEvent):
    """Resource lock and GPU event."""

    event_type: str = "resource.event"
    component: str = "resource"


@dataclass(frozen=True, slots=True)
class ArtifactEvent(TypedEvent):
    """Artifact save/availability event."""

    event_type: str = "artifact.event"
    component: str = "artifact"


class EventRingBuffer:
    """Fixed-size in-memory event history."""

    def __init__(self, size: int = 1000) -> None:
        self._events: deque[TypedEvent] = deque(maxlen=size)

    def append(self, event: TypedEvent) -> None:
        """Push newest event; oldest is dropped automatically."""
        self._events.append(event)

    def get_recent(self, limit: int = 200) -> list[TypedEvent]:
        """Get recent events, newest first."""
        if limit <= 0:
            return []
        return list(reversed(list(self._events)[-limit:]))

    def get_by_job(self, job_id: str, limit: int = 500) -> list[TypedEvent]:
        """Get events for one job in chronological order."""
        filtered = [event for event in self._events if event.job_id == job_id]
        if limit <= 0:
            return []
        return filtered[-limit:]
