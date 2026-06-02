"""
Observability domain models — typed system events and metric snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class ObservabilityEvent:
    """Typed event emitted by backend components."""

    event_id: str = field(default_factory=lambda: str(uuid4()))
    event_type: str = "system.event"
    component: str = "system"
    level: str = "info"
    message: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    job_id: str | None = None
    correlation_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MetricPoint:
    """Current value for a metric key."""

    name: str
    kind: str
    value: float
    unit: str = ""
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

