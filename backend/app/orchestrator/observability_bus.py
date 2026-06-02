"""
In-process observability event bus with recent-event buffering and metrics.

Design:
- FIFO global buffer for recent events to support "what just happened?" debugging.
- Per-job timeline buffers to support end-to-end traceability per job id.
- Subscriber fan-out for live consumers (e.g. WebSocket streaming).
- Embedded metrics registry for counters, gauges, and histogram summaries.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from app.domain.observability import MetricPoint, ObservabilityEvent

logger = logging.getLogger(__name__)

ObservabilitySubscriber = Callable[[ObservabilityEvent], Any]


class MetricsRegistry:
    """
    Lightweight in-memory metrics registry.

    - Counters: monotonically increasing totals (events/jobs/errors).
    - Gauges: latest snapshot values (queue depth, active jobs, memory).
    - Histograms: statistical timing/value summaries (count/sum/min/max/avg/last).
    """

    def __init__(self) -> None:
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = defaultdict(float)
        # Histogram storage by metric name:
        # count/sum for averages, min/max range, and last observed value.
        self._hist: dict[str, dict[str, float]] = defaultdict(
            lambda: {
                "count": 0.0,
                "sum": 0.0,
                "min": 0.0,
                "max": 0.0,
                "last": 0.0,
            }
        )

    def inc(self, name: str, value: float = 1.0) -> None:
        self._counters[name] += value

    def set_gauge(self, name: str, value: float) -> None:
        self._gauges[name] = value

    def observe(self, name: str, value: float) -> None:
        stats = self._hist[name]
        if stats["count"] == 0:
            stats["min"] = value
            stats["max"] = value
        else:
            stats["min"] = min(stats["min"], value)
            stats["max"] = max(stats["max"], value)
        stats["count"] += 1
        stats["sum"] += value
        stats["last"] = value

    def get_counter(self, name: str) -> float:
        return self._counters.get(name, 0.0)

    def get_gauge(self, name: str) -> float:
        return self._gauges.get(name, 0.0)

    def get_hist_avg(self, name: str) -> float:
        stats = self._hist.get(name)
        if not stats or stats["count"] == 0:
            return 0.0
        return stats["sum"] / stats["count"]

    def snapshot(self) -> dict[str, list[MetricPoint]]:
        now = datetime.now(timezone.utc)
        counters = [
            MetricPoint(name=name, kind="counter", value=value, updated_at=now)
            for name, value in sorted(self._counters.items())
        ]
        gauges = [
            MetricPoint(name=name, kind="gauge", value=value, updated_at=now)
            for name, value in sorted(self._gauges.items())
        ]
        histograms: list[MetricPoint] = []
        for name, stats in sorted(self._hist.items()):
            if stats["count"] == 0:
                continue
            histograms.append(
                MetricPoint(
                    name=name,
                    kind="histogram",
                    value=stats["last"],
                    updated_at=now,
                )
            )
            histograms.append(
                MetricPoint(
                    name=f"{name}.avg",
                    kind="histogram",
                    value=stats["sum"] / stats["count"],
                    updated_at=now,
                )
            )
            histograms.append(
                MetricPoint(
                    name=f"{name}.max",
                    kind="histogram",
                    value=stats["max"],
                    updated_at=now,
                )
            )
        return {"counters": counters, "gauges": gauges, "histograms": histograms}


class ObservabilityBus:
    """
    Event bus for structured observability events.

    Maintains dual buffers (global recent events + per-job timelines), records
    metrics, and broadcasts events to async subscribers. `publish` is async and
    awaits subscribers; `publish_sync` records immediately and schedules
    subscribers only when an event loop is available.
    """

    def __init__(self, max_events: int = 1000, job_timeline_size: int = 300) -> None:
        self._subscribers: list[ObservabilitySubscriber] = []
        self._recent_events: deque[ObservabilityEvent] = deque(maxlen=max_events)
        # Per-job timelines are bounded independently to keep local debugging context.
        self._job_events: dict[str, deque[ObservabilityEvent]] = defaultdict(
            lambda: deque(maxlen=job_timeline_size)
        )
        self.metrics = MetricsRegistry()

    def subscribe(self, callback: ObservabilitySubscriber) -> None:
        self._subscribers.append(callback)

    def unsubscribe(self, callback: ObservabilitySubscriber) -> None:
        self._subscribers = [s for s in self._subscribers if s is not callback]

    async def publish(self, event: ObservabilityEvent) -> None:
        self._record(event)
        for subscriber in self._subscribers:
            try:
                result = subscriber(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("Observability subscriber failed for %s", event.event_id)

    def publish_sync(self, event: ObservabilityEvent) -> None:
        """
        Publish from sync/threaded contexts (non-blocking for subscribers).

        If no event loop is active, the event is still recorded in buffers/metrics,
        but async subscribers are not notified.
        """
        self._record(event)
        loop = _try_get_loop()
        if not loop:
            return
        for subscriber in self._subscribers:
            try:
                result = subscriber(event)
                if asyncio.iscoroutine(result):
                    loop.create_task(result)
            except Exception:
                logger.exception("Observability subscriber failed for %s", event.event_id)

    def recent_events(self, limit: int = 100) -> list[ObservabilityEvent]:
        return list(self._recent_events)[-limit:]

    def job_timeline(self, job_id: str, limit: int = 200) -> list[ObservabilityEvent]:
        timeline = self._job_events.get(job_id, deque())
        return list(timeline)[-limit:]

    def timed(self, metric_name: str) -> "_TimerContext":
        return _TimerContext(self.metrics, metric_name)

    def _record(self, event: ObservabilityEvent) -> None:
        self._recent_events.append(event)
        if event.job_id:
            self._job_events[event.job_id].append(event)
        if event.level == "error":
            self.metrics.inc("observability.error_events")


class _TimerContext:
    def __init__(self, registry: MetricsRegistry, metric_name: str) -> None:
        self._registry = registry
        self._metric_name = metric_name
        self._start = 0.0

    def __enter__(self) -> "_TimerContext":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        elapsed = time.perf_counter() - self._start
        self._registry.observe(self._metric_name, elapsed)


def _try_get_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


observability_bus = ObservabilityBus()
