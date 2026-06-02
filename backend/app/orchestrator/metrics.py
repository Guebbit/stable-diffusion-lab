"""In-memory metrics registry for observability snapshots."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import quantiles

from app.domain.events import TypedEvent


@dataclass
class MetricsRegistry:
    """Stores counters, gauges, and histograms in memory."""

    counters: dict[str, int] = field(
        default_factory=lambda: {
            "jobs_submitted": 0,
            "jobs_completed": 0,
            "jobs_failed": 0,
            "model_loads": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "oom_errors": 0,
        }
    )
    gauges: dict[str, float] = field(
        default_factory=lambda: {
            "queue_depth": 0,
            "active_jobs": 0,
            "cached_pipelines": 0,
            "cuda_allocated_mb": 0,
            "cuda_reserved_mb": 0,
            "disk_free_mb": 0,
        }
    )
    histograms: dict[str, deque[float]] = field(
        default_factory=lambda: {
            "queue_wait_time": deque(maxlen=1000),
            "job_execution_time": deque(maxlen=1000),
            "model_load_duration": deque(maxlen=1000),
            "pipeline_acquisition_time": deque(maxlen=1000),
            "artifact_save_duration": deque(maxlen=1000),
        }
    )

    def inc_counter(self, name: str, value: int = 1) -> None:
        """Increment a counter."""
        self.counters[name] = self.counters.get(name, 0) + value

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge value."""
        self.gauges[name] = value

    def set_counter(self, name: str, value: int) -> None:
        """Set counter directly when source-of-truth is external state.

        Use this for snapshot-synchronized counters (for example PipelineCache
        hit/miss totals) and use ``inc_counter`` for event-driven increments.
        """
        self.counters[name] = value

    def observe_histogram(self, name: str, value: float) -> None:
        """Record one histogram observation."""
        series = self.histograms.setdefault(name, deque(maxlen=1000))
        series.append(value)

    def snapshot(self) -> dict[str, dict[str, object]]:
        """Return current counters, gauges, and histogram percentiles."""
        histogram_view: dict[str, dict[str, float]] = {}
        for key, values in self.histograms.items():
            series = list(values)
            if not series:
                histogram_view[key] = {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
                continue
            if len(series) == 1:
                one = float(series[0])
                histogram_view[key] = {"count": 1, "p50": one, "p95": one, "p99": one}
                continue
            q = quantiles(series, n=100, method="inclusive")
            histogram_view[key] = {
                "count": float(len(series)),
                "p50": q[49],
                "p95": q[94],
                "p99": q[98],
            }

        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": histogram_view,
        }

    def record_event(self, event: TypedEvent) -> None:
        """Update metrics from a typed observability event."""
        event_type = event.event_type
        if event_type == "job.enqueued":
            self.inc_counter("jobs_submitted")
        elif event_type == "job.completed":
            self.inc_counter("jobs_completed")
        elif event_type == "job.failed":
            self.inc_counter("jobs_failed")
        elif event_type == "job.model_loaded":
            self.inc_counter("model_loads")
        elif event_type == "resource.oom":
            self.inc_counter("oom_errors")
