from __future__ import annotations

from app.domain.observability import ObservabilityEvent
from app.orchestrator.observability_bus import MetricsRegistry, ObservabilityBus


def test_metrics_registry_tracks_counter_gauge_histogram() -> None:
    registry = MetricsRegistry()
    registry.inc("jobs.submitted")
    registry.set_gauge("jobs.active", 2)
    registry.observe("job.execution_time_seconds", 1.2)
    registry.observe("job.execution_time_seconds", 0.8)

    assert registry.get_counter("jobs.submitted") == 1
    assert registry.get_gauge("jobs.active") == 2
    assert registry.get_hist_avg("job.execution_time_seconds") == 1.0


def test_observability_bus_buffers_recent_and_job_timeline() -> None:
    bus = ObservabilityBus(max_events=3)
    # max_events=3 keeps only the newest three global events (FIFO eviction).
    bus.publish_sync(ObservabilityEvent(event_type="one", job_id="job-1", correlation_id="job-1"))
    bus.publish_sync(ObservabilityEvent(event_type="two", job_id="job-2", correlation_id="job-2"))
    bus.publish_sync(ObservabilityEvent(event_type="three", job_id="job-1", correlation_id="job-1"))
    bus.publish_sync(ObservabilityEvent(event_type="four", job_id="job-1", correlation_id="job-1"))

    assert [e.event_type for e in bus.recent_events(limit=10)] == ["two", "three", "four"]
    assert [e.event_type for e in bus.job_timeline("job-1", limit=10)] == [
        "one",
        "three",
        "four",
    ]
