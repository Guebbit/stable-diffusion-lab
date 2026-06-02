from app.orchestrator.metrics import MetricsRegistry


def test_metrics_registry_snapshot_includes_percentiles() -> None:
    metrics = MetricsRegistry()
    metrics.inc_counter("jobs_submitted")
    metrics.set_gauge("queue_depth", 3)
    for value in (1.0, 2.0, 3.0, 4.0):
        metrics.observe_histogram("job_execution_time", value)

    snapshot = metrics.snapshot()

    assert snapshot["counters"]["jobs_submitted"] == 1
    assert snapshot["gauges"]["queue_depth"] == 3
    assert snapshot["histograms"]["job_execution_time"]["p50"] >= 2.0
    assert (
        snapshot["histograms"]["job_execution_time"]["p99"]
        >= snapshot["histograms"]["job_execution_time"]["p95"]
    )
