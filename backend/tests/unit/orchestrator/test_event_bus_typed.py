import pytest

from app.domain.events import JobEvent
from app.orchestrator.event_bus import EventBus


@pytest.mark.asyncio
async def test_event_bus_stores_and_filters_typed_events() -> None:
    bus = EventBus()

    await bus.publish_event(JobEvent(event_type="job.started", job_id="job-1"))
    await bus.publish_event(JobEvent(event_type="job.completed", job_id="job-1"))
    await bus.publish_event(JobEvent(event_type="job.started", job_id="job-2"))

    recent = bus.get_recent_events(limit=2)
    assert [event.job_id for event in recent] == ["job-2", "job-1"]

    timeline = bus.get_job_events(job_id="job-1", limit=10)
    assert [event.event_type for event in timeline] == ["job.started", "job.completed"]
