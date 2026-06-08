"""Tests for the typed events system (events.py)."""

import pytest
from datetime import datetime, timezone

from app.domain.events import (
    TypedEvent,
    SystemEvent,
    JobEvent,
    ModelEvent,
    ResourceEvent,
    ArtifactEvent,
    EventRingBuffer,
)


# --- TypedEvent base ---

class TestTypedEvent:
    def test_default_values(self):
        """TypedEvent should have sensible defaults."""
        event = TypedEvent()
        assert event.event_id is not None
        assert event.event_type == "system.event"
        assert isinstance(event.timestamp, datetime)
        assert event.correlation_id is None
        assert event.component == "system"
        assert event.level == "info"
        assert event.job_id is None
        assert event.message == ""
        assert event.payload == {}

    def test_custom_values(self):
        """TypedEvent should accept custom values."""
        event = TypedEvent(
            event_type="custom.event",
            component="custom",
            level="error",
            job_id="job-123",
            message="Test message",
            payload={"key": "value"},
        )
        assert event.event_type == "custom.event"
        assert event.component == "custom"
        assert event.level == "error"
        assert event.job_id == "job-123"
        assert event.message == "Test message"
        assert event.payload == {"key": "value"}

    def test_to_dict_serializes_timestamp(self):
        """to_dict should serialize timestamp as ISO-8601 string."""
        event = TypedEvent()
        d = event.to_dict()
        assert "timestamp" in d
        assert isinstance(d["timestamp"], str)
        # Should be parseable
        datetime.fromisoformat(d["timestamp"])

    def test_to_dict_preserves_event_id(self):
        """to_dict should preserve event_id."""
        event = TypedEvent(event_id="test-id")
        d = event.to_dict()
        assert d["event_id"] == "test-id"


# --- TypedEvent subclasses ---

class TestTypedEventSubclasses:
    def test_system_event_defaults(self):
        """SystemEvent should override event_type and component."""
        event = SystemEvent()
        assert event.event_type == "system.event"
        assert event.component == "system"

    def test_job_event_defaults(self):
        """JobEvent should override event_type and component."""
        event = JobEvent()
        assert event.event_type == "job.event"
        assert event.component == "job"

    def test_model_event_defaults(self):
        """ModelEvent should override event_type and component."""
        event = ModelEvent()
        assert event.event_type == "model.event"
        assert event.component == "model"

    def test_resource_event_defaults(self):
        """ResourceEvent should override event_type and component."""
        event = ResourceEvent()
        assert event.event_type == "resource.event"
        assert event.component == "resource"

    def test_artifact_event_defaults(self):
        """ArtifactEvent should override event_type and component."""
        event = ArtifactEvent()
        assert event.event_type == "artifact.event"
        assert event.component == "artifact"


# --- EventRingBuffer ---

class TestEventRingBuffer:
    def test_append_and_get_recent(self):
        """Events should be stored and retrieved."""
        buffer = EventRingBuffer(size=10)
        e1 = TypedEvent(event_type="job.event", message="first")
        e2 = TypedEvent(event_type="job.event", message="second")
        buffer.append(e1)
        buffer.append(e2)
        events = buffer.get_recent(10)
        assert len(events) == 2
        assert events[-1].message == "second"  # Most recent last in list

    def test_get_recent_limited(self):
        """get_recent should respect the limit."""
        buffer = EventRingBuffer(size=100)
        for i in range(10):
            buffer.append(TypedEvent(message=f"event-{i}"))
        events = buffer.get_recent(5)
        assert len(events) == 5

    def test_ring_buffer_overflow(self):
        """Oldest events should be dropped when buffer is full."""
        buffer = EventRingBuffer(size=3)
        for i in range(5):
            buffer.append(TypedEvent(message=f"event-{i}"))
        events = buffer.get_recent(100)
        assert len(events) == 3
        # First two events dropped
        assert events[0].message == "event-2"

    def test_get_by_job(self):
        """get_by_job should return events matching the job_id."""
        buffer = EventRingBuffer(size=100)
        job1 = "job-1"
        job2 = "job-2"
        buffer.append(TypedEvent(job_id=job1, message="job1-a"))
        buffer.append(TypedEvent(job_id=job2, message="job2-a"))
        buffer.append(TypedEvent(job_id=job1, message="job1-b"))
        events = buffer.get_by_job(job1)
        assert len(events) == 2
        assert all(e.job_id == job1 for e in events)
        assert events[0].message == "job1-a"
        assert events[1].message == "job1-b"

    def test_get_by_job_empty(self):
        """get_by_job should return empty list for unknown job_id."""
        buffer = EventRingBuffer(size=100)
        buffer.append(TypedEvent(job_id="job-1"))
        events = buffer.get_by_job("job-999")
        assert events == []

    def test_get_recent_zero_limit(self):
        """get_recent with limit=0 should return empty list."""
        buffer = EventRingBuffer(size=10)
        buffer.append(TypedEvent())
        assert buffer.get_recent(0) == []

    def test_get_by_job_zero_limit(self):
        """get_by_job with limit=0 should return empty list."""
        buffer = EventRingBuffer(size=10)
        buffer.append(TypedEvent())
        assert buffer.get_by_job("any", limit=0) == []