"""Tests for the SSE event hub (api/sse/hub.py)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import AsyncGenerator

from app.sse import SSEHub, SSEClient, EventRingBuffer
from app.events import JobEvent, SystemEvent, TypedEvent


# --- SSEClient ---

class TestSSEClient:
    @pytest.fixture
    def mock_write(self):
        """Create a mock async iterator for write."""
        mock_iter = AsyncMock()
        mock_iter.send = AsyncMock()
        mock_iter.asend = AsyncMock(side_effect=StopAsyncIteration)
        mock_iter.__anext__ = AsyncMock(side_effect=StopAsyncIteration)
        return mock_iter

    @pytest.fixture
    def mock_headers(self):
        return {"Connection": "keep-alive"}

    @pytest.fixture
    def mock_receive(self):
        """Create a mock receive callable."""
        return AsyncMock()

    @pytest.fixture
    def mock_send(self):
        """Create a mock send callable."""
        send = AsyncMock()
        send.return_value = None
        return send

    @pytest.mark.asyncio
    async def test_sse_client_initialization(self, mock_headers, mock_receive, mock_send):
        """SSEClient should initialize with correct parameters."""
        client = SSEClient(
            client_id="test-client",
            headers=mock_headers,
            receive=mock_receive,
            send=mock_send,
        )
        assert client.client_id == "test-client"
        assert client.subscribed_event_types == set()
        assert client.buffer_size == 50
        assert client.connected is False

    @pytest.mark.asyncio
    async def test_subscribe_event_types(self, mock_headers, mock_receive, mock_send):
        """subscribe_event_types should add types to the subscription list."""
        client = SSEClient(
            client_id="test-client",
            headers=mock_headers,
            receive=mock_receive,
            send=mock_send,
        )
        await client.subscribe_event_types(["job.event", "model.event"])
        assert "job.event" in client.subscribed_event_types
        assert "model.event" in client.subscribed_event_types
        assert client.connected is True

    @pytest.mark.asyncio
    async def test_unsubscribe_event_types(self, mock_headers, mock_receive, mock_send):
        """unsubscribe_event_types should remove types from the subscription."""
        client = SSEClient(
            client_id="test-client",
            headers=mock_headers,
            receive=mock_receive,
            send=mock_send,
        )
        await client.subscribe_event_types(["job.event", "model.event"])
        await client.unsubscribe_event_types(["job.event"])
        assert "job.event" not in client.subscribed_event_types
        assert "model.event" in client.subscribed_event_types


# --- SSEHub ---

class TestSSEHub:
    @pytest.fixture
    def sse_hub(self):
        """Create a SSEHub instance."""
        return SSEHub(max_clients=100, max_buffer_size=10)

    def test_sse_hub_initialization(self):
        """SSEHub should initialize with empty subscribers and buffer."""
        hub = SSEHub(max_clients=50, max_buffer_size=20)
        assert hub.max_clients == 50
        assert hub.max_buffer_size == 20
        # hub should have an internal event buffer
        assert hasattr(hub, '_event_buffer')
        assert hasattr(hub, '_subscribers')

    def test_publish_event(self, sse_hub):
        """publish_event should add the event to the buffer."""
        event = JobEvent(message="Test event")
        hub = SSEHub(max_clients=10, max_buffer_size=10)
        hub.publish(event)
        events = hub.get_recent_events()
        assert len(events) == 1
        assert events[0].message == "Test event"

    def test_publish_system_event(self, sse_hub):
        """publish_system should create and publish a SystemEvent."""
        hub = SSEHub(max_clients=10, max_buffer_size=10)
        hub.publish_system("System started")
        events = hub.get_recent_events()
        assert len(events) == 1
        assert isinstance(events[0], SystemEvent)
        assert events[0].message == "System started"

    def test_get_recent_events(self, sse_hub):
        """get_recent_events should return the configured number of events."""
        hub = SSEHub(max_clients=10, max_buffer_size=10)
        for i in range(5):
            hub.publish(JobEvent(message=f"Event {i}"))
        events = hub.get_recent_events(3)
        assert len(events) == 3

    def test_get_recent_events_empty(self, sse_hub):
        """get_recent_events should return empty list for no events."""
        hub = SSEHub(max_clients=10, max_buffer_size=10)
        events = hub.get_recent_events()
        assert events == []

    def test_limit_zero_returns_empty(self, sse_hub):
        """get_recent_events with limit=0 should return empty list."""
        hub = SSEHub(max_clients=10, max_buffer_size=10)
        hub.publish(JobEvent(message="Test"))
        events = hub.get_recent_events(0)
        assert events == []

    def test_get_job_events(self, sse_hub):
        """get_job_events should filter by job_id."""
        hub = SSEHub(max_clients=10, max_buffer_size=100)
        hub.publish(JobEvent(job_id="job-1", message="job1-event"))
        hub.publish(JobEvent(job_id="job-2", message="job2-event"))
        hub.publish(JobEvent(job_id="job-1", message="job1-event-2"))
        events = hub.get_job_events("job-1")
        assert len(events) == 2
        assert all(e.job_id == "job-1" for e in events)
        assert events[0].message == "job1-event"

    def test_get_job_events_empty(self, sse_hub):
        """get_job_events should return empty list for unknown job_id."""
        hub = SSEHub(max_clients=10, max_buffer_size=100)
        hub.publish(JobEvent(job_id="job-1", message="Test"))
        events = hub.get_job_events("job-999")
        assert events == []