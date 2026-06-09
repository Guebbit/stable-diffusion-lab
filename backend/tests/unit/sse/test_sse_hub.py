"""Tests for the SSE event hub (api/sse/hub.py)."""

import asyncio
import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.api.sse.hub import SSEHub
from app.domain.events import JobEvent


class TestSSEHubInitialization:
    def test_initial_state(self):
        """SSEHub should start with no connections."""
        hub = SSEHub()
        assert hub.connection_count == 0
        assert hub._global_streams == []
        assert hub._streams == {}

    def test_connection_count_includes_filtered_and_global(self):
        """connection_count should sum all stream queues."""
        hub = SSEHub()
        hub._global_streams.append(asyncio.Queue())
        hub._global_streams.append(asyncio.Queue())
        hub._streams["job"] = [asyncio.Queue()]
        assert hub.connection_count == 3


class TestSSEHubBroadcast:
    @pytest.mark.asyncio
    async def test_broadcast_typed_event_to_global_streams(self):
        """broadcast_typed should send event to all global streams."""
        hub = SSEHub()
        queue1 = asyncio.Queue()
        queue2 = asyncio.Queue()
        hub._global_streams.append(queue1)
        hub._global_streams.append(queue2)

        event = JobEvent(
            event_type="job.started",
            job_id="test-123",
            message="Job started",
        )
        await hub.broadcast_typed(event)

        # Both queues should receive the event
        data1 = queue1.get_nowait()
        data2 = queue2.get_nowait()
        assert data1.startswith("data: ")
        assert data2.startswith("data: ")

        payload1 = json.loads(data1[6:])
        payload2 = json.loads(data2[6:])
        assert payload1["job_id"] == "test-123"
        assert payload2["message"] == "Job started"
        assert payload1["type"] == "job.started"

    @pytest.mark.asyncio
    async def test_broadcast_typed_event_to_filtered_streams(self):
        """broadcast_typed should send event to streams subscribed to the category."""
        hub = SSEHub()
        job_queue = asyncio.Queue()
        model_queue = asyncio.Queue()
        hub._streams["job"] = [job_queue]
        hub._streams["model"] = [model_queue]

        event = JobEvent(
            event_type="job.completed",
            job_id="test-456",
            message="Job completed",
        )
        await hub.broadcast_typed(event)

        # Only the job queue should receive the event (category is "job")
        data = job_queue.get_nowait()
        assert job_queue.qsize() == 0
        assert model_queue.qsize() == 0  # model queue should be empty

        payload = json.loads(data[6:])
        assert payload["type"] == "job.completed"



class TestSSEHubCreateStream:
    @pytest.mark.asyncio
    async def test_create_stream_returns_streaming_response(self):
        """create_stream should return a StreamingResponse."""
        from fastapi.responses import StreamingResponse

        hub = SSEHub()
        mock_request = MagicMock()
        mock_request.is_disconnected = AsyncMock(return_value=False)

        response = await hub.create_stream(mock_request)
        assert isinstance(response, StreamingResponse)
        assert response.media_type == "text/event-stream"

    @pytest.mark.asyncio
    async def test_create_stream_includes_correct_headers(self):
        """create_stream should set SSE-appropriate headers."""
        hub = SSEHub()
        mock_request = MagicMock()
        mock_request.is_disconnected = AsyncMock(return_value=False)

        response = await hub.create_stream(mock_request)
        assert response.headers["Cache-Control"] == "no-cache"
        assert response.headers["Connection"] == "keep-alive"
        assert response.headers["X-Accel-Buffering"] == "no"

    @pytest.mark.asyncio
    async def test_create_stream_global_increments_count(self):
        """Creating a global stream should increase connection_count."""
        hub = SSEHub()
        mock_request = MagicMock()
        mock_request.is_disconnected = AsyncMock(return_value=False)

        assert hub.connection_count == 0
        await hub.create_stream(mock_request)
        assert hub.connection_count == 1

    @pytest.mark.asyncio
    async def test_create_stream_filtered_subscribes_to_category(self):
        """Creating a filtered stream should register under the category."""
        hub = SSEHub()
        mock_request = MagicMock()
        mock_request.is_disconnected = AsyncMock(return_value=False)

        await hub.create_stream(mock_request, subscribe={"job", "model"})
        assert "job" in hub._streams
        assert "model" in hub._streams
        assert len(hub._streams["job"]) == 1
        assert len(hub._streams["model"]) == 1


class TestEventPayload:
    @pytest.mark.asyncio
    async def test_broadcast_includes_type_field(self):
        """Broadcasted event should include the event_type as 'type' in payload."""
        hub = SSEHub()
        queue = asyncio.Queue()
        hub._global_streams.append(queue)

        event = JobEvent(
            event_type="job.failed",
            job_id="err-1",
            message="GPU OOM",
            level="error",
        )
        await hub.broadcast_typed(event)

        data = queue.get_nowait()
        payload = json.loads(data[6:])
        assert payload["type"] == "job.failed"
        assert payload["level"] == "error"
        assert payload["job_id"] == "err-1"

    @pytest.mark.asyncio
    async def test_broadcast_includes_timestamp(self):
        """Broadcasted event should include an ISO timestamp."""
        hub = SSEHub()
        queue = asyncio.Queue()
        hub._global_streams.append(queue)

        event = JobEvent(event_type="job.started", job_id="t-1", message="started")
        await hub.broadcast_typed(event)

        data = queue.get_nowait()
        payload = json.loads(data[6:])
        assert "timestamp" in payload