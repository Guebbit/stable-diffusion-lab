"""
ComfyUI WebSocket + REST client — handles all communication with ComfyUI server.

ComfyUI's API works differently from typical REST services:
1. Submit workflow JSON via POST /prompt → get a prompt_id
2. Connect to WebSocket at /ws → receive real-time progress events
3. On completion, download output images via GET /view

This client encapsulates the full communication lifecycle.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Callable

import httpx


logger = logging.getLogger(__name__)


class ComfyUIClient:
    """
    WebSocket + HTTP client for ComfyUI server communication.

    Lifecycle of a workflow execution:
    1. queue_workflow() → submits workflow, returns prompt_id
    2. wait_for_completion() → listens on WebSocket for progress/completion
    3. download_output() → fetches generated images/videos from ComfyUI

    Thread safety: Each workflow execution gets its own WebSocket connection.
    """

    def __init__(self, base_url: str = "http://localhost:8188") -> None:
        """
        Initialize ComfyUI client.

        Args:
            base_url: ComfyUI server URL (default: http://localhost:8188).
        """
        self._base_url = base_url.rstrip("/")
        self._client_id = str(uuid.uuid4())

    @property
    def base_url(self) -> str:
        """Base URL of the ComfyUI server."""
        return self._base_url

    @property
    def ws_url(self) -> str:
        """WebSocket URL for real-time events."""
        http_url = self._base_url.replace("http://", "ws://").replace("https://", "wss://")
        return f"{http_url}/ws?clientId={self._client_id}"

    async def queue_workflow(self, workflow: dict[str, Any]) -> str:
        """
        Submit a workflow to ComfyUI's execution queue.

        Args:
            workflow: Complete ComfyUI workflow JSON (node graph).

        Returns:
            prompt_id: Unique identifier for tracking this execution.
        """
        payload = {
            "prompt": workflow,
            "client_id": self._client_id,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self._base_url}/prompt",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

        prompt_id = result["prompt_id"]
        logger.info("Queued ComfyUI workflow: %s", prompt_id)
        return prompt_id

    async def wait_for_completion(
        self,
        prompt_id: str,
        on_progress: Callable[[int, int], None] | None = None,
        timeout: float = 300.0,
    ) -> dict[str, Any]:
        """
        Wait for a workflow to complete via WebSocket events.

        Connects to ComfyUI's WebSocket and listens for events until
        the specified prompt_id is fully executed.

        Args:
            prompt_id: The prompt ID to wait for.
            on_progress: Optional callback(current_step, total_steps).
            timeout: Maximum time to wait (seconds).

        Returns:
            Dict with output node results from ComfyUI.

        Raises:
            TimeoutError: If execution takes longer than timeout.
            RuntimeError: If ComfyUI reports an execution error.
        """
        # Deferred import: websockets is optional (only needed when ComfyUI is enabled)
        import websockets

        async with websockets.connect(self.ws_url) as ws:
            deadline = asyncio.get_event_loop().time() + timeout

            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    raise TimeoutError(f"ComfyUI workflow {prompt_id} timed out after {timeout}s")

                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=remaining)
                except asyncio.TimeoutError:
                    raise TimeoutError(f"ComfyUI workflow {prompt_id} timed out after {timeout}s")

                data = json.loads(message)
                event_type = data.get("type")

                # Progress update
                if event_type == "progress" and on_progress:
                    value = data.get("data", {}).get("value", 0)
                    max_value = data.get("data", {}).get("max", 1)
                    on_progress(value, max_value)

                # Execution complete for our prompt
                if event_type == "executed":
                    event_data = data.get("data", {})
                    if event_data.get("prompt_id") == prompt_id:
                        return event_data.get("output", {})

                # Execution error
                if event_type == "execution_error":
                    event_data = data.get("data", {})
                    if event_data.get("prompt_id") == prompt_id:
                        error_msg = event_data.get("exception_message", "Unknown error")
                        raise RuntimeError(f"ComfyUI execution error: {error_msg}")

    async def download_output(
        self,
        filename: str,
        subfolder: str = "",
        output_type: str = "output",
    ) -> bytes:
        """
        Download a generated file from ComfyUI's output directory.

        Args:
            filename: Name of the output file.
            subfolder: Subfolder within ComfyUI's output directory.
            output_type: ComfyUI folder type ("output", "input", "temp").

        Returns:
            Raw file bytes.
        """
        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": output_type,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(f"{self._base_url}/view", params=params)
            response.raise_for_status()

        return response.content

    async def upload_image(self, image_path: Path) -> str:
        """
        Upload an image to ComfyUI's input directory.

        Required for img2img workflows where ComfyUI needs to access the source image.

        Args:
            image_path: Local path to the image file.

        Returns:
            Filename as stored by ComfyUI (use in workflow LoadImage node).
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            with open(image_path, "rb") as f:
                files = {"image": (image_path.name, f, "image/png")}
                response = await client.post(
                    f"{self._base_url}/upload/image",
                    files=files,
                )
                response.raise_for_status()

        result = response.json()
        return result.get("name", image_path.name)

    async def health_check(self) -> bool:
        """Check if ComfyUI server is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self._base_url}/system_stats")
                return response.status_code == 200
        except (httpx.ConnectError, httpx.ReadTimeout):
            return False

    async def interrupt(self) -> None:
        """Send interrupt signal to cancel current ComfyUI execution."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(f"{self._base_url}/interrupt")
        logger.info("Sent interrupt to ComfyUI")
