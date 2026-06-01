"""
BentoML HTTP client — shared client with retry logic for all BentoML adapters.

All BentoML adapters delegate to this client for HTTP communication with
the BentoML runner process. Provides consistent error handling, timeouts,
and retry behavior across all adapter types.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx


logger = logging.getLogger(__name__)

# Default timeout per job type (seconds)
DEFAULT_TIMEOUTS: dict[str, float] = {
    "text_to_image": 120.0,
    "image_to_image": 120.0,
    "video": 300.0,
    "vision": 60.0,
    "llm": 120.0,
    "model_load": 180.0,
}


class BentoMLClient:
    """
    HTTP client for communicating with BentoML service endpoints.

    Handles:
    - Request serialization (JSON + multipart for images)
    - Retry with exponential backoff (3 attempts)
    - Configurable timeouts per operation type
    - Health checking

    All BentoML adapters receive this client via constructor injection.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:3000",
        max_retries: int = 3,
    ) -> None:
        """
        Initialize the BentoML client.

        Args:
            base_url: Base URL of the BentoML service (e.g., "http://localhost:3000").
            max_retries: Number of retry attempts for failed requests.
        """
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries

    async def post_json(
        self,
        endpoint: str,
        payload: dict[str, Any],
        timeout_key: str = "text_to_image",
    ) -> dict[str, Any]:
        """
        Send a JSON POST request to the BentoML service.

        Retries on connection errors and 5xx responses.

        Args:
            endpoint: API path (e.g., "/generate").
            payload: JSON-serializable request body.
            timeout_key: Key into DEFAULT_TIMEOUTS for operation-specific timeout.

        Returns:
            Parsed JSON response as a dict.

        Raises:
            httpx.HTTPStatusError: On non-retryable HTTP errors (4xx).
            ConnectionError: After all retries exhausted.
        """
        url = f"{self._base_url}{endpoint}"
        timeout = DEFAULT_TIMEOUTS.get(timeout_key, 120.0)

        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    return response.json()

            except (httpx.ConnectError, httpx.ReadTimeout) as exc:
                last_error = exc
                if attempt < self._max_retries:
                    wait = 2**attempt  # Exponential backoff: 2s, 4s, 8s
                    logger.warning(
                        "BentoML request failed (attempt %d/%d): %s. Retrying in %ds",
                        attempt,
                        self._max_retries,
                        exc,
                        wait,
                    )
                    import asyncio

                    await asyncio.sleep(wait)

            except httpx.HTTPStatusError as exc:
                if exc.response.status_code >= 500 and attempt < self._max_retries:
                    last_error = exc
                    import asyncio

                    await asyncio.sleep(2**attempt)
                else:
                    raise

        raise ConnectionError(
            f"BentoML service unreachable after {self._max_retries} attempts: {last_error}"
        )

    async def post_multipart(
        self,
        endpoint: str,
        data: dict[str, Any],
        files: dict[str, tuple[str, bytes, str]],
        timeout_key: str = "image_to_image",
    ) -> dict[str, Any]:
        """
        Send a multipart POST request (for image uploads).

        Args:
            endpoint: API path.
            data: Form fields.
            files: File fields as {name: (filename, content_bytes, mime_type)}.
            timeout_key: Key into DEFAULT_TIMEOUTS.

        Returns:
            Parsed JSON response.
        """
        url = f"{self._base_url}{endpoint}"
        timeout = DEFAULT_TIMEOUTS.get(timeout_key, 120.0)

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, data=data, files=files)
            response.raise_for_status()
            return response.json()

    async def health_check(self) -> bool:
        """
        Check if the BentoML service is reachable and healthy.

        Returns True if the service responds, False otherwise.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self._base_url}/healthz")
                return response.status_code == 200
        except (httpx.ConnectError, httpx.ReadTimeout):
            return False
