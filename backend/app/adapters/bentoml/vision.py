"""
BentoML vision adapter — delegates image captioning to a BentoML runner.

Implements VisionProvider protocol. Sends the image and optional prompt,
receives the generated caption text.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.adapters.bentoml.client import BentoMLClient


logger = logging.getLogger(__name__)


class BentoMLVisionAdapter:
    """Image captioning via BentoML service."""

    def __init__(self, client: BentoMLClient) -> None:
        self._client = client

    async def caption(
        self,
        image_path: Path,
        model_id: str,
        prompt: str = "",
    ) -> str:
        """
        Request image captioning from BentoML service.

        Sends the image as base64 and receives a text caption.
        """
        import base64

        image_bytes = image_path.read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "model_id": model_id,
            "image": image_b64,
            "prompt": prompt,
        }

        response = await self._client.post_json(
            "/api/v1/generate/caption",
            payload=payload,
            timeout_key="vision",
        )

        return response.get("caption", "")
