"""
BentoML video adapter — delegates video generation to a BentoML runner.

Implements VideoProvider protocol. Sends parameters, receives video bytes.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from app.adapters.bentoml.client import BentoMLClient
from app.domain.protocols import ProgressCallback
from app.domain.value_objects import ArtifactReference, GenerationParams


logger = logging.getLogger(__name__)


class BentoMLVideoAdapter:
    """Video generation via BentoML service."""

    def __init__(self, client: BentoMLClient) -> None:
        self._client = client

    async def generate(
        self,
        params: GenerationParams,
        model_id: str,
        output_dir: Path,
        source_image_path: Path | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> list[ArtifactReference]:
        """
        Request video generation from BentoML service.

        Sends generation parameters and optional source image,
        receives video bytes, saves as MP4 locally.
        """
        import base64

        payload: dict[str, Any] = {
            "model_id": model_id,
            "prompt": params.prompt,
            "negative_prompt": params.negative_prompt,
            "width": params.width,
            "height": params.height,
            "num_inference_steps": params.num_inference_steps,
            "seed": params.seed,
        }

        # Include source image if provided (image-to-video)
        if source_image_path is not None:
            image_bytes = source_image_path.read_bytes()
            payload["source_image"] = base64.b64encode(image_bytes).decode("utf-8")

        response = await self._client.post_json(
            "/api/v1/generate/video",
            payload=payload,
            timeout_key="video",
        )

        # Save video bytes to local file
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_id = uuid.uuid4()
        filename = f"{artifact_id}.mp4"
        file_path = output_dir / filename

        video_bytes = base64.b64decode(response["video_data"])
        file_path.write_bytes(video_bytes)

        return [ArtifactReference(
            artifact_id=artifact_id,
            job_id=uuid.UUID(int=0),
            file_path=str(file_path),
            media_type="video/mp4",
            width=params.width,
            height=params.height,
            size_bytes=len(video_bytes),
        )]
