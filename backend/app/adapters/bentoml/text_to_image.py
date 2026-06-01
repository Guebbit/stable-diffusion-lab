"""
BentoML text-to-image adapter — delegates generation to a BentoML runner.

Implements TextToImageProvider protocol. Serializes GenerationParams to JSON,
sends to the BentoML service, downloads generated images, and saves locally.
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


class BentoMLTextToImageAdapter:
    """
    Text-to-image via BentoML service.

    The BentoML runner handles pipeline loading and inference in its own process.
    This adapter is a thin HTTP client that sends params and receives image bytes.
    """

    def __init__(self, client: BentoMLClient) -> None:
        self._client = client

    async def generate(
        self,
        params: GenerationParams,
        model_id: str,
        output_dir: Path,
        on_progress: ProgressCallback | None = None,
    ) -> list[ArtifactReference]:
        """
        Request text-to-image generation from BentoML service.

        Sends generation parameters as JSON, receives generated images
        as base64-encoded bytes, saves them locally.
        """
        payload = {
            "model_id": model_id,
            "prompt": params.prompt,
            "negative_prompt": params.negative_prompt,
            "width": params.width,
            "height": params.height,
            "num_inference_steps": params.num_inference_steps,
            "guidance_scale": params.guidance_scale,
            "seed": params.seed,
            "num_images": params.num_images,
        }

        response = await self._client.post_json(
            "/api/v1/generate/text-to-image",
            payload=payload,
            timeout_key="text_to_image",
        )

        # Parse response and save images locally
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts = self._save_response_artifacts(response, output_dir, params)

        return artifacts

    @staticmethod
    def _save_response_artifacts(
        response: dict[str, Any],
        output_dir: Path,
        params: GenerationParams,
    ) -> list[ArtifactReference]:
        """Save base64-encoded images from BentoML response to disk."""
        import base64

        artifacts: list[ArtifactReference] = []

        for image_data in response.get("images", []):
            artifact_id = uuid.uuid4()
            filename = f"{artifact_id}.png"
            file_path = output_dir / filename

            # Decode and save image bytes
            image_bytes = base64.b64decode(image_data["data"])
            file_path.write_bytes(image_bytes)

            artifacts.append(
                ArtifactReference(
                    artifact_id=artifact_id,
                    job_id=uuid.UUID(int=0),
                    file_path=str(file_path),
                    media_type="image/png",
                    width=params.width,
                    height=params.height,
                    size_bytes=len(image_bytes),
                )
            )

        return artifacts
