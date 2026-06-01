"""
BentoML image-to-image adapter — delegates img2img to a BentoML runner.

Implements ImageToImageProvider protocol. Uploads the source image as multipart,
sends generation parameters, and saves the result locally.
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


class BentoMLImageToImageAdapter:
    """Image-to-image via BentoML service with source image upload."""

    def __init__(self, client: BentoMLClient) -> None:
        self._client = client

    async def generate(
        self,
        params: GenerationParams,
        model_id: str,
        source_image_path: Path,
        output_dir: Path,
        strength: float = 0.75,
        on_progress: ProgressCallback | None = None,
    ) -> list[ArtifactReference]:
        """
        Request image-to-image generation from BentoML service.

        Uploads source image as multipart and sends generation parameters.
        """
        import base64

        # Encode source image for JSON transport
        image_bytes = source_image_path.read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

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
            "strength": strength,
            "source_image": image_b64,
        }

        response = await self._client.post_json(
            "/api/v1/generate/image-to-image",
            payload=payload,
            timeout_key="image_to_image",
        )

        # Save response images
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
