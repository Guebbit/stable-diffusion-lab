"""
Direct image-to-image adapter — transforms images using diffusion pipelines.

Loads a Stable Diffusion img2img pipeline and runs inference locally.
Implements ImageToImageProvider protocol from app.domain.protocols.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from uuid import uuid4

from app.domain.protocols import ProgressCallback
from app.domain.value_objects import ArtifactReference, GenerationParams

logger = logging.getLogger(__name__)


class DirectImageToImageAdapter:
    """Image-to-image inference using diffusers StableDiffusionImg2ImgPipeline."""

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
        Run image-to-image inference.

        Loads the model pipeline (if not cached), processes the source image,
        and writes output files to output_dir.
        """
        # TODO: Implement full pipeline loading and inference
        # This follows the same pattern as DirectTextToImageAdapter
        raise NotImplementedError("DirectImageToImageAdapter.generate is not yet implemented")
