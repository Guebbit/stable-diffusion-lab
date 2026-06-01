"""
Direct video generation adapter — generates video from text/image prompts.

Uses video diffusion models (e.g., AnimateDiff, SVD) via diffusers.
Implements VideoProvider protocol from app.domain.protocols.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from app.domain.protocols import ProgressCallback
from app.domain.value_objects import ArtifactReference, GenerationParams

logger = logging.getLogger(__name__)


class DirectVideoAdapter:
    """Video generation using diffusers video pipelines."""

    async def generate(
        self,
        params: GenerationParams,
        model_id: str,
        output_dir: Path,
        source_image_path: Path | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> ArtifactReference:
        """
        Run video generation inference.

        Loads the video model pipeline (if not cached), generates frames,
        and exports the video file to output_dir.
        """
        # TODO: Implement video pipeline loading and inference
        raise NotImplementedError("DirectVideoAdapter.generate is not yet implemented")
