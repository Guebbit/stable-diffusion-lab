"""
Direct vision/captioning adapter — generates text descriptions from images.

Uses BLIP, BLIP-2, or similar vision-language models via transformers.
Implements VisionProvider protocol from app.domain.protocols.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DirectVisionAdapter:
    """Image captioning using transformers vision-language models (BLIP/BLIP-2)."""

    async def caption(
        self,
        image_path: Path,
        model_id: str,
        prompt: str = "",
    ) -> str:
        """
        Generate a text caption for the given image.

        Loads the vision model (if not cached), processes the image,
        and returns a natural language description.
        """
        # TODO: Implement vision model loading and inference
        raise NotImplementedError("DirectVisionAdapter.caption is not yet implemented")
