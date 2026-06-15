"""
Direct vision/captioning adapter — generates text descriptions from images.

Uses BLIP, BLIP-2, or similar vision-language models via HuggingFace transformers.
Implements VisionProvider protocol from app.domain.protocols.

Vision models are lighter than diffusion pipelines (1-4 GB), so they share
the same PipelineCache but tagged with a "vision" category.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from app.adapters.direct.pipeline_cache import PipelineCache


logger = logging.getLogger(__name__)


class DirectVisionAdapter:
    """
    Image captioning using transformers vision-language models.

    Supports BLIP, BLIP-2, and similar models that implement the
    image-to-text pipeline pattern. The optional prompt parameter
    enables guided captioning (e.g., "Describe the lighting").
    """

    def __init__(self, pipeline_cache: PipelineCache) -> None:
        self._cache = pipeline_cache

    async def caption(
        self,
        image_path: Path,
        model_id: str,
        prompt: str = "",
    ) -> str:
        """
        Generate a text caption for the given image.

        Loads the vision model via PipelineCache, processes the image,
        and returns a natural language description.

        Args:
            image_path: Path to the image file to caption.
            model_id: HuggingFace model ID (e.g., "Salesforce/blip2-opt-2.7b").
            prompt: Optional guidance prompt for conditional captioning.

        Returns:
            Generated caption text.
        """
        # Get vision model components from cache
        model_components = await self._cache.get_or_load(
            model_id,
            loader=lambda: asyncio.to_thread(self._build_pipeline, model_id),
            category="vision",
        )

        # Run inference on thread pool (model forward pass is blocking)
        caption_text = await asyncio.to_thread(
            self._run_inference, model_components, image_path, prompt
        )
        return caption_text

    @staticmethod
    def _build_pipeline(model_id: str) -> tuple[Any, int]:
        """
        Build vision model components — called by PipelineCache on cache miss.

        Returns a dict with 'model' and 'processor' keys, plus estimated VRAM.
        We store both in the cache as a single entry since they're always used together.
        """
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            from app.infrastructure.config.settings import get_settings
            if not get_settings().allow_cpu_fallback:
                raise RuntimeError(
                    "CUDA not available — vision job aborted. "
                    "Set ALLOW_CPU_FALLBACK=true to allow CPU inference."
                )
            logger.warning("CUDA not available — running vision on CPU (ALLOW_CPU_FALLBACK=true)")
        dtype = torch.float16 if device == "cuda" else torch.float32

        logger.info("Building vision pipeline: %s → %s", model_id, device)

        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).to(device)

        # Estimate VRAM from model parameters
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        estimated_vram_mb = param_bytes // (1024 * 1024)

        components = {"model": model, "processor": processor, "device": device}
        return components, estimated_vram_mb

    @staticmethod
    def _run_inference(
        components: dict[str, Any],
        image_path: Path,
        prompt: str,
    ) -> str:
        """Synchronous vision inference — runs on thread pool."""
        import torch
        from PIL import Image

        model = components["model"]
        processor = components["processor"]
        device = components["device"]

        # Load and prepare image
        image = Image.open(image_path).convert("RGB")

        # Process inputs — some models accept optional text prompts
        if prompt:
            inputs = processor(images=image, text=prompt, return_tensors="pt")
        else:
            inputs = processor(images=image, return_tensors="pt")

        # Move inputs to model device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # Generate caption
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        # Decode generated tokens to text
        caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        return caption.strip()
