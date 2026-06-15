"""
Direct image-to-image adapter — transforms images using diffusion pipelines.

Loads a Stable Diffusion img2img pipeline and runs inference locally.
Implements ImageToImageProvider protocol from app.domain.protocols.

Uses the shared PipelineCache for efficient model lifecycle management.
Same pattern as text-to-image but with a source image input and strength parameter.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from app.adapters.base import build_diffusers_step_callback, save_artifacts_from_pil_images
from app.adapters.direct.pipeline_cache import PipelineCache
from app.domain.protocols import ProgressCallback
from app.domain.value_objects import ArtifactReference, GenerationParams


logger = logging.getLogger(__name__)


class DirectImageToImageAdapter:
    """
    Image-to-image inference using diffusers img2img pipelines.

    Takes a source image + prompt and produces a transformed output.
    The 'strength' parameter controls how much the source image is preserved
    (0.0 = identical to source, 1.0 = ignore source completely).
    """

    def __init__(self, pipeline_cache: PipelineCache) -> None:
        self._cache = pipeline_cache

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

        Loads the model via PipelineCache, processes the source image,
        applies the diffusion transformation, and saves outputs.
        """
        # Get img2img pipeline from cache
        pipeline = await self._cache.get_or_load(
            f"{model_id}:img2img",
            loader=lambda: asyncio.to_thread(self._build_pipeline, model_id),
            category="diffusion",
        )

        # Run inference on thread pool
        artifacts = await asyncio.to_thread(
            self._run_inference,
            pipeline,
            params,
            source_image_path,
            output_dir,
            strength,
            on_progress,
        )
        return artifacts

    @staticmethod
    def _build_pipeline(model_id: str) -> tuple[Any, int]:
        """
        Build an img2img pipeline — called by PipelineCache on cache miss.

        Uses AutoPipelineForImage2Image which auto-detects the correct
        pipeline class (SD1.5, SDXL, etc.) from the model config.
        """
        import torch
        from diffusers import AutoPipelineForImage2Image

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            from app.infrastructure.config.settings import get_settings
            if not get_settings().allow_cpu_fallback:
                raise RuntimeError(
                    "CUDA not available — img2img job aborted. "
                    "Set ALLOW_CPU_FALLBACK=true to allow CPU inference."
                )
            logger.warning("CUDA not available — running img2img on CPU (ALLOW_CPU_FALLBACK=true)")
        dtype = torch.float16 if device == "cuda" else torch.float32

        logger.info("Building img2img pipeline: %s → %s", model_id, device)

        pipeline = AutoPipelineForImage2Image.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
        ).to(device)
        if hasattr(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()
        if hasattr(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing()

        # Estimate VRAM from UNet parameters (main VRAM consumer)
        param_bytes = sum(p.numel() * p.element_size() for p in pipeline.unet.parameters())
        estimated_vram_mb = (param_bytes * 2) // (1024 * 1024)

        return pipeline, estimated_vram_mb

    @staticmethod
    def _run_inference(
        pipeline: Any,
        params: GenerationParams,
        source_image_path: Path,
        output_dir: Path,
        strength: float,
        on_progress: ProgressCallback | None,
    ) -> list[ArtifactReference]:
        """Synchronous img2img inference — runs on thread pool."""
        import torch
        from PIL import Image

        # Load and resize source image to target dimensions
        source_image = Image.open(source_image_path).convert("RGB")
        source_image = source_image.resize((params.width, params.height), Image.LANCZOS)

        # Resolve seed
        seed = params.seed if params.seed is not None else int(time.time() * 1000) % (2**32)
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)

        # img2img has fewer effective steps due to strength
        effective_steps = max(int(params.num_inference_steps * strength), 1)

        # Step progress callback — built by shared utility
        step_callback = build_diffusers_step_callback(on_progress, effective_steps)

        # Run the img2img pipeline
        result = pipeline(
            prompt=params.prompt,
            negative_prompt=params.negative_prompt or None,
            image=source_image,
            strength=strength,
            num_inference_steps=params.num_inference_steps,
            guidance_scale=params.guidance_scale,
            num_images_per_prompt=params.num_images,
            generator=generator,
            callback_on_step_end=step_callback,
        )

        return save_artifacts_from_pil_images(result.images, output_dir, params)
