"""
Direct sketch-to-ink adapter — converts sketches using T2I adapters or img2img fallback.

T2I adapter models (e.g. TencentARC/t2i-adapter-sketch-sdxl-1.0) are helper models
that cannot be loaded standalone: they require a base diffusion model alongside them.
This adapter handles both cases:
  - T2I adapter: loads adapter weights + base model via StableDiffusionXLAdapterPipeline.
  - Standalone model: falls back to AutoPipelineForImage2Image (standard img2img).

The base_model_id is resolved at job-creation time from the model's requirements field
and passed through job params so the adapter can load both models together.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from app.adapters.base import build_diffusers_step_callback, save_artifacts_from_pil_images
from app.adapters.direct.pipeline_cache import PipelineCache
from app.domain.protocols import ProgressCallback
from app.domain.value_objects import ArtifactReference, GenerationParams

logger = logging.getLogger(__name__)

# Sentinel strings stored as the first element of the cached pipeline tuple
_T2I_ADAPTER = "t2i_adapter"
_IMG2IMG = "img2img"


class DirectSketchToInkAdapter:
    """
    Sketch-to-ink inference supporting T2I adapter + base model pipelines.

    When base_model_id is supplied (resolved from the model's requires_base_model
    requirement), loads a T2IAdapter conditioned on the sketch + a full SDXL pipeline.
    Falls back to standard img2img if base_model_id is absent or loading fails.
    """

    def __init__(self, pipeline_cache: PipelineCache) -> None:
        self._cache = pipeline_cache

    async def generate(
        self,
        params: GenerationParams,
        model_id: str,
        source_image_path: Path,
        output_dir: Path,
        strength: float = 0.9,
        base_model_id: str | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> list[ArtifactReference]:
        cache_key = f"{model_id}:{base_model_id}:sketch2ink"
        pipeline_tuple = await self._cache.get_or_load(
            cache_key,
            loader=lambda: asyncio.to_thread(
                self._build_pipeline, model_id, base_model_id
            ),
            category="diffusion",
        )
        return await asyncio.to_thread(
            self._run_inference,
            pipeline_tuple,
            params,
            source_image_path,
            output_dir,
            strength,
            on_progress,
        )

    @staticmethod
    def _build_pipeline(model_id: str, base_model_id: str | None) -> tuple:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            from app.infrastructure.config.settings import get_settings
            if not get_settings().allow_cpu_fallback:
                raise RuntimeError(
                    "CUDA not available — sketch-to-ink job aborted. "
                    "Set ALLOW_CPU_FALLBACK=true to allow CPU inference."
                )
            logger.warning("CUDA not available — running sketch-to-ink on CPU (ALLOW_CPU_FALLBACK=true)")
        dtype = torch.float16 if device == "cuda" else torch.float32

        if base_model_id:
            try:
                from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter

                logger.info(
                    "Building T2I adapter sketch pipeline: %s + %s → %s",
                    model_id,
                    base_model_id,
                    device,
                )
                adapter = T2IAdapter.from_pretrained(model_id, torch_dtype=dtype)
                pipeline = StableDiffusionXLAdapterPipeline.from_pretrained(
                    base_model_id,
                    adapter=adapter,
                    torch_dtype=dtype,
                ).to(device)
                param_bytes = sum(
                    p.numel() * p.element_size() for p in pipeline.unet.parameters()
                )
                return (_T2I_ADAPTER, pipeline), (param_bytes * 2) // (1024 * 1024)
            except Exception as exc:
                logger.warning(
                    "T2I adapter loading failed (%s); falling back to img2img", exc
                )

        # Fallback: standard img2img pipeline
        from diffusers import AutoPipelineForImage2Image

        logger.info("Building img2img fallback sketch pipeline: %s → %s", model_id, device)
        pipeline = AutoPipelineForImage2Image.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
        ).to(device)
        param_bytes = sum(p.numel() * p.element_size() for p in pipeline.unet.parameters())
        return (_IMG2IMG, pipeline), (param_bytes * 2) // (1024 * 1024)

    @staticmethod
    def _run_inference(
        pipeline_tuple: tuple,
        params: GenerationParams,
        source_image_path: Path,
        output_dir: Path,
        strength: float,
        on_progress: ProgressCallback | None,
    ) -> list[ArtifactReference]:
        import torch
        from PIL import Image

        pipeline_type, pipeline = pipeline_tuple

        source = Image.open(source_image_path).convert("RGB")
        source = source.resize((params.width, params.height), Image.LANCZOS)

        seed = params.seed if params.seed is not None else int(time.time() * 1000) % (2**32)
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)

        num_steps = params.num_inference_steps
        step_callback = build_diffusers_step_callback(on_progress, num_steps)
        prompt = params.prompt or "high quality illustration, clean linework"

        if pipeline_type == _T2I_ADAPTER:
            result = pipeline(
                prompt=prompt,
                negative_prompt=params.negative_prompt or None,
                image=source,
                num_inference_steps=num_steps,
                guidance_scale=params.guidance_scale,
                adapter_conditioning_scale=0.8,
                generator=generator,
                callback_on_step_end=step_callback,
            )
        else:
            result = pipeline(
                prompt=prompt,
                negative_prompt=params.negative_prompt or None,
                image=source,
                strength=strength,
                num_inference_steps=num_steps,
                guidance_scale=params.guidance_scale,
                generator=generator,
                callback_on_step_end=step_callback,
            )

        return save_artifacts_from_pil_images(result.images, output_dir, params)
