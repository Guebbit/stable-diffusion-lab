"""
Direct sketch-to-ink adapter — converts sketches using ControlNet, T2I adapters, or img2img.

Helper models cannot be loaded standalone; they require a base diffusion model alongside them.
This adapter handles three cases based on model_type + base_model_id from job params:
  - controlnet: ControlNetModel + StableDiffusionControlNetPipeline (SD1.5 lineart etc.)
  - t2i_adapter: T2IAdapter + StableDiffusionXLAdapterPipeline (SDXL sketch adapters)
  - fallback: AutoPipelineForImage2Image when no base model or loading fails

Both model_type and base_model_id are resolved at job-creation time from the model's
requirements and model_type fields and threaded through job params.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from app.adapters.base import (
    apply_pipeline_to_device,
    build_diffusers_step_callback,
    build_legacy_diffusers_step_callback,
    estimate_pipeline_vram_mb,
    hf_token,
    resolve_model_path,
    save_artifacts_from_pil_images,
)
from app.adapters.direct.pipeline_cache import PipelineCache
from app.domain.protocols import ProgressCallback
from app.domain.value_objects import ArtifactReference, GenerationParams

logger = logging.getLogger(__name__)


def _apply_lora(pipeline: object, lora_model_id: str | None, lora_strength: float) -> None:
    """Load and fuse LoRA weights into a pipeline if a LoRA model ID is provided."""
    if not lora_model_id:
        return
    if not hasattr(pipeline, "load_lora_weights"):
        logger.warning("Pipeline does not support LoRA — skipping %s", lora_model_id)
        return
    lora_path = resolve_model_path(lora_model_id)
    logger.info("Loading LoRA weights: %s (strength=%.2f)", lora_path, lora_strength)
    pipeline.load_lora_weights(lora_path)  # type: ignore[union-attr]
    if hasattr(pipeline, "fuse_lora"):
        pipeline.fuse_lora(lora_scale=lora_strength)  # type: ignore[union-attr]


# Sentinel strings stored as the first element of the cached pipeline tuple
_T2I_ADAPTER = "t2i_adapter"
_CONTROLNET = "controlnet"
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
        model_type: str | None = None,
        adapter_conditioning_scale: float = 0.9,
        lora_model_id: str | None = None,
        lora_strength: float = 0.8,
        on_progress: ProgressCallback | None = None,
    ) -> list[ArtifactReference]:
        cache_key = f"{model_id}:{base_model_id}:{lora_model_id or ''}:sketch2ink"
        pipeline_tuple = await self._cache.get_or_load(
            cache_key,
            loader=lambda: asyncio.to_thread(
                self._build_pipeline, model_id, base_model_id, model_type, lora_model_id, lora_strength
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
            adapter_conditioning_scale,
            on_progress,
        )

    @staticmethod
    def _build_pipeline(
        model_id: str,
        base_model_id: str | None,
        model_type: str | None = None,
        lora_model_id: str | None = None,
        lora_strength: float = 0.8,
    ) -> tuple:
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
        # bfloat16 on CUDA: required for FLUX-based models, safe for SD1.5/SDXL on Ampere+.
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        if base_model_id and model_type == "controlnet":
            try:
                from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

                logger.info(
                    "Building ControlNet sketch pipeline: %s + %s → %s",
                    model_id,
                    base_model_id,
                    device,
                )
                controlnet = ControlNetModel.from_pretrained(
                    resolve_model_path(model_id), torch_dtype=dtype, token=hf_token()
                )
                from app.infrastructure.config.settings import get_settings
                pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                    resolve_model_path(base_model_id),
                    controlnet=controlnet,
                    torch_dtype=dtype,
                    safety_checker=None,
                    token=hf_token(),
                    low_cpu_mem_usage=True,
                )
                pipeline = apply_pipeline_to_device(pipeline, device, get_settings().pipeline_offload)
                _apply_lora(pipeline, lora_model_id, lora_strength)
                return (_CONTROLNET, pipeline), estimate_pipeline_vram_mb(pipeline)
            except Exception as exc:
                logger.warning(
                    "ControlNet loading failed (%s); falling back to img2img", exc
                )

        if base_model_id and model_type != "controlnet":
            try:
                from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter

                logger.info(
                    "Building T2I adapter sketch pipeline: %s + %s → %s",
                    model_id,
                    base_model_id,
                    device,
                )
                adapter = T2IAdapter.from_pretrained(
                    resolve_model_path(model_id), torch_dtype=dtype, token=hf_token()
                )
                from app.infrastructure.config.settings import get_settings
                pipeline = StableDiffusionXLAdapterPipeline.from_pretrained(
                    resolve_model_path(base_model_id),
                    adapter=adapter,
                    torch_dtype=dtype,
                    token=hf_token(),
                    low_cpu_mem_usage=True,
                )
                pipeline = apply_pipeline_to_device(pipeline, device, get_settings().pipeline_offload)
                _apply_lora(pipeline, lora_model_id, lora_strength)
                return (_T2I_ADAPTER, pipeline), estimate_pipeline_vram_mb(pipeline)
            except Exception as exc:
                logger.warning(
                    "T2I adapter loading failed (%s); falling back to img2img", exc
                )

        # Fallback: standard img2img pipeline
        from diffusers import AutoPipelineForImage2Image

        fallback_model = base_model_id or model_id
        fallback_path = resolve_model_path(fallback_model)
        logger.info("Building img2img fallback sketch pipeline: %s → %s", fallback_path, device)
        from app.infrastructure.config.settings import get_settings
        pipeline = AutoPipelineForImage2Image.from_pretrained(
            fallback_path,
            torch_dtype=dtype,
            safety_checker=None,
            token=hf_token(),
            low_cpu_mem_usage=True,
        )
        pipeline = apply_pipeline_to_device(pipeline, device, get_settings().pipeline_offload)
        _apply_lora(pipeline, lora_model_id, lora_strength)
        return (_IMG2IMG, pipeline), estimate_pipeline_vram_mb(pipeline)

    @staticmethod
    def _run_inference(
        pipeline_tuple: tuple,
        params: GenerationParams,
        source_image_path: Path,
        output_dir: Path,
        strength: float,
        adapter_conditioning_scale: float,
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
        prompt = params.prompt or "high quality illustration, clean linework"

        if pipeline_type == _CONTROLNET:
            # StableDiffusionControlNetPipeline uses the legacy callback API
            legacy_callback = build_legacy_diffusers_step_callback(on_progress, num_steps)
            result = pipeline(
                prompt=prompt,
                negative_prompt=params.negative_prompt or None,
                image=source,
                num_inference_steps=num_steps,
                guidance_scale=params.guidance_scale,
                controlnet_conditioning_scale=adapter_conditioning_scale,
                generator=generator,
                callback=legacy_callback,
                callback_steps=1,
            )
        elif pipeline_type == _T2I_ADAPTER:
            # StableDiffusionXLAdapterPipeline also uses the legacy callback API
            legacy_callback = build_legacy_diffusers_step_callback(on_progress, num_steps)
            result = pipeline(
                prompt=prompt,
                negative_prompt=params.negative_prompt or None,
                image=source,
                num_inference_steps=num_steps,
                guidance_scale=params.guidance_scale,
                adapter_conditioning_scale=adapter_conditioning_scale,
                generator=generator,
                callback=legacy_callback,
                callback_steps=1,
            )
        else:
            step_callback = build_diffusers_step_callback(on_progress, num_steps)
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
