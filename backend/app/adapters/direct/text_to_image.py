"""
Direct text-to-image adapter using HuggingFace Diffusers.

Implements the TextToImageProvider protocol by loading and calling
diffusion pipelines directly in-process. Uses PipelineCache for
efficient model lifecycle management.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
from pathlib import Path
from typing import Any

from app.adapters.base import build_diffusers_step_callback, estimate_pipeline_vram_mb, hf_token, load_pipeline, resolve_model_path, save_artifacts_from_pil_images
from app.adapters.direct.pipeline_cache import PipelineCache
from app.domain.protocols import ProgressCallback
from app.domain.value_objects import ArtifactReference, GenerationParams


logger = logging.getLogger(__name__)


class DirectTextToImageAdapter:
    """
    Generates images from text prompts using diffusers pipelines.

    Implements TextToImageProvider protocol. Runs inference on a thread pool
    to avoid blocking the async event loop.

    Pipeline lifecycle is managed by the shared PipelineCache:
    - Pipelines are loaded lazily on first use via cache.get_or_load()
    - LRU eviction frees VRAM when cache is full
    - No direct pipeline management in this adapter
    """

    def __init__(self, pipeline_cache: PipelineCache) -> None:
        self._cache = pipeline_cache

    async def generate(
        self,
        params: GenerationParams,
        model_id: str,
        output_dir: Path,
        lora_model_id: str | None = None,
        lora_strength: float = 0.8,
        on_progress: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
    ) -> list[ArtifactReference]:
        """
        Run text-to-image generation.

        Loads the model via PipelineCache (lazy, LRU-managed), runs inference
        in a thread pool, saves outputs to disk, returns artifact references.
        """
        cache_key = f"{model_id}:{lora_model_id or ''}"
        pipeline = await self._cache.get_or_load(
            cache_key,
            loader=lambda: asyncio.to_thread(self._build_pipeline, model_id, lora_model_id, lora_strength),
            category="diffusion",
        )

        # Run blocking inference on thread pool to keep event loop responsive
        artifacts = await asyncio.to_thread(
            self._run_inference, pipeline, params, output_dir, on_progress, cancel_event
        )
        return artifacts

    @staticmethod
    def _build_pipeline(model_id: str, lora_model_id: str | None = None, lora_strength: float = 0.8) -> tuple[Any, int]:
        """
        Build a text-to-image pipeline — called by PipelineCache on cache miss.

        All device placement, offload strategy, and quantization logic is delegated to
        load_pipeline() in base.py, which fixes three bugs present in the old inline code:
          1. device_map="balanced" caused T5-XXL and other sub-components to land in CPU
             RAM even when VRAM was sufficient — filled all system RAM for FLUX (~22 GB).
          2. float16 caused NaN overflow in FLUX attention layers; bfloat16 is now used.
          3. sequential_cpu used device_map="sequential" (accelerate distribution mode)
             instead of enable_sequential_cpu_offload() — a completely different mechanism
             that actually achieves ~1-2 GB peak VRAM.
        """
        import torch
        from diffusers import DiffusionPipeline
        from app.infrastructure.config.settings import get_settings

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            if not get_settings().allow_cpu_fallback:
                raise RuntimeError(
                    "CUDA not available — text-to-image job aborted. "
                    "Set ALLOW_CPU_FALLBACK=true to allow CPU inference."
                )
            logger.warning("CUDA not available — running text-to-image on CPU (ALLOW_CPU_FALLBACK=true)")

        # bfloat16 on CUDA: required for FLUX (float16 overflows its attention layers due to
        # limited exponent range), and safe for SD1.x/SDXL on Ampere+ GPUs (RTX 3000+).
        # CPU always uses float32 — bfloat16 ops are not accelerated on most CPUs.
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        settings = get_settings()
        model_path = resolve_model_path(model_id)
        logger.info("Building text-to-image pipeline: %s → %s", model_path, device)

        pipeline = load_pipeline(
            DiffusionPipeline,
            model_path,
            device,
            dtype,
            settings.pipeline_offload,
            settings.quantize_mode,
            hf_token(),
            safety_checker=None,
        )

        # VAE tiling decodes large images tile-by-tile — avoids OOM on high-res outputs.
        # Attention slicing processes one head at a time — reduces peak VRAM mid-step.
        if hasattr(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()
        if hasattr(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing()

        if lora_model_id and hasattr(pipeline, "load_lora_weights"):
            lora_path = resolve_model_path(lora_model_id)
            logger.info("Loading LoRA weights: %s (strength=%.2f)", lora_path, lora_strength)
            pipeline.load_lora_weights(lora_path)
            if hasattr(pipeline, "fuse_lora"):
                pipeline.fuse_lora(lora_scale=lora_strength)

        return pipeline, estimate_pipeline_vram_mb(pipeline)

    @staticmethod
    def _run_inference(
        pipeline: Any,
        params: GenerationParams,
        output_dir: Path,
        on_progress: ProgressCallback | None,
        cancel_event: threading.Event | None = None,
    ) -> list[ArtifactReference]:
        """Synchronous inference execution — runs on thread pool."""
        import torch

        # Resolve seed (deterministic if provided, random-ish otherwise)
        seed = params.seed if params.seed is not None else int(time.time() * 1000) % (2**32)
        # pipeline.device can be 'meta' when CPU offload is active; fall back to cuda/cpu
        gen_device = pipeline.device if str(pipeline.device) != "meta" else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        generator = torch.Generator(device=gen_device).manual_seed(seed)

        # Step progress callback — raises GenerationCancelledError when cancel_event is set
        step_callback = build_diffusers_step_callback(
            on_progress, params.num_inference_steps, cancel_event=cancel_event
        )

        # Filter kwargs to only what this pipeline's __call__ actually accepts.
        # Some models (e.g. FLUX) omit negative_prompt and guidance_scale.
        supported = set(inspect.signature(pipeline.__call__).parameters)
        all_kwargs: dict[str, Any] = {
            "prompt": params.prompt,
            "negative_prompt": params.negative_prompt or None,
            "width": params.width,
            "height": params.height,
            "num_inference_steps": params.num_inference_steps,
            "guidance_scale": params.guidance_scale,
            "num_images_per_prompt": params.num_images,
            "generator": generator,
            "callback_on_step_end": step_callback,
        }
        call_kwargs = {k: v for k, v in all_kwargs.items() if k in supported}
        result = pipeline(**call_kwargs)

        return save_artifacts_from_pil_images(result.images, output_dir, params)
