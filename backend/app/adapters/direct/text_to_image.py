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

from app.adapters.base import apply_pipeline_to_device, build_diffusers_step_callback, estimate_pipeline_vram_mb, hf_token, resolve_model_path, save_artifacts_from_pil_images
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

        Returns (pipeline_object, estimated_vram_mb) tuple.
        Heavy imports are deferred to here to keep app startup fast.
        """
        import torch
        from diffusers import DiffusionPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            from app.infrastructure.config.settings import get_settings
            if not get_settings().allow_cpu_fallback:
                raise RuntimeError(
                    "CUDA not available — text-to-image job aborted. "
                    "Set ALLOW_CPU_FALLBACK=true to allow CPU inference."
                )
            logger.warning("CUDA not available — running text-to-image on CPU (ALLOW_CPU_FALLBACK=true)")
        dtype = torch.float16 if device == "cuda" else torch.float32

        model_path = resolve_model_path(model_id)
        logger.info("Building text-to-image pipeline: %s → %s", model_path, device)

        from app.infrastructure.config.settings import get_settings
        offload = get_settings().pipeline_offload

        if device == "cuda" and offload in ("model_cpu", "sequential_cpu"):
            # Load with device_map so each component is streamed directly to its
            # target device as it initialises — avoids staging the full model in
            # CPU RAM (transformer + T5 together easily exceed 16 GB and trigger
            # the OOM killer before enable_model_cpu_offload() is even called).
            # device_map is mutually exclusive with the manual offload helpers.
            pipeline = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
                safety_checker=None,
                token=hf_token(),
                device_map="balanced",
            )
        else:
            pipeline = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
                safety_checker=None,
                token=hf_token(),
                low_cpu_mem_usage=True,
            )
            pipeline = apply_pipeline_to_device(pipeline, device, offload)

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
