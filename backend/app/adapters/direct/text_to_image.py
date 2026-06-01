"""
Direct text-to-image adapter using HuggingFace Diffusers.

Implements the TextToImageProvider protocol by loading and calling
diffusion pipelines directly in-process. Uses PipelineCache for
efficient model lifecycle management.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from app.adapters.direct.pipeline_cache import PipelineCache
from app.domain.protocols import ProgressCallback
from app.domain.value_objects import ArtifactReference, GenerationParams, JobProgress


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
        on_progress: ProgressCallback | None = None,
    ) -> list[ArtifactReference]:
        """
        Run text-to-image generation.

        Loads the model via PipelineCache (lazy, LRU-managed), runs inference
        in a thread pool, saves outputs to disk, returns artifact references.
        """
        # Get pipeline from cache (loads if not cached, evicts LRU if full)
        pipeline = await self._cache.get_or_load(
            model_id,
            loader=lambda: asyncio.to_thread(self._build_pipeline, model_id),
            category="diffusion",
        )

        # Run blocking inference on thread pool to keep event loop responsive
        artifacts = await asyncio.to_thread(
            self._run_inference, pipeline, params, output_dir, on_progress
        )
        return artifacts

    @staticmethod
    def _build_pipeline(model_id: str) -> tuple[Any, int]:
        """
        Build a text-to-image pipeline — called by PipelineCache on cache miss.

        Returns (pipeline_object, estimated_vram_mb) tuple.
        Heavy imports are deferred to here to keep app startup fast.
        """
        import torch
        from diffusers import DiffusionPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        logger.info("Building text-to-image pipeline: %s → %s", model_id, device)

        pipeline = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
        ).to(device)

        # Estimate VRAM: count parameters × bytes per element
        param_bytes = sum(p.numel() * p.element_size() for p in pipeline.unet.parameters())
        estimated_vram_mb = (param_bytes * 2) // (1024 * 1024)  # ×2 for activations

        return pipeline, estimated_vram_mb

    @staticmethod
    def _run_inference(
        pipeline: Any,
        params: GenerationParams,
        output_dir: Path,
        on_progress: ProgressCallback | None,
    ) -> list[ArtifactReference]:
        """Synchronous inference execution — runs on thread pool."""
        import torch

        # Resolve seed (deterministic if provided, random-ish otherwise)
        seed = params.seed if params.seed is not None else int(time.time() * 1000) % (2**32)
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)

        # Step progress callback — forwards to orchestrator via on_progress
        def step_callback(pipe: Any, step: int, timestep: Any, kwargs: Any) -> Any:
            if on_progress:
                progress = JobProgress(
                    job_id=uuid.UUID(int=0),  # Replaced by orchestrator
                    status="running",
                    progress_percent=int((step / params.num_inference_steps) * 100),
                    current_step=step,
                    total_steps=params.num_inference_steps,
                )
                on_progress(progress)
            return kwargs

        # Run the diffusion pipeline
        result = pipeline(
            prompt=params.prompt,
            negative_prompt=params.negative_prompt or None,
            width=params.width,
            height=params.height,
            num_inference_steps=params.num_inference_steps,
            guidance_scale=params.guidance_scale,
            num_images_per_prompt=params.num_images,
            generator=generator,
            callback_on_step_end=step_callback,
        )

        # Save outputs and build artifact references
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts: list[ArtifactReference] = []

        for image in result.images:
            artifact_id = uuid.uuid4()
            filename = f"{artifact_id}.png"
            file_path = output_dir / filename
            image.save(file_path)

            artifacts.append(
                ArtifactReference(
                    artifact_id=artifact_id,
                    job_id=uuid.UUID(int=0),  # Set by orchestrator
                    file_path=str(file_path),
                    media_type="image/png",
                    width=params.width,
                    height=params.height,
                    size_bytes=file_path.stat().st_size,
                )
            )

        return artifacts
