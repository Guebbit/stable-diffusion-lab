"""
Direct text-to-image adapter using HuggingFace Diffusers.

Implements the TextToImageProvider protocol by loading and calling
diffusion pipelines directly in-process. Handles pipeline caching
to avoid redundant loads.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from app.domain.protocols import ProgressCallback, TextToImageProvider
from app.domain.value_objects import ArtifactReference, GenerationParams, JobProgress

logger = logging.getLogger(__name__)


class DirectTextToImageAdapter:
    """
    Generates images from text prompts using diffusers pipelines.

    Implements TextToImageProvider protocol. Runs inference on a thread pool
    to avoid blocking the async event loop.

    Pipeline lifecycle:
    - Pipelines are loaded lazily on first use
    - Only one pipeline is kept in memory at a time (GPU constraint)
    - Switching models triggers unload → load automatically
    """

    def __init__(self) -> None:
        self._current_model_id: str | None = None
        self._pipeline: Any = None

    async def generate(
        self,
        params: GenerationParams,
        model_id: str,
        output_dir: Path,
        on_progress: ProgressCallback | None = None,
    ) -> list[ArtifactReference]:
        """
        Run text-to-image generation.

        Loads the model if not already loaded, runs inference in a thread pool,
        saves outputs to disk, and returns artifact references.
        """
        # Ensure correct model is loaded
        if self._current_model_id != model_id:
            await self._load_pipeline(model_id)

        # Run blocking inference on thread pool to keep event loop responsive
        artifacts = await asyncio.to_thread(
            self._run_inference, params, output_dir, on_progress
        )
        return artifacts

    async def _load_pipeline(self, model_id: str) -> None:
        """Load a diffusion pipeline (heavy operation, done on thread pool)."""
        await asyncio.to_thread(self._load_pipeline_sync, model_id)

    def _load_pipeline_sync(self, model_id: str) -> None:
        """Synchronous pipeline loading — runs on thread pool."""
        # Import heavy libraries only inside the adapter
        import torch
        from diffusers import DiffusionPipeline

        logger.info("Loading pipeline: %s", model_id)

        # Unload previous pipeline to free VRAM
        if self._pipeline is not None:
            del self._pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        self._pipeline = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
        ).to(device)

        self._current_model_id = model_id
        logger.info("Pipeline loaded: %s on %s", model_id, device)

    def _run_inference(
        self,
        params: GenerationParams,
        output_dir: Path,
        on_progress: ProgressCallback | None,
    ) -> list[ArtifactReference]:
        """Synchronous inference execution — runs on thread pool."""
        import torch

        # Resolve seed
        seed = params.seed if params.seed is not None else int(time.time() * 1000) % (2**32)
        generator = torch.Generator(device=self._pipeline.device).manual_seed(seed)

        # Callback for step progress
        def step_callback(pipe: Any, step: int, timestep: Any, kwargs: Any) -> Any:
            if on_progress:
                progress = JobProgress(
                    job_id=uuid.UUID(int=0),  # Will be replaced by orchestrator
                    status="running",
                    progress_percent=int((step / params.num_inference_steps) * 100),
                    current_step=step,
                    total_steps=params.num_inference_steps,
                )
                on_progress(progress)
            return kwargs

        # Run the pipeline
        result = self._pipeline(
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
        artifacts: list[ArtifactReference] = []
        for i, image in enumerate(result.images):
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
