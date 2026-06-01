"""
Direct video generation adapter — generates video from text/image prompts.

Uses video diffusion models (AnimateDiff, Stable Video Diffusion, CogVideoX)
via the HuggingFace diffusers library. Implements VideoProvider protocol.

Video generation is the heaviest VRAM consumer (8-24 GB). The ResourceCoordinator
guarantees exclusive GPU access during video inference.
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


class DirectVideoAdapter:
    """
    Video generation using diffusers video pipelines.

    Supports both text-to-video and image-to-video (source_image_path optional).
    Outputs a single MP4 file using imageio/ffmpeg for frame-to-video encoding.
    """

    def __init__(self, pipeline_cache: PipelineCache) -> None:
        self._cache = pipeline_cache

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

        Loads the video pipeline via PipelineCache, generates frames,
        encodes to MP4, and returns an artifact reference.

        Args:
            params: Generation parameters (prompt, steps, etc.).
            model_id: HuggingFace model ID for the video pipeline.
            output_dir: Directory to save the output video file.
            source_image_path: Optional source image for image-to-video.
            on_progress: Optional callback for progress reporting.

        Returns:
            ArtifactReference pointing to the generated video file.
        """
        # Get video pipeline from cache
        pipeline = await self._cache.get_or_load(
            model_id,
            loader=lambda: asyncio.to_thread(self._build_pipeline, model_id),
            category="video",
        )

        # Run inference on thread pool
        artifact = await asyncio.to_thread(
            self._run_inference,
            pipeline,
            params,
            output_dir,
            source_image_path,
            on_progress,
        )
        return artifact

    @staticmethod
    def _build_pipeline(model_id: str) -> tuple[Any, int]:
        """
        Build a video generation pipeline — called by PipelineCache on cache miss.

        Uses DiffusionPipeline.from_pretrained which auto-detects the pipeline type
        (AnimateDiffPipeline, StableVideoDiffusionPipeline, etc.).
        Applies model CPU offloading if VRAM is limited.
        """
        import torch
        from diffusers import DiffusionPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        logger.info("Building video pipeline: %s → %s", model_id, device)

        pipeline = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )

        # Video pipelines are huge — use CPU offloading if on GPU
        # This moves sub-models to GPU only during their forward pass
        if device == "cuda":
            pipeline.enable_model_cpu_offload()
        else:
            pipeline = pipeline.to(device)

        # Estimate VRAM (video models are large: 8-24 GB with offloading)
        total_params = sum(p.numel() for p in pipeline.unet.parameters())
        bytes_per_param = 2 if dtype == torch.float16 else 4
        estimated_vram_mb = (total_params * bytes_per_param) // (1024 * 1024)

        return pipeline, estimated_vram_mb

    @staticmethod
    def _run_inference(
        pipeline: Any,
        params: GenerationParams,
        output_dir: Path,
        source_image_path: Path | None,
        on_progress: ProgressCallback | None,
    ) -> ArtifactReference:
        """Synchronous video inference — runs on thread pool."""
        import torch
        from diffusers.utils import export_to_video

        # Resolve seed
        seed = params.seed if params.seed is not None else int(time.time() * 1000) % (2**32)
        generator = torch.Generator(device="cpu").manual_seed(seed)

        # Build pipeline kwargs
        pipeline_kwargs: dict[str, Any] = {
            "num_inference_steps": params.num_inference_steps,
            "generator": generator,
        }

        # Text-to-video vs image-to-video
        if source_image_path is not None:
            from PIL import Image

            source_image = Image.open(source_image_path).convert("RGB")
            source_image = source_image.resize((params.width, params.height), Image.LANCZOS)
            pipeline_kwargs["image"] = source_image
        else:
            pipeline_kwargs["prompt"] = params.prompt
            if params.negative_prompt:
                pipeline_kwargs["negative_prompt"] = params.negative_prompt

        # Add dimensions if supported by the pipeline
        pipeline_kwargs["width"] = params.width
        pipeline_kwargs["height"] = params.height

        # Step callback for progress (video reports per-frame)
        def step_callback(pipe: Any, step: int, timestep: Any, kwargs: Any) -> Any:
            if on_progress:
                progress = JobProgress(
                    job_id=uuid.UUID(int=0),
                    status="running",
                    progress_percent=int((step / params.num_inference_steps) * 100),
                    current_step=step,
                    total_steps=params.num_inference_steps,
                    message=f"Generating frame batch (step {step})",
                )
                on_progress(progress)
            return kwargs

        pipeline_kwargs["callback_on_step_end"] = step_callback

        # Run video generation
        result = pipeline(**pipeline_kwargs)

        # Export frames to MP4
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_id = uuid.uuid4()
        filename = f"{artifact_id}.mp4"
        file_path = output_dir / filename

        # result.frames is a list of PIL Images or numpy arrays
        frames = result.frames[0] if hasattr(result, "frames") else result.images
        export_to_video(frames, str(file_path), fps=8)

        return ArtifactReference(
            artifact_id=artifact_id,
            job_id=uuid.UUID(int=0),
            file_path=str(file_path),
            media_type="video/mp4",
            width=params.width,
            height=params.height,
            size_bytes=file_path.stat().st_size,
        )
