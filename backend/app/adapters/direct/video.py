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

from app.adapters.base import estimate_pipeline_vram_mb, hf_token, load_pipeline, resolve_model_path
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
    ) -> list[ArtifactReference]:
        """
        Run video generation inference.

        Loads the video pipeline via PipelineCache, generates frames,
        encodes to MP4, and returns a single-element artifact list.
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
        return [artifact]

    @staticmethod
    def _build_pipeline(model_id: str) -> tuple[Any, int]:
        """
        Build a video generation pipeline — called by PipelineCache on cache miss.

        Auto-detects the pipeline type (AnimateDiffPipeline, StableVideoDiffusionPipeline,
        CogVideoX, etc.) via DiffusionPipeline.from_pretrained. Video models can be very
        large (CogVideoX is ~18 GB), so load_pipeline() is used instead of the old
        from_pretrained + apply_pipeline_to_device pattern — which staged through CPU RAM
        and would fill all system RAM for large video models in PIPELINE_OFFLOAD=none mode.
        """
        import torch
        from diffusers import DiffusionPipeline
        from app.infrastructure.config.settings import get_settings

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            if not get_settings().allow_cpu_fallback:
                raise RuntimeError(
                    "CUDA not available — video job aborted. "
                    "Set ALLOW_CPU_FALLBACK=true to allow CPU inference."
                )
            logger.warning("CUDA not available — running video on CPU (ALLOW_CPU_FALLBACK=true)")

        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        settings = get_settings()
        model_path = resolve_model_path(model_id)
        logger.info("Building video pipeline: %s → %s", model_path, device)

        pipeline = load_pipeline(
            DiffusionPipeline,
            model_path,
            device,
            dtype,
            settings.pipeline_offload,
            settings.quantize_mode,
            hf_token(),
        )

        return pipeline, estimate_pipeline_vram_mb(pipeline)

    @staticmethod
    def _run_inference(
        pipeline: Any,
        params: GenerationParams,
        output_dir: Path,
        source_image_path: Path | None,
        on_progress: ProgressCallback | None,
    ) -> ArtifactReference:  # single artifact; generate() wraps it in a list
        """Synchronous video inference — runs on thread pool."""
        import torch
        from diffusers.utils import export_to_video

        # Resolve seed
        seed = params.seed if params.seed is not None else int(time.time() * 1000) % (2**32)
        # CPU generator: SVD and AnimateDiff move it to the correct device internally;
        # passing a CUDA generator causes device-mismatch errors on some pipeline versions
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
