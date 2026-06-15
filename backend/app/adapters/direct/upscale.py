"""
Direct upscale adapter — upscales images using a diffusion upscaling pipeline.

Implements UpscaleProvider protocol from app.domain.protocols.
Uses the shared PipelineCache for model lifecycle management.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

from app.adapters.direct.pipeline_cache import PipelineCache
from app.domain.protocols import ProgressCallback
from app.domain.value_objects import ArtifactReference

logger = logging.getLogger(__name__)


class DirectUpscaleAdapter:
    """
    Image upscaling using diffusion upscaling pipelines (e.g. StableDiffusionUpscalePipeline).

    The pipeline produces a fixed-resolution upscaled output; a final PIL resize
    is applied if the requested scale_factor yields different target dimensions.
    """

    def __init__(self, pipeline_cache: PipelineCache) -> None:
        self._cache = pipeline_cache

    async def upscale(
        self,
        image_path: Path,
        model_id: str,
        output_dir: Path,
        scale_factor: float = 2.0,
        on_progress: ProgressCallback | None = None,
    ) -> list[ArtifactReference]:
        pipeline = await self._cache.get_or_load(
            f"{model_id}:upscale",
            loader=lambda: asyncio.to_thread(self._build_pipeline, model_id),
            category="diffusion",
        )
        return await asyncio.to_thread(
            self._run_inference, pipeline, image_path, output_dir, scale_factor
        )

    @staticmethod
    def _build_pipeline(model_id: str) -> tuple:
        import torch
        from diffusers import StableDiffusionUpscalePipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        logger.info("Building upscale pipeline: %s → %s", model_id, device)

        try:
            pipeline = StableDiffusionUpscalePipeline.from_pretrained(
                model_id, torch_dtype=dtype
            ).to(device)
            param_bytes = sum(p.numel() * p.element_size() for p in pipeline.unet.parameters())
            return pipeline, (param_bytes * 2) // (1024 * 1024)
        except (ValueError, OSError):
            # Model is not a dedicated upscaler (missing low_res_scheduler / watermarker).
            # Fall back to PIL-only upscaling — no ML pipeline needed.
            logger.warning(
                "Model %s cannot be loaded as StableDiffusionUpscalePipeline; "
                "using PIL Lanczos upscaling instead",
                model_id,
            )
            return None, 0

    @staticmethod
    def _run_inference(
        pipeline,
        image_path: Path,
        output_dir: Path,
        scale_factor: float,
    ) -> list[ArtifactReference]:
        from PIL import Image

        source = Image.open(image_path).convert("RGB")
        target_w = int(source.width * scale_factor)
        target_h = int(source.height * scale_factor)

        if pipeline is not None:
            result = pipeline(
                prompt="",
                image=source,
                num_inference_steps=20,
                guidance_scale=0.0,
            )
            upscaled = result.images[0]
            if upscaled.size != (target_w, target_h):
                upscaled = upscaled.resize((target_w, target_h), Image.LANCZOS)
        else:
            upscaled = source.resize((target_w, target_h), Image.LANCZOS)

        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_id = uuid.uuid4()
        file_path = output_dir / f"{artifact_id}.png"
        upscaled.save(file_path)

        return [
            ArtifactReference(
                artifact_id=artifact_id,
                job_id=uuid.UUID(int=0),
                file_path=str(file_path),
                media_type="image/png",
                width=target_w,
                height=target_h,
                size_bytes=file_path.stat().st_size,
            )
        ]
