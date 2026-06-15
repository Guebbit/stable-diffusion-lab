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
        prompt: str = "",
        noise_level: int = 20,
        num_inference_steps: int = 20,
        on_progress: ProgressCallback | None = None,
    ) -> list[ArtifactReference]:
        pipeline = await self._cache.get_or_load(
            f"{model_id}:upscale",
            loader=lambda: asyncio.to_thread(self._build_pipeline, model_id),
            category="diffusion",
        )
        return await asyncio.to_thread(
            DirectUpscaleAdapter._run_inference,
            pipeline, image_path, output_dir,
            scale_factor, prompt, noise_level, num_inference_steps,
        )

    @staticmethod
    def _build_pipeline(model_id: str) -> tuple:
        import torch
        from diffusers import StableDiffusionUpscalePipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            from app.infrastructure.config.settings import get_settings
            if not get_settings().allow_cpu_fallback:
                raise RuntimeError(
                    "CUDA not available — upscale job aborted. "
                    "Set ALLOW_CPU_FALLBACK=true to allow CPU inference."
                )
            logger.warning("CUDA not available — running upscale on CPU (ALLOW_CPU_FALLBACK=true)")
        dtype = torch.float16 if device == "cuda" else torch.float32
        logger.info("Building upscale pipeline: %s → %s", model_id, device)

        try:
            pipeline = StableDiffusionUpscalePipeline.from_pretrained(
                model_id, torch_dtype=dtype
            ).to(device)
            # VAE tiling decodes in tiles to avoid OOM on large upscaled outputs.
            # Attention slicing reduces peak VRAM during UNet forward passes.
            pipeline.enable_vae_tiling()
            pipeline.enable_attention_slicing()
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

    # The x4 upscaler was trained on 128×128 low-res patches → 512×512 output.
    # Feeding larger images in one shot is OOD and causes polygon/block artifacts.
    _TILE_SIZE = 128   # low-res tile size (model training resolution)
    _OVERLAP = 16      # overlap in low-res pixels to blend seams away

    @staticmethod
    def _blend_weights(size: int):
        import numpy as np
        ramp = np.minimum(np.arange(size), size - 1 - np.arange(size))
        plateau = max(size // 4, 1)
        ramp = np.minimum(ramp, plateau).astype(np.float32)
        return (1.0 - np.cos(np.pi * ramp / plateau)) / 2.0 + 1e-6

    @classmethod
    def _upscale_tiled(
        cls,
        pipeline,
        low_res,
        target_w: int,
        target_h: int,
        prompt: str,
        noise_level: int,
        num_inference_steps: int,
    ) -> "Image.Image":
        import numpy as np
        from PIL import Image

        lr_w, lr_h = low_res.size
        out_canvas = np.zeros((target_h, target_w, 3), dtype=np.float32)
        weight_canvas = np.zeros((target_h, target_w), dtype=np.float32)

        stride = cls._TILE_SIZE - cls._OVERLAP
        ys = list(range(0, lr_h, stride))
        xs = list(range(0, lr_w, stride))

        for y in ys:
            for x in xs:
                x_end = min(x + cls._TILE_SIZE, lr_w)
                y_end = min(y + cls._TILE_SIZE, lr_h)
                x_start = max(0, x_end - cls._TILE_SIZE)
                y_start = max(0, y_end - cls._TILE_SIZE)

                tile = low_res.crop((x_start, y_start, x_end, y_end))
                actual_w, actual_h = tile.size
                if tile.size != (cls._TILE_SIZE, cls._TILE_SIZE):
                    padded = Image.new("RGB", (cls._TILE_SIZE, cls._TILE_SIZE))
                    padded.paste(tile, (0, 0))
                    tile = padded

                result = pipeline(
                    prompt=prompt,
                    image=tile,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=7.5,
                    noise_level=noise_level,
                )
                upscaled_tile = result.images[0]

                out_x = x_start * 4
                out_y = y_start * 4
                out_w = actual_w * 4
                out_h = actual_h * 4

                tile_arr = np.array(
                    upscaled_tile.crop((0, 0, out_w, out_h)), dtype=np.float32
                )
                wy = cls._blend_weights(out_h)[:, np.newaxis]
                wx = cls._blend_weights(out_w)[np.newaxis, :]
                weight = wy * wx

                out_canvas[out_y:out_y + out_h, out_x:out_x + out_w] += (
                    tile_arr * weight[:, :, np.newaxis]
                )
                weight_canvas[out_y:out_y + out_h, out_x:out_x + out_w] += weight

        weight_canvas = np.maximum(weight_canvas, 1e-6)[:, :, np.newaxis]
        result_arr = np.clip(out_canvas / weight_canvas, 0, 255).astype(np.uint8)
        return Image.fromarray(result_arr)

    @classmethod
    def _run_inference(
        cls,
        pipeline,
        image_path: Path,
        output_dir: Path,
        scale_factor: float,
        prompt: str = "",
        noise_level: int = 20,
        num_inference_steps: int = 20,
    ) -> list[ArtifactReference]:
        from PIL import Image

        source = Image.open(image_path).convert("RGB")
        target_w = int(source.width * scale_factor)
        target_h = int(source.height * scale_factor)

        if pipeline is not None:
            # Low-res input for the 4× upscaler: target / 4.
            lr_w = max(target_w // 4, 1)
            lr_h = max(target_h // 4, 1)
            low_res = source.resize((lr_w, lr_h), Image.LANCZOS)

            if lr_w <= cls._TILE_SIZE and lr_h <= cls._TILE_SIZE:
                # Small enough to process in one shot.
                result = pipeline(
                    prompt=prompt,
                    image=low_res,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=7.5,
                    noise_level=noise_level,
                )
                upscaled = result.images[0]
                if upscaled.size != (target_w, target_h):
                    upscaled = upscaled.resize((target_w, target_h), Image.LANCZOS)
            else:
                upscaled = cls._upscale_tiled(
                    pipeline, low_res, target_w, target_h,
                    prompt, noise_level, num_inference_steps,
                )
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
