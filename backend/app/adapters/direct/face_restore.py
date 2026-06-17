"""
Direct face-restoration adapter — sharpens/repairs faces using GFPGAN.

Used as an optional final step in the upscale pipeline (after SD x4 upscaling).
GFPGAN detects all faces in the image, runs each through a GAN trained on
high-quality faces, then blends the result back onto the full image.

Implements no public Protocol (it's only called by UpscalePipeline internally),
but follows the same PipelineCache + asyncio.to_thread pattern as other adapters.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

from app.adapters.direct.pipeline_cache import PipelineCache
from app.domain.value_objects import ArtifactReference

logger = logging.getLogger(__name__)

# Maps HuggingFace model_id → direct download URL for the .pth weight file.
# GFPGAN's GFPGANer accepts a URL and auto-downloads on first use.
_MODEL_URLS: dict[str, str] = {
    "TencentARC/GFPGANv1.4": (
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
    ),
    "TencentARC/GFPGANv1.3": (
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
    ),
}


class DirectFaceRestoreAdapter:
    """
    Face restoration using GFPGAN (Tencent ARC).

    Weights auto-download to ~/.cache/gfpgan/ on first use.
    The restorer is cached in PipelineCache under category 'face_restore'
    to avoid reloading on every pipeline run.

    Note on fidelity_weight (0.0 → 1.0):
      0.0 = pure AI restoration (maximum sharpening, may alter identity)
      1.0 = preserve original pixel values (no restoration effect)
    Sweet spot for most images: 0.5
    """

    def __init__(self, pipeline_cache: PipelineCache) -> None:
        self._cache = pipeline_cache

    async def restore(
        self,
        image_path: Path,
        model_id: str,
        output_dir: Path,
        fidelity: float = 0.5,
    ) -> list[ArtifactReference]:
        """
        Restore faces in the image at image_path.

        Loads the GFPGAN restorer from cache (or builds it on first call),
        runs face detection + enhancement, and saves the result to output_dir.
        """
        # Resolve model URL from registry; fall back to model_id itself
        # (allows local paths or direct URLs to be passed as model_id)
        model_url = _MODEL_URLS.get(model_id, model_id)

        restorer = await self._cache.get_or_load(
            f"{model_id}:face_restore",
            loader=lambda: asyncio.to_thread(self._build_restorer, model_url),
            category="face_restore",
        )

        return await asyncio.to_thread(
            DirectFaceRestoreAdapter._run_inference,
            restorer, image_path, output_dir, fidelity,
        )

    @staticmethod
    def _build_restorer(model_url: str) -> tuple:
        """Build a GFPGANer instance — called on cache miss, runs in thread pool."""
        from gfpgan import GFPGANer

        logger.info("Building GFPGAN restorer from: %s", model_url)

        # upscale=1 because we handle resolution separately in the pipeline.
        # bg_upsampler=None avoids loading a second heavy model just for background.
        restorer = GFPGANer(
            model_path=model_url,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )

        # GFPGAN is light (~350 MB); report its approximate VRAM footprint.
        estimated_vram_mb = 350
        return restorer, estimated_vram_mb

    @staticmethod
    def _run_inference(
        restorer,
        image_path: Path,
        output_dir: Path,
        fidelity: float,
    ) -> list[ArtifactReference]:
        """Synchronous face restoration — runs in thread pool."""
        import cv2
        import numpy as np
        from PIL import Image

        # GFPGAN expects a BGR numpy array (OpenCV convention)
        source = Image.open(image_path).convert("RGB")
        bgr = cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR)

        # enhance() returns (cropped_faces, restored_faces, restored_img)
        # paste_back=True blends the restored faces back onto the full image.
        _, _, restored_bgr = restorer.enhance(
            bgr,
            has_aligned=False,     # let GFPGAN detect and align faces
            only_center_face=False,# restore every face in the image
            paste_back=True,
            weight=fidelity,       # fidelity_weight: higher = less restoration
        )

        if restored_bgr is None:
            # No faces detected — return the source image unchanged
            logger.warning("GFPGAN: no faces detected in %s, returning source", image_path)
            restored_bgr = bgr

        restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
        result = Image.fromarray(restored_rgb)

        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_id = uuid.uuid4()
        file_path = output_dir / f"{artifact_id}.png"
        result.save(file_path)

        return [
            ArtifactReference(
                artifact_id=artifact_id,
                job_id=uuid.UUID(int=0),  # placeholder — face restore runs inside a pipeline step, not as a standalone job
                file_path=str(file_path),
                media_type="image/png",
                width=result.width,
                height=result.height,
                size_bytes=file_path.stat().st_size,
            )
        ]
