"""
UpscalePipeline — multi-step image upscaling orchestrator.

Chains up to three steps into a single UPSCALE job:
  1. [Optional] Enhancement — img2img to improve input quality BEFORE upscaling.
     Useful for very noisy or blurry sources where the upscaler would amplify
     existing artefacts. Strength ~0.4 preserves structure while cleaning up.
  2. [Required] Upscale — SD x4 diffusion upscaler (tiled, cosine blending).
     Takes the (optionally enhanced) image and increases resolution.
  3. [Optional] Face Restore — GFPGAN to repair faces AFTER upscaling.
     The upscaler can distort facial features; GFPGAN corrects them.

SOLID responsibilities:
  - UpscalePipelineConfig: single source of truth for pipeline parameters (SRP)
  - UpscalePipeline: orchestrates the three adapters (SRP)
  - Individual adapters handle their own model loading and inference (OCP/DIP)

Each step's adapter is injected (DI) — no static imports, no hidden coupling.
Intermediate outputs live in a temp dir so only the final file reaches output_dir.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from app.domain.protocols import ProgressCallback
from app.domain.value_objects import ArtifactReference, GenerationParams

logger = logging.getLogger(__name__)


@dataclass
class UpscalePipelineConfig:
    """
    All parameters needed to drive the full upscale pipeline.

    Acts as the single source of truth — constructed once from job_params
    and passed down instead of threading individual arguments everywhere.
    """

    # ── Step 2 (required): SD x4 diffusion upscaler ──────────────────────
    upscale_model_id: str
    scale_factor: float = 2.0
    prompt: str = ""
    noise_level: int = 20
    num_inference_steps: int = 20

    # ── Step 1 (optional): img2img enhancement before upscaling ──────────
    # Set enhance_model_id to a non-None value to enable this step.
    enhance_model_id: str | None = None
    enhance_strength: float = 0.4       # keep low to preserve structure
    enhance_steps: int = 20             # fewer steps = faster + less change
    enhance_guidance_scale: float = 7.5

    # ── Step 3 (optional): GFPGAN face restoration after upscaling ───────
    # Set face_restore_model_id to enable. Only useful if the image has faces.
    face_restore_model_id: str | None = None
    # fidelity 0.0 = max restoration, 1.0 = preserve original
    face_restore_fidelity: float = 0.5


class UpscalePipeline:
    """
    Compound orchestrator for the three-step upscale pipeline.

    Registered in AdapterRegistry under JobType.UPSCALE, replacing the bare
    DirectUpscaleAdapter. Exposes the same `upscale()` interface so the
    PipelineExecutor needs no knowledge of the pipeline's internal structure.

    Intermediate files live in a TemporaryDirectory so only the final artifact
    lands in output_dir. The temp dir is cleaned up automatically even on error.
    """

    def __init__(
        self,
        upscale_adapter,    # DirectUpscaleAdapter
        img2img_adapter,    # DirectImageToImageAdapter — used for enhancement
        face_restore_adapter,  # DirectFaceRestoreAdapter
    ) -> None:
        self._upscale = upscale_adapter
        self._img2img = img2img_adapter
        self._face_restore = face_restore_adapter

    async def upscale(
        self,
        image_path: Path,
        model_id: str,
        output_dir: Path,
        # Upscale step params
        scale_factor: float = 2.0,
        prompt: str = "",
        noise_level: int = 20,
        num_inference_steps: int = 20,
        # Enhancement step params (all ignored when enhance_model_id is None)
        enhance_model_id: str | None = None,
        enhance_strength: float = 0.4,
        # Face restore step params (ignored when face_restore_model_id is None)
        face_restore_model_id: str | None = None,
        face_restore_fidelity: float = 0.5,
        on_progress: ProgressCallback | None = None,
    ) -> list[ArtifactReference]:
        """
        Run the full pipeline for the given image.

        Args:
            image_path: Source image to process.
            model_id: ID of the upscale_image-capable model (step 2).
            output_dir: Where the final artifact is written.
            enhance_model_id: If set, run img2img enhancement first (step 1).
            face_restore_model_id: If set, run GFPGAN restoration last (step 3).
        """
        config = UpscalePipelineConfig(
            upscale_model_id=model_id,
            scale_factor=scale_factor,
            prompt=prompt,
            noise_level=noise_level,
            num_inference_steps=num_inference_steps,
            enhance_model_id=enhance_model_id,
            enhance_strength=enhance_strength,
            face_restore_model_id=face_restore_model_id,
            face_restore_fidelity=face_restore_fidelity,
        )

        logger.info(
            "UpscalePipeline starting: enhance=%s, upscale=%s, face_restore=%s",
            config.enhance_model_id or "skip",
            config.upscale_model_id,
            config.face_restore_model_id or "skip",
        )

        # Temp dir holds intermediate results.
        # Using a plain context manager is fine in async code because all
        # awaits complete before the `with` block exits.
        with tempfile.TemporaryDirectory() as _tmp:
            tmp = Path(_tmp)
            current_path = image_path

            # ── Step 1: Enhancement (optional) ───────────────────────────
            if config.enhance_model_id:
                logger.info("Step 1/3: running enhancement with %s", config.enhance_model_id)
                current_path = await self._run_enhance(
                    current_path, config, tmp / "step1_enhance"
                )
            else:
                logger.info("Step 1/3: enhancement skipped")

            # ── Step 2: Upscale (required) ────────────────────────────────
            # Route upscale output through temp so only the final artifact (post-face-restore) lands in output_dir
            step2_out = tmp / "step2_upscale" if config.face_restore_model_id else output_dir
            logger.info("Step 2/3: upscaling with %s (×%s)", config.upscale_model_id, config.scale_factor)
            refs = await self._upscale.upscale(
                current_path,
                config.upscale_model_id,
                step2_out,
                scale_factor=config.scale_factor,
                prompt=config.prompt,
                noise_level=config.noise_level,
                num_inference_steps=config.num_inference_steps,
                on_progress=on_progress,
            )
            current_path = Path(refs[0].file_path)

            # ── Step 3: Face Restore (optional) ──────────────────────────
            if config.face_restore_model_id:
                logger.info("Step 3/3: face restore with %s", config.face_restore_model_id)
                refs = await self._face_restore.restore(
                    current_path,
                    config.face_restore_model_id,
                    output_dir,
                    fidelity=config.face_restore_fidelity,
                )
            else:
                logger.info("Step 3/3: face restore skipped")

            return refs

    # ── Private helpers ───────────────────────────────────────────────────

    async def _run_enhance(
        self,
        image_path: Path,
        config: UpscalePipelineConfig,
        step_dir: Path,
    ) -> Path:
        """
        Run the img2img enhancement step and return the output image path.

        We read the source dimensions so the enhancement preserves the original
        size — no resizing, just quality improvement.
        """
        from PIL import Image as _PILImage

        step_dir.mkdir(parents=True, exist_ok=True)

        with _PILImage.open(image_path) as img:
            w, h = img.size

        gen_params = GenerationParams(
            prompt=config.prompt,
            negative_prompt="",
            width=w,
            height=h,
            num_inference_steps=config.enhance_steps,
            guidance_scale=config.enhance_guidance_scale,
            seed=None,
            num_images=1,
        )

        refs = await self._img2img.generate(
            gen_params,
            config.enhance_model_id,   # type: ignore[arg-type]  # guarded by caller
            image_path,
            step_dir,
            strength=config.enhance_strength,
        )
        return Path(refs[0].file_path)
