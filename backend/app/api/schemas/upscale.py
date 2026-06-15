"""Schemas for the dedicated upscale pipeline endpoint."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class UpscalePipelineRequest(BaseModel):
    """
    Request body for the /upscale/run endpoint (multipart fields arrive via Form).

    Maps directly to UpscalePipelineConfig in the adapter layer:
      Step 1 (optional) — Enhancement: enhance_model_id / enhance_strength
      Step 2 (required) — Upscale:    model_id / scale_factor / prompt / noise_level / num_inference_steps
      Step 3 (optional) — Face Restore: face_restore_model_id / face_restore_fidelity
    """
    model_config = ConfigDict(protected_namespaces=())

    # ── Step 2: SD x4 upscaler (required) ────────────────────────────────
    model_id: str = Field(..., description="ID of an upscale_image-capable model")
    scale_factor: float = Field(2.0, ge=1.0, le=8.0, description="Target upscale multiplier")
    prompt: str = Field(
        "",
        description="Optional text prompt to guide detail synthesis (SD-based upscalers only)",
    )
    noise_level: int = Field(
        20,
        ge=0,
        le=350,
        description="Noise conditioning augmentation — higher values add more AI-generated detail",
    )
    num_inference_steps: int = Field(20, ge=1, le=150, description="Denoising steps")

    # ── Step 1: img2img enhancement (optional) ────────────────────────────
    enhance_model_id: str | None = Field(
        None,
        description="ID of an image_to_image-capable model; if set, runs before upscaling",
    )
    enhance_strength: float = Field(
        0.4,
        ge=0.05,
        le=0.95,
        description="img2img strength for enhancement step (low = subtle cleanup)",
    )

    # ── Step 3: face restoration (optional) ──────────────────────────────
    face_restore_model_id: str | None = Field(
        None,
        description="ID of a face_restore-capable model (e.g. GFPGAN); if set, runs after upscaling",
    )
    face_restore_fidelity: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Fidelity weight: 0.0 = max restoration, 1.0 = preserve original",
    )
