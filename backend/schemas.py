"""
Shared API schemas and generation type aliases.

These Pydantic models define the "shape" of data flowing between frontend ↔ backend.
They handle validation automatically (e.g. min/max values, required fields).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# Type aliases — these restrict values to a known set of options (like enums but lighter)
ModelSource = Literal["huggingface", "civitai"]                          # Where models come from
GenerationTask = Literal["text2img", "img2img", "sketch2ink"]           # What kind of generation
ModelFamily = Literal["sd15", "sdxl"]                                   # Architecture family
ImageWorkflowPreset = Literal["general", "recolor", "style-transfer", "upscale"]  # img2img presets


# ─── Request/Response models ───────────────────────────────────────────────

class ModelLoadRequest(BaseModel):
    """Payload used to pre-load a model (so generation is faster afterwards)."""

    model_id: str = Field(..., description="HuggingFace repo ID or CivitAI model version ID")
    model_source: ModelSource = Field("huggingface")
    task: GenerationTask = Field("text2img")


class ModelLoadResponse(BaseModel):
    """API response returned after a model load attempt."""

    success: bool
    model_id: str
    message: str


class GenerationRequest(BaseModel):
    """Text-to-image request schema — everything the frontend sends to /api/generate."""

    prompt: str = Field(..., min_length=1)                              # What to generate
    negative_prompt: Optional[str] = None                               # What to avoid
    model_id: str = Field(..., description="HuggingFace repo ID or CivitAI model version ID")
    model_source: ModelSource = Field("huggingface")
    width: int = Field(512, ge=64, le=2048)                            # Image width in pixels
    height: int = Field(512, ge=64, le=2048)                           # Image height in pixels
    num_inference_steps: int = Field(20, ge=1, le=150)                 # Denoising iterations (quality)
    guidance_scale: float = Field(7.5, ge=1.0, le=30.0)                # Prompt adherence strength
    seed: Optional[int] = None                                         # For reproducibility
    num_images: int = Field(1, ge=1, le=4)                             # How many images to generate


class GeneratedImage(BaseModel):
    """Single gallery image item returned to the frontend."""

    id: str                         # Unique identifier
    url: str                        # Base64 data URL (embedded PNG)
    prompt: str
    negative_prompt: Optional[str]
    model_id: str
    width: int
    height: int
    seed: int                       # The seed used (for "recreate this" feature)
    created_at: str


class GenerationResponse(BaseModel):
    """Shared response format used by all generation workflows."""

    images: list[GeneratedImage]    # The generated images
    model_id: str                   # Which model produced them
    elapsed_seconds: float          # How long generation took


class BackendStatus(BaseModel):
    """Status payload used by the frontend health panel."""

    status: Literal["ok", "loading", "error"]
    loaded_model: Optional[str] = None   # Currently cached model (or None if cold start)
    device: str                          # "cuda" or "cpu"
    message: Optional[str] = None
