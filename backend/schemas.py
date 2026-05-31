"""
Shared API schemas and generation type aliases.

These Pydantic models define the "shape" of data flowing between frontend ↔ backend.
They handle validation automatically (e.g. min/max values, required fields).

How Pydantic works (short version):
  - You define a class that inherits from BaseModel.
  - Each class attribute is a typed field.
  - When the API receives JSON, Pydantic parses + validates it automatically.
  - Invalid data raises a 422 error before your code even runs.
  - Field(...) marks a field as required; Field("default") sets a default.
  - ge/le = greater-than-or-equal / less-than-or-equal (numeric bounds).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# ─── Type aliases (used as "lightweight enums") ────────────────────────────
#
# Literal["a", "b"] = a string that can only be "a" or "b".
# This gives us auto-validation: if the frontend sends an unknown value the
# API returns 422 immediately, before any business logic runs.

ModelSource = Literal["huggingface", "civitai"]
# huggingface → model is a repo on huggingface.co (many files, loaded with from_pretrained)
# civitai     → model is a single .safetensors checkpoint downloaded from civitai.com

GenerationTask = Literal["text2img", "img2img", "sketch2ink"]
# text2img   → pure text prompt → new image
# img2img    → reference image + text prompt → transformed image
# sketch2ink → hand-drawn sketch + text prompt → cleaned ink render (uses ControlNet)

ModelFamily = Literal["sd15", "sdxl"]
# sd15 = Stable Diffusion 1.5 — smaller, faster, 512px native resolution
# sdxl = Stable Diffusion XL   — larger, higher quality, 1024px native resolution
# This matters because ControlNet weights are architecture-specific (can't mix families)

ImageWorkflowPreset = Literal["general", "recolor", "style-transfer", "upscale"]
# Preset names the frontend sends to choose img2img generation defaults.
# Each preset maps to a different (strength, steps, guidance) combination
# in IMAGE_WORKFLOW_DEFAULTS inside image_service.py.


# ─── Request/Response models ───────────────────────────────────────────────
#
# These classes describe what JSON the frontend sends (Request) or receives (Response).
# FastAPI uses them to auto-generate OpenAPI docs at /docs.

class ModelLoadRequest(BaseModel):
    """Payload used to pre-load a model (so generation is faster afterwards).
    Loading a large model takes 10-30s; this lets the UI trigger loading early
    and show a spinner while the user is still filling in the prompt form."""

    # HuggingFace example: "stabilityai/stable-diffusion-2-1"
    # CivitAI example: "128713" (the numeric version ID from the URL)
    model_id: str = Field(..., description="HuggingFace repo ID or CivitAI model version ID")
    model_source: ModelSource = Field("huggingface")   # Defaults to HuggingFace if not specified
    task: GenerationTask = Field("text2img")            # Pre-load for a specific task type


class ModelLoadResponse(BaseModel):
    """API response returned after a model load attempt."""

    success: bool       # True = model is ready, False = something went wrong
    model_id: str       # Echo back which model was loaded (for UI confirmation)
    message: str        # Human-readable status or error message


class GenerationRequest(BaseModel):
    """Text-to-image request schema — everything the frontend sends to /api/generate.
    This is used only for the pure text-to-image workflow (JSON body).
    Image-based workflows use multipart form data instead (see main.py)."""

    prompt: str = Field(..., min_length=1)              # What to generate — REQUIRED, cannot be empty
    negative_prompt: Optional[str] = None               # What to avoid (e.g. "blurry, low quality")
    model_id: str = Field(..., description="HuggingFace repo ID or CivitAI model version ID")
    model_source: ModelSource = Field("huggingface")

    # Image dimensions — must be multiples of 8 (SD latent space requirement).
    # 512×512 is safe for SD 1.5; 1024×1024 is typical for SDXL.
    width: int = Field(512, ge=64, le=2048)
    height: int = Field(512, ge=64, le=2048)

    # More steps = higher quality but slower. 20-30 is a good balance.
    # The diffusion process starts from pure noise and iteratively refines
    # the image over num_inference_steps denoising steps.
    num_inference_steps: int = Field(20, ge=1)

    # Guidance scale (CFG) = how literally the model follows the prompt.
    # Low (~3-5): creative, ignores prompt partially.
    # Medium (~7-9): good balance (default 7.5).
    # High (>12): very prompt-literal but can look over-saturated.
    guidance_scale: float = Field(7.5, ge=1.0)

    # Seed controls the initial noise tensor. Same seed + same params = same output.
    # None = generate a random seed each time.
    seed: Optional[int] = None

    # How many images to generate in one call (batched on the GPU).
    num_images: int = Field(1, ge=1)


class GeneratedImage(BaseModel):
    """Single gallery image item returned to the frontend.
    The image bytes are embedded as a base64 data URL so the frontend
    can display them directly without needing a separate file endpoint."""

    id: str                         # UUID — lets the frontend use this as a React key / store key
    url: str                        # "data:image/png;base64,..." — the actual image, embedded
    prompt: str                     # The prompt that generated this image (for display)
    negative_prompt: Optional[str]  # The negative prompt used, if any
    model_id: str                   # Which model produced this image
    width: int                      # Final rendered width in pixels
    height: int                     # Final rendered height in pixels
    seed: int                       # The exact seed used — user can paste this back to recreate
    created_at: str                 # ISO-8601 timestamp, e.g. "2024-05-01T12:00:00Z"


class GenerationResponse(BaseModel):
    """Shared response format used by all three generation workflows
    (text2img, img2img, sketch2ink). The frontend always expects this shape."""

    images: list[GeneratedImage]    # One entry per generated image (batch)
    model_id: str                   # Which model was used (echoed for the frontend to display)
    elapsed_seconds: float          # Total wall-clock time for generation (not counting model load)


class BackendStatus(BaseModel):
    """Status payload used by the frontend health / connection panel.
    The frontend polls /api/status periodically to show if the backend is alive."""

    status: Literal["ok", "loading", "error"]
    loaded_model: Optional[str] = None   # Cache key of the active pipeline, or None if cold
    device: str                          # "cuda" (GPU) or "cpu" (slow fallback)
    message: Optional[str] = None        # Optional extra info (e.g. error details)
