"""
FastAPI routes for text-to-image and image-guided Stable Diffusion workflows.

This is the HTTP layer — it receives requests from the Vue frontend,
validates inputs, calls the AI pipeline, and returns results as JSON.

Request lifecycle (simplified):
  1. Browser sends HTTP request (JSON or multipart form-data)
  2. FastAPI validates the request body against the Pydantic schema
  3. We call ensure_model() to make sure the right pipeline is warm
  4. We call the pipeline with the request parameters
  5. We serialize the output PIL images → base64 data URLs
  6. We return a GenerationResponse JSON to the browser

Three generation endpoints:
 1. POST /api/generate               → text-to-image (prompt → image), JSON body
 2. POST /api/generate-from-image    → img2img (image + prompt → new image), multipart
 3. POST /api/generate-sketch-to-ink → ControlNet sketch cleanup, multipart

Why multipart for image endpoints?
  JSON can only carry text. Multipart form-data lets us send binary image data
  alongside text fields in a single HTTP request. FastAPI's File() + Form() handle
  the parsing automatically.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from image_service import (
    prepare_input_image,
    read_uploaded_image,
    resolve_img2img_settings,
    resolve_seed,
    serialize_images,
)
from model_service import ensure_model, get_active_pipeline, get_device, get_loaded_model_key
from schemas import (
    BackendStatus,
    GenerationRequest,
    GenerationResponse,
    ImageWorkflowPreset,
    ModelLoadRequest,
    ModelLoadResponse,
    ModelSource,
)

# Configure root logger — INFO level means we see helpful startup/request messages
# but not verbose DEBUG-level torch internals.
# We call basicConfig first as a fallback, then also attach a StreamHandler directly
# so our app-level logger emits output even when uvicorn has already claimed the
# root logger's handlers (basicConfig is a no-op if handlers already exist).
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s - %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

# ─── App setup ─────────────────────────────────────────────────────────────

# FastAPI auto-generates interactive docs at /docs (Swagger UI) and /redoc.
# The title/description/version appear there.
app = FastAPI(
    title="Stable Diffusion Lab API",
    description="Backend API for Stable Diffusion image generation",
    version="1.0.0",
)

# CORS (Cross-Origin Resource Sharing):
#   Browsers block JS fetch() calls to a different origin by default for security.
#   Our Vue frontend (e.g. http://localhost:5173) calls this backend (http://localhost:8000),
#   which is a different origin, so we need CORS headers.
#   allow_origins=["*"] means any origin is permitted — fine for a local dev tool.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Accept requests from any domain
    allow_credentials=True,   # Allow cookies / auth headers if needed
    allow_methods=["*"],      # Allow GET, POST, OPTIONS, etc.
    allow_headers=["*"],      # Allow any request header (e.g. Content-Type, Authorization)
)


# ─── Shared helpers ────────────────────────────────────────────────────────

def _build_generator(seed: Optional[int]) -> tuple[torch.Generator, int]:
    """Build a seeded torch.Generator for deterministic diffusion sampling.

    What is a torch.Generator?
      It's a random number generator object that the diffusion pipeline uses
      to create the initial noise tensor. By seeding it explicitly we ensure
      that the same seed always produces the same starting noise → same output.

    Why device-specific?
      On CUDA, the RNG state lives on the GPU. We must create the generator
      on the same device as the pipeline's tensors or we'd get a device mismatch.
    """

    seed_value = resolve_seed(seed)      # Use provided seed or auto-generate one
    generator = torch.Generator(device=get_device()).manual_seed(seed_value)
    return generator, seed_value


def _build_generation_response(
    *,
    output_images,
    prompt: str,
    negative_prompt: Optional[str],
    model_id: str,
    width: int,
    height: int,
    seed: int,
    elapsed: float,
) -> GenerationResponse:
    """Assemble a standard GenerationResponse from pipeline output and metadata.

    This is a shared helper used by all three generation endpoints so they all
    return exactly the same response format. Keyword-only arguments (the * forces
    callers to use keyword syntax) prevent mistakes from argument ordering.
    """

    # Convert raw PIL images to base64 data URLs with full metadata attached
    images = serialize_images(
        output_images,
        prompt=prompt,
        negative_prompt=negative_prompt,
        model_id=model_id,
        width=width,
        height=height,
        seed=seed,
    )
    return GenerationResponse(
        images=images,
        model_id=model_id,
        elapsed_seconds=round(elapsed, 2),   # 2 decimal places is enough precision
    )


# ─── Health / status endpoint ──────────────────────────────────────────────

@app.get("/api/status", response_model=BackendStatus)
async def get_status() -> BackendStatus:
    """Return backend health and the currently cached pipeline identifier.

    The Vue frontend polls this endpoint periodically to:
      - Show a "connected / disconnected" indicator
      - Display which model is currently warm (loaded in memory)
      - Show whether the backend is running on GPU or CPU
    """

    return BackendStatus(
        status="ok",
        loaded_model=get_loaded_model_key(),   # None if no model has been loaded yet
        device=get_device(),                   # "cuda" or "cpu"
    )


# ─── Model pre-loading endpoint ───────────────────────────────────────────

@app.post("/api/models/load", response_model=ModelLoadResponse)
async def load_model(request: ModelLoadRequest) -> ModelLoadResponse:
    """Pre-load a model pipeline so the next generation request starts faster.

    Why does this exist?
      Loading a large model (especially SDXL) can take 15–60 seconds on first run.
      By providing a dedicated load endpoint, the frontend can:
        1. Trigger loading while the user is still filling in the prompt form.
        2. Show a progress spinner without blocking the generation form.
        3. Confirm the model is ready before the user hits "generate".
    """

    logger.info(
        "POST /api/models/load — model_id=%s  source=%s  task=%s",
        request.model_id, request.model_source, request.task,
    )

    # ControlNet requires HuggingFace because we need to load two separate repos
    # (base model + ControlNet weights). CivitAI is single-file and doesn't support this.
    if request.task == "sketch2ink" and request.model_source != "huggingface":
        raise HTTPException(
            status_code=400,
            detail="Sketch to ink currently supports HuggingFace SD 1.5 and SDXL base models only.",
        )

    try:
        start_load = time.time()
        # ensure_model is idempotent: if the model is already loaded it returns immediately
        ensure_model(request.model_id, request.model_source, request.task)
        load_elapsed = time.time() - start_load
        logger.info(
            "Model '%s' ready in %.2fs on %s",
            request.model_id, load_elapsed, get_device(),
        )
        return ModelLoadResponse(
            success=True,
            model_id=request.model_id,
            message=f"Model '{request.model_id}' loaded successfully on {get_device()}.",
        )
    except Exception as exc:
        logger.exception("Failed to load model %s", request.model_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ─── Text-to-image generation ─────────────────────────────────────────────

@app.post("/api/generate", response_model=GenerationResponse)
async def generate_images(request: GenerationRequest) -> GenerationResponse:
    """Generate images from a text prompt using a text-to-image pipeline.

    This is the "classic" Stable Diffusion workflow:
      describe what you want in words → model generates it from scratch.

    Internally the model:
      1. Encodes the prompt text into embedding vectors (via the text encoder / CLIP)
      2. Starts from a pure random noise tensor (seeded by the generator)
      3. Iteratively denoises over num_inference_steps iterations, guided by the prompt
      4. Decodes the resulting latent tensor back to pixel space (via the VAE decoder)

    This endpoint receives a JSON body (no file upload needed).
    """

    # Ensure the correct pipeline is loaded (no-op if already warm)
    try:
        ensure_model(request.model_id, request.model_source, task="text2img")
        pipeline = get_active_pipeline()
    except Exception as exc:
        logger.exception("Failed to load model for generation")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Build a seeded RNG so we can reproduce this image later if needed
    generator, seed_value = _build_generator(request.seed)

    logger.info(
        "text2img — model=%s  %dx%d  steps=%d  cfg=%.1f  images=%d  seed=%d",
        request.model_id, request.width, request.height,
        request.num_inference_steps, request.guidance_scale,
        request.num_images, seed_value,
    )
    start = time.time()
    try:
        # Calling the pipeline like a function runs the entire diffusion loop.
        # This is where the GPU does its work — can take seconds to minutes.
        output = pipeline(
            prompt=request.prompt,                           # What to generate
            negative_prompt=request.negative_prompt or "",  # What to avoid (empty string = none)
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,  # More steps = higher quality, slower
            guidance_scale=request.guidance_scale,            # CFG: how closely to follow the prompt
            num_images_per_prompt=request.num_images,         # Generate multiple images in one pass
            generator=generator,                              # Seeded RNG for reproducibility
        )
    except Exception as exc:
        logger.exception("Image generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed = time.time() - start
    logger.info("text2img done — %.2fs  %d image(s)", elapsed, request.num_images)
    return _build_generation_response(
        output_images=output.images,  # type: ignore[arg-type, union-attr]
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        model_id=request.model_id,
        width=request.width,
        height=request.height,
        seed=seed_value,
        elapsed=elapsed,
    )


# ─── Image-to-image generation ────────────────────────────────────────────

@app.post("/api/generate-from-image", response_model=GenerationResponse)
async def generate_from_image(
    # FastAPI splits multipart form-data into File() (binary upload) and Form() (text fields)
    image: UploadFile = File(...),                                # The reference image (binary)
    prompt: str = Form(..., min_length=1),                        # Required text field
    model_id: str = Form(...),
    model_source: ModelSource = Form("huggingface"),
    negative_prompt: Optional[str] = Form(None),
    workflow_preset: ImageWorkflowPreset = Form("general"),       # Which preset to start from
    strength: Optional[float] = Form(None, ge=0.1, le=1.0),      # Override preset strength
    num_inference_steps: Optional[int] = Form(None, ge=1, le=150),
    guidance_scale: Optional[float] = Form(None, ge=1.0, le=30.0),
    width: Optional[int] = Form(None),    # If None, keep the uploaded image's original width
    height: Optional[int] = Form(None),
    seed: Optional[int] = Form(None),
    num_images: int = Form(1, ge=1, le=4),
) -> GenerationResponse:
    """Generate images from an uploaded reference image using img2img diffusion.

    How img2img works (conceptually):
      1. The reference image is encoded into latent space (compressed representation).
      2. Gaussian noise is added to that latent — how much depends on `strength`.
         strength=0.3 → mild noise (output stays close to input)
         strength=0.9 → heavy noise (output is mostly prompt-driven, little original left)
      3. The diffusion model denoises from that noisy starting point guided by the prompt.
      4. The resulting latent is decoded back to pixels.

    The key difference from text2img: we start partway through the denoising schedule
    (not from pure noise) so the output is "guided" by the original image's structure.
    """

    try:
        ensure_model(model_id, model_source, task="img2img")
        pipeline = get_active_pipeline()
    except Exception as exc:
        logger.exception("Failed to load model for image-to-image generation")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Decode the uploaded bytes → PIL Image in RGB mode (converts away any alpha channel)
    input_image = await read_uploaded_image(image)
    # Resize to model-safe dimensions (multiples of 8, within 64–2048 bounds)
    input_image, target_width, target_height = prepare_input_image(input_image, width, height)

    # Merge preset defaults with any explicit user overrides
    resolved_strength, resolved_steps, resolved_guidance = resolve_img2img_settings(
        workflow_preset, strength, num_inference_steps, guidance_scale
    )

    generator, seed_value = _build_generator(seed)

    logger.info(
        "img2img — model=%s  preset=%s  strength=%.2f  steps=%d  images=%d  seed=%d",
        model_id, workflow_preset, resolved_strength, resolved_steps, num_images, seed_value,
    )
    start = time.time()
    try:
        output = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            image=input_image,                # The reference image to start from
            strength=resolved_strength,       # How much to transform (0.3=subtle, 0.9=dramatic)
            num_inference_steps=resolved_steps,
            guidance_scale=resolved_guidance,
            num_images_per_prompt=num_images,
            generator=generator,
        )
    except Exception as exc:
        logger.exception("Image-to-image generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed = time.time() - start
    logger.info("img2img done — %.2fs  %d image(s)", elapsed, num_images)
    return _build_generation_response(
        output_images=output.images,  # type: ignore[arg-type, union-attr]
        prompt=prompt,
        negative_prompt=negative_prompt,
        model_id=model_id,
        width=target_width,
        height=target_height,
        seed=seed_value,
        elapsed=elapsed,
    )


# ─── Sketch-to-ink (ControlNet) generation ─────────────────────────────────

@app.post("/api/generate-sketch-to-ink", response_model=GenerationResponse)
async def generate_sketch_to_ink(
    image: UploadFile = File(...),
    prompt: str = Form(..., min_length=1),
    model_id: str = Form(...),
    model_source: ModelSource = Form("huggingface"),
    negative_prompt: Optional[str] = Form(None),
    # controlnet_conditioning_scale: how strongly the sketch constrains the output.
    # 1.0 = standard adherence; >1.0 = more rigid sketch-following; <1.0 = looser.
    controlnet_conditioning_scale: float = Form(1.1, ge=0.1, le=2.0),
    num_inference_steps: int = Form(28, ge=1, le=150),
    guidance_scale: float = Form(8.0, ge=1.0, le=30.0),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    seed: Optional[int] = Form(None),
    num_images: int = Form(1, ge=1, le=4),
) -> GenerationResponse:
    """Generate cleaned ink output from a hand-drawn sketch using ControlNet.

    How ControlNet sketch-to-ink works:
      1. Your sketch image is the "conditioning image" passed to the ControlNet branch.
      2. ControlNet extracts structural/edge information from the sketch.
      3. These structural activations are injected into the diffusion U-Net at multiple
         resolutions, forcing the model to respect the sketch's composition and line structure.
      4. The text prompt controls style, colors, and content details.
      5. The result looks like your sketch — but drawn in a clean, rendered style.

    Only HuggingFace models are supported here because we need separate model repos
    for the base model AND the ControlNet weights (two from_pretrained() calls).
    CivitAI provides only single-file checkpoints which don't support this pattern.
    """

    if model_source != "huggingface":
        raise HTTPException(
            status_code=400,
            detail="Sketch to ink currently supports HuggingFace SD 1.5 and SDXL base models only.",
        )

    try:
        # Loading sketch2ink pipeline is heavier than others:
        # it downloads + loads BOTH the base model AND the ControlNet model
        ensure_model(model_id, model_source, task="sketch2ink")
        pipeline = get_active_pipeline()
    except Exception as exc:
        logger.exception("Failed to load model for sketch-to-ink generation")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    input_image = await read_uploaded_image(image)
    input_image, target_width, target_height = prepare_input_image(input_image, width, height)
    generator, seed_value = _build_generator(seed)

    logger.info(
        "sketch2ink — model=%s  cnet_scale=%.2f  steps=%d  images=%d  seed=%d",
        model_id, controlnet_conditioning_scale, num_inference_steps, num_images, seed_value,
    )
    start = time.time()
    try:
        output = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            image=input_image,                                          # The sketch (conditioning image)
            controlnet_conditioning_scale=controlnet_conditioning_scale, # Sketch influence strength
            width=target_width,
            height=target_height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            generator=generator,
        )
    except Exception as exc:
        logger.exception("Sketch-to-ink generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed = time.time() - start
    logger.info("sketch2ink done — %.2fs  %d image(s)", elapsed, num_images)
    return _build_generation_response(
        output_images=output.images,  # type: ignore[arg-type, union-attr]
        prompt=prompt,
        negative_prompt=negative_prompt,
        model_id=model_id,
        width=target_width,
        height=target_height,
        seed=seed_value,
        elapsed=elapsed,
    )


# ─── Global error handler ─────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception) -> JSONResponse:
    """Catch any unhandled exception and return a stable JSON error payload.

    Without this, FastAPI's default behavior for unexpected errors is to return
    an HTML "500 Internal Server Error" page. That breaks the frontend because
    it expects JSON in every response and can't parse an HTML error body.

    This handler ensures the frontend always gets a predictable {"detail": "..."} shape
    even when something completely unexpected crashes the server.
    """

    logger.exception("Unhandled exception")
    return JSONResponse(status_code=500, content={"detail": str(exc)})
