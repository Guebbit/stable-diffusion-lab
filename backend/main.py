"""
FastAPI routes for text-to-image and image-guided Stable Diffusion workflows.

This is the HTTP layer — it receives requests from the Vue frontend,
validates inputs, calls the AI pipeline, and returns results as JSON.

Three generation endpoints:
 1. /api/generate          → text-to-image (prompt → image)
 2. /api/generate-from-image → img2img (image + prompt → new image)
 3. /api/generate-sketch-to-ink → ControlNet sketch cleanup
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── App setup ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Stable Diffusion Lab API",
    description="Backend API for Stable Diffusion image generation",
    version="1.0.0",
)

# CORS: allow the frontend (any origin) to call these endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Shared helpers ────────────────────────────────────────────────────────

def _build_generator(seed: Optional[int]) -> tuple[torch.Generator, int]:
    """Build a seeded torch generator for deterministic diffusion sampling.
    The "generator" controls the random noise that the diffusion process starts from.
    Same seed = same starting noise = same output (reproducibility)."""

    seed_value = resolve_seed(seed)
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
    """Create a standard API response from generated images and metadata.
    Wraps raw PIL images into base64 payloads the frontend can display."""

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
        elapsed_seconds=round(elapsed, 2),
    )


# ─── Health / status endpoint ──────────────────────────────────────────────

@app.get("/api/status", response_model=BackendStatus)
async def get_status() -> BackendStatus:
    """Return backend health and currently cached pipeline identifier.
    The frontend polls this to show connection status and loaded model name."""

    return BackendStatus(
        status="ok",
        loaded_model=get_loaded_model_key(),
        device=get_device(),
    )


# ─── Model pre-loading endpoint ───────────────────────────────────────────

@app.post("/api/models/load", response_model=ModelLoadResponse)
async def load_model(request: ModelLoadRequest) -> ModelLoadResponse:
    """Pre-load a model pipeline so the next generation request starts faster.
    Loading a large model can take 10-30s; this lets the UI show a loading state."""

    if request.task == "sketch2ink" and request.model_source != "huggingface":
        raise HTTPException(
            status_code=400,
            detail="Sketch to ink currently supports HuggingFace SD 1.5 and SDXL base models only.",
        )

    try:
        ensure_model(request.model_id, request.model_source, request.task)
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
    """Generate images from text prompt using a text-to-image pipeline.
    This is the "classic" SD workflow: describe what you want → get an image."""

    try:
        ensure_model(request.model_id, request.model_source, task="text2img")
        pipeline = get_active_pipeline()
    except Exception as exc:
        logger.exception("Failed to load model for generation")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    generator, seed_value = _build_generator(request.seed)

    start = time.time()
    try:
        # Call the pipeline like a function — this is where the AI does its work.
        # It runs the full denoising loop (num_inference_steps iterations).
        output = pipeline(
            prompt=request.prompt,                       # What to generate
            negative_prompt=request.negative_prompt or "",  # What to avoid
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,  # Quality vs speed tradeoff
            guidance_scale=request.guidance_scale,       # How literally to follow the prompt
            num_images_per_prompt=request.num_images,    # Batch size
            generator=generator,                        # Seed for reproducibility
        )
    except Exception as exc:
        logger.exception("Image generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _build_generation_response(
        output_images=output.images,  # type: ignore[arg-type, union-attr]
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        model_id=request.model_id,
        width=request.width,
        height=request.height,
        seed=seed_value,
        elapsed=time.time() - start,
    )


# ─── Image-to-image generation ────────────────────────────────────────────

@app.post("/api/generate-from-image", response_model=GenerationResponse)
async def generate_from_image(
    image: UploadFile = File(...),
    prompt: str = Form(..., min_length=1),
    model_id: str = Form(...),
    model_source: ModelSource = Form("huggingface"),
    negative_prompt: Optional[str] = Form(None),
    workflow_preset: ImageWorkflowPreset = Form("general"),
    strength: Optional[float] = Form(None, ge=0.1, le=1.0),
    num_inference_steps: Optional[int] = Form(None, ge=1, le=150),
    guidance_scale: Optional[float] = Form(None, ge=1.0, le=30.0),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    seed: Optional[int] = Form(None),
    num_images: int = Form(1, ge=1, le=4),
) -> GenerationResponse:
    """Generate images from an uploaded reference image using img2img.
    img2img adds noise to your image then denoises it guided by the prompt.
    "strength" controls how much of the original is preserved (lower = more similar)."""

    try:
        ensure_model(model_id, model_source, task="img2img")
        pipeline = get_active_pipeline()
    except Exception as exc:
        logger.exception("Failed to load model for image-to-image generation")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Read and resize the uploaded image to model-safe dimensions
    input_image = await read_uploaded_image(image)
    input_image, target_width, target_height = prepare_input_image(input_image, width, height)
    # Apply workflow preset defaults (user overrides win)
    resolved_strength, resolved_steps, resolved_guidance = resolve_img2img_settings(
        workflow_preset, strength, num_inference_steps, guidance_scale
    )
    generator, seed_value = _build_generator(seed)

    start = time.time()
    try:
        output = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            image=input_image,                  # The reference image to transform
            strength=resolved_strength,         # How much to change (0.3=subtle, 0.9=dramatic)
            num_inference_steps=resolved_steps,
            guidance_scale=resolved_guidance,
            num_images_per_prompt=num_images,
            generator=generator,
        )
    except Exception as exc:
        logger.exception("Image-to-image generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _build_generation_response(
        output_images=output.images,  # type: ignore[arg-type, union-attr]
        prompt=prompt,
        negative_prompt=negative_prompt,
        model_id=model_id,
        width=target_width,
        height=target_height,
        seed=seed_value,
        elapsed=time.time() - start,
    )


# ─── Sketch-to-ink (ControlNet) generation ─────────────────────────────────

@app.post("/api/generate-sketch-to-ink", response_model=GenerationResponse)
async def generate_sketch_to_ink(
    image: UploadFile = File(...),
    prompt: str = Form(..., min_length=1),
    model_id: str = Form(...),
    model_source: ModelSource = Form("huggingface"),
    negative_prompt: Optional[str] = Form(None),
    controlnet_conditioning_scale: float = Form(1.1, ge=0.1, le=2.0),
    num_inference_steps: int = Form(28, ge=1, le=150),
    guidance_scale: float = Form(8.0, ge=1.0, le=30.0),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    seed: Optional[int] = Form(None),
    num_images: int = Form(1, ge=1, le=4),
) -> GenerationResponse:
    """Generate cleaned ink output from a sketch using ControlNet conditioning.
    ControlNet takes a "control image" (your sketch) and uses its edges/lines
    to spatially guide what the diffusion model generates.
    controlnet_conditioning_scale = how strongly the sketch constrains output."""

    if model_source != "huggingface":
        raise HTTPException(
            status_code=400,
            detail="Sketch to ink currently supports HuggingFace SD 1.5 and SDXL base models only.",
        )

    try:
        ensure_model(model_id, model_source, task="sketch2ink")
        pipeline = get_active_pipeline()
    except Exception as exc:
        logger.exception("Failed to load model for sketch-to-ink generation")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    input_image = await read_uploaded_image(image)
    input_image, target_width, target_height = prepare_input_image(input_image, width, height)
    generator, seed_value = _build_generator(seed)

    start = time.time()
    try:
        output = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            image=input_image,                                    # The sketch/scribble
            controlnet_conditioning_scale=controlnet_conditioning_scale,  # Sketch influence strength
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

    return _build_generation_response(
        output_images=output.images,  # type: ignore[arg-type, union-attr]
        prompt=prompt,
        negative_prompt=negative_prompt,
        model_id=model_id,
        width=target_width,
        height=target_height,
        seed=seed_value,
        elapsed=time.time() - start,
    )


# ─── Global error handler ─────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception) -> JSONResponse:
    """Return a stable JSON error payload for unexpected backend exceptions.
    Without this, FastAPI would return HTML error pages that break the frontend."""

    logger.exception("Unhandled exception")
    return JSONResponse(status_code=500, content={"detail": str(exc)})
