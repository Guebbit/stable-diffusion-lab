"""FastAPI backend for text-to-image and image-guided Stable Diffusion workflows."""

from __future__ import annotations

import io
import logging
import os
import re
import time
import uuid
from base64 import b64encode
from pathlib import Path
from typing import Literal, Optional

import requests
import torch

# Diffusers pipeline classes used across text, img2img, and sketch-conditioned workflows.
# - AutoPipelineForImage2Image: auto-picks img2img class from model metadata.
# - ControlNetModel: injects structure-preserving conditioning from sketch-like inputs.
from diffusers import (
    AutoPipelineForImage2Image,
    ControlNetModel,
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Stable Diffusion Lab API",
    description="Backend API for Stable Diffusion image generation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_CACHE_DIR = Path(os.environ.get("MODELS_CACHE_DIR", "/app/models_cache"))
HF_TOKEN = os.environ.get("HF_TOKEN", "")
CIV_TOKEN = os.environ.get("CIV_TOKEN", "")

# The currently loaded Diffusers pipeline object.
# A pipeline is the high-level class that wires model weights + scheduler + inference call.
_pipeline: Optional[DiffusionPipeline] = None
_loaded_model_id: Optional[str] = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info("Using device: %s", _device)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

ModelSource = Literal["huggingface", "civitai"]
GenerationTask = Literal["text2img", "img2img", "sketch2ink"]
ModelFamily = Literal["sd15", "sdxl"]
ImageWorkflowPreset = Literal["general", "recolor", "style-transfer", "upscale"]

SKETCH_CONTROLNET_MODELS: dict[ModelFamily, str] = {
    "sd15": "lllyasviel/control_v11p_sd15_scribble",
    "sdxl": "xinsir/controlnet-scribble-sdxl-1.0",
}

# Preset defaults for generic image-guided generation workflows.
# These defaults are intentionally mild so users can still control behavior from the UI.
IMAGE_WORKFLOW_DEFAULTS: dict[ImageWorkflowPreset, dict[str, float | int]] = {
    "general": {"strength": 0.6, "num_inference_steps": 20, "guidance_scale": 7.5},
    "recolor": {"strength": 0.45, "num_inference_steps": 24, "guidance_scale": 7.0},
    "style-transfer": {"strength": 0.72, "num_inference_steps": 30, "guidance_scale": 8.0},
    "upscale": {"strength": 0.3, "num_inference_steps": 18, "guidance_scale": 6.5},
}


class ModelLoadRequest(BaseModel):
    """Payload used to load a model for a specific generation task."""
    model_id: str = Field(..., description="HuggingFace repo ID or CivitAI model version ID")
    model_source: ModelSource = Field("huggingface")
    task: GenerationTask = Field("text2img")


class ModelLoadResponse(BaseModel):
    """API response returned after a model load attempt."""
    success: bool
    model_id: str
    message: str


class GenerationRequest(BaseModel):
    """Text-to-image request schema."""
    prompt: str = Field(..., min_length=1)
    negative_prompt: Optional[str] = None
    model_id: str = Field(..., description="HuggingFace repo ID or CivitAI model version ID")
    model_source: ModelSource = Field("huggingface")
    width: int = Field(512, ge=64, le=2048)
    height: int = Field(512, ge=64, le=2048)
    num_inference_steps: int = Field(20, ge=1, le=150)
    guidance_scale: float = Field(7.5, ge=1.0, le=30.0)
    seed: Optional[int] = None
    num_images: int = Field(1, ge=1, le=4)


class GeneratedImage(BaseModel):
    """Single gallery image item returned to the frontend."""
    id: str
    url: str
    prompt: str
    negative_prompt: Optional[str]
    model_id: str
    width: int
    height: int
    seed: int
    created_at: str


class GenerationResponse(BaseModel):
    """Shared response format used by all generation workflows."""
    images: list[GeneratedImage]
    model_id: str
    elapsed_seconds: float


class BackendStatus(BaseModel):
    """Status payload used by the frontend health panel."""
    status: Literal["ok", "loading", "error"]
    loaded_model: Optional[str] = None
    device: str
    message: Optional[str] = None


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _resolve_cache_subpath(relative_path: str) -> Path:
    """Resolve a model cache path and block path traversal outside the cache directory."""
    candidate = (MODELS_CACHE_DIR / relative_path).resolve()
    cache_root = MODELS_CACHE_DIR.resolve()
    if candidate != cache_root and cache_root not in candidate.parents:
        raise RuntimeError(f"Model cache path escapes cache root directory: {relative_path}")
    return candidate


def _normalize_civitai_model_version_id(model_version_id: str) -> str:
    """Validate and normalize CivitAI model IDs to a numeric string."""
    normalized = model_version_id.strip()
    if not re.fullmatch(r"\d+", normalized):
        raise RuntimeError(f"CivitAI model version ID must be numeric, got: {model_version_id}")
    return normalized


def _load_huggingface_pipeline(model_id: str, task: GenerationTask) -> DiffusionPipeline:
    """Load a HuggingFace Diffusers pipeline for the requested generation task."""
    dtype = torch.float16 if _device == "cuda" else torch.float32
    local_path = _resolve_cache_subpath(model_id)
    source = str(local_path) if local_path.exists() else model_id
    logger.info("Loading HuggingFace model from: %s", source)
    if task == "sketch2ink":
        model_family = _resolve_model_family(model_id)
        controlnet_model_id = SKETCH_CONTROLNET_MODELS[model_family]
        logger.info("Loading sketch ControlNet model from: %s", controlnet_model_id)
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model_id,
            torch_dtype=dtype,
            cache_dir=str(MODELS_CACHE_DIR),
            token=HF_TOKEN or None,
        )
        if model_family == "sdxl":
            pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
                source,
                controlnet=controlnet,
                torch_dtype=dtype,
                cache_dir=str(MODELS_CACHE_DIR),
                token=HF_TOKEN or None,
            )
        else:
            pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                source,
                controlnet=controlnet,
                torch_dtype=dtype,
                cache_dir=str(MODELS_CACHE_DIR),
                token=HF_TOKEN or None,
            )
    elif task == "img2img":
        pipeline = AutoPipelineForImage2Image.from_pretrained(
            source,
            torch_dtype=dtype,
            cache_dir=str(MODELS_CACHE_DIR),
            token=HF_TOKEN or None,
        )
    else:
        pipeline = DiffusionPipeline.from_pretrained(
            source,
            torch_dtype=dtype,
            cache_dir=str(MODELS_CACHE_DIR),
            token=HF_TOKEN or None,
        )
    pipeline = pipeline.to(_device)
    if _device == "cuda":
        pipeline.enable_attention_slicing()
    return pipeline


def _download_civitai_model(model_version_id: str) -> Path:
    """Download (or reuse cached) CivitAI checkpoint file."""
    MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    normalized_model_version_id = _normalize_civitai_model_version_id(model_version_id)
    dest = _resolve_cache_subpath(f"civitai_{normalized_model_version_id}.safetensors")
    if dest.exists():
        logger.info("CivitAI model already cached at %s", dest)
        return dest

    url = f"https://civitai.com/api/download/models/{normalized_model_version_id}"
    headers: dict[str, str] = {}
    if CIV_TOKEN:
        headers["Authorization"] = f"Bearer {CIV_TOKEN}"

    logger.info("Downloading CivitAI model version %s …", model_version_id)
    response = requests.get(url, headers=headers, stream=True, timeout=600)
    if response.status_code != 200:
        raise RuntimeError(
            f"CivitAI download failed with status {response.status_code}: {response.text[:200]}"
        )

    with open(dest, "wb") as fh:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            fh.write(chunk)

    logger.info("CivitAI model downloaded to %s", dest)
    return dest


def _detect_pipeline_class(checkpoint_path: Path):
    """
    Inspect checkpoint tensor keys to detect SD 1.x vs SDXL.
    SDXL checkpoints always contain 'conditioner.*' keys; SD 1.x/2.x do not.
    Falls back to StableDiffusionPipeline when detection is inconclusive.
    """
    keys: list[str] = []

    # Prefer safetensors (fast, no pickle)
    try:
        from safetensors import safe_open
        with safe_open(str(checkpoint_path), framework="pt", device="cpu") as f:
            keys = list(f.keys())
    except Exception:
        pass

    # Fallback: load ckpt (pickle), inspect state_dict keys only
    if not keys:
        try:
            ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
            state_dict = ckpt.get("state_dict", ckpt)
            keys = list(state_dict.keys())
        except Exception:
            logger.warning(
                "Could not inspect checkpoint keys; defaulting to StableDiffusionPipeline"
            )
            return StableDiffusionPipeline

    if any("conditioner" in k for k in keys):
        return StableDiffusionXLPipeline

    return StableDiffusionPipeline


def _load_civitai_pipeline(model_version_id: str, task: GenerationTask) -> DiffusionPipeline:
    """Load CivitAI checkpoint into the right pipeline class for text or img2img tasks."""
    if task == "sketch2ink":
        raise RuntimeError(
            "Sketch to ink currently supports HuggingFace SD 1.5 and SDXL base models only."
        )

    checkpoint_path = _download_civitai_model(model_version_id)
    dtype = torch.float16 if _device == "cuda" else torch.float32
    logger.info("Loading CivitAI checkpoint from %s", checkpoint_path)

    pipeline_class = _detect_pipeline_class(checkpoint_path)
    if task == "img2img":
        if pipeline_class.__name__ == "StableDiffusionXLPipeline":
            pipeline_class = StableDiffusionXLImg2ImgPipeline
        else:
            pipeline_class = StableDiffusionImg2ImgPipeline
    logger.info("Auto-detected pipeline class: %s", pipeline_class.__name__)

    kwargs: dict = {"torch_dtype": dtype}
    # SDXL pipelines do not accept safety_checker args
    if pipeline_class not in {StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline}:
        kwargs["safety_checker"] = None
        kwargs["requires_safety_checker"] = False

    pipeline = pipeline_class.from_single_file(str(checkpoint_path), **kwargs)
    pipeline = pipeline.to(_device)
    if _device == "cuda":
        pipeline.enable_attention_slicing()
    return pipeline


def _resolve_model_family(model_id: str) -> ModelFamily:
    """Infer whether a model ID is SD 1.5-like or SDXL-like for ControlNet compatibility."""
    normalized = model_id.lower()
    if "sdxl" in normalized or "stable-diffusion-xl" in normalized:
        return "sdxl"
    # Fallback for model IDs like "stable-diffusion-xl-base-1.0": match "xl" only as a token,
    # bounded by -, _, /, or string edges so names like "pixel" do not count as SDXL markers.
    if re.search(r"(?:^|[-_/])xl(?:$|[-_/])", normalized):
        return "sdxl"
    return "sd15"


def _ensure_model(model_id: str, model_source: ModelSource, task: GenerationTask = "text2img") -> None:
    """Load and cache the requested model pipeline if it is not already active."""
    global _pipeline, _loaded_model_id

    cache_key = f"{task}:{model_source}:{model_id}"
    if _loaded_model_id == cache_key and _pipeline is not None:
        return

    if model_source == "huggingface":
        _pipeline = _load_huggingface_pipeline(model_id, task)
    else:
        _pipeline = _load_civitai_pipeline(model_id, task)

    _loaded_model_id = cache_key
    logger.info("Model ready: %s", cache_key)


def _normalize_size(value: int) -> int:
    """Clamp to model-safe bounds and align to 8px as required by SD latent scaling."""
    bounded = max(64, min(2048, value))
    return (bounded // 8) * 8


def _resolve_seed(seed: Optional[int]) -> int:
    """Return provided seed or derive a 32-bit timestamp-based seed for torch generators."""
    return seed if seed is not None else int(time.time()) % (2**32)


def _serialize_images(
    output_images: list[Image.Image],
    prompt: str,
    negative_prompt: Optional[str],
    model_id: str,
    width: int,
    height: int,
    seed: int,
) -> list[GeneratedImage]:
    """Convert PIL images into gallery-ready base64 payloads used by the frontend."""
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    images: list[GeneratedImage] = []
    for pil_image in output_images:
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        data_url = "data:image/png;base64," + b64encode(buf.getvalue()).decode()
        images.append(
            GeneratedImage(
                id=str(uuid.uuid4()),
                url=data_url,
                prompt=prompt,
                negative_prompt=negative_prompt,
                model_id=model_id,
                width=width,
                height=height,
                seed=seed,
                created_at=created_at,
            )
        )
    return images


async def _read_uploaded_image(uploaded_file: UploadFile) -> Image.Image:
    """Read an uploaded file and convert it to RGB for Diffusers image conditioning."""
    if not uploaded_file.content_type or not uploaded_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    try:
        image_bytes = await uploaded_file.read()
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image upload") from exc


def _prepare_input_image(
    input_image: Image.Image, width: Optional[int], height: Optional[int]
) -> tuple[Image.Image, int, int]:
    """Resize uploaded image to model-safe dimensions while preserving request intent."""
    target_width = _normalize_size(width if width is not None else input_image.width)
    target_height = _normalize_size(height if height is not None else input_image.height)

    if input_image.width != target_width or input_image.height != target_height:
        # Diffusion latents require dimensions divisible by 8, so we snap before inference.
        input_image = input_image.resize((target_width, target_height))
    return input_image, target_width, target_height


def _resolve_img2img_settings(
    workflow_preset: ImageWorkflowPreset,
    strength: Optional[float],
    num_inference_steps: Optional[int],
    guidance_scale: Optional[float],
) -> tuple[float, int, float]:
    """
    Resolve inference settings for image-guided generation.

    Key knobs:
    - strength: how much the output can diverge from uploaded pixels.
    - num_inference_steps: denoising iterations (quality vs latency).
    - guidance_scale: how strongly the prompt steers the result (CFG).
    """
    defaults = IMAGE_WORKFLOW_DEFAULTS[workflow_preset]
    resolved_strength = strength if strength is not None else float(defaults["strength"])
    resolved_steps = (
        num_inference_steps
        if num_inference_steps is not None
        else int(defaults["num_inference_steps"])
    )
    resolved_guidance = guidance_scale if guidance_scale is not None else float(
        defaults["guidance_scale"]
    )
    return resolved_strength, resolved_steps, resolved_guidance


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/status", response_model=BackendStatus)
async def get_status() -> BackendStatus:
    """Return backend health and currently cached pipeline identifier."""
    return BackendStatus(
        status="ok",
        loaded_model=_loaded_model_id,
        device=_device,
    )


@app.post("/api/models/load", response_model=ModelLoadResponse)
async def load_model(request: ModelLoadRequest) -> ModelLoadResponse:
    """Pre-load a model pipeline so the next generation request starts faster."""
    if request.task == "sketch2ink" and request.model_source != "huggingface":
        raise HTTPException(
            status_code=400,
            detail="Sketch to ink currently supports HuggingFace SD 1.5 and SDXL base models only.",
        )

    try:
        _ensure_model(request.model_id, request.model_source, request.task)
        return ModelLoadResponse(
            success=True,
            model_id=request.model_id,
            message=f"Model '{request.model_id}' loaded successfully on {_device}.",
        )
    except Exception as exc:
        logger.exception("Failed to load model %s", request.model_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/generate", response_model=GenerationResponse)
async def generate_images(request: GenerationRequest) -> GenerationResponse:
    """Generate images from text prompt using a text-to-image pipeline."""
    try:
        _ensure_model(request.model_id, request.model_source, task="text2img")
    except Exception as exc:
        logger.exception("Failed to load model for generation")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    assert _pipeline is not None

    seed = _resolve_seed(request.seed)
    # torch.Generator lets us reproduce output when the same seed/settings are reused.
    generator = torch.Generator(device=_device).manual_seed(seed)

    start = time.time()
    try:
        output = _pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or "",
            # width/height apply only to text-to-image because no reference image is provided.
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            num_images_per_prompt=request.num_images,
            generator=generator,
        )
    except Exception as exc:
        logger.exception("Image generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed = time.time() - start
    images = _serialize_images(
        output.images,  # type: ignore[arg-type, union-attr]
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        model_id=request.model_id,
        width=request.width,
        height=request.height,
        seed=seed,
    )

    return GenerationResponse(
        images=images,
        model_id=request.model_id,
        elapsed_seconds=round(elapsed, 2),
    )


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
    """
    Generate images from an uploaded reference image (img2img).

    `workflow_preset` lets frontend presets use consistent backend defaults.
    Explicit request values always take precedence over preset defaults.
    """

    try:
        _ensure_model(model_id, model_source, task="img2img")
    except Exception as exc:
        logger.exception("Failed to load model for image-to-image generation")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    assert _pipeline is not None

    input_image = await _read_uploaded_image(image)
    input_image, target_width, target_height = _prepare_input_image(input_image, width, height)
    resolved_strength, resolved_steps, resolved_guidance = _resolve_img2img_settings(
        workflow_preset, strength, num_inference_steps, guidance_scale
    )

    seed_value = _resolve_seed(seed)
    # Seed controls deterministic noise initialization for img2img denoising.
    generator = torch.Generator(device=_device).manual_seed(seed_value)

    start = time.time()
    try:
        output = _pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            # `image` is the uploaded conditioning image that anchors composition.
            image=input_image,
            # `strength` defines how aggressively we redraw the uploaded image.
            strength=resolved_strength,
            num_inference_steps=resolved_steps,
            guidance_scale=resolved_guidance,
            num_images_per_prompt=num_images,
            generator=generator,
        )
    except Exception as exc:
        logger.exception("Image-to-image generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed = time.time() - start
    images = _serialize_images(
        output.images,  # type: ignore[arg-type, union-attr]
        prompt=prompt,
        negative_prompt=negative_prompt,
        model_id=model_id,
        width=target_width,
        height=target_height,
        seed=seed_value,
    )

    return GenerationResponse(
        images=images,
        model_id=model_id,
        elapsed_seconds=round(elapsed, 2),
    )


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
    """
    Generate cleaned ink-style output from a sketch using a ControlNet-conditioned pipeline.

    ControlNet keeps the sketch structure while allowing text prompt styling on top.
    """
    if model_source != "huggingface":
        raise HTTPException(
            status_code=400,
            detail="Sketch to ink currently supports HuggingFace SD 1.5 and SDXL base models only.",
        )

    try:
        _ensure_model(model_id, model_source, task="sketch2ink")
    except Exception as exc:
        logger.exception("Failed to load model for sketch-to-ink generation")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    assert _pipeline is not None

    input_image = await _read_uploaded_image(image)
    input_image, target_width, target_height = _prepare_input_image(input_image, width, height)

    seed_value = _resolve_seed(seed)
    # Same seed + same sketch + same prompt => reproducible outputs.
    generator = torch.Generator(device=_device).manual_seed(seed_value)

    start = time.time()
    try:
        output = _pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            # The sketch is the ControlNet condition input.
            image=input_image,
            # Higher values enforce sketch structure more strongly.
            controlnet_conditioning_scale=controlnet_conditioning_scale,
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
    images = _serialize_images(
        output.images,  # type: ignore[arg-type, union-attr]
        prompt=prompt,
        negative_prompt=negative_prompt,
        model_id=model_id,
        width=target_width,
        height=target_height,
        seed=seed_value,
    )

    return GenerationResponse(
        images=images,
        model_id=model_id,
        elapsed_seconds=round(elapsed, 2),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception) -> JSONResponse:
    """Return a stable JSON error payload for unexpected backend exceptions."""
    logger.exception("Unhandled exception")
    return JSONResponse(status_code=500, content={"detail": str(exc)})