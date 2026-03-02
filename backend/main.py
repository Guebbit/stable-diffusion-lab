"""
Stable Diffusion Lab - FastAPI Backend

Provides REST API endpoints to:
- Load models from HuggingFace Hub or CivitAI
- Generate images via Stable Diffusion pipelines
"""

from __future__ import annotations

import io
import logging
import os
import time
import uuid
from base64 import b64encode
from pathlib import Path
from typing import Literal, Optional

import requests
import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS_CACHE_DIR = Path(os.environ.get("MODELS_CACHE_DIR", "/app/models_cache"))
CIVITAI_API_KEY = os.environ.get("CIVITAI_API_KEY", "")

_pipeline: Optional[DiffusionPipeline] = None
_loaded_model_id: Optional[str] = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info("Using device: %s", _device)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

ModelSource = Literal["huggingface", "civitai"]


class ModelLoadRequest(BaseModel):
    model_id: str = Field(..., description="HuggingFace repo ID or CivitAI model version ID")
    model_source: ModelSource = Field("huggingface")


class ModelLoadResponse(BaseModel):
    success: bool
    model_id: str
    message: str


class GenerationRequest(BaseModel):
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
    images: list[GeneratedImage]
    model_id: str
    elapsed_seconds: float


class BackendStatus(BaseModel):
    status: Literal["ok", "loading", "error"]
    loaded_model: Optional[str] = None
    device: str
    message: Optional[str] = None


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _load_huggingface_pipeline(model_id: str) -> DiffusionPipeline:
    logger.info("Loading HuggingFace model: %s", model_id)
    dtype = torch.float16 if _device == "cuda" else torch.float32
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        cache_dir=str(MODELS_CACHE_DIR),
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipeline = pipeline.to(_device)
    if _device == "cuda":
        pipeline.enable_attention_slicing()
    return pipeline


def _download_civitai_model(model_version_id: str) -> Path:
    """Download a CivitAI model checkpoint and return its local path."""
    MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_CACHE_DIR / f"civitai_{model_version_id}.safetensors"
    if dest.exists():
        logger.info("CivitAI model already cached at %s", dest)
        return dest

    url = f"https://civitai.com/api/download/models/{model_version_id}"
    headers: dict[str, str] = {}
    if CIVITAI_API_KEY:
        headers["Authorization"] = f"Bearer {CIVITAI_API_KEY}"

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


def _load_civitai_pipeline(model_version_id: str) -> DiffusionPipeline:
    checkpoint_path = _download_civitai_model(model_version_id)
    dtype = torch.float16 if _device == "cuda" else torch.float32
    logger.info("Loading CivitAI checkpoint from %s", checkpoint_path)
    pipeline = StableDiffusionPipeline.from_single_file(
        str(checkpoint_path),
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipeline = pipeline.to(_device)
    if _device == "cuda":
        pipeline.enable_attention_slicing()
    return pipeline


def _ensure_model(model_id: str, model_source: ModelSource) -> None:
    global _pipeline, _loaded_model_id

    cache_key = f"{model_source}:{model_id}"
    if _loaded_model_id == cache_key and _pipeline is not None:
        return

    if model_source == "huggingface":
        _pipeline = _load_huggingface_pipeline(model_id)
    else:
        _pipeline = _load_civitai_pipeline(model_id)

    _loaded_model_id = cache_key
    logger.info("Model ready: %s", cache_key)


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/status", response_model=BackendStatus)
async def get_status() -> BackendStatus:
    return BackendStatus(
        status="ok",
        loaded_model=_loaded_model_id,
        device=_device,
    )


@app.post("/api/models/load", response_model=ModelLoadResponse)
async def load_model(request: ModelLoadRequest) -> ModelLoadResponse:
    try:
        _ensure_model(request.model_id, request.model_source)
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
    try:
        _ensure_model(request.model_id, request.model_source)
    except Exception as exc:
        logger.exception("Failed to load model for generation")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    assert _pipeline is not None

    seed = request.seed if request.seed is not None else int(time.time()) % (2**32)
    generator = torch.Generator(device=_device).manual_seed(seed)

    start = time.time()
    try:
        output = _pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
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
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    images: list[GeneratedImage] = []
    for pil_image in output.images:  # type: ignore[union-attr]
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        data_url = "data:image/png;base64," + b64encode(buf.getvalue()).decode()
        images.append(
            GeneratedImage(
                id=str(uuid.uuid4()),
                url=data_url,
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                model_id=request.model_id,
                width=request.width,
                height=request.height,
                seed=seed,
                created_at=created_at,
            )
        )

    return GenerationResponse(
        images=images,
        model_id=request.model_id,
        elapsed_seconds=round(elapsed, 2),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception")
    return JSONResponse(status_code=500, content={"detail": str(exc)})
