"""
Model and pipeline lifecycle helpers for Stable Diffusion workflows.

This is the AI brain of the backend. It handles:
 - Loading models from HuggingFace or CivitAI into GPU/CPU memory
 - Keeping exactly ONE pipeline loaded at a time (to avoid OOM)
 - Choosing the right pipeline class depending on the task
   (text2img, img2img, or sketch2ink via ControlNet)
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

import requests
import torch

# "diffusers" is the HuggingFace library that wraps Stable Diffusion models
# into easy-to-use "pipeline" objects you can call like functions.
from diffusers import (
    AutoPipelineForImage2Image,   # Auto-detects the right img2img pipeline class
    ControlNetModel,              # Extra neural network that adds spatial conditioning (e.g. sketch edges)
    DiffusionPipeline,            # Base class for all diffusion pipelines
    StableDiffusionControlNetPipeline,       # SD 1.5 + ControlNet
    StableDiffusionImg2ImgPipeline,          # SD 1.5 image-to-image
    StableDiffusionPipeline,                 # SD 1.5 text-to-image
    StableDiffusionXLControlNetPipeline,     # SDXL + ControlNet
    StableDiffusionXLImg2ImgPipeline,        # SDXL image-to-image
    StableDiffusionXLPipeline,               # SDXL text-to-image
)

from schemas import GenerationTask, ModelFamily, ModelSource

logger = logging.getLogger(__name__)

# --- Configuration from environment variables ---
MODELS_CACHE_DIR = Path(os.environ.get("MODELS_CACHE_DIR", "/app/models_cache"))
HF_TOKEN = os.environ.get("HF_TOKEN", "")    # Optional token for private HuggingFace models
CIV_TOKEN = os.environ.get("CIV_TOKEN", "")  # Optional token for CivitAI downloads

# ControlNet model IDs for sketch-to-ink (one per model family).
# These are pre-trained networks that understand line drawings / scribbles.
SKETCH_CONTROLNET_MODELS: dict[ModelFamily, str] = {
    "sd15": "lllyasviel/control_v11p_sd15_scribble",
    "sdxl": "xinsir/controlnet-scribble-sdxl-1.0",
}

# --- Module-level state: only ONE pipeline active at a time ---
_pipeline: Optional[DiffusionPipeline] = None   # The currently loaded pipeline object
_loaded_model_id: Optional[str] = None           # Cache key so we don't reload the same model
_device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU if available, else CPU (slow)

logger.info("Using device: %s", _device)


# ─── Public getters (used by main.py to check state) ───────────────────────

def get_device() -> str:
    """Return the active torch device used by the backend."""
    return _device


def get_loaded_model_key() -> Optional[str]:
    """Return the cache key of the currently loaded model pipeline."""
    return _loaded_model_id


def get_active_pipeline() -> DiffusionPipeline:
    """Return the active pipeline and fail fast when no model is loaded yet."""
    if _pipeline is None:
        raise RuntimeError("No model pipeline is loaded")
    return _pipeline


# ─── CivitAI helpers (downloading .safetensors checkpoints) ────────────────

def _resolve_civitai_checkpoint_path(normalized_model_version_id: int) -> Path:
    """Return a cache path for a validated CivitAI model version ID.
    Also prevents path-traversal attacks (no ../ escaping the cache dir)."""

    checkpoint_path = (MODELS_CACHE_DIR / f"civitai_{normalized_model_version_id}.safetensors").resolve()
    cache_root = MODELS_CACHE_DIR.resolve()
    if cache_root not in checkpoint_path.parents:
        raise RuntimeError("Resolved CivitAI checkpoint path escaped model cache directory")
    return checkpoint_path


def _normalize_civitai_model_version_id(model_version_id: str) -> int:
    """Validate and normalize CivitAI model IDs to an integer value.
    CivitAI identifies each model variant with a numeric "version ID"."""

    normalized = model_version_id.strip()
    if not re.fullmatch(r"\d+", normalized):
        raise RuntimeError(f"CivitAI model version ID must be numeric, got: {model_version_id}")
    return int(normalized)


# ─── Model family detection ────────────────────────────────────────────────

def _resolve_model_family(model_id: str) -> ModelFamily:
    """Infer whether a model is SD 1.5 or SDXL based on its name.
    This matters because ControlNet models are architecture-specific:
    you can't use an SD 1.5 ControlNet with an SDXL base model."""

    normalized = model_id.lower()
    if "sdxl" in normalized or "stable-diffusion-xl" in normalized:
        return "sdxl"
    if re.search(r"(?:^|[-_/])xl(?:$|[-_/])", normalized):
        return "sdxl"
    return "sd15"


# ─── HuggingFace pipeline loading ──────────────────────────────────────────

def _load_huggingface_pipeline(model_id: str, task: GenerationTask) -> DiffusionPipeline:
    """Load a HuggingFace Diffusers pipeline for the requested generation task.

    Depending on the task we pick different pipeline classes:
    - text2img: generic DiffusionPipeline (works for both SD 1.5 and SDXL)
    - img2img: AutoPipelineForImage2Image (auto-selects correct class)
    - sketch2ink: ControlNet pipeline (needs a separate ControlNet model + base model)
    """

    # float16 = half precision = faster & less VRAM on GPU; float32 for CPU compatibility
    dtype = torch.float16 if _device == "cuda" else torch.float32
    logger.info("Loading HuggingFace model from: %s", model_id)

    if task == "sketch2ink":
        # ControlNet = an extra network injected into the diffusion U-Net.
        # It receives a "conditioning image" (the sketch) and guides generation.
        model_family = _resolve_model_family(model_id)
        controlnet_model_id = SKETCH_CONTROLNET_MODELS[model_family]
        logger.info("Loading sketch ControlNet model from: %s", controlnet_model_id)

        # Load the ControlNet weights (separate from the base model)
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model_id,
            torch_dtype=dtype,
            cache_dir=str(MODELS_CACHE_DIR),
            token=HF_TOKEN or None,
        )
        # Combine the base model + ControlNet into one pipeline
        if model_family == "sdxl":
            pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
                model_id,
                controlnet=controlnet,
                torch_dtype=dtype,
                cache_dir=str(MODELS_CACHE_DIR),
                token=HF_TOKEN or None,
            )
        else:
            pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                model_id,
                controlnet=controlnet,
                torch_dtype=dtype,
                cache_dir=str(MODELS_CACHE_DIR),
                token=HF_TOKEN or None,
            )
    elif task == "img2img":
        # img2img: takes an existing image + prompt, blends them with diffusion
        pipeline = AutoPipelineForImage2Image.from_pretrained(
            model_id,
            torch_dtype=dtype,
            cache_dir=str(MODELS_CACHE_DIR),
            token=HF_TOKEN or None,
        )
    else:
        # text2img: standard text-to-image generation (the default)
        pipeline = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            cache_dir=str(MODELS_CACHE_DIR),
            token=HF_TOKEN or None,
        )

    # Move the pipeline (all model weights) to the target device
    pipeline = pipeline.to(_device)
    # Attention slicing = process attention in chunks → saves VRAM at slight speed cost
    if _device == "cuda":
        pipeline.enable_attention_slicing()
    return pipeline


# ─── CivitAI pipeline loading ──────────────────────────────────────────────

def _download_civitai_model(model_version_id: str) -> Path:
    """Download (or reuse cached) CivitAI checkpoint file.
    CivitAI hosts community models as single .safetensors files (not HF repos)."""

    MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    normalized_model_version_id = _normalize_civitai_model_version_id(model_version_id)
    checkpoint_path = _resolve_civitai_checkpoint_path(normalized_model_version_id)

    # Skip download if already cached
    if checkpoint_path.exists():
        logger.info("CivitAI model already cached at %s", checkpoint_path)
        return checkpoint_path

    # Stream-download the checkpoint (can be several GB)
    url = f"https://civitai.com/api/download/models/{normalized_model_version_id}"
    headers: dict[str, str] = {}
    if CIV_TOKEN:
        headers["Authorization"] = "Bearer " + CIV_TOKEN

    logger.info("Downloading CivitAI model version %s …", model_version_id)
    response = requests.get(url, headers=headers, stream=True, timeout=600)
    if response.status_code != 200:
        raise RuntimeError(
            f"CivitAI download failed with status {response.status_code}: {response.text[:200]}"
        )

    # Write in 1MB chunks to avoid loading the entire model into RAM at once
    with open(checkpoint_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)

    logger.info("CivitAI model downloaded to %s", checkpoint_path)
    return checkpoint_path


def _detect_pipeline_class(checkpoint_path: Path) -> type[DiffusionPipeline]:
    """Inspect checkpoint keys to figure out if the model is SD 1.x or SDXL.
    SDXL checkpoints contain "conditioner" keys that SD 1.x ones don't have."""

    keys: list[str] = []

    try:
        # safetensors = efficient model format; we peek at the tensor names without loading weights
        from safetensors import safe_open

        with safe_open(str(checkpoint_path), framework="pt", device="cpu") as f:
            keys = list(f.keys())
    except Exception:
        logger.warning("Could not inspect checkpoint keys safely; defaulting to SD 1.x pipeline")
        return StableDiffusionPipeline

    # "conditioner" is a key pattern unique to SDXL architecture
    if any("conditioner" in key for key in keys):
        return StableDiffusionXLPipeline

    return StableDiffusionPipeline


def _load_civitai_pipeline(model_version_id: str, task: GenerationTask) -> DiffusionPipeline:
    """Load CivitAI checkpoint into the right pipeline class.
    Unlike HuggingFace (many files in a repo), CivitAI = single checkpoint file."""

    if task == "sketch2ink":
        raise RuntimeError(
            "Sketch to ink currently supports HuggingFace SD 1.5 and SDXL base models only."
        )

    checkpoint_path = _download_civitai_model(model_version_id)
    dtype = torch.float16 if _device == "cuda" else torch.float32
    logger.info("Loading CivitAI checkpoint from %s", checkpoint_path)

    # Auto-detect whether this is SD 1.x or SDXL from the checkpoint structure
    pipeline_class = _detect_pipeline_class(checkpoint_path)
    # If task is img2img, swap to the corresponding img2img pipeline class
    if task == "img2img":
        if pipeline_class.__name__ == "StableDiffusionXLPipeline":
            pipeline_class = StableDiffusionXLImg2ImgPipeline
        else:
            pipeline_class = StableDiffusionImg2ImgPipeline
    logger.info("Auto-detected pipeline class: %s", pipeline_class.__name__)

    kwargs: dict = {"torch_dtype": dtype}
    # SD 1.x has an optional NSFW safety checker; disable it for CivitAI models
    if pipeline_class not in {StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline}:
        kwargs["safety_checker"] = None
        kwargs["requires_safety_checker"] = False

    # from_single_file = load from one .safetensors file (CivitAI format)
    pipeline = pipeline_class.from_single_file(str(checkpoint_path), **kwargs)
    pipeline = pipeline.to(_device)
    if _device == "cuda":
        pipeline.enable_attention_slicing()
    return pipeline


# ─── Main entry point: model loading orchestrator ──────────────────────────

def ensure_model(model_id: str, model_source: ModelSource, task: GenerationTask = "text2img") -> None:
    """Load and cache the requested model pipeline if it is not already active.
    This is a "lazy singleton": if the same model+task is already loaded, skip.
    If a different model is requested, it replaces the old one (only 1 at a time)."""

    global _pipeline, _loaded_model_id

    # Cache key combines task+source+id so switching tasks forces a reload
    cache_key = f"{task}:{model_source}:{model_id}"
    if _loaded_model_id == cache_key and _pipeline is not None:
        return  # Already loaded, nothing to do

    # Pick the right loader based on where the model lives
    if model_source == "huggingface":
        _pipeline = _load_huggingface_pipeline(model_id, task)
    else:
        _pipeline = _load_civitai_pipeline(model_id, task)

    _loaded_model_id = cache_key
    logger.info("Model ready: %s", cache_key)
