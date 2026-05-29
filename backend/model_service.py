"""Model and pipeline lifecycle helpers for Stable Diffusion workflows."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

import requests
import torch
from diffusers import (
    AutoPipelineForImage2Image,
    ControlNetModel,
    DiffusionPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)

from schemas import GenerationTask, ModelFamily, ModelSource

logger = logging.getLogger(__name__)

MODELS_CACHE_DIR = Path(os.environ.get("MODELS_CACHE_DIR", "/app/models_cache"))
HF_TOKEN = os.environ.get("HF_TOKEN", "")
CIV_TOKEN = os.environ.get("CIV_TOKEN", "")

SKETCH_CONTROLNET_MODELS: dict[ModelFamily, str] = {
    "sd15": "lllyasviel/control_v11p_sd15_scribble",
    "sdxl": "xinsir/controlnet-scribble-sdxl-1.0",
}

_pipeline: Optional[DiffusionPipeline] = None
_loaded_model_id: Optional[str] = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info("Using device: %s", _device)


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


def _resolve_model_family(model_id: str) -> ModelFamily:
    """Infer SD 1.5-like vs SDXL-like model IDs for ControlNet compatibility."""

    normalized = model_id.lower()
    if "sdxl" in normalized or "stable-diffusion-xl" in normalized:
        return "sdxl"
    if re.search(r"(?:^|[-_/])xl(?:$|[-_/])", normalized):
        return "sdxl"
    return "sd15"


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
    destination = _resolve_cache_subpath(f"civitai_{normalized_model_version_id}.safetensors")
    if destination.exists():
        logger.info("CivitAI model already cached at %s", destination)
        return destination

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

    with open(destination, "wb") as file_handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            file_handle.write(chunk)

    logger.info("CivitAI model downloaded to %s", destination)
    return destination


def _detect_pipeline_class(checkpoint_path: Path):
    """Inspect checkpoint keys and infer whether the model is SD 1.x or SDXL."""

    keys: list[str] = []

    try:
        from safetensors import safe_open

        with safe_open(str(checkpoint_path), framework="pt", device="cpu") as file_handle:
            keys = list(file_handle.keys())
    except Exception:
        pass

    if not keys:
        try:
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
            state_dict = checkpoint.get("state_dict", checkpoint)
            keys = list(state_dict.keys())
        except Exception:
            logger.warning(
                "Could not inspect checkpoint keys; defaulting to StableDiffusionPipeline"
            )
            return StableDiffusionPipeline

    if any("conditioner" in key for key in keys):
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
    if pipeline_class not in {StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline}:
        kwargs["safety_checker"] = None
        kwargs["requires_safety_checker"] = False

    pipeline = pipeline_class.from_single_file(str(checkpoint_path), **kwargs)
    pipeline = pipeline.to(_device)
    if _device == "cuda":
        pipeline.enable_attention_slicing()
    return pipeline


def ensure_model(model_id: str, model_source: ModelSource, task: GenerationTask = "text2img") -> None:
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
