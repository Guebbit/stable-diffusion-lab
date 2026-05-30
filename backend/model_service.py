"""
Model and pipeline lifecycle helpers for Stable Diffusion workflows.

This is the AI brain of the backend. It handles:
 - Loading models from HuggingFace or CivitAI into GPU/CPU memory
 - Keeping exactly ONE pipeline loaded at a time (to avoid OOM)
 - Choosing the right pipeline class depending on the task
   (text2img, img2img, or sketch2ink via ControlNet)

What is a "pipeline"?
  In HuggingFace Diffusers, a pipeline is an object that bundles together:
    - The VAE      (encodes images into latent space, decodes back to pixels)
    - The U-Net    (the main denoising neural network)
    - The text encoder (converts your prompt string into embeddings)
    - The scheduler (controls the noise removal schedule across steps)
  You call it like a function: output = pipeline(prompt=...) → PIL images.

Why only ONE pipeline at a time?
  Each pipeline occupies several GB of GPU VRAM (e.g. SD 1.5 ≈ 4 GB float16).
  Loading a second one without unloading the first would cause CUDA out-of-memory.
  The module-level singleton (_pipeline) ensures the old pipeline is garbage-collected
  before the new one is loaded.
"""

from __future__ import annotations

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
from logging_config import get_logger

# Attach a StreamHandler via the shared helper so logs surface even when uvicorn
# has already claimed the root logger (logging.basicConfig is a no-op in that case).
logger = get_logger(__name__)

# ─── Runtime configuration (from environment variables) ────────────────────
#
# Using env vars instead of hard-coded paths means Docker / deployment configs
# can override these without touching source code.

MODELS_CACHE_DIR = Path(os.environ.get("MODELS_CACHE_DIR", "/app/models_cache"))
# HF_TOKEN is needed for gated/private HuggingFace models (e.g. Meta Llama).
# For most public Stable Diffusion models it can be left empty.
HF_TOKEN = os.environ.get("HF_TOKEN", "")
# CIV_TOKEN is the CivitAI API key. Required for most downloads since CivitAI
# added mandatory authentication to their download API.
CIV_TOKEN = os.environ.get("CIV_TOKEN", "")

# ─── ControlNet model IDs for sketch-to-ink ────────────────────────────────
#
# ControlNet models are architecture-specific:
#   - SD 1.5 ControlNet CANNOT be used with SDXL base models and vice versa.
# We keep one ControlNet model per model family so we can pick the right one
# based on which base model the user selected.
SKETCH_CONTROLNET_MODELS: dict[ModelFamily, str] = {
    "sd15": "lllyasviel/control_v11p_sd15_scribble",
    # xinsir's scribble ControlNet is fine-tuned for SDXL
    "sdxl": "xinsir/controlnet-scribble-sdxl-1.0",
}

# ─── Module-level singleton state ──────────────────────────────────────────
#
# These three variables represent the complete runtime state of this module.
# They live at module scope (not inside a class) for simplicity.
_pipeline: Optional[DiffusionPipeline] = None   # The currently loaded pipeline object
_loaded_model_id: Optional[str] = None          # Cache key (task:source:model_id)
# Detect GPU at startup. torch.cuda.is_available() returns True only if
# NVIDIA CUDA drivers are installed and a compatible GPU is present.
_device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info("Using device: %s", _device)


# ─── Public getters (used by main.py to read state) ────────────────────────

def get_device() -> str:
    """Return the torch device string ("cuda" or "cpu") in use by this backend."""
    return _device


def get_loaded_model_key() -> Optional[str]:
    """Return the composite cache key of the currently loaded pipeline, or None.
    The key has the form "task:source:model_id" (e.g. "text2img:huggingface:runwayml/stable-diffusion-v1-5").
    The frontend displays this to tell the user which model is warm."""
    return _loaded_model_id


def get_active_pipeline() -> DiffusionPipeline:
    """Return the active pipeline, raising RuntimeError if nothing is loaded yet.
    Callers in main.py should always call ensure_model() before this."""
    if _pipeline is None:
        raise RuntimeError("No model pipeline is loaded")
    return _pipeline


# ─── CivitAI helpers (path validation + downloading) ──────────────────────

def _resolve_civitai_checkpoint_path(normalized_model_version_id: int) -> Path:
    """Return the expected local cache path for a CivitAI checkpoint.

    Security: .resolve() expands symlinks and ".." components.
    We then verify the resolved path is still inside MODELS_CACHE_DIR.
    This prevents a path-traversal attack where a crafted model_id like
    "../../etc/passwd" could escape the cache directory.
    """

    checkpoint_path = (
        MODELS_CACHE_DIR / f"civitai_{normalized_model_version_id}.safetensors"
    ).resolve()
    cache_root = MODELS_CACHE_DIR.resolve()

    # parents is a list of all ancestor directories of the resolved path.
    # If cache_root is not in that list, the path has escaped the cache dir.
    if cache_root not in checkpoint_path.parents:
        raise RuntimeError("Resolved CivitAI checkpoint path escaped model cache directory")
    return checkpoint_path


def _normalize_civitai_model_version_id(model_version_id: str) -> int:
    """Validate and normalize CivitAI model version IDs to a plain integer.

    CivitAI model version IDs are purely numeric (e.g. "128713").
    We strip whitespace and reject anything that isn't digits-only to prevent
    injection attacks (e.g. someone passing "123; rm -rf /").
    """

    normalized = model_version_id.strip()
    # fullmatch means the ENTIRE string must match, no leftover characters
    if not re.fullmatch(r"\d+", normalized):
        raise RuntimeError(f"CivitAI model version ID must be numeric, got: {model_version_id}")
    return int(normalized)


# ─── Model family detection ────────────────────────────────────────────────

def _resolve_model_family(model_id: str) -> ModelFamily:
    """Infer whether a model is SD 1.5-family or SDXL-family from its name string.

    Why do we need this?
      Diffusers has separate pipeline classes for SD 1.5 and SDXL.
      ControlNet models are architecture-specific — you can't mix them.
      Rather than asking the user to specify the family explicitly, we detect it
      from common naming conventions used on HuggingFace and CivitAI.

    Detection rules (applied in order):
      1. Contains "sdxl" or "stable-diffusion-xl" → SDXL
      2. Contains a word-boundary "xl" token → SDXL (e.g. "my-model-xl-v2")
      3. Otherwise → SD 1.5 (safe default for most models)
    """

    normalized = model_id.lower()
    if "sdxl" in normalized or "stable-diffusion-xl" in normalized:
        return "sdxl"
    # Word-boundary check: "xl" must be surrounded by start/end or [-_/] separators.
    # Avoids false matches on words like "textile" that happen to contain "xl".
    if re.search(r"(?:^|[-_/])xl(?:$|[-_/])", normalized):
        return "sdxl"
    return "sd15"


# ─── HuggingFace pipeline loading ──────────────────────────────────────────

def _load_huggingface_pipeline(model_id: str, task: GenerationTask) -> DiffusionPipeline:
    """Load a HuggingFace Diffusers pipeline for the requested generation task.

    HuggingFace models live in "model repositories" (many files: config, weights, tokenizer…).
    from_pretrained() downloads them (first call) or loads from cache (subsequent calls).

    Pipeline class chosen per task:
      text2img   → DiffusionPipeline.from_pretrained — handles both SD 1.5 and SDXL automatically
      img2img    → AutoPipelineForImage2Image         — auto-wraps the text2img model for img2img
      sketch2ink → ControlNet pipeline                — needs an additional ControlNet model

    dtype choice:
      float16 (half precision) on CUDA: uses half the VRAM and runs ~2× faster
      float32 (full precision) on CPU:  required because most CPUs lack float16 support
    """

    dtype = torch.float16 if _device == "cuda" else torch.float32
    logger.info("Loading HuggingFace model from: %s", model_id)

    if task == "sketch2ink":
        # ─── ControlNet loading ─────────────────────────────────────────────
        # ControlNet works by injecting an extra "conditioning branch" into the U-Net.
        # The conditioning branch receives the sketch image and produces activations
        # that are added to the U-Net's own activations at multiple resolution levels.
        # This forces the U-Net to respect the sketch's spatial structure while still
        # following the text prompt for style/content.

        # First determine the architecture so we pick the matching ControlNet weights
        model_family = _resolve_model_family(model_id)
        controlnet_model_id = SKETCH_CONTROLNET_MODELS[model_family]
        logger.info("Loading sketch ControlNet model from: %s", controlnet_model_id)

        # Load ControlNet weights independently from the base model
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model_id,
            torch_dtype=dtype,
            cache_dir=str(MODELS_CACHE_DIR),
            token=HF_TOKEN or None,  # None = use anonymous access
        )

        # Combine base model + ControlNet into one callable pipeline object
        if model_family == "sdxl":
            pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
                model_id,
                controlnet=controlnet,   # Inject the ControlNet branch
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
        # ─── img2img loading ────────────────────────────────────────────────
        # AutoPipelineForImage2Image inspects the model config and wraps it
        # with the correct img2img class (SD15 or SDXL) automatically.
        # This avoids us having to detect the family here.
        pipeline = AutoPipelineForImage2Image.from_pretrained(
            model_id,
            torch_dtype=dtype,
            cache_dir=str(MODELS_CACHE_DIR),
            token=HF_TOKEN or None,
        )

    else:
        # ─── text2img loading ────────────────────────────────────────────────
        # DiffusionPipeline.from_pretrained is the most generic loader.
        # It reads the model's model_index.json to figure out which pipeline
        # subclass to instantiate (SD15, SDXL, etc.) automatically.
        pipeline = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            cache_dir=str(MODELS_CACHE_DIR),
            token=HF_TOKEN or None,
        )

    # Move all model weights to the target device (GPU or CPU).
    # This is what actually allocates VRAM / RAM for the tensors.
    pipeline = pipeline.to(_device)

    # Attention slicing splits the self-attention computation into smaller chunks
    # processed sequentially. Slightly slower per step, but can cut peak VRAM by ~30%.
    # Only useful on GPU; on CPU memory pressure is less critical.
    if _device == "cuda":
        pipeline.enable_attention_slicing()

    return pipeline


# ─── CivitAI pipeline loading ──────────────────────────────────────────────

def _download_civitai_model(model_version_id: str) -> Path:
    """Download (or reuse cached) a CivitAI model checkpoint.

    CivitAI model format vs HuggingFace format:
      HuggingFace models are multi-file repositories (weights split across many shards,
      plus config JSONs, tokenizers, etc.).
      CivitAI models are typically a SINGLE .safetensors file containing everything.
      from_pretrained() won't work for these — we use from_single_file() instead.

    Streaming download:
      Model files can be 2–8+ GB. Streaming with iter_content() writes chunks
      directly to disk without buffering the whole file in RAM first.
    """

    MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Sanitise and convert the version ID to an integer
    normalized_model_version_id = _normalize_civitai_model_version_id(model_version_id)
    checkpoint_path = _resolve_civitai_checkpoint_path(normalized_model_version_id)

    # If we already downloaded this version, skip the download entirely
    if checkpoint_path.exists():
        logger.info("CivitAI model already cached at %s", checkpoint_path)
        return checkpoint_path

    # Build the download URL and auth headers
    url = f"https://civitai.com/api/download/models/{normalized_model_version_id}"
    headers: dict[str, str] = {}
    if CIV_TOKEN:
        # ****** authentication as required by CivitAI API
        headers["Authorization"] = "Bearer " + CIV_TOKEN

    logger.info("Downloading CivitAI model version %s …", model_version_id)
    # stream=True tells requests NOT to download the full body into memory yet;
    # we consume it in chunks below.
    # timeout=600 = allow up to 10 minutes for large downloads.
    response = requests.get(url, headers=headers, stream=True, timeout=600)
    if response.status_code != 200:
        raise RuntimeError(
            f"CivitAI download failed with status {response.status_code}: {response.text[:200]}"
        )

    # Write the file in 1MB chunks to keep RAM usage constant regardless of file size
    with open(checkpoint_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)

    logger.info("CivitAI model downloaded to %s", checkpoint_path)
    return checkpoint_path


def _detect_pipeline_class(checkpoint_path: Path) -> type[DiffusionPipeline]:
    """Inspect a .safetensors checkpoint to determine SD family (1.5 vs XL).

    Why do we peek at tensor names instead of just trying to load?
      If we guess wrong (e.g. try to load SDXL with SD15 pipeline), the model
      loads partially and then fails mid-way with cryptic shape errors.
      Inspecting the key names first is cheap and gives us a definitive answer.

    How it works:
      .safetensors has a header section with all tensor names; we can read just
      the names (metadata) without loading the actual weight data.
      SDXL introduced a "conditioner" module (dual text encoders combined).
      SD 1.5 has "cond_stage_model" instead. So "conditioner" in keys → SDXL.
    """

    keys: list[str] = []

    try:
        # safe_open = read-only access to a safetensors file.
        # framework="pt" → PyTorch tensors; device="cpu" → don't allocate VRAM just to peek.
        from safetensors import safe_open

        with safe_open(str(checkpoint_path), framework="pt", device="cpu") as f:
            keys = list(f.keys())   # Only reads the metadata header, not the weights
    except Exception:
        # If inspection fails for any reason, default to SD 1.5 (the safer choice)
        logger.warning("Could not inspect checkpoint keys safely; defaulting to SD 1.x pipeline")
        return StableDiffusionPipeline

    # "conditioner" is present in all SDXL checkpoint key paths
    if any("conditioner" in key for key in keys):
        return StableDiffusionXLPipeline

    return StableDiffusionPipeline


def _load_civitai_pipeline(model_version_id: str, task: GenerationTask) -> DiffusionPipeline:
    """Load a CivitAI single-file checkpoint into the appropriate pipeline class.

    CivitAI + ControlNet is not supported because ControlNet requires loading
    two separate checkpoints (base + ControlNet) in a very specific way that
    CivitAI single-file format doesn't accommodate cleanly.
    """

    if task == "sketch2ink":
        raise RuntimeError(
            "Sketch to ink currently supports HuggingFace SD 1.5 and SDXL base models only."
        )

    # Download (or load from cache) the .safetensors file
    checkpoint_path = _download_civitai_model(model_version_id)

    dtype = torch.float16 if _device == "cuda" else torch.float32
    logger.info("Loading CivitAI checkpoint from %s", checkpoint_path)

    # Peek at tensor names to decide SD15 vs SDXL pipeline class
    pipeline_class = _detect_pipeline_class(checkpoint_path)

    # If the task is img2img, swap to the matching img2img subclass.
    # StableDiffusionXLPipeline → StableDiffusionXLImg2ImgPipeline
    # StableDiffusionPipeline   → StableDiffusionImg2ImgPipeline
    if task == "img2img":
        if pipeline_class.__name__ == "StableDiffusionXLPipeline":
            pipeline_class = StableDiffusionXLImg2ImgPipeline
        else:
            pipeline_class = StableDiffusionImg2ImgPipeline

    logger.info("Auto-detected pipeline class: %s", pipeline_class.__name__)

    kwargs: dict = {"torch_dtype": dtype}

    # SD 1.x includes an optional NSFW safety checker that adds latency and is
    # generally not wanted for creative/art workflows. Disable it explicitly.
    # SDXL doesn't have a safety checker so we skip this for SDXL classes.
    if pipeline_class not in {StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline}:
        kwargs["safety_checker"] = None
        kwargs["requires_safety_checker"] = False

    # from_single_file = load all model components from a single .safetensors
    # (as opposed to from_pretrained which reads from a multi-file repo directory)
    pipeline = pipeline_class.from_single_file(str(checkpoint_path), **kwargs)
    pipeline = pipeline.to(_device)

    if _device == "cuda":
        pipeline.enable_attention_slicing()

    return pipeline


# ─── Main entry point: model loading orchestrator ──────────────────────────

def ensure_model(model_id: str, model_source: ModelSource, task: GenerationTask = "text2img") -> None:
    """Load and cache the requested model pipeline if it is not already active.

    This implements a "lazy singleton" pattern:
      - If the exact same (task, source, model_id) combination is already loaded → no-op.
      - If a different model (or same model for a different task) is requested →
        replace the old pipeline (freeing its VRAM) and load the new one.

    Why include task in the cache key?
      The same base model (e.g. "runwayml/stable-diffusion-v1-5") needs a *different*
      pipeline object for text2img vs img2img vs sketch2ink.
      Without task in the key, switching tasks would silently use the wrong pipeline type.

    Side effects:
      Sets the module-level _pipeline and _loaded_model_id variables.
      The old pipeline goes out of scope and is garbage-collected (freeing GPU memory).
    """

    global _pipeline, _loaded_model_id

    # Composite cache key: "task:source:model_id"
    # Example: "text2img:huggingface:stabilityai/stable-diffusion-2-1"
    cache_key = f"{task}:{model_source}:{model_id}"

    if _loaded_model_id == cache_key and _pipeline is not None:
        # Cache hit — same model+task already warm, nothing to do
        logger.info("Model already loaded: %s (skipping reload)", cache_key)
        return

    # If a different model was loaded, log the swap so it's visible in the container
    if _loaded_model_id is not None:
        logger.info("Swapping model: %s → %s", _loaded_model_id, cache_key)
    else:
        logger.info("Loading model: %s", cache_key)

    # Route to the correct loader based on where the model comes from
    if model_source == "huggingface":
        _pipeline = _load_huggingface_pipeline(model_id, task)
    else:
        _pipeline = _load_civitai_pipeline(model_id, task)

    _loaded_model_id = cache_key
    logger.info("Model ready: %s", cache_key)
