"""
Vision model service for image captioning/description.

This module loads and manages vision-language models (BLIP, BLIP-2, ViT-GPT2)
that take an image as input and produce a natural language description.

How image captioning works (conceptual overview):
  1. The image is fed through a vision encoder (e.g. ViT — Vision Transformer)
     which converts the pixel grid into a sequence of visual embeddings.
  2. These visual embeddings are passed to a language model (GPT-2, OPT, etc.)
     which generates a text description token by token.
  3. The result is a natural language sentence describing the image content.

BLIP vs BLIP-2:
  - BLIP (Bootstrapped Language-Image Pre-training): single-stage model,
    directly fine-tuned for captioning. Fast but less detailed.
  - BLIP-2: two-stage model with a "Q-Former" bridge between vision and language.
    More parameters, more detailed descriptions, but heavier.

Why only one vision model at a time?
  Same reason as the diffusion pipelines: VRAM is limited. Each vision model
  occupies 1-4 GB, so we keep exactly one loaded and swap when the user changes.
"""

from __future__ import annotations

import time
from typing import Optional

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BlipForConditionalGeneration,
    BlipProcessor,
)

from logging_config import get_logger

logger = get_logger(__name__)

# ─── Module-level singleton state ──────────────────────────────────────────
#
# Similar pattern to model_service.py: one model + processor at a time.
_vision_model: Optional[object] = None
_vision_processor: Optional[object] = None
_loaded_vision_model_id: Optional[str] = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _load_vision_model(model_id: str) -> None:
    """Load a vision captioning model and its processor into memory.

    Uses HuggingFace transformers' Auto classes to detect the right model type.
    Moves the model to GPU if available for faster inference.

    Supported model architectures:
      - BLIP (BlipForConditionalGeneration) — most common for captioning
      - BLIP-2 (Blip2ForConditionalGeneration) — larger, more detailed
      - ViT-GPT2 (VisionEncoderDecoderModel) — lightweight alternative

    The Auto classes (AutoProcessor, AutoModelForVision2Seq) handle detection
    automatically based on the model's config.json on HuggingFace Hub.
    """
    global _vision_model, _vision_processor, _loaded_vision_model_id

    # Skip reload if already loaded
    if _loaded_vision_model_id == model_id:
        return

    logger.info("Loading vision model: %s", model_id)

    # Free previous model from GPU memory before loading new one
    _vision_model = None
    _vision_processor = None
    _loaded_vision_model_id = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        # Try BLIP-specific loading first (most captioning models are BLIP-based)
        _vision_processor = BlipProcessor.from_pretrained(model_id)
        _vision_model = BlipForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
        ).to(_device)
    except Exception:
        # Fallback: use generic Auto classes for other architectures
        from transformers import AutoModelForVision2Seq

        _vision_processor = AutoProcessor.from_pretrained(model_id)
        _vision_model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
        ).to(_device)

    _loaded_vision_model_id = model_id
    logger.info("Vision model '%s' loaded on %s", model_id, _device)


def describe_image(image: Image.Image, model_id: str) -> tuple[str, float]:
    """Generate a text description of the provided image.

    Args:
        image: PIL Image in RGB mode (already preprocessed by the caller).
        model_id: HuggingFace model ID for the vision model to use.

    Returns:
        Tuple of (description_text, elapsed_seconds).

    Pipeline steps:
      1. Load model if not already cached.
      2. Preprocess image into model-ready tensors (resize, normalize, etc.).
      3. Run model.generate() to produce token IDs.
      4. Decode token IDs back into a human-readable string.
    """
    _load_vision_model(model_id)

    start = time.time()

    # Preprocess: resize + normalize image into the tensor format the model expects.
    # return_tensors="pt" gives us PyTorch tensors ready for the model.
    inputs = _vision_processor(images=image, return_tensors="pt").to(_device)  # type: ignore[union-attr]

    # Generate caption tokens — max_new_tokens caps output length to avoid runaway generation.
    # The model autoregressively predicts one token at a time until it outputs <EOS> or hits max.
    with torch.no_grad():
        output_ids = _vision_model.generate(**inputs, max_new_tokens=150)  # type: ignore[union-attr]

    # Decode the integer token IDs back to readable text, stripping special tokens like <PAD>
    description = _vision_processor.decode(output_ids[0], skip_special_tokens=True)  # type: ignore[union-attr]

    elapsed = time.time() - start
    logger.info("Image described in %.2fs using '%s'", elapsed, model_id)

    return description.strip(), elapsed
