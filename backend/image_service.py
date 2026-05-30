"""
Image preprocessing and response-serialization helpers.

This file handles everything between "raw image data" and "what the AI/frontend needs":
 - Reading uploads → PIL images
 - Resizing to model-safe dimensions (multiples of 8, bounded)
 - Choosing generation settings based on workflow presets
 - Converting generated PIL images → base64 data URLs for the frontend gallery

Why a separate file?
  main.py (routes) calls these helpers so it stays focused on HTTP logic,
  not image manipulation math. Single-responsibility principle.
"""

from __future__ import annotations

import io
import time
import uuid
from base64 import b64encode
from typing import Optional

from fastapi import HTTPException, UploadFile
from PIL import Image

from schemas import GeneratedImage, ImageWorkflowPreset

# ─── Workflow preset defaults ──────────────────────────────────────────────
#
# img2img has three key parameters. These presets give sensible starting values
# for different use-cases. The user can always override any of them explicitly.
#
# strength (0.0 – 1.0):
#   How much the AI is allowed to change the input image.
#   0.0 = output is identical to input (no diffusion at all)
#   1.0 = input image is completely ignored (like pure text-to-image)
#   Practical sweet-spot: 0.3 (subtle) to 0.8 (heavy transformation)
#
# num_inference_steps:
#   How many denoising iterations the diffusion model runs.
#   More steps → sharper, more refined output, but slower.
#   For img2img the effective steps is actually (steps × strength) because
#   we start partway through the denoising schedule, not from pure noise.
#
# guidance_scale (CFG scale):
#   How strictly the model follows the text prompt.
#   Too low → prompt is barely respected; too high → artifacts / over-sharpening.
IMAGE_WORKFLOW_DEFAULTS: dict[ImageWorkflowPreset, dict[str, float | int]] = {
    # General purpose: balanced transform that respects the original image
    "general":        {"strength": 0.60, "num_inference_steps": 20, "guidance_scale": 7.5},
    # Recolor: low strength so composition/structure is preserved, only palette changes
    "recolor":        {"strength": 0.45, "num_inference_steps": 24, "guidance_scale": 7.0},
    # Style transfer: higher strength so artistic style can override the original look
    "style-transfer": {"strength": 0.72, "num_inference_steps": 30, "guidance_scale": 8.0},
    # Upscale / refine: very low strength — we just want texture/detail added, not re-composition
    "upscale":        {"strength": 0.30, "num_inference_steps": 18, "guidance_scale": 6.5},
}


# ─── Size helpers ──────────────────────────────────────────────────────────

def normalize_size(value: int) -> int:
    """Clamp to model-safe bounds and align to 8px.

    Why multiples of 8?
      Stable Diffusion encodes images into a "latent space" that is 8× smaller
      than the pixel space in each dimension (the VAE downsampling factor is 8).
      If the pixel dimensions aren't divisible by 8, the tensor reshape inside
      the VAE encoder fails with a shape mismatch error.

    Why 64–2048 bounds?
      Below 64px the model has no useful context to work with.
      Above 2048px VRAM consumption becomes impractical on consumer hardware.
    """

    bounded = max(64, min(2048, value))     # Clamp to safe range
    return (bounded // 8) * 8              # Floor-align to nearest 8px boundary


# ─── Seed helper ──────────────────────────────────────────────────────────

def resolve_seed(seed: Optional[int]) -> int:
    """Return provided seed or derive a 32-bit timestamp-based seed.

    How seeds work in diffusion:
      The generation process starts from a tensor of random Gaussian noise.
      The seed initialises the RNG that creates that noise tensor.
      Same seed + same parameters = same noise tensor = same output image.
      This lets users "save" a result and reproduce it exactly later.

    Why timestamp % 2**32?
      torch.Generator.manual_seed() accepts 32-bit integers.
      time.time() returns a float like 1716211200.123456;
      the modulo brings it into the valid 32-bit unsigned range.
    """

    return seed if seed is not None else int(time.time()) % (2**32)


# ─── Image serialisation ──────────────────────────────────────────────────

def serialize_images(
    output_images: list[Image.Image],
    prompt: str,
    negative_prompt: Optional[str],
    model_id: str,
    width: int,
    height: int,
    seed: int,
) -> list[GeneratedImage]:
    """Convert PIL images into gallery-ready base64 payloads for the frontend.

    Why base64 data URLs instead of saving files?
      Saving to disk requires managing file paths, cleanup, and a static-file
      server route. Embedding the image as a data URL ("data:image/png;base64,...")
      lets the frontend display it straight from the JSON response with no extra
      round-trips and no filesystem state to manage.

    Trade-off: base64 is ~33% larger than raw bytes. Fine for a dev/lab tool;
    for production you'd want a blob store + pre-signed URLs instead.
    """

    # One ISO-8601 timestamp shared by all images in this batch
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    images: list[GeneratedImage] = []
    for pil_image in output_images:
        # Write the PIL image into an in-memory byte buffer as PNG.
        # We never touch the filesystem — BytesIO acts like an open file in RAM.
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")

        # b64encode returns bytes; .decode() converts to a plain Python str.
        # Prepend the data URL prefix so browsers / <img> tags recognise the format.
        data_url = "data:image/png;base64," + b64encode(buf.getvalue()).decode()

        images.append(
            GeneratedImage(
                id=str(uuid.uuid4()),   # Random UUID — unique key for the frontend store
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


# ─── Upload helpers ───────────────────────────────────────────────────────

async def read_uploaded_image(uploaded_file: UploadFile) -> Image.Image:
    """Read a multipart-uploaded file and decode it into a PIL Image in RGB mode.

    Why convert to RGB?
      - Diffusers pipelines expect 3-channel (R, G, B) tensors.
      - PNG files can be RGBA (4 channels — adds transparency).
      - Grayscale images are 1 channel.
      - .convert("RGB") normalises all of these to exactly 3 channels.
      If we skip this step we'd get tensor shape errors inside the model.
    """

    # content_type is set by the browser when uploading, e.g. "image/jpeg"
    if not uploaded_file.content_type or not uploaded_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    try:
        # Read all file bytes into memory, then decode as image.
        # BytesIO wraps the bytes so PIL can treat them like a regular file.
        image_bytes = await uploaded_file.read()
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image upload") from exc


def prepare_input_image(
    input_image: Image.Image,
    width: Optional[int],
    height: Optional[int],
) -> tuple[Image.Image, int, int]:
    """Resize uploaded image to model-safe dimensions and return (image, w, h).

    Decision logic:
      1. If the caller provided explicit width/height → use those (clamped + aligned).
      2. Otherwise keep the image's original dimensions (clamped + aligned).
    The returned width/height are the values actually used (echoed back in the response
    so the frontend knows the final rendered size).
    """

    # Use the requested size, or fall back to the image's own dimensions
    target_width = normalize_size(width if width is not None else input_image.width)
    target_height = normalize_size(height if height is not None else input_image.height)

    # Only resize if the dimensions actually changed (avoids unnecessary pixel resampling)
    if input_image.width != target_width or input_image.height != target_height:
        input_image = input_image.resize((target_width, target_height))

    return input_image, target_width, target_height


# ─── img2img settings resolver ────────────────────────────────────────────

def resolve_img2img_settings(
    workflow_preset: ImageWorkflowPreset,
    strength: Optional[float],
    num_inference_steps: Optional[int],
    guidance_scale: Optional[float],
) -> tuple[float, int, float]:
    """Resolve effective img2img settings: preset defaults + any explicit overrides.

    This implements a "defaults with overrides" pattern:
      - Start from the preset (e.g. "recolor" has low strength by default).
      - Apply any non-None user-provided values on top.
      - User values always win over preset defaults.

    Why this approach?
      It lets the frontend offer one-click presets ("recolor", "style-transfer")
      while still letting power users fine-tune individual sliders without
      having to specify every parameter every time.
    """

    defaults = IMAGE_WORKFLOW_DEFAULTS[workflow_preset]

    # Each line: use the explicit value if provided, otherwise fall back to preset default.
    # Explicit float() / int() casts ensure correct types even when defaults are stored as floats.
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
