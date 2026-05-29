"""Image preprocessing and response-serialization helpers."""

from __future__ import annotations

import io
import time
import uuid
from base64 import b64encode
from typing import Optional

from fastapi import HTTPException, UploadFile
from PIL import Image

from schemas import GeneratedImage, ImageWorkflowPreset

IMAGE_WORKFLOW_DEFAULTS: dict[ImageWorkflowPreset, dict[str, float | int]] = {
    "general": {"strength": 0.6, "num_inference_steps": 20, "guidance_scale": 7.5},
    "recolor": {"strength": 0.45, "num_inference_steps": 24, "guidance_scale": 7.0},
    "style-transfer": {"strength": 0.72, "num_inference_steps": 30, "guidance_scale": 8.0},
    "upscale": {"strength": 0.3, "num_inference_steps": 18, "guidance_scale": 6.5},
}


def normalize_size(value: int) -> int:
    """Clamp to model-safe bounds and align to 8px as required by SD latent scaling."""

    bounded = max(64, min(2048, value))
    return (bounded // 8) * 8


def resolve_seed(seed: Optional[int]) -> int:
    """Return provided seed or derive a 32-bit timestamp-based seed."""

    return seed if seed is not None else int(time.time()) % (2**32)


def serialize_images(
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


async def read_uploaded_image(uploaded_file: UploadFile) -> Image.Image:
    """Read an uploaded file and convert it to RGB for Diffusers image conditioning."""

    if not uploaded_file.content_type or not uploaded_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    try:
        image_bytes = await uploaded_file.read()
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image upload") from exc


def prepare_input_image(
    input_image: Image.Image,
    width: Optional[int],
    height: Optional[int],
) -> tuple[Image.Image, int, int]:
    """Resize uploaded image to model-safe dimensions while preserving request intent."""

    target_width = normalize_size(width if width is not None else input_image.width)
    target_height = normalize_size(height if height is not None else input_image.height)

    if input_image.width != target_width or input_image.height != target_height:
        input_image = input_image.resize((target_width, target_height))
    return input_image, target_width, target_height


def resolve_img2img_settings(
    workflow_preset: ImageWorkflowPreset,
    strength: Optional[float],
    num_inference_steps: Optional[int],
    guidance_scale: Optional[float],
) -> tuple[float, int, float]:
    """Resolve effective img2img settings using preset defaults + explicit overrides."""

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
