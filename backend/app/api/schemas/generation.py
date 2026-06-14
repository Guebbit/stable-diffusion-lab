"""Generation request/response schemas."""
from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class TextToImageRequest(BaseModel):
    """Request body for text-to-image generation."""
    model_config = ConfigDict(protected_namespaces=())
    prompt: str = Field(..., min_length=1, description="The text prompt to generate from")
    negative_prompt: str = Field("", description="What to avoid in the generated image")
    model_id: str = Field(..., description="ID of the model to use for generation")
    width: int = Field(512, ge=64, le=2048, multiple_of=8, description="Output width in pixels")
    height: int = Field(512, ge=64, le=2048, multiple_of=8, description="Output height in pixels")
    num_inference_steps: int = Field(20, ge=1, le=150, description="Denoising steps")
    guidance_scale: float = Field(7.5, ge=1.0, le=30.0, description="CFG scale")
    seed: int | None = Field(None, description="Random seed for reproducibility")
    num_images: int = Field(1, ge=1, le=8, description="Number of images to generate")


class ImageToImageRequest(BaseModel):
    """Request body for image-to-image generation."""
    model_config = ConfigDict(protected_namespaces=())
    prompt: str = Field(..., min_length=1, description="The text prompt guiding transformation")
    negative_prompt: str = Field("", description="What to avoid in the generated image")
    model_id: str = Field(..., description="ID of the model to use")
    strength: float = Field(0.75, ge=0.0, le=1.0, description="How much to transform the input")
    num_inference_steps: int = Field(20, ge=1, le=150, description="Denoising steps")
    guidance_scale: float = Field(7.5, ge=1.0, le=30.0, description="CFG scale")
    seed: int | None = Field(None, description="Random seed for reproducibility")
    num_images: int = Field(1, ge=1, le=8, description="Number of images to generate")


class DescribeRequest(BaseModel):
    """Request body for image description/captioning."""
    model_config = ConfigDict(protected_namespaces=())
    model_id: str = Field(..., description="ID of the vision model to use for captioning")


class UpscaleRequest(BaseModel):
    """Request body for image upscaling."""
    model_config = ConfigDict(protected_namespaces=())
    model_id: str = Field(..., description="ID of the upscale model to use")
    scale_factor: float = Field(2.0, ge=1.0, le=4.0, description="Upscale factor")


class RecolorRequest(BaseModel):
    """Request body for image recoloring."""
    model_config = ConfigDict(protected_namespaces=())
    prompt: str = Field(..., min_length=1, description="Color guidance prompt")
    model_id: str = Field(..., description="ID of the model to use for recoloring")
    strength: float = Field(0.75, ge=0.0, le=1.0, description="How much to recolor")


class SketchToInkRequest(BaseModel):
    """Request body for sketch-to-ink generation."""
    model_config = ConfigDict(protected_namespaces=())
    model_id: str = Field(..., description="ID of the model to use for sketch-to-ink")


class JobResponse(BaseModel):
    """Response returned when a job is successfully enqueued."""
    job_id: UUID
    status: str = "pending"
    message: str = "Job submitted"
    correlation_id: str | None = None
