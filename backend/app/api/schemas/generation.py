"""
Generation-related request/response schemas.

Defines the contracts for all generation endpoints (text-to-image,
image-to-image, video, captioning, LLM). These schemas are the
stable public API — internal implementation may change freely.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class TextToImageRequest(BaseModel):
    """Request body for text-to-image generation."""

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

    prompt: str = Field(..., min_length=1, description="The text prompt guiding transformation")
    negative_prompt: str = Field("", description="What to avoid in the generated image")
    model_id: str = Field(..., description="ID of the model to use")
    strength: float = Field(0.75, ge=0.0, le=1.0, description="How much to transform the input")
    num_inference_steps: int = Field(20, ge=1, le=150, description="Denoising steps")
    guidance_scale: float = Field(7.5, ge=1.0, le=30.0, description="CFG scale")
    seed: int | None = Field(None, description="Random seed for reproducibility")
    num_images: int = Field(1, ge=1, le=8, description="Number of images to generate")


class VideoGenerationRequest(BaseModel):
    """Request body for video generation."""

    prompt: str = Field(..., min_length=1, description="Text prompt for video content")
    negative_prompt: str = Field("", description="What to avoid in the generated video")
    model_id: str = Field(..., description="ID of the video model to use")
    width: int = Field(512, ge=64, le=2048, multiple_of=8, description="Frame width in pixels")
    height: int = Field(512, ge=64, le=2048, multiple_of=8, description="Frame height in pixels")
    num_frames: int = Field(16, ge=1, le=120, description="Number of video frames")
    num_inference_steps: int = Field(20, ge=1, le=150, description="Denoising steps")
    guidance_scale: float = Field(7.5, ge=1.0, le=30.0, description="CFG scale")
    seed: int | None = Field(None, description="Random seed for reproducibility")


class VideoCaptioningRequest(BaseModel):
    """Request body for image/video captioning (vision tasks)."""

    model_id: str = Field(..., description="ID of the vision model to use")
    prompt: str = Field("", description="Optional prompt to guide captioning")


class LLMChatRequest(BaseModel):
    """Request body for local LLM inference."""

    model_id: str = Field(..., description="ID of the LLM model to use")
    messages: list[dict[str, str]] = Field(
        ..., min_length=1, description="Chat messages in OpenAI-compatible format"
    )
    max_tokens: int = Field(512, ge=1, le=8192, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")


class JobResponse(BaseModel):
    """Response returned when a job is successfully enqueued."""

    job_id: UUID
    status: str = "pending"
    message: str = "Job submitted"
    correlation_id: str | None = None


class JobStatusResponse(BaseModel):
    """Detailed job status response with progress information."""

    id: UUID
    job_type: str
    status: str
    progress_percent: int = 0
    current_step: int = 0
    total_steps: int = 0
    message: str = ""
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
