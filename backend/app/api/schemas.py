"""
API request/response schemas — Pydantic models for the HTTP boundary.

These define what the client sends and receives. They are separate from
domain value objects (which are internal) and ORM models (which are DB-specific).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


# --- Generation schemas ---


class TextToImageRequest(BaseModel):
    """Request body for POST /api/v1/generation/text-to-image."""

    prompt: str = Field(..., min_length=1, description="What to generate")
    negative_prompt: str = Field("", description="What to avoid")
    model_id: str = Field(..., description="Model identifier (HuggingFace repo or CivitAI ID)")
    width: int = Field(512, ge=64, le=2048, description="Output width in pixels")
    height: int = Field(512, ge=64, le=2048, description="Output height in pixels")
    num_inference_steps: int = Field(20, ge=1, le=150, description="Denoising steps")
    guidance_scale: float = Field(7.5, ge=1.0, le=30.0, description="CFG scale")
    seed: int | None = Field(None, description="Random seed (None = random)")
    num_images: int = Field(1, ge=1, le=4, description="Batch size")


class ImageToImageRequest(BaseModel):
    """Request body for POST /api/v1/generation/image-to-image."""

    prompt: str = Field(..., min_length=1)
    negative_prompt: str = ""
    model_id: str = Field(...)
    strength: float = Field(0.75, ge=0.0, le=1.0, description="Denoising strength")
    num_inference_steps: int = Field(20, ge=1, le=150)
    guidance_scale: float = Field(7.5, ge=1.0, le=30.0)
    seed: int | None = None
    num_images: int = Field(1, ge=1, le=4)


# --- Job schemas ---


class JobResponse(BaseModel):
    """Response returned when a job is submitted (202 Accepted)."""

    job_id: UUID
    status: str
    message: str = "Job submitted successfully"


class JobStatusResponse(BaseModel):
    """Full job status response."""

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
    error: str = ""
    result: dict[str, Any] = Field(default_factory=dict)


# --- Model schemas ---


class ModelRegistryResponse(BaseModel):
    """Single model entry returned by the catalog API."""

    id: UUID
    model_id: str
    name: str
    source: str
    family: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    source_url: str = ""
    status: str
    size_bytes: int = 0
    is_verified: bool = False
    created_at: datetime


class ModelRegisterRequest(BaseModel):
    """Request body for POST /api/v1/models."""

    model_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    source: str = Field(..., description="huggingface | civitai | local")
    family: str = Field("custom", description="sd15 | sdxl | flux | custom")
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    source_url: str = ""


# --- System schemas ---


class SystemStatusResponse(BaseModel):
    """Response for GET /api/v1/system/status."""

    status: str = "ok"
    version: str
    device: str
    gpu_busy: bool = False
    loaded_models: list[str] = Field(default_factory=list)
    pending_jobs: int = 0
