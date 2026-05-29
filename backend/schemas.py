"""Shared API schemas and generation type aliases."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

ModelSource = Literal["huggingface", "civitai"]
GenerationTask = Literal["text2img", "img2img", "sketch2ink"]
ModelFamily = Literal["sd15", "sdxl"]
ImageWorkflowPreset = Literal["general", "recolor", "style-transfer", "upscale"]


class ModelLoadRequest(BaseModel):
    """Payload used to load a model for a specific generation task."""

    model_id: str = Field(..., description="HuggingFace repo ID or CivitAI model version ID")
    model_source: ModelSource = Field("huggingface")
    task: GenerationTask = Field("text2img")


class ModelLoadResponse(BaseModel):
    """API response returned after a model load attempt."""

    success: bool
    model_id: str
    message: str


class GenerationRequest(BaseModel):
    """Text-to-image request schema."""

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
    """Single gallery image item returned to the frontend."""

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
    """Shared response format used by all generation workflows."""

    images: list[GeneratedImage]
    model_id: str
    elapsed_seconds: float


class BackendStatus(BaseModel):
    """Status payload used by the frontend health panel."""

    status: Literal["ok", "loading", "error"]
    loaded_model: Optional[str] = None
    device: str
    message: Optional[str] = None
