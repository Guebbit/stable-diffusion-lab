"""
Model registry request/response schemas.

Defines the contracts for model catalog endpoints — registering,
listing, downloading, and managing AI models from various sources.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ModelRegisterRequest(BaseModel):
    """Request to register a new model in the catalog."""

    model_id: str = Field(..., min_length=1, description="Unique model identifier (e.g., HF repo ID)")
    name: str = Field(..., min_length=1, description="Human-readable display name")
    source: str = Field(..., description="Origin: huggingface, civitai, or local")
    family: str = Field("custom", description="Architecture family: sd15, sdxl, flux, custom")
    description: str = Field("", description="Brief description of the model")
    tags: list[str] = Field(default_factory=list, description="Searchable tags")
    source_url: str = Field("", description="URL where the model can be downloaded from")


class ModelDownloadRequest(BaseModel):
    """Request to start downloading a registered model."""

    force: bool = Field(False, description="Re-download even if already present")


class ModelRegistryResponse(BaseModel):
    """Full model information as returned by the API."""

    id: UUID
    model_id: str
    name: str
    source: str
    family: str
    description: str
    tags: list[str]
    source_url: str
    status: str
    size_bytes: int | None = None
    download_progress: int = 0
    is_verified: bool = False
    created_at: datetime
    updated_at: datetime | None = None
