"""
Model registry request/response schemas.

Defines the contracts for model catalog endpoints — registering,
listing, downloading, and managing AI models from various sources.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelRegisterRequest(BaseModel):
    """Request to register a new model in the catalog."""

    model_config = ConfigDict(protected_namespaces=())

    model_id: str = Field(
        ..., min_length=1, description="Unique model identifier (e.g., HF repo ID)"
    )
    name: str = Field(..., min_length=1, description="Human-readable display name")
    source: str = Field(..., description="Origin: huggingface, civitai, github, or local")
    family: str = Field("custom", description="Architecture family: sd15, sdxl, flux, custom")
    variant: str = Field("", description="Model variant: fp16, fp32, gguf-q4, etc.")
    model_type: str = Field(
        "base_diffusion",
        description=(
            "Coarse category for FE filtering: base_diffusion | lora | controlnet | "
            "t2i_adapter | ip_adapter | vae | upscaler | face_restore | vision_language"
        ),
    )
    compatible_bases: list[str] = Field(
        default_factory=list,
        description=(
            "Base model families this model requires, e.g. ['sdxl'] or ['sd1.5','sdxl']. "
            "Empty for base_diffusion models."
        ),
    )
    description: str = Field("", description="Full description of the model")
    short_description: str = Field("", description="One-line summary for compact UI contexts (max 200 chars)")
    tags: list[str] = Field(default_factory=list, description="Searchable tags")
    source_url: str = Field("", description="URL where the model can be downloaded from")
    preferred_name: str = Field("", description="Display name override for frontend")
    capabilities: list[str] = Field(
        default_factory=list,
        description="What the model can do: text_to_image, image_to_image, etc.",
    )


class ModelDownloadRequest(BaseModel):
    """Request to start downloading a registered model."""

    model_config = ConfigDict(protected_namespaces=())

    force: bool = Field(False, description="Re-download even if already present")
    variant: str = Field("", description="Specific variant to download (e.g., fp16)")


class ModelRegistryResponse(BaseModel):
    """Full model information as returned by the API."""

    model_config = ConfigDict(protected_namespaces=())

    id: UUID
    model_id: str
    name: str
    preferred_name: str | None = None
    source: str
    family: str
    variant: str = ""
    model_type: str = "base_diffusion"
    compatible_bases: list[str] = Field(default_factory=list)
    description: str
    short_description: str = ""
    tags: list[str]
    source_url: str
    capabilities: list[str] = []
    status: str
    total_size_bytes: int = 0
    download_size_bytes: int | None = None
    last_error: str | None = None
    recommended_vram_min_gb: int | None = None
    recommended_vram_max_gb: int | None = None
    license: str = ""
    base_model: str = ""
    precision: str = ""
    requirements: dict[str, object] = Field(default_factory=dict)
    notes: str = ""
    is_verified: bool = False
    last_verified_at: datetime | None = None
    created_at: datetime
    updated_at: datetime | None = None