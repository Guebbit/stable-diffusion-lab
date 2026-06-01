"""
Artifact/gallery request/response schemas.

Defines the contracts for artifact endpoints — listing generated outputs,
serving files, and managing gallery metadata (favorites, ratings, notes).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ArtifactListQuery(BaseModel):
    """Query parameters for listing artifacts."""

    model_name: str | None = Field(None, description="Filter by model name")
    media_type: str | None = Field(None, description="Filter by media type (image/png, video/mp4)")
    is_favorite: bool | None = Field(None, description="Filter by favorite status")
    limit: int = Field(50, ge=1, le=100, description="Maximum results per page")
    offset: int = Field(0, ge=0, description="Number of results to skip")


class ArtifactResponse(BaseModel):
    """Full artifact information as returned by the API."""

    id: UUID
    job_id: UUID
    file_path: str
    thumbnail_path: str = ""
    media_type: str = "image/png"
    size_bytes: int = 0
    width: int = 0
    height: int = 0
    duration_seconds: float | None = None
    prompt: str = ""
    negative_prompt: str = ""
    seed: int = 0
    model_name: str = ""
    model_id_ref: str = ""
    generation_params: dict[str, Any] = {}
    is_favorite: bool = False
    rating: int = 0
    notes: str = ""
    created_at: datetime
    updated_at: datetime | None = None


class ArtifactUpdateRequest(BaseModel):
    """Partial update for artifact gallery metadata."""

    is_favorite: bool | None = Field(None, description="Mark/unmark as favorite")
    rating: int | None = Field(None, ge=0, le=5, description="Rating from 0-5 stars")
    notes: str | None = Field(None, description="User annotation/notes")
