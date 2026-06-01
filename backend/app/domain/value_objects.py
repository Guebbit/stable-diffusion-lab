"""
Domain value objects — immutable data containers shared across layers.

Value objects carry data but have no identity of their own. They are
compared by value (all fields match), not by reference. Use these to
pass structured data between layers without coupling to ORM models or
Pydantic request/response schemas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID


@dataclass(frozen=True, slots=True)
class GenerationParams:
    """Parameters for any image/video generation job."""

    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    seed: int | None = None
    num_images: int = 1
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelIdentifier:
    """Uniquely identifies a model regardless of source."""

    model_id: str
    source: str
    family: str = ""
    variant: str = ""


@dataclass(frozen=True, slots=True)
class JobProgress:
    """Snapshot of a job's progress, emitted to WebSocket clients."""

    job_id: UUID
    status: str
    progress_percent: int = 0
    current_step: int = 0
    total_steps: int = 0
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """Pointer to a generated artifact on disk."""

    artifact_id: UUID
    job_id: UUID
    file_path: str
    thumbnail_path: str = ""
    media_type: str = "image/png"
    width: int = 0
    height: int = 0
    size_bytes: int = 0


@dataclass(frozen=True, slots=True)
class DownloadProgress:
    """Tracks progress of a model download."""

    model_id: str
    downloaded_bytes: int = 0
    total_bytes: int | None = None
    percentage: int = 0
    speed_bytes_per_sec: float = 0.0
    is_complete: bool = False
    error: str = ""
