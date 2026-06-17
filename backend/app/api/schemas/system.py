"""System observability response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RuntimeStatus(BaseModel):
    """Runtime status details."""

    version: str
    uptime_seconds: float
    python_version: str
    torch_version: str | None = None
    cuda_available: bool


class GPUStatus(BaseModel):
    """GPU/resource state."""

    busy: bool
    current_job_id: str | None = None
    device_name: str
    estimated_vram_used_mb: int
    vram_budget_mb: int
    cuda_allocated_mb: float
    cuda_reserved_mb: float
    last_activity: datetime | None = None


class ModelStatus(BaseModel):
    """Model cache and load state."""

    loaded_models: list[str] = Field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    warm: bool = False
    cache_size: int = 0
    last_load_at: datetime | None = None
    last_unload_at: datetime | None = None
    last_load_failed: bool = False


class JobQueueStatus(BaseModel):
    """Job queue counters/state."""

    pending: int = 0
    running: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    queue_depth: int = 0
    oldest_pending_age_seconds: float | None = None
    current_job_id: str | None = None


class ArtifactStatus(BaseModel):
    """Artifact storage status."""

    recent_saves_count: int = 0
    storage_health: str = "ok"
    disk_free_mb: float = 0.0


class StatusResponse(BaseModel):
    """System status with model statistics."""

    status: str = "ok"
    device: str = "cpu"
    total_models: int = 0
    models_by_family: dict[str, int] = Field(default_factory=dict)
    models_disk_bytes: int = 0
    disk_free_bytes: int = 0


class HealthResponse(BaseModel):
    """Health warnings and blockers."""

    status: str = "ok"
    warnings: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    correlation_id: str | None = None


class SystemStateSnapshot(BaseModel):
    """Comprehensive system state snapshot."""

    runtime: RuntimeStatus
    gpu: GPUStatus
    models: ModelStatus
    jobs: JobQueueStatus
    artifacts: ArtifactStatus
    health: HealthResponse
    correlation_id: str | None = None


class TypedEventResponse(BaseModel):
    """Typed observability event payload."""

    event_id: str
    event_type: str
    timestamp: datetime
    correlation_id: str | None = None
    component: str
    level: str
    job_id: str | None = None
    message: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class JobTimelineResponse(BaseModel):
    """Per-job event timeline."""

    job_id: str
    events: list[TypedEventResponse] = Field(default_factory=list)


class MetricsSnapshot(BaseModel):
    """In-memory metrics snapshot."""

    counters: dict[str, int] = Field(default_factory=dict)
    gauges: dict[str, float] = Field(default_factory=dict)
    histograms: dict[str, dict[str, float]] = Field(default_factory=dict)
