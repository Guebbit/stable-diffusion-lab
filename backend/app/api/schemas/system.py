"""
System status and health-check response schemas.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RuntimeSnapshot(BaseModel):
    app_name: str
    version: str
    uptime_seconds: int = 0
    python_version: str
    inference_backend: str


class GpuSnapshot(BaseModel):
    available: bool = False
    device: str = "cpu"
    device_name: str = "cpu"
    busy: bool = False
    active_job_id: str | None = None
    estimated_vram_used_mb: int = 0
    vram_budget_mb: int = 0
    cuda_allocated_mb: float = 0
    cuda_reserved_mb: float = 0


class ModelCacheSnapshot(BaseModel):
    loaded_models: list[str] = Field(default_factory=list)
    active_model: str | None = None
    cache_size: int = 0
    max_cached: int = 0
    estimated_vram_usage_mb: int = 0


class JobQueueSnapshot(BaseModel):
    pending: int = 0
    running: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    active_job_id: str | None = None
    oldest_pending_age_seconds: int | None = None


class HealthSnapshot(BaseModel):
    status: str = "ok"
    warnings: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class SystemStatusResponse(BaseModel):
    """System health and resource information."""

    status: str = Field("ok", description="Overall system status")
    version: str = Field(..., description="Application version string")
    device: str = Field(..., description="Active inference device (cuda/cpu/mps)")
    gpu_busy: bool = Field(False, description="Whether GPU is currently occupied by a job")
    loaded_models: list[str] = Field(default_factory=list, description="Currently loaded model IDs")
    pending_jobs: int = Field(0, description="Number of jobs waiting in the queue")
    runtime: RuntimeSnapshot
    gpu: GpuSnapshot
    models: ModelCacheSnapshot
    jobs: JobQueueSnapshot
    health: HealthSnapshot


class SystemEventResponse(BaseModel):
    event_id: str
    event_type: str
    component: str
    level: str
    message: str
    timestamp: datetime
    job_id: str | None = None
    correlation_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class MetricPointResponse(BaseModel):
    name: str
    kind: str
    value: float
    unit: str = ""
    updated_at: datetime


class MetricsSnapshotResponse(BaseModel):
    counters: list[MetricPointResponse] = Field(default_factory=list)
    gauges: list[MetricPointResponse] = Field(default_factory=list)
    histograms: list[MetricPointResponse] = Field(default_factory=list)
