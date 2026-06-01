"""
System status and health-check response schemas.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SystemStatusResponse(BaseModel):
    """System health and resource information."""

    status: str = Field("ok", description="Overall system status")
    version: str = Field(..., description="Application version string")
    device: str = Field(..., description="Active inference device (cuda/cpu/mps)")
    gpu_busy: bool = Field(False, description="Whether GPU is currently occupied by a job")
    loaded_models: list[str] = Field(default_factory=list, description="Currently loaded model IDs")
    pending_jobs: int = Field(0, description="Number of jobs waiting in the queue")
