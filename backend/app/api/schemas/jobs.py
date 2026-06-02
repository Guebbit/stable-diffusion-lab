"""
Job-related request/response schemas.

Defines the contracts for job queue endpoints — listing, status,
cancellation, retry, and event history queries.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class JobListQuery(BaseModel):
    """Query parameters for listing jobs."""

    model_config = ConfigDict(protected_namespaces=())

    status: str | None = Field(
        None, description="Filter: pending, running, completed, failed, cancelled"
    )
    job_type: str | None = Field(
        None, description="Filter: text_to_image, model_download, etc."
    )
    model_id: str | None = Field(None, description="Filter by associated model UUID")
    limit: int = Field(50, ge=1, le=100, description="Maximum results per page")
    offset: int = Field(0, ge=0, description="Number of results to skip")


class JobDetailResponse(BaseModel):
    """Full job details including progress and timing."""

    id: UUID
    job_type: str
    status: str
    progress_percent: int = 0
    current_step: int = 0
    total_steps: int = 0
    message: str = ""
    priority: int = 0
    attempt: int = 1
    max_attempts: int = 1
    params: dict[str, Any] = {}
    result: dict[str, Any] = {}
    error: str = ""
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    timeout_at: datetime | None = None


class JobEventResponse(BaseModel):
    """A single job state transition event."""

    id: UUID
    job_id: UUID
    from_status: str
    to_status: str
    message: str = ""
    metadata: dict[str, Any] = {}
    created_at: datetime


class JobCancelResponse(BaseModel):
    """Response after requesting job cancellation."""

    job_id: UUID
    status: str
    message: str = "Cancellation requested"
