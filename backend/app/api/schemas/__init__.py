"""
API schema package — Pydantic models for request/response contracts.

Each file corresponds to a route group. Schemas define the public API surface
and are decoupled from internal domain models and ORM entities.
"""

from app.api.schemas.artifacts import (
    ArtifactListQuery,
    ArtifactResponse,
    ArtifactUpdateRequest,
)
from app.api.schemas.common import ErrorResponse, PaginatedResponse
from app.api.schemas.generation import (
    ImageToImageRequest,
    JobResponse,
    JobStatusResponse,
    LLMChatRequest,
    TextToImageRequest,
    VideoCaptioningRequest,
    VideoGenerationRequest,
)
from app.api.schemas.jobs import (
    JobCancelResponse,
    JobDetailResponse,
    JobEventResponse,
    JobListQuery,
)
from app.api.schemas.models import (
    ModelDownloadRequest,
    ModelRegistryResponse,
    ModelRegisterRequest,
)
from app.api.schemas.system import (
    GPUStatus,
    HealthResponse,
    JobQueueStatus,
    JobTimelineResponse,
    MetricsSnapshot,
    ModelStatus,
    RuntimeStatus,
    SystemStateSnapshot,
    TypedEventResponse,
)

__all__ = [
    "ArtifactListQuery",
    "ArtifactResponse",
    "ArtifactUpdateRequest",
    "ErrorResponse",
    "ImageToImageRequest",
    "JobCancelResponse",
    "JobDetailResponse",
    "JobEventResponse",
    "JobListQuery",
    "JobResponse",
    "JobStatusResponse",
    "LLMChatRequest",
    "ModelDownloadRequest",
    "ModelRegistryResponse",
    "ModelRegisterRequest",
    "PaginatedResponse",
    "GPUStatus",
    "HealthResponse",
    "JobQueueStatus",
    "JobTimelineResponse",
    "MetricsSnapshot",
    "ModelStatus",
    "RuntimeStatus",
    "SystemStateSnapshot",
    "TextToImageRequest",
    "TypedEventResponse",
    "VideoCaptioningRequest",
    "VideoGenerationRequest",
]
