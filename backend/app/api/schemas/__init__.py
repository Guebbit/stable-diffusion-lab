"""
API schema package — Pydantic models for request/response contracts.

Each file corresponds to a route group. Schemas define the public API surface
and are decoupled from internal domain models and ORM entities.
"""

from app.api.schemas.generation import (
    ImageToImageRequest,
    JobResponse,
    JobStatusResponse,
    TextToImageRequest,
    VideoCaptioningRequest,
    VideoGenerationRequest,
)
from app.api.schemas.models import (
    ModelDownloadRequest,
    ModelRegistryResponse,
    ModelRegisterRequest,
)
from app.api.schemas.system import SystemStatusResponse

__all__ = [
    "ImageToImageRequest",
    "JobResponse",
    "JobStatusResponse",
    "ModelDownloadRequest",
    "ModelRegistryResponse",
    "ModelRegisterRequest",
    "SystemStatusResponse",
    "TextToImageRequest",
    "VideoCaptioningRequest",
    "VideoGenerationRequest",
]
