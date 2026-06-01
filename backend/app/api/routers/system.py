"""
System router — health check, status, and diagnostics.
"""

from __future__ import annotations

import torch
from fastapi import APIRouter

from app.api.schemas import SystemStatusResponse
from app.infrastructure.config.settings import get_settings

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status() -> SystemStatusResponse:
    """
    Return current system status.

    Used by the frontend health panel to show connection state,
    GPU availability, and active model info.
    """
    settings = get_settings()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return SystemStatusResponse(
        status="ok",
        version=settings.app_version,
        device=device,
        gpu_busy=False,  # TODO: Wire to ResourceCoordinator
        loaded_models=[],  # TODO: Wire to ModelManager
        pending_jobs=0,  # TODO: Wire to JobRepository count
    )
