"""
Artifacts router — endpoints for the generated output gallery.

Provides listing, metadata access, file serving, and gallery management
(favorites, ratings, notes) for all generated artifacts.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.artifacts import ArtifactResponse, ArtifactUpdateRequest
from app.api.schemas.common import PaginatedResponse
from app.infrastructure.database.repositories import ArtifactRepository
from app.infrastructure.database.session import get_async_session
from app.services.artifact_service import ArtifactService

router = APIRouter(prefix="/artifacts", tags=["artifacts"])


def _get_artifact_service(session: AsyncSession = Depends(get_async_session)) -> ArtifactService:
    """Dependency injection for ArtifactService."""
    return ArtifactService(artifact_repository=ArtifactRepository(session))


@router.get("/", response_model=PaginatedResponse[ArtifactResponse])
async def list_artifacts(
    model_name: str | None = Query(None, description="Filter by model name"),
    media_type: str | None = Query(None, description="Filter by media type"),
    is_favorite: bool | None = Query(None, description="Filter by favorite status"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    service: ArtifactService = Depends(_get_artifact_service),
) -> PaginatedResponse[ArtifactResponse]:
    """List artifacts with optional filtering and pagination."""
    artifacts, total = await service.list_artifacts(
        model_name=model_name,
        media_type=media_type,
        is_favorite=is_favorite,
        limit=limit,
        offset=offset,
    )
    items = [_to_response(a) for a in artifacts]
    return PaginatedResponse(
        items=items, total=total, limit=limit, offset=offset, has_more=(offset + limit < total)
    )


@router.get("/{artifact_id}", response_model=ArtifactResponse)
async def get_artifact(
    artifact_id: UUID,
    service: ArtifactService = Depends(_get_artifact_service),
) -> ArtifactResponse:
    """Get metadata for a specific artifact."""
    artifact = await service.get_artifact(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return _to_response(artifact)


@router.patch("/{artifact_id}", response_model=ArtifactResponse)
async def update_artifact(
    artifact_id: UUID,
    request: ArtifactUpdateRequest,
    service: ArtifactService = Depends(_get_artifact_service),
) -> ArtifactResponse:
    """Update gallery metadata (favorite, rating, notes) for an artifact."""
    artifact = await service.update_metadata(
        artifact_id=artifact_id,
        is_favorite=request.is_favorite,
        rating=request.rating,
        notes=request.notes,
    )
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return _to_response(artifact)


@router.get("/{artifact_id}/file")
async def get_artifact_file(
    artifact_id: UUID,
    service: ArtifactService = Depends(_get_artifact_service),
) -> FileResponse:
    """Serve the raw file for an artifact."""
    artifact = await service.get_artifact(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    file_path = Path(artifact.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Artifact file not found on disk")
    return FileResponse(str(file_path), media_type=artifact.media_type)


@router.delete("/{artifact_id}", status_code=204, response_model=None)
async def delete_artifact(
    artifact_id: UUID,
    service: ArtifactService = Depends(_get_artifact_service),
) -> None:
    """Delete an artifact (file + record)."""
    deleted = await service.delete_artifact(artifact_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Artifact not found")


def _to_response(a: object) -> ArtifactResponse:
    """Map an ArtifactRecord to its API response schema."""
    return ArtifactResponse(
        id=a.id,  # type: ignore[attr-defined]
        job_id=a.job_id,  # type: ignore[attr-defined]
        file_path=f"/api/v1/artifacts/{a.id}/file",  # type: ignore[attr-defined]
        thumbnail_path=a.thumbnail_path,  # type: ignore[attr-defined]
        media_type=a.media_type,  # type: ignore[attr-defined]
        size_bytes=a.size_bytes,  # type: ignore[attr-defined]
        width=a.width,  # type: ignore[attr-defined]
        height=a.height,  # type: ignore[attr-defined]
        duration_seconds=a.duration_seconds,  # type: ignore[attr-defined]
        prompt=a.prompt,  # type: ignore[attr-defined]
        negative_prompt=a.negative_prompt,  # type: ignore[attr-defined]
        seed=a.seed,  # type: ignore[attr-defined]
        model_name=a.model_name,  # type: ignore[attr-defined]
        model_id_ref=a.model_id_ref,  # type: ignore[attr-defined]
        generation_params=a.generation_params,  # type: ignore[attr-defined]
        is_favorite=a.is_favorite,  # type: ignore[attr-defined]
        rating=a.rating,  # type: ignore[attr-defined]
        notes=a.notes,  # type: ignore[attr-defined]
        created_at=a.created_at,  # type: ignore[attr-defined]
        updated_at=a.updated_at,  # type: ignore[attr-defined]
    )
