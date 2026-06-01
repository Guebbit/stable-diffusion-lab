"""
Artifact service — gallery management operations.

Handles artifact listing, metadata updates (favorite/rating/notes),
and artifact deletion with filesystem cleanup.
"""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import UUID

from app.infrastructure.database.models import ArtifactRecord
from app.infrastructure.database.repositories import ArtifactRepository

logger = logging.getLogger(__name__)


class ArtifactService:
    """
    Manages artifact gallery operations.

    Operations:
    - list: paginated, filtered artifact queries
    - get: single artifact detail
    - update_metadata: gallery curation (favorite, rating, notes)
    - delete: remove artifact file and DB record
    """

    def __init__(self, artifact_repository: ArtifactRepository) -> None:
        self._artifact_repo = artifact_repository

    async def get_artifact(self, artifact_id: UUID) -> ArtifactRecord | None:
        """Get a single artifact by ID."""
        return await self._artifact_repo.get_by_id(artifact_id)

    async def list_artifacts(
        self,
        model_name: str | None = None,
        media_type: str | None = None,
        is_favorite: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[ArtifactRecord], int]:
        """List artifacts with filtering and pagination."""
        return await self._artifact_repo.list_filtered(
            model_name=model_name,
            media_type=media_type,
            is_favorite=is_favorite,
            limit=limit,
            offset=offset,
        )

    async def update_metadata(
        self,
        artifact_id: UUID,
        is_favorite: bool | None = None,
        rating: int | None = None,
        notes: str | None = None,
    ) -> ArtifactRecord | None:
        """Update gallery metadata for an artifact. Only non-None fields are changed."""
        artifact = await self._artifact_repo.get_by_id(artifact_id)
        if not artifact:
            return None

        updates: dict = {}
        if is_favorite is not None:
            updates["is_favorite"] = is_favorite
        if rating is not None:
            updates["rating"] = rating
        if notes is not None:
            updates["notes"] = notes

        if updates:
            await self._artifact_repo.update(artifact_id, **updates)
            artifact = await self._artifact_repo.get_by_id(artifact_id)

        return artifact

    async def delete_artifact(self, artifact_id: UUID) -> bool:
        """Delete an artifact — remove file from disk and DB record."""
        artifact = await self._artifact_repo.get_by_id(artifact_id)
        if not artifact:
            return False

        # Delete the actual file from disk
        file_path = Path(artifact.file_path)
        if file_path.exists():
            file_path.unlink()
            logger.info("Deleted artifact file: %s", file_path)

        # Delete thumbnail if exists
        if artifact.thumbnail_path:
            thumb_path = Path(artifact.thumbnail_path)
            if thumb_path.exists():
                thumb_path.unlink()

        # Delete DB record
        await self._artifact_repo.delete(artifact_id)
        logger.info("Deleted artifact: %s", artifact_id)
        return True
