"""
Model Resolver — translates user-facing model identifiers to canonical slugs.

Single responsibility: given a model identifier (slug or DB UUID), look up the
canonical HuggingFace/Civitai model_id in the repository. If the identifier is
not found, it passes through unchanged so adapters can attempt direct loading.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import UUID

from app.infrastructure.database.repositories import ModelRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class ModelResolver:
    """Resolve user-facing model identifiers to canonical model IDs."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._repo = ModelRepository(session)

    async def resolve(self, identifier: str | UUID) -> str:
        """
        Resolve an identifier to a canonical model_id.

        Args:
            identifier: A model slug (e.g. ``runwayml/stable-diffusion-v1-5``)
                or a database UUID string.

        Returns:
            The canonical model_id string. If the identifier is not found in
            the repository, it is returned as-is so adapters can attempt
            direct loading from remote sources.
        """
        # Try DB lookup first (handles both UUID and slug paths)
        model = await self._repo.get_by_model_id(str(identifier))
        if model is not None:
            return model.model_id

        # Fallback: try UUID lookup
        if isinstance(identifier, str):
            try:
                uuid_val = UUID(identifier)
                model = await self._repo.get_by_id(uuid_val)
                if model is not None:
                    return model.model_id
            except ValueError:
                pass

        # Not found — pass through for direct/remote loading
        logger.warning(
            "Model '%s' not found in repository, using as-is for direct loading",
            identifier,
        )
        return str(identifier)