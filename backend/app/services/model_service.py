"""
Model service — model catalog CRUD operations.

Single responsibility: manage the model registry (metadata).
Job orchestration (download queuing, etc.) is delegated to JobCreator.
"""

from __future__ import annotations

import logging

from app.domain.enums import ModelStatus
from app.infrastructure.database.models import ModelRecord
from app.infrastructure.database.repositories import ModelRepository
from app.infrastructure.storage.storage_manager import StorageManager

logger = logging.getLogger(__name__)


class ModelService:
    """
    Manages the model catalog (metadata only).

    Operations:
    - register: add a model to the catalog (metadata only)
    - list: query the catalog
    - get: inspect a specific model
    - delete: remove model files and catalog entry

    Job orchestration (download, load, etc.) is handled by JobCreator.
    """

    def __init__(
        self,
        model_repository: ModelRepository,
        storage_manager: StorageManager,
    ) -> None:
        self._model_repo = model_repository
        self._storage = storage_manager

    async def register_model(
        self,
        model_id: str,
        name: str,
        source: str,
        family: str = "custom",
        variant: str = "",
        description: str = "",
        tags: list[str] | None = None,
        source_url: str = "",
        capabilities: list[str] | None = None,
    ) -> ModelRecord:
        """Register a new model in the catalog (metadata only, no download)."""
        existing = await self._model_repo.get_by_model_id(model_id)
        if existing:
            logger.warning("Model %s already registered", model_id)
            return existing

        record = ModelRecord(
            model_id=model_id,
            name=name,
            source=source,
            family=family,
            variant=variant,
            description=description,
            tags=tags or [],
            source_url=source_url,
            capabilities=capabilities or [],
            status=ModelStatus.NOT_DOWNLOADED,
        )
        record = await self._model_repo.create(record)
        logger.info("Registered model: %s (%s)", name, model_id)
        return record

    async def list_models(
        self,
        source: str | None = None,
        capabilities: list[str] | None = None,
    ) -> list[ModelRecord]:
        """List all models in the catalog, optionally filtered by source and/or capabilities."""
        return await self._model_repo.list_all(source=source, capabilities=capabilities)

    async def get_model(self, model_id: str) -> ModelRecord | None:
        """Get detailed info about a specific model."""
        return await self._model_repo.get_by_model_id(model_id)

    async def delete_model(self, model_id: str) -> None:
        """Remove a model from catalog and disk."""
        model = await self._model_repo.get_by_model_id(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")

        # Delete files from disk
        self._storage.delete_model_files(model.source, model_id)

        # Delete DB record
        await self._model_repo.delete(model.id)
        logger.info("Deleted model: %s", model_id)
