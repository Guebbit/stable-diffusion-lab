"""
Model service — model registry and lifecycle management.

Handles model catalog operations: register, download, load, unload, delete.
Coordinates between the model repository (DB) and storage manager (filesystem).
"""

from __future__ import annotations

import logging
from uuid import UUID

from app.domain.enums import JobStatus, JobType, ModelStatus
from app.infrastructure.database.models import JobRecord, ModelRecord
from app.infrastructure.database.repositories import JobRepository, ModelRepository
from app.infrastructure.storage.storage_manager import StorageManager

logger = logging.getLogger(__name__)


class ModelService:
    """
    Manages the model catalog and lifecycle.

    Operations:
    - register: add a model to the catalog (metadata only)
    - download: queue a download job for a registered model
    - load: load model weights into GPU/CPU memory
    - unload: free model from memory
    - delete: remove model files and catalog entry
    - list: query the catalog
    - inspect: get detailed info about a model
    """

    def __init__(
        self,
        model_repository: ModelRepository,
        job_repository: JobRepository,
        storage_manager: StorageManager,
    ) -> None:
        self._model_repo = model_repository
        self._job_repo = job_repository
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
        preferred_name: str | None = None,
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
            preferred_name=preferred_name,
        )
        record = await self._model_repo.create(record)
        logger.info("Registered model: %s (%s)", name, model_id)
        return record

    async def request_download(self, model_id: str) -> UUID:
        """Queue a download job for a registered model."""
        model = await self._model_repo.get_by_model_id(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found in registry")

        # Create a download job
        job = JobRecord(
            job_type=JobType.MODEL_DOWNLOAD,
            status=JobStatus.PENDING,
            model_id=model.id,
            params={"model_id": model_id, "source": model.source},
        )
        job = await self._job_repo.create(job)

        # Update model status
        await self._model_repo.update_status(model_id, ModelStatus.DOWNLOADING)
        logger.info("Queued download for model: %s (job: %s)", model_id, job.id)
        return job.id

    async def list_models(self, source: str | None = None) -> list[ModelRecord]:
        """List all models in the catalog, optionally filtered by source."""
        return await self._model_repo.list_all(source=source)

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
