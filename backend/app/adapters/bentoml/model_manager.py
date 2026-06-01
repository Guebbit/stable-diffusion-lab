"""
BentoML model manager — triggers model load/unload via BentoML service API.

Implements ModelManager protocol. The BentoML runner owns its own GPU memory,
so load/unload requests are forwarded to the runner's management endpoints.
"""

from __future__ import annotations

import logging

from app.adapters.bentoml.client import BentoMLClient


logger = logging.getLogger(__name__)


class BentoMLModelManager:
    """
    Model lifecycle management via BentoML service.

    Unlike DirectModelManager (which controls local PipelineCache),
    this manager sends HTTP requests to the BentoML runner to control
    model loading in the runner's own process and GPU memory.
    """

    def __init__(self, client: BentoMLClient) -> None:
        self._client = client
        self._loaded_models: set[str] = set()

    async def load_model(self, model_id: str, device: str = "cuda") -> None:
        """Request the BentoML runner to load a model."""
        await self._client.post_json(
            "/api/v1/models/load",
            payload={"model_id": model_id, "device": device},
            timeout_key="model_load",
        )
        self._loaded_models.add(model_id)
        logger.info("BentoML model loaded: %s", model_id)

    async def unload_model(self, model_id: str) -> None:
        """Request the BentoML runner to unload a model."""
        await self._client.post_json(
            "/api/v1/models/unload",
            payload={"model_id": model_id},
            timeout_key="model_load",
        )
        self._loaded_models.discard(model_id)
        logger.info("BentoML model unloaded: %s", model_id)

    def is_loaded(self, model_id: str) -> bool:
        """Check if a model is loaded in the BentoML runner (local cache of state)."""
        return model_id in self._loaded_models

    def get_loaded_models(self) -> list[str]:
        """Return IDs of models believed to be loaded in the BentoML runner."""
        return list(self._loaded_models)
