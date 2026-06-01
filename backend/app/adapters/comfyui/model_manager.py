"""
ComfyUI model manager — triggers model loading via ComfyUI's management API.

Implements ModelManager protocol. ComfyUI manages its own model cache,
so these calls inform ComfyUI which models to preload/unload.
"""

from __future__ import annotations

import logging

from app.adapters.comfyui.client import ComfyUIClient


logger = logging.getLogger(__name__)


class ComfyUIModelManager:
    """
    Model lifecycle management via ComfyUI server.

    ComfyUI handles its own model loading/caching internally. This manager
    provides a protocol-compatible interface, though ComfyUI's model management
    is more limited than the direct backend (no explicit unload API).
    """

    def __init__(self, client: ComfyUIClient) -> None:
        self._client = client
        self._loaded_models: set[str] = set()

    async def load_model(self, model_id: str, device: str = "cuda") -> None:
        """
        Pre-load a model in ComfyUI (best-effort).

        ComfyUI loads models on first use. This call is a hint that
        the model will be used soon. No-op if ComfyUI doesn't support preloading.
        """
        # ComfyUI loads models lazily — record intent for is_loaded() tracking
        self._loaded_models.add(model_id)
        logger.info("ComfyUI model marked as loaded: %s (lazy-loaded on use)", model_id)

    async def unload_model(self, model_id: str) -> None:
        """
        Request ComfyUI to free model from memory (best-effort).

        ComfyUI manages its own cache; explicit unload may not be supported.
        We send a free memory request as a hint.
        """
        self._loaded_models.discard(model_id)
        # ComfyUI's /free endpoint can free memory
        try:
            import httpx

            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    f"{self._client._base_url}/free",
                    json={"unload_models": True},
                )
        except Exception:
            logger.warning("Failed to request ComfyUI model unload for %s", model_id)

    def is_loaded(self, model_id: str) -> bool:
        """Check if a model has been loaded (based on local tracking)."""
        return model_id in self._loaded_models

    def get_loaded_models(self) -> list[str]:
        """Return models believed to be loaded in ComfyUI."""
        return list(self._loaded_models)
