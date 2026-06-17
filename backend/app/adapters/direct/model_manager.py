"""
Direct model manager — handles model load/unload lifecycle for in-process inference.

Implements the ModelManager protocol from app.domain.protocols.
Coordinates with PipelineCache to control which models are in GPU memory.

This is a thin orchestration layer — actual loading logic lives in the adapters,
but the ModelManager provides a unified interface for the service layer to
load/unload models regardless of type (diffusion, vision, LLM).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.adapters.direct.pipeline_cache import PipelineCache


logger = logging.getLogger(__name__)


class DirectModelManager:
    """
    Model lifecycle management for the direct inference backend.

    Responsibilities:
    - load_model: Trigger pipeline loading into GPU/CPU via PipelineCache
    - unload_model: Trigger pipeline eviction from PipelineCache
    - is_loaded / get_loaded_models: Query cache state

    Does NOT handle model downloading or registry updates — that's ModelService's job.
    """

    def __init__(self, pipeline_cache: PipelineCache) -> None:
        self._cache = pipeline_cache

    async def load_model(self, model_id: str, device: str = "cuda") -> None:
        """
        Load a model into GPU/CPU memory.

        If the model is already loaded, this is a no-op.
        If the cache is full, the LRU pipeline is evicted first.

        Args:
            model_id: HuggingFace model ID or local path.
            device: Target device ("cuda", "cpu", or "auto").
        """
        if self._cache.is_loaded(model_id):
            logger.debug("Model already loaded: %s", model_id)
            return

        # Build a generic loader — we don't know the model type here,
        # so we use DiffusionPipeline.from_pretrained which auto-detects
        async def _loader() -> tuple[Any, int]:
            return await asyncio.to_thread(self._load_sync, model_id, device)

        await self._cache.get_or_load(model_id, _loader, category="generic")

    async def unload_model(self, model_id: str) -> None:
        """
        Remove a model from GPU/CPU memory.

        Frees VRAM immediately. The model can be re-loaded later.
        """
        await self._cache.evict(model_id)
        logger.info("Model unloaded: %s", model_id)

    def is_loaded(self, model_id: str) -> bool:
        """Check if a model is currently in GPU/CPU memory."""
        return self._cache.is_loaded(model_id)

    def get_loaded_models(self) -> list[str]:
        """Return IDs of all currently loaded models."""
        return self._cache.get_loaded_ids()

    @staticmethod
    def _load_sync(model_id: str, device: str) -> tuple[Any, int]:
        """
        Synchronous model loading — runs on thread pool.

        Uses DiffusionPipeline.from_pretrained as a generic loader.
        Specific adapters will use their own loaders via PipelineCache.get_or_load()
        with more tailored pipeline classes.
        """
        import torch
        from diffusers import DiffusionPipeline

        # Resolve device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            from app.infrastructure.config.settings import get_settings
            if not get_settings().allow_cpu_fallback:
                raise RuntimeError(
                    "CUDA not available — model load aborted. "
                    "Set ALLOW_CPU_FALLBACK=true to allow CPU inference."
                )
            logger.warning("CUDA not available — loading model on CPU (ALLOW_CPU_FALLBACK=true)")

        dtype = torch.float16 if device == "cuda" else torch.float32

        logger.info("Loading model: %s → %s (%s)", model_id, device, dtype)

        from app.adapters.base import apply_pipeline_to_device
        from app.infrastructure.config.settings import get_settings
        pipeline = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
            low_cpu_mem_usage=True,
        )
        pipeline = apply_pipeline_to_device(pipeline, device, get_settings().pipeline_offload)

        # Estimate VRAM usage (rough: parameter count × bytes per param)
        estimated_vram_mb = _estimate_pipeline_vram(pipeline, dtype)

        return pipeline, estimated_vram_mb


def _estimate_pipeline_vram(pipeline: Any, dtype: Any) -> int:
    """
    Estimate VRAM usage of a loaded pipeline in MB.

    Uses parameter count × bytes-per-element as a rough approximation.
    Actual usage may be higher due to activations and optimizer state,
    but this is sufficient for cache budget decisions.
    """
    import torch

    total_bytes = 0
    bytes_per_element = 2 if dtype == torch.float16 else 4

    # Sum parameters across all sub-models in the pipeline
    for attr_name in dir(pipeline):
        attr = getattr(pipeline, attr_name, None)
        if hasattr(attr, "parameters"):
            try:
                param_count = sum(p.numel() for p in attr.parameters())
                total_bytes += param_count * bytes_per_element
            except Exception:
                pass

    return total_bytes // (1024 * 1024)  # Convert to MB
