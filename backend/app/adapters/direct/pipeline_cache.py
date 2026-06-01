"""
Pipeline cache — LRU cache for loaded ML pipelines with VRAM budget enforcement.

Why this exists: Loading a diffusion pipeline takes 10-60 seconds and consumes
2-12 GB VRAM. Re-loading for every job is unacceptable. This cache keeps the
most recently used pipeline(s) in memory and evicts least-recently-used ones
when the budget is exceeded.

All direct adapters share a single PipelineCache instance (injected via constructor).
Adapters never load pipelines directly — they always go through the cache.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable


logger = logging.getLogger(__name__)


@dataclass
class LoadedPipeline:
    """Metadata wrapper around a loaded ML pipeline object."""

    # The actual pipeline object (DiffusionPipeline, transformers model, etc.)
    pipeline: Any

    # Identifier used to load this pipeline (HF repo ID or local path)
    model_id: str

    # Category tag — allows differentiated eviction priorities
    # e.g., "diffusion", "vision", "llm", "video"
    category: str = "diffusion"

    # Estimated VRAM usage in megabytes (used for budget tracking)
    estimated_vram_mb: int = 0

    # When this pipeline was last used (for LRU ordering)
    last_used: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PipelineCache:
    """
    LRU cache for loaded ML pipelines with VRAM budget enforcement.

    Eviction policy:
    - Single-GPU default: max 1 pipeline in memory (safest for 8-16 GB cards)
    - If VRAM budget allows, keep up to `max_cached` pipelines simultaneously
    - Least-recently-used pipeline is evicted first when capacity is reached
    - Eviction deletes the Python object and calls torch.cuda.empty_cache()

    Thread safety:
    - Uses an asyncio Lock to serialize load/evict operations
    - Multiple concurrent reads (is_loaded, get) are safe without the lock
    """

    def __init__(
        self,
        max_cached: int = 1,
        vram_budget_mb: int = 0,
    ) -> None:
        """
        Initialize the pipeline cache.

        Args:
            max_cached: Maximum number of pipelines to keep loaded simultaneously.
                        Default 1 = only one pipeline in memory at a time.
            vram_budget_mb: Optional VRAM budget in MB. 0 = no budget enforcement,
                           rely only on max_cached count.
        """
        self._max_cached = max_cached
        self._vram_budget_mb = vram_budget_mb

        # OrderedDict maintains insertion/access order for LRU tracking
        self._cache: OrderedDict[str, LoadedPipeline] = OrderedDict()

        # Serialize load/evict operations to prevent race conditions
        self._lock = asyncio.Lock()

    # --- Public API ---

    async def get_or_load(
        self,
        model_id: str,
        loader: Callable[[], Awaitable[tuple[Any, int]]],
        category: str = "diffusion",
    ) -> Any:
        """
        Get a cached pipeline or load it (evicting others if needed).

        Args:
            model_id: Unique identifier for the model (HF repo ID or local path).
            loader: Async callable that loads and returns (pipeline_object, estimated_vram_mb).
                    Only called if the model is not already cached.
            category: Category tag for eviction priority differentiation.

        Returns:
            The loaded pipeline object, ready for inference.
        """
        # Fast path: already cached — move to end (most recently used)
        if model_id in self._cache:
            self._cache.move_to_end(model_id)
            entry = self._cache[model_id]
            entry.last_used = datetime.now(timezone.utc)
            logger.debug("Cache hit: %s", model_id)
            return entry.pipeline

        # Slow path: need to load — acquire lock to serialize
        async with self._lock:
            # Double-check after acquiring lock (another coroutine may have loaded it)
            if model_id in self._cache:
                self._cache.move_to_end(model_id)
                return self._cache[model_id].pipeline

            # Evict if at capacity
            await self._evict_if_needed()

            # Load the pipeline
            logger.info("Cache miss — loading: %s", model_id)
            pipeline, vram_mb = await loader()

            # Store in cache
            self._cache[model_id] = LoadedPipeline(
                pipeline=pipeline,
                model_id=model_id,
                category=category,
                estimated_vram_mb=vram_mb,
                last_used=datetime.now(timezone.utc),
            )

            logger.info(
                "Cached: %s (%d MB, %d/%d slots used)",
                model_id,
                vram_mb,
                len(self._cache),
                self._max_cached,
            )

            return pipeline

    async def evict(self, model_id: str) -> None:
        """
        Explicitly remove a specific pipeline from the cache.

        Used by DirectModelManager.unload_model() to free resources on demand.
        """
        async with self._lock:
            if model_id in self._cache:
                self._evict_entry(model_id)
                logger.info("Evicted on request: %s", model_id)

    async def evict_all(self) -> None:
        """Remove all pipelines from cache — used during shutdown."""
        async with self._lock:
            model_ids = list(self._cache.keys())
            for model_id in model_ids:
                self._evict_entry(model_id)
            logger.info("Evicted all %d cached pipelines", len(model_ids))

    def is_loaded(self, model_id: str) -> bool:
        """Check if a model is currently in the cache (lock-free read)."""
        return model_id in self._cache

    def get_loaded_ids(self) -> list[str]:
        """Return IDs of all currently cached models."""
        return list(self._cache.keys())

    @property
    def current_vram_usage_mb(self) -> int:
        """Total estimated VRAM usage of all cached pipelines."""
        return sum(entry.estimated_vram_mb for entry in self._cache.values())

    @property
    def size(self) -> int:
        """Number of pipelines currently cached."""
        return len(self._cache)

    # --- Internal helpers ---

    async def _evict_if_needed(self) -> None:
        """Evict LRU pipelines until we're under capacity."""
        # Count-based eviction
        while len(self._cache) >= self._max_cached:
            # Pop the least recently used (first item in OrderedDict)
            oldest_id = next(iter(self._cache))
            self._evict_entry(oldest_id)

        # VRAM budget eviction (if budget is set)
        if self._vram_budget_mb > 0:
            while self._cache and self.current_vram_usage_mb > self._vram_budget_mb:
                oldest_id = next(iter(self._cache))
                self._evict_entry(oldest_id)

    def _evict_entry(self, model_id: str) -> None:
        """Remove a single entry from cache and free GPU memory."""
        entry = self._cache.pop(model_id, None)
        if entry is None:
            return

        # Delete the pipeline object to release GPU memory
        del entry.pipeline

        # Clear CUDA cache to actually free the VRAM
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info(
            "Evicted from cache: %s (freed ~%d MB)",
            model_id,
            entry.estimated_vram_mb,
        )
