"""
Adapter registry — maps (job_type, inference_backend) → concrete adapter instance.

The orchestrator/worker uses this registry to find the correct adapter for a job.
This decouples the worker from concrete adapter classes entirely — the worker
only knows about Protocols, and the registry handles resolution.

Backend selection strategy:
1. If the request specifies a backend → use that
2. If not, use the configured default for that job type
3. If the configured backend is unavailable → fall back to direct
"""

from __future__ import annotations

import logging
from typing import Any

from app.domain.enums import InferenceBackend, JobType
from app.domain.protocols import (
    ImageToImageProvider,
    LLMProvider,
    RecolorProvider,
    SketchToInkProvider,
    TextToImageProvider,
    UpscaleProvider,
    VideoProvider,
    VisionProvider,
)


logger = logging.getLogger(__name__)

# Mapping from JobType to the expected Protocol class.
# Used for runtime validation at registration time to catch wiring errors early.
_PROTOCOL_MAP: dict[JobType, type[Any]] = {
    JobType.TEXT_TO_IMAGE: TextToImageProvider,
    JobType.IMAGE_TO_IMAGE: ImageToImageProvider,
    JobType.IMAGE_ANALYSIS: VisionProvider,
    JobType.VIDEO_GENERATION: VideoProvider,
    JobType.LLM_INFERENCE: LLMProvider,
    JobType.UPSCALE: UpscaleProvider,
    JobType.RECOLOR: RecolorProvider,
    JobType.SKETCH_TO_INK: SketchToInkProvider,
}


class AdapterRegistry:
    """
    Central registry mapping (job_type, backend) → adapter instance.

    Usage:
        registry = AdapterRegistry(default_backend=InferenceBackend.DIRECT_PYTHON)
        registry.register(JobType.TEXT_TO_IMAGE, InferenceBackend.DIRECT_PYTHON, adapter)
        provider = registry.get_provider(JobType.TEXT_TO_IMAGE)

    The registry always returns a concrete adapter that satisfies the
    corresponding Protocol (TextToImageProvider, VideoProvider, etc.).
    """

    def __init__(self, default_backend: InferenceBackend = InferenceBackend.DIRECT_PYTHON) -> None:
        """
        Initialize the registry.

        Args:
            default_backend: The fallback backend used when no specific backend
                           is requested. Typically DIRECT_PYTHON for local-first.
        """
        self._default_backend = default_backend
        # Nested mapping: {job_type: {backend: adapter_instance}}
        self._registry: dict[JobType, dict[InferenceBackend, Any]] = {}

    def register(
        self,
        job_type: JobType,
        backend: InferenceBackend,
        adapter: Any,
        *,
        validate_protocol: bool = True,
    ) -> None:
        """
        Register an adapter for a specific (job_type, backend) pair.

        Args:
            job_type: The kind of work this adapter handles.
            backend: Which inference engine this adapter uses.
            adapter: The concrete adapter instance (must satisfy the relevant Protocol).
            validate_protocol: If True, verify the adapter implements the expected
                Protocol at registration time. Enable this to catch wiring errors
                early instead of failing silently at runtime.

        Raises:
            TypeError: If validate_protocol is True and the adapter does not
                satisfy the expected Protocol for the given job_type.
        """
        if validate_protocol:
            expected_protocol = _PROTOCOL_MAP.get(job_type)
            if expected_protocol is not None and not isinstance(adapter, expected_protocol):
                raise TypeError(
                    f"Adapter for {job_type} must implement {expected_protocol.__name__}. "
                    f"Got {type(adapter).__name__} which is missing required methods."
                )

        if job_type not in self._registry:
            self._registry[job_type] = {}
        self._registry[job_type][backend] = adapter
        logger.debug("Registered adapter: %s/%s → %s", job_type, backend, type(adapter).__name__)

    def get_provider(
        self,
        job_type: JobType,
        backend: InferenceBackend | None = None,
    ) -> Any:
        """
        Get the adapter for a given job type and optional backend preference.

        Resolution order:
        1. Exact match: (job_type, requested_backend)
        2. Default: (job_type, self._default_backend)
        3. Fallback: (job_type, DIRECT_PYTHON)
        4. Raises if nothing found

        Args:
            job_type: The type of inference work to perform.
            backend: Optional backend preference. If None, uses default.

        Returns:
            Adapter instance satisfying the relevant Protocol.

        Raises:
            ValueError: If no adapter is registered for this job_type.
        """
        backends_for_type = self._registry.get(job_type)
        if not backends_for_type:
            raise ValueError(
                f"No adapters registered for job type: {job_type}. "
                f"Available types: {list(self._registry.keys())}"
            )

        # Try requested backend first
        if backend and backend in backends_for_type:
            return backends_for_type[backend]

        # Try configured default
        if self._default_backend in backends_for_type:
            if backend and backend != self._default_backend:
                logger.warning(
                    "Requested backend %s not available for %s, falling back to %s",
                    backend,
                    job_type,
                    self._default_backend,
                )
            return backends_for_type[self._default_backend]

        # Last resort: DIRECT_PYTHON
        if InferenceBackend.DIRECT_PYTHON in backends_for_type:
            logger.warning("Falling back to DIRECT_PYTHON for %s", job_type)
            return backends_for_type[InferenceBackend.DIRECT_PYTHON]

        # Nothing found — pick any available
        available_backend = next(iter(backends_for_type))
        logger.warning("No preferred backend for %s, using %s", job_type, available_backend)
        return backends_for_type[available_backend]

    def is_available(self, job_type: JobType, backend: InferenceBackend) -> bool:
        """Check if a specific (job_type, backend) combination is registered."""
        return backend in self._registry.get(job_type, {})

    def get_available_backends(self, job_type: JobType) -> list[InferenceBackend]:
        """List all registered backends for a given job type."""
        return list(self._registry.get(job_type, {}).keys())

    def get_registered_types(self) -> list[JobType]:
        """List all job types that have at least one registered adapter."""
        return list(self._registry.keys())