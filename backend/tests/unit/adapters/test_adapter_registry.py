from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.adapters.adapter_registry import AdapterRegistry
from app.domain.enums import InferenceBackend, JobType
from app.domain.value_objects import ArtifactReference, GenerationParams, JobProgress


def _make_dummy_t2i_adapter() -> Any:
    """Create a dummy adapter that satisfies TextToImageProvider protocol."""
    adapter = MagicMock()
    adapter.generate = AsyncMock(return_value=[ArtifactReference("test.png", 100, "png")])
    return adapter


def _make_non_compliant_adapter() -> Any:
    """Create an adapter that does NOT satisfy TextToImageProvider protocol."""
    return object()


# -- Registration behavior tests (protocol validation disabled) --


def test_register_and_retrieve_adapter() -> None:
    registry = AdapterRegistry(default_backend=InferenceBackend.DIRECT_PYTHON)
    adapter = object()

    registry.register(
        JobType.TEXT_TO_IMAGE, InferenceBackend.DIRECT_PYTHON, adapter, validate_protocol=False
    )

    assert registry.get_provider(JobType.TEXT_TO_IMAGE) is adapter
    assert registry.is_available(JobType.TEXT_TO_IMAGE, InferenceBackend.DIRECT_PYTHON)


def test_multiple_backends_per_job_type() -> None:
    registry = AdapterRegistry(default_backend=InferenceBackend.BENTOML)
    direct_adapter = object()
    bento_adapter = object()

    registry.register(
        JobType.TEXT_TO_IMAGE, InferenceBackend.DIRECT_PYTHON, direct_adapter, validate_protocol=False
    )
    registry.register(
        JobType.TEXT_TO_IMAGE, InferenceBackend.BENTOML, bento_adapter, validate_protocol=False
    )

    assert registry.get_provider(JobType.TEXT_TO_IMAGE) is bento_adapter
    assert (
        registry.get_provider(JobType.TEXT_TO_IMAGE, InferenceBackend.DIRECT_PYTHON)
        is direct_adapter
    )
    assert registry.get_available_backends(JobType.TEXT_TO_IMAGE) == [
        InferenceBackend.DIRECT_PYTHON,
        InferenceBackend.BENTOML,
    ]


def test_unregistered_job_type_raises_error() -> None:
    registry = AdapterRegistry()

    with pytest.raises(ValueError, match="No adapters registered for job type"):
        registry.get_provider(JobType.VIDEO_GENERATION)


# -- Runtime protocol validation tests --


def test_register_valid_adapter_passes_validation() -> None:
    """An adapter that implements the protocol registers successfully."""
    registry = AdapterRegistry()
    adapter = _make_dummy_t2i_adapter()

    registry.register(JobType.TEXT_TO_IMAGE, InferenceBackend.DIRECT_PYTHON, adapter)

    assert registry.get_provider(JobType.TEXT_TO_IMAGE) is adapter


def test_register_invalid_adapter_raises_type_error() -> None:
    """An adapter missing required methods raises TypeError at registration time."""
    registry = AdapterRegistry()
    bad_adapter = _make_non_compliant_adapter()

    with pytest.raises(TypeError, match="must implement TextToImageProvider"):
        registry.register(JobType.TEXT_TO_IMAGE, InferenceBackend.DIRECT_PYTHON, bad_adapter)


def test_register_without_validation_skips_check() -> None:
    """When validate_protocol=False, even non-compliant adapters register."""
    registry = AdapterRegistry()
    bad_adapter = _make_non_compliant_adapter()

    registry.register(
        JobType.TEXT_TO_IMAGE,
        InferenceBackend.DIRECT_PYTHON,
        bad_adapter,
        validate_protocol=False,
    )
    assert registry.get_provider(JobType.TEXT_TO_IMAGE) is bad_adapter


def test_model_lifecycle_jobs_skip_validation_gracefully() -> None:
    """JobTypes without a protocol mapping (MODEL_DOWNLOAD, MODEL_LOAD) always pass."""
    registry = AdapterRegistry()
    handler = object()

    registry.register(JobType.MODEL_DOWNLOAD, InferenceBackend.DIRECT_PYTHON, handler)
    registry.register(JobType.MODEL_LOAD, InferenceBackend.DIRECT_PYTHON, handler)

    assert registry.is_available(JobType.MODEL_DOWNLOAD, InferenceBackend.DIRECT_PYTHON)
    assert registry.is_available(JobType.MODEL_LOAD, InferenceBackend.DIRECT_PYTHON)
