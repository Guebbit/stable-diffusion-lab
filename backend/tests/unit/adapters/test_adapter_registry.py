from __future__ import annotations

import pytest

from app.adapters.adapter_registry import AdapterRegistry
from app.domain.enums import InferenceBackend, JobType


def test_register_and_retrieve_adapter() -> None:
    registry = AdapterRegistry(default_backend=InferenceBackend.DIRECT_PYTHON)
    adapter = object()

    registry.register(JobType.TEXT_TO_IMAGE, InferenceBackend.DIRECT_PYTHON, adapter)

    assert registry.get_provider(JobType.TEXT_TO_IMAGE) is adapter
    assert registry.is_available(JobType.TEXT_TO_IMAGE, InferenceBackend.DIRECT_PYTHON)


def test_multiple_backends_per_job_type() -> None:
    registry = AdapterRegistry(default_backend=InferenceBackend.BENTOML)
    direct_adapter = object()
    bento_adapter = object()

    registry.register(JobType.TEXT_TO_IMAGE, InferenceBackend.DIRECT_PYTHON, direct_adapter)
    registry.register(JobType.TEXT_TO_IMAGE, InferenceBackend.BENTOML, bento_adapter)

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
