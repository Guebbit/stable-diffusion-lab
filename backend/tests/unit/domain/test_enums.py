from __future__ import annotations

from app.domain.enums import (
    GenerationTask,
    InferenceBackend,
    JobStatus,
    JobType,
    ModelFamily,
    ModelSource,
    ModelStatus,
)


def test_enum_values_match_expected_contract() -> None:
    assert set(ModelSource) == {ModelSource.HUGGINGFACE, ModelSource.CIVITAI, ModelSource.LOCAL}
    assert set(ModelFamily) == {
        ModelFamily.SD15,
        ModelFamily.SDXL,
        ModelFamily.FLUX,
        ModelFamily.CUSTOM,
    }
    assert set(ModelStatus) == {
        ModelStatus.NOT_DOWNLOADED,
        ModelStatus.DOWNLOADING,
        ModelStatus.DOWNLOAD_PAUSED,
        ModelStatus.DOWNLOADED,
        ModelStatus.LOADING,
        ModelStatus.LOADED,
        ModelStatus.ERROR,
    }
    assert set(JobStatus) == {
        JobStatus.PENDING,
        JobStatus.RUNNING,
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    }
    assert set(JobType) == {
        JobType.TEXT_TO_IMAGE,
        JobType.IMAGE_TO_IMAGE,
        JobType.IMAGE_CAPTIONING,
        JobType.VIDEO_GENERATION,
        JobType.LLM_INFERENCE,
        JobType.MODEL_DOWNLOAD,
        JobType.MODEL_LOAD,
    }
    assert set(GenerationTask) == {
        GenerationTask.TEXT_TO_IMAGE,
        GenerationTask.IMAGE_TO_IMAGE,
        GenerationTask.CAPTIONING,
        GenerationTask.VIDEO,
        GenerationTask.LLM_CHAT,
    }
    assert set(InferenceBackend) == {
        InferenceBackend.DIRECT_PYTHON,
        InferenceBackend.BENTOML,
        InferenceBackend.COMFYUI,
    }


def test_enums_behave_like_strings() -> None:
    assert isinstance(JobStatus.PENDING, str)
    assert str(JobStatus.PENDING) == "pending"
    assert JobStatus.PENDING == "pending"
    assert InferenceBackend.DIRECT_PYTHON.upper() == "DIRECT_PYTHON"


def test_enum_values_are_unique_within_each_enum() -> None:
    enum_groups = [
        ModelSource,
        ModelFamily,
        ModelStatus,
        JobStatus,
        JobType,
        GenerationTask,
        InferenceBackend,
    ]
    for enum_group in enum_groups:
        values = [item.value for item in enum_group]
        assert len(values) == len(set(values))


def test_model_and_backend_enum_values_do_not_overlap() -> None:
    model_values = {item.value for item in ModelSource} | {item.value for item in ModelFamily} | {
        item.value for item in ModelStatus
    }
    backend_values = {item.value for item in InferenceBackend}
    assert model_values.isdisjoint(backend_values)
