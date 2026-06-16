from __future__ import annotations

from app.domain.enums import (
    GenerationMode,
    GenerationTask,
    InferenceBackend,
    JobStatus,
    JobType,
    ModelCapability,
    ModelFamily,
    ModelSource,
    ModelStatus,
)


def test_enum_values_match_expected_contract() -> None:
    assert set(ModelSource) == {ModelSource.HUGGINGFACE, ModelSource.CIVITAI, ModelSource.GITHUB, ModelSource.LOCAL}
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
    assert set(GenerationMode) == {
        GenerationMode.TEXT_TO_IMAGE,
        GenerationMode.IMAGE_TO_IMAGE,
        GenerationMode.UPSCALE,
        GenerationMode.DESCRIBE,
        GenerationMode.RECOLOR,
        GenerationMode.SKETCH_TO_INK,
    }
    assert set(ModelCapability) == {
        ModelCapability.TEXT_TO_IMAGE,
        ModelCapability.IMAGE_TO_IMAGE,
        ModelCapability.UPSCALE_IMAGE,
        ModelCapability.FACE_RESTORE,
        ModelCapability.DESCRIBE,
        ModelCapability.RECOLOR,
        ModelCapability.SKETCH_TO_INK,
    }
    assert set(JobType) == {
        JobType.TEXT_TO_IMAGE,
        JobType.IMAGE_TO_IMAGE,
        JobType.IMAGE_ANALYSIS,
        JobType.UPSCALE,
        JobType.RECOLOR,
        JobType.SKETCH_TO_INK,
        JobType.VIDEO_GENERATION,
        JobType.LLM_INFERENCE,
        JobType.MODEL_DOWNLOAD,
        JobType.MODEL_DELETE,
        JobType.MODEL_LOAD,
        JobType.MODEL_REFRESH,
    }
    assert set(GenerationTask) == {
        GenerationTask.TEXT_TO_IMAGE,
        GenerationTask.IMAGE_TO_IMAGE,
        GenerationTask.IMAGE_ANALYSIS,
        GenerationTask.UPSCALE,
        GenerationTask.RECOLOR,
        GenerationTask.SKETCH_TO_INK,
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
        GenerationMode,
        ModelCapability,
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


def test_generation_mode_and_capability_are_aligned() -> None:
    """Most GenerationMode values must exist verbatim in ModelCapability.

    Exceptions (intentional splits):
    - GenerationMode.UPSCALE ("upscale") maps to ModelCapability.UPSCALE_IMAGE
      ("upscale_image") at the model level.
    ModelCapability is a strict superset — it also includes model-specific
    capabilities like FACE_RESTORE that have no corresponding user-facing mode.
    """
    _INTENTIONAL_SPLITS = {"upscale"}
    mode_values = {m.value for m in GenerationMode} - _INTENTIONAL_SPLITS
    capability_values = {c.value for c in ModelCapability}
    uncovered = mode_values - capability_values
    assert not uncovered, f"GenerationMode values missing from ModelCapability: {uncovered}"
