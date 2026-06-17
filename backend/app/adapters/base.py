"""
Base adapter utilities — shared logic for generation adapters.

Provides reusable helpers for progress tracking and artifact persistence
so each concrete adapter only describes its unique workflow steps.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any


def resolve_model_path(model_id: str) -> str:
    """
    Return the local filesystem path for a downloaded model, or model_id as-is.

    Download jobs store files at:
      settings.models_path / <source> / <model_id parts>
    e.g. /app/storage/models/huggingface/black-forest-labs/FLUX.1-dev

    Returning the local path lets from_pretrained() load entirely from disk,
    which avoids HuggingFace network calls — critical for gated models where
    a 401 would be raised even when files are already present locally.
    """
    from app.infrastructure.config.settings import get_settings
    models_path = get_settings().models_path
    parts = [p for p in model_id.replace("\\", "/").split("/") if p]
    for source in ("huggingface", "civitai", "github", "local"):
        candidate = models_path.joinpath(source, *parts)
        if candidate.is_dir() and any(candidate.iterdir()):
            return str(candidate)
    return model_id


def hf_token() -> str | None:
    """Return the HuggingFace API token from settings (None if not configured)."""
    from app.infrastructure.config.settings import get_settings
    token = get_settings().huggingface_token
    return token or None

from app.domain.protocols import ProgressCallback
from app.domain.value_objects import ArtifactReference, GenerationParams, JobProgress


# Type alias for a raw step callback (current_step, total_steps)
StepCallback = Callable[[int, int], None]


class GenerationCancelledError(Exception):
    """Raised by the step callback when cooperative cancellation is requested."""


def build_diffusers_step_callback(
    on_progress: ProgressCallback | None,
    total_steps: int,
    cancel_event: Any = None,  # threading.Event | None — avoided circular import
) -> Callable[..., Any]:
    """
    Return a diffusers-compatible step callback: (pipe, step, timestep, kwargs) -> kwargs.

    Forwards JobProgress on each diffusion step. Raises GenerationCancelledError
    if cancel_event is set — this propagates out of the pipeline call cooperatively.
    Safe to call from a thread pool (the ProgressCallback is invoked synchronously).
    """

    def _callback(pipe: Any, step: int, timestep: Any, kwargs: Any) -> Any:
        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelledError("Generation cancelled by user")
        if on_progress is None:
            return kwargs
        on_progress(
            JobProgress(
                job_id=uuid.UUID(int=0),
                status="running",
                progress_percent=int((step / max(total_steps, 1)) * 100),
                current_step=step,
                total_steps=total_steps,
            )
        )
        return kwargs

    return _callback


def build_legacy_diffusers_step_callback(
    on_progress: ProgressCallback | None,
    total_steps: int,
) -> Callable[..., None]:
    """
    Return an old-style diffusers step callback: (step, timestep, latents) -> None.

    Use this for pipelines that pre-date callback_on_step_end, such as
    StableDiffusionXLAdapterPipeline. Pass as callback=..., callback_steps=1.
    """

    def _callback(step: int, timestep: Any, latents: Any) -> None:
        if on_progress is None:
            return
        on_progress(
            JobProgress(
                job_id=uuid.UUID(int=0),
                status="running",
                progress_percent=int((step / max(total_steps, 1)) * 100),
                current_step=step,
                total_steps=total_steps,
            )
        )

    return _callback


def build_step_callback(
    on_progress: ProgressCallback | None,
    total_steps: int,
) -> StepCallback:
    """
    Return a (current, total) → None callback that forwards JobProgress.

    Wraps the user-provided ProgressCallback so adapters don't repeat
    the JobProgress construction boilerplate.
    """

    def _callback(current: int, total: int) -> None:
        if on_progress is None:
            return
        on_progress(
            JobProgress(
                job_id=uuid.UUID(int=0),
                status="running",
                progress_percent=int((current / max(total, 1)) * 100),
                current_step=current,
                total_steps=total,
            )
        )

    return _callback


def save_artifacts_from_bytes(
    image_bytes_list: list[bytes],
    output_dir: Path,
    params: GenerationParams,
) -> list[ArtifactReference]:
    """
    Save raw image bytes to disk and return ArtifactReference list.

    Each image is saved as a unique PNG file. The caller is responsible
    for fetching the bytes (network download, pipeline output, etc.).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[ArtifactReference] = []

    for image_bytes in image_bytes_list:
        artifact_id = uuid.uuid4()
        local_filename = f"{artifact_id}.png"
        file_path = output_dir / local_filename
        file_path.write_bytes(image_bytes)

        artifacts.append(
            ArtifactReference(
                artifact_id=artifact_id,
                job_id=uuid.UUID(int=0),
                file_path=str(file_path),
                media_type="image/png",
                width=params.width,
                height=params.height,
                size_bytes=len(image_bytes),
            )
        )

    return artifacts


def save_artifacts_from_pil_images(
    images: list[Any],
    output_dir: Path,
    params: GenerationParams,
) -> list[ArtifactReference]:
    """
    Save PIL Image objects to disk and return ArtifactReference list.

    Used by Direct adapters after diffusers pipeline returns PIL images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[ArtifactReference] = []

    for image in images:
        artifact_id = uuid.uuid4()
        local_filename = f"{artifact_id}.png"
        file_path = output_dir / local_filename
        image.save(file_path)

        artifacts.append(
            ArtifactReference(
                artifact_id=artifact_id,
                job_id=uuid.UUID(int=0),
                file_path=str(file_path),
                media_type="image/png",
                width=params.width,
                height=params.height,
                size_bytes=file_path.stat().st_size,
            )
        )

    return artifacts
