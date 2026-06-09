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

from app.domain.protocols import ProgressCallback
from app.domain.value_objects import ArtifactReference, GenerationParams, JobProgress


# Type alias for a raw step callback (current_step, total_steps)
StepCallback = Callable[[int, int], None]


def build_diffusers_step_callback(
    on_progress: ProgressCallback | None,
    total_steps: int,
) -> Callable[..., Any]:
    """
    Return a diffusers-compatible step callback: (pipe, step, timestep, kwargs) -> kwargs.

    Forwards JobProgress on each diffusion step. Safe to call from a thread pool
    (the ProgressCallback is invoked synchronously).
    """

    def _callback(pipe: Any, step: int, timestep: Any, kwargs: Any) -> Any:
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
