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


class InsufficientVRAMError(RuntimeError):
    """Raised when free VRAM is too low to load a pipeline with PIPELINE_OFFLOAD=none."""


def estimate_pipeline_vram_mb(pipeline: Any) -> int:
    """
    Estimate VRAM usage in MB for a diffusers pipeline.

    Checks .transformer first (FLUX, DiT-based) then .unet (SD 1.x/2.x/XL).
    Returns 0 if neither attribute is found.
    """
    component = getattr(pipeline, "transformer", None) or getattr(pipeline, "unet", None)
    if component is None:
        return 0
    param_bytes = sum(p.numel() * p.element_size() for p in component.parameters())
    return (param_bytes * 2) // (1024 * 1024)


def apply_pipeline_to_device(pipeline: Any, device: str, offload_strategy: str) -> Any:
    """
    Move a diffusers pipeline to the target device using the configured offload strategy.

    Called after from_pretrained() (model still on CPU) — safe to estimate VRAM here.

    Strategies (set via PIPELINE_OFFLOAD env var):
      none           — move entire pipeline to VRAM; raises InsufficientVRAMError if it won't fit.
      model_cpu      — stream sub-models to GPU one at a time (~3-5 GB peak VRAM).
      sequential_cpu — stream layer-by-layer (~1-2 GB peak VRAM, slowest).
    """
    import logging
    logger = logging.getLogger(__name__)

    if device != "cuda":
        return pipeline.to(device)

    if offload_strategy == "none":
        import torch
        needed_mb = estimate_pipeline_vram_mb(pipeline)
        if needed_mb > 0:
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            free_mb = free_bytes // (1024 * 1024)
            total_mb = total_bytes // (1024 * 1024)
            if free_mb < needed_mb:
                raise InsufficientVRAMError(
                    f"Not enough VRAM to load pipeline: need ~{needed_mb} MB, "
                    f"only {free_mb} MB free of {total_mb} MB total. "
                    "Free GPU memory (stop other jobs or models) or set "
                    "PIPELINE_OFFLOAD=model_cpu in your .env."
                )
            logger.info(
                "VRAM check passed: need ~%d MB, %d MB free — loading to GPU", needed_mb, free_mb
            )
        return pipeline.to(device)

    if offload_strategy == "sequential_cpu":
        pipeline.enable_sequential_cpu_offload()
        return pipeline

    # Default: model_cpu
    pipeline.enable_model_cpu_offload()
    return pipeline


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
