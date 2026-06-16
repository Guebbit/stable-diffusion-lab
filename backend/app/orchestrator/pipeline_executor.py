"""
Pipeline Executor — routes a job to the correct adapter and runs inference.

Single responsibility: given a job type and parameters, select the right
adapter from the registry, construct the GenerationParams and invoke the
adapter's generate/caption method.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from app.adapters.adapter_registry import AdapterRegistry
from app.domain.enums import InferenceBackend, JobType
from app.domain.value_objects import GenerationParams
from app.infrastructure.config.settings import get_settings

from app.domain.value_objects import ArtifactReference

if TYPE_CHECKING:
    from app.domain.value_objects import JobProgress

logger = logging.getLogger(__name__)


class PipelineExecutor:
    """Execute inference pipelines by delegating to the correct adapter."""

    def __init__(self, adapter_registry: AdapterRegistry) -> None:
        self._adapter_registry = adapter_registry

    async def execute(
        self,
        job_id: UUID,
        job_type: str,
        params: dict,
        on_progress: None | callable = None,
    ) -> list[ArtifactReference]:
        """
        Route a job to the correct adapter and execute it.

        Args:
            job_id: The job UUID (used for output paths).
            job_type: JobType enum value as string.
            params: Serialized job parameters.
            on_progress: Optional callback for progress updates.
        """
        settings = get_settings()
        correlation_id = params.get("correlation_id")
        backend_str = params.get("backend")
        backend = InferenceBackend(backend_str) if backend_str else None
        _loop = asyncio.get_running_loop()
        output_dir = settings.artifacts_path / str(job_id)

        progress_cb = on_progress or _make_dummy_progress(correlation_id, job_id)

        job_type_enum = JobType(job_type)

        if job_type_enum == JobType.TEXT_TO_IMAGE:
            return await self._run_text_to_image(
                job_id, backend, params, output_dir, progress_cb
            )
        elif job_type_enum == JobType.IMAGE_TO_IMAGE:
            return await self._run_image_to_image(
                job_id, backend, params, output_dir, progress_cb
            )
        elif job_type_enum == JobType.IMAGE_ANALYSIS:
            await self._run_image_analysis(job_id, backend, params)
        elif job_type_enum == JobType.UPSCALE:
            return await self._run_upscale(job_id, backend, params, output_dir, progress_cb)
        elif job_type_enum == JobType.RECOLOR:
            return await self._run_recolor(job_id, backend, params, output_dir, progress_cb)
        elif job_type_enum == JobType.SKETCH_TO_INK:
            return await self._run_sketch_to_ink(job_id, backend, params, output_dir, progress_cb)
        elif job_type_enum == JobType.VIDEO_GENERATION:
            return await self._run_video(job_id, backend, params, output_dir, progress_cb)
        elif job_type_enum == JobType.LLM_INFERENCE:
            await self._run_llm(job_id, backend, params)
        else:
            raise ValueError(
                f"PipelineExecutor does not handle job type '{job_type}'. "
                "Use ModelOperationHandler for model lifecycle jobs."
            )
        return []

    # ── Private dispatch methods ──────────────────────────────────────

    async def _run_text_to_image(
        self,
        job_id: UUID,
        backend: InferenceBackend | None,
        params: dict,
        output_dir: Path,
        on_progress: callable,
    ) -> list[ArtifactReference]:
        adapter = self._adapter_registry.get_provider(JobType.TEXT_TO_IMAGE, backend)
        gen_params = _build_gen_params(params)
        return await adapter.generate(
            gen_params,
            params["model_id"],
            output_dir,
            lora_model_id=params.get("lora_model_id"),
            lora_strength=float(params.get("lora_strength", 0.8)),
            on_progress=on_progress,
        )

    async def _run_image_to_image(
        self,
        job_id: UUID,
        backend: InferenceBackend | None,
        params: dict,
        output_dir: Path,
        on_progress: callable,
    ) -> list[ArtifactReference]:
        adapter = self._adapter_registry.get_provider(JobType.IMAGE_TO_IMAGE, backend)
        gen_params = _build_gen_params(params)
        return await adapter.generate(
            gen_params,
            params["model_id"],
            Path(params["source_image_path"]),
            output_dir,
            strength=params.get("strength", 0.75),
            on_progress=on_progress,
        )

    async def _run_image_analysis(
        self,
        job_id: UUID,
        backend: InferenceBackend | None,
        params: dict,
    ) -> list[ArtifactReference]:
        adapter = self._adapter_registry.get_provider(JobType.IMAGE_ANALYSIS, backend)
        await adapter.caption(
            Path(params["image_path"]),
            params["model_id"],
            prompt=params.get("prompt", ""),
        )
        return []

    async def _run_upscale(
        self,
        job_id: UUID,
        backend: InferenceBackend | None,
        params: dict,
        output_dir: Path,
        on_progress: callable,
    ) -> list[ArtifactReference]:
        adapter = self._adapter_registry.get_provider(JobType.UPSCALE, backend)
        return await adapter.upscale(
            Path(params["image_path"]),
            params["model_id"],
            output_dir,
            # Upscale step
            scale_factor=float(params.get("scale_factor", 2.0)),
            prompt=params.get("prompt", ""),
            noise_level=int(params.get("noise_level", 20)),
            num_inference_steps=int(params.get("num_inference_steps", 20)),
            # Enhancement step (None = skip)
            enhance_model_id=params.get("enhance_model_id"),
            enhance_strength=float(params.get("enhance_strength", 0.4)),
            # Face restore step (None = skip)
            face_restore_model_id=params.get("face_restore_model_id"),
            face_restore_fidelity=float(params.get("face_restore_fidelity", 0.5)),
            on_progress=on_progress,
        )

    async def _run_recolor(
        self,
        job_id: UUID,
        backend: InferenceBackend | None,
        params: dict,
        output_dir: Path,
        on_progress: callable,
    ) -> list[ArtifactReference]:
        adapter = self._adapter_registry.get_provider(JobType.RECOLOR, backend)
        image_path = Path(params["image_path"])
        gen_params = _build_gen_params_from_image(params, image_path)
        return await adapter.generate(
            gen_params,
            params["model_id"],
            image_path,
            output_dir,
            strength=float(params.get("strength", 0.75)),
            on_progress=on_progress,
        )

    async def _run_sketch_to_ink(
        self,
        job_id: UUID,
        backend: InferenceBackend | None,
        params: dict,
        output_dir: Path,
        on_progress: callable,
    ) -> list[ArtifactReference]:
        adapter = self._adapter_registry.get_provider(JobType.SKETCH_TO_INK, backend)
        image_path = Path(params["image_path"])
        gen_params = _build_gen_params_from_image(params, image_path)
        return await adapter.generate(
            gen_params,
            params["model_id"],
            image_path,
            output_dir,
            strength=0.9,
            base_model_id=params.get("base_model_id"),
            model_type=params.get("model_type"),
            adapter_conditioning_scale=float(params.get("adapter_conditioning_scale", 0.9)),
            lora_model_id=params.get("lora_model_id"),
            lora_strength=float(params.get("lora_strength", 0.8)),
            on_progress=on_progress,
        )

    async def _run_video(
        self,
        job_id: UUID,
        backend: InferenceBackend | None,
        params: dict,
        output_dir: Path,
        on_progress: callable,
    ) -> list[ArtifactReference]:
        adapter = self._adapter_registry.get_provider(JobType.VIDEO_GENERATION, backend)
        gen_params = _build_gen_params(params, default_steps=25)
        source_path = (
            Path(params["source_image_path"]) if params.get("source_image_path") else None
        )
        return await adapter.generate(
            gen_params,
            params["model_id"],
            output_dir,
            source_image_path=source_path,
            on_progress=on_progress,
        )

    async def _run_llm(
        self,
        job_id: UUID,
        backend: InferenceBackend | None,
        params: dict,
    ) -> list[ArtifactReference]:
        adapter = self._adapter_registry.get_provider(JobType.LLM_INFERENCE, backend)
        await adapter.generate(
            params["messages"],
            params["model_id"],
            max_tokens=params.get("max_tokens", 512),
            temperature=params.get("temperature", 0.7),
        )
        return []


# ── Module-level helpers ────────────────────────────────────────────────

def _make_dummy_progress(correlation_id: str | None, job_id: UUID) -> callable:
    """Return a no-op progress callback."""

    def _no_op(progress: "JobProgress") -> None:
        pass

    return _no_op


def _build_gen_params(params: dict, default_steps: int = 20) -> GenerationParams:
    """Extract GenerationParams from a serialized params dict."""
    return GenerationParams(
        prompt=params.get("prompt", ""),
        negative_prompt=params.get("negative_prompt", ""),
        width=params.get("width", 512),
        height=params.get("height", 512),
        num_inference_steps=params.get("num_inference_steps", default_steps),
        guidance_scale=params.get("guidance_scale", 7.5),
        seed=params.get("seed"),
        num_images=params.get("num_images", 1),
    )


def _build_gen_params_from_image(params: dict, image_path: Path) -> GenerationParams:
    """Like _build_gen_params but falls back to source image dimensions when width/height are absent."""
    width = params.get("width")
    height = params.get("height")
    if not (width and height):
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception:
            width, height = 512, 512
    return GenerationParams(
        prompt=params.get("prompt", ""),
        negative_prompt=params.get("negative_prompt", ""),
        width=int(width),
        height=int(height),
        num_inference_steps=params.get("num_inference_steps", 20),
        guidance_scale=params.get("guidance_scale", 7.5),
        seed=params.get("seed"),
        num_images=1,
    )