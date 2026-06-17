"""
Inference provider protocols — the adapter contracts.

These Python Protocols define what each inference capability must look like.
The service layer depends ONLY on these protocols, never on concrete adapter
implementations. This is the Dependency Inversion Principle in action.

Adding a new backend (e.g., TensorRT) means writing a new class that satisfies
the relevant Protocol — no changes to services or API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from app.domain.value_objects import ArtifactReference, GenerationParams, JobProgress


# --- Callback type for progress reporting ---

class ProgressCallback(Protocol):
    """Signature for progress reporting callbacks passed into adapters."""

    def __call__(self, progress: JobProgress) -> None: ...


# --- Text-to-Image ---

@runtime_checkable
class TextToImageProvider(Protocol):
    """Generates images from a text prompt."""

    async def generate(
        self,
        params: GenerationParams,
        model_id: str,
        output_dir: Path,
        on_progress: ProgressCallback | None = None,
    ) -> list[ArtifactReference]:
        """Run text-to-image inference. Returns references to saved artifacts."""
        ...


# --- Image-to-Image ---

@runtime_checkable
class ImageToImageProvider(Protocol):
    """Transforms an input image guided by a text prompt."""

    async def generate(
        self,
        params: GenerationParams,
        model_id: str,
        source_image_path: Path,
        output_dir: Path,
        strength: float = 0.75,
        on_progress: ProgressCallback | None = None,
    ) -> list[ArtifactReference]:
        """Run image-to-image inference."""
        ...


# --- Vision / Captioning ---

@runtime_checkable
class VisionProvider(Protocol):
    """Generates text descriptions from images."""

    async def caption(
        self,
        image_path: Path,
        model_id: str,
        prompt: str = "",
    ) -> str:
        """Return a text caption/description of the image."""
        ...


# --- Video Generation ---

@runtime_checkable
class VideoProvider(Protocol):
    """Generates video from text or image inputs."""

    async def generate(
        self,
        params: GenerationParams,
        model_id: str,
        output_dir: Path,
        source_image_path: Path | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> list[ArtifactReference]:
        """Run video generation. Returns list with a single video artifact reference."""
        ...


# --- Upscale ---

@runtime_checkable
class UpscaleProvider(Protocol):
    """Upscales an input image using a dedicated upscaling pipeline."""

    async def upscale(
        self,
        image_path: Path,
        model_id: str,
        output_dir: Path,
        scale_factor: float = 2.0,
        on_progress: ProgressCallback | None = None,
    ) -> list[ArtifactReference]:
        """Upscale the image and return references to saved artifacts."""
        ...


# --- Recolor ---

@runtime_checkable
class RecolorProvider(Protocol):
    """Re-colorizes an image guided by a text prompt via img2img diffusion."""

    async def generate(
        self,
        params: GenerationParams,
        model_id: str,
        source_image_path: Path,
        output_dir: Path,
        strength: float = 0.75,
        on_progress: ProgressCallback | None = None,
    ) -> list[ArtifactReference]:
        """Run recolor inference."""
        ...


# --- Sketch-to-Ink ---

@runtime_checkable
class SketchToInkProvider(Protocol):
    """Converts sketch / line-art images to styled ink-look outputs."""

    async def generate(
        self,
        params: GenerationParams,
        model_id: str,
        source_image_path: Path,
        output_dir: Path,
        strength: float = 0.9,
        base_model_id: str | None = None,
        model_type: str | None = None,
        adapter_conditioning_scale: float = 0.9,
        lora_model_id: str | None = None,
        lora_strength: float = 0.8,
        on_progress: ProgressCallback | None = None,
    ) -> list[ArtifactReference]:
        """Run sketch-to-ink inference."""
        ...


# --- Local LLM ---

@runtime_checkable
class LLMProvider(Protocol):
    """Runs local LLM inference (chat completions)."""

    async def generate(
        self,
        messages: list[dict[str, str]],
        model_id: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Generate a text completion from the message history."""
        ...


# --- Source Provider (model download/discovery) ---

@runtime_checkable
class SourceProvider(Protocol):
    """Resolves model metadata and downloads model files from an external source."""

    async def resolve_model_info(self, model_id: str) -> dict[str, Any]:
        """Fetch file manifest and metadata for a model."""
        ...

    async def download_file(
        self,
        model_id: str,
        file_path: str,
        destination: Path,
        resume_from_byte: int = 0,
        on_progress: Any = None,
    ) -> dict[str, Any]:
        """Download a single model file. Returns {sha256, size_bytes}."""
        ...

    def supports_resume(self) -> bool:
        """Whether this source supports byte-range resume."""
        ...


# --- Model Lifecycle ---

@runtime_checkable
class ModelManager(Protocol):
    """Handles model loading/unloading on the inference device."""

    async def load_model(self, model_id: str, device: str = "cuda") -> None:
        """Load a model into memory (GPU/CPU)."""
        ...

    async def unload_model(self, model_id: str) -> None:
        """Remove a model from memory, freeing resources."""
        ...

    def is_loaded(self, model_id: str) -> bool:
        """Check if a model is currently in memory."""
        ...

    def get_loaded_models(self) -> list[str]:
        """Return IDs of all currently loaded models."""
        ...
