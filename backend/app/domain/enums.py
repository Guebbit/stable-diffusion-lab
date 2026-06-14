"""
Domain enumerations and literal types.

Central source of truth for all categorical values in the system.
Other layers import these — never redefine string literals inline.
"""

from enum import StrEnum


class ModelSource(StrEnum):
    """Where a model originates from."""

    HUGGINGFACE = "huggingface"
    CIVITAI = "civitai"
    LOCAL = "local"


class ModelFamily(StrEnum):
    """Architecture family of a diffusion model."""

    SD15 = "sd15"
    SDXL = "sdxl"
    FLUX = "flux"
    CUSTOM = "custom"


class ModelStatus(StrEnum):
    """Lifecycle status of a model in the registry."""

    NOT_DOWNLOADED = "not_downloaded"
    DOWNLOADING = "downloading"
    DOWNLOAD_PAUSED = "download_paused"
    DOWNLOADED = "downloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"


class JobStatus(StrEnum):
    """State machine states for a background job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GenerationMode(StrEnum):
    """High-level generation mode presented to the user."""

    TEXT_TO_IMAGE = "text_to_image"
    IMAGE_TO_IMAGE = "image_to_image"
    UPSCALE = "upscale"
    DESCRIBE = "describe"
    RECOLOR = "recolor"
    SKETCH_TO_INK = "sketch_to_ink"


class ModelCapability(StrEnum):
    """Capabilities a model supports, used to filter by generation mode."""

    TEXT_TO_IMAGE = "text_to_image"
    IMAGE_TO_IMAGE = "image_to_image"
    UPSCALE = "upscale"
    DESCRIBE = "describe"
    RECOLOR = "recolor"
    SKETCH_TO_INK = "sketch_to_ink"


class JobType(StrEnum):
    """The kind of work a job performs."""

    TEXT_TO_IMAGE = "text_to_image"
    IMAGE_TO_IMAGE = "image_to_image"
    IMAGE_ANALYSIS = "image_analysis"
    UPSCALE = "upscale"
    RECOLOR = "recolor"
    SKETCH_TO_INK = "sketch_to_ink"
    VIDEO_GENERATION = "video_generation"
    LLM_INFERENCE = "llm_inference"
    MODEL_DOWNLOAD = "model_download"
    MODEL_DELETE = "model_delete"
    MODEL_LOAD = "model_load"
    MODEL_REFRESH = "model_refresh"


class GenerationTask(StrEnum):
    """High-level generation task type for the API."""

    TEXT_TO_IMAGE = "text_to_image"
    IMAGE_TO_IMAGE = "image_to_image"
    IMAGE_ANALYSIS = "image_analysis"
    UPSCALE = "upscale"
    RECOLOR = "recolor"
    SKETCH_TO_INK = "sketch_to_ink"
    VIDEO = "video"
    LLM_CHAT = "llm_chat"


class InferenceBackend(StrEnum):
    """Which inference engine handles execution."""

    DIRECT_PYTHON = "direct_python"
    BENTOML = "bentoml"
    COMFYUI = "comfyui"
