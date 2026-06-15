"""
Direct inference adapters — run AI models via Python (diffusers, transformers, torch).

This sub-package contains one adapter class per capability (text-to-image,
image-to-image, vision, video, LLM). Each adapter implements the corresponding
Protocol from app.domain.protocols.

All adapters share a PipelineCache instance for efficient model lifecycle management.
"""

from app.adapters.direct.face_restore import DirectFaceRestoreAdapter
from app.adapters.direct.image_to_image import DirectImageToImageAdapter
from app.adapters.direct.llm import DirectLLMAdapter
from app.adapters.direct.model_manager import DirectModelManager
from app.adapters.direct.pipeline_cache import PipelineCache
from app.adapters.direct.sketch_to_ink import DirectSketchToInkAdapter
from app.adapters.direct.text_to_image import DirectTextToImageAdapter
from app.adapters.direct.upscale import DirectUpscaleAdapter
from app.adapters.direct.upscale_pipeline import UpscalePipeline
from app.adapters.direct.video import DirectVideoAdapter
from app.adapters.direct.vision import DirectVisionAdapter


__all__ = [
    "DirectFaceRestoreAdapter",
    "DirectImageToImageAdapter",
    "DirectLLMAdapter",
    "DirectModelManager",
    "DirectSketchToInkAdapter",
    "DirectTextToImageAdapter",
    "DirectUpscaleAdapter",
    "UpscalePipeline",
    "DirectVideoAdapter",
    "DirectVisionAdapter",
    "PipelineCache",
]
