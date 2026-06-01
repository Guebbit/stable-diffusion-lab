"""
Direct inference adapters — run AI models via Python (diffusers, transformers, torch).

This sub-package contains one adapter class per capability (text-to-image,
image-to-image, vision, video, LLM). Each adapter implements the corresponding
Protocol from app.domain.protocols.
"""

from app.adapters.direct.image_to_image import DirectImageToImageAdapter
from app.adapters.direct.llm import DirectLLMAdapter
from app.adapters.direct.text_to_image import DirectTextToImageAdapter
from app.adapters.direct.video import DirectVideoAdapter
from app.adapters.direct.vision import DirectVisionAdapter

__all__ = [
    "DirectImageToImageAdapter",
    "DirectLLMAdapter",
    "DirectTextToImageAdapter",
    "DirectVideoAdapter",
    "DirectVisionAdapter",
]
