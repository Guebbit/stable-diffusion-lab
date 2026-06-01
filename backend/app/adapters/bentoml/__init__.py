"""
BentoML inference adapter — delegates to a BentoML service endpoint.

BentoML is useful when you want:
- Process isolation (OOM crashes don't kill the API)
- Adaptive batching for higher throughput
- Model serving with monitoring
- Quantization/optimization via ONNX, TensorRT

All adapters in this package implement the same Protocols as their direct
counterparts but delegate execution to a remote BentoML runner via HTTP.
"""

from app.adapters.bentoml.client import BentoMLClient
from app.adapters.bentoml.image_to_image import BentoMLImageToImageAdapter
from app.adapters.bentoml.llm import BentoMLLLMAdapter
from app.adapters.bentoml.model_manager import BentoMLModelManager
from app.adapters.bentoml.text_to_image import BentoMLTextToImageAdapter
from app.adapters.bentoml.video import BentoMLVideoAdapter
from app.adapters.bentoml.vision import BentoMLVisionAdapter


__all__ = [
    "BentoMLClient",
    "BentoMLImageToImageAdapter",
    "BentoMLLLMAdapter",
    "BentoMLModelManager",
    "BentoMLTextToImageAdapter",
    "BentoMLVideoAdapter",
    "BentoMLVisionAdapter",
]
