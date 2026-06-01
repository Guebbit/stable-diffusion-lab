"""
ComfyUI inference adapter — submits workflows to a ComfyUI server.

ComfyUI is useful when you want:
- Complex multi-model workflows (ControlNet, LoRA, IP-Adapter, face restore)
- Visual workflow composition via node graphs
- Access to ComfyUI's extensive community node ecosystem
- AnimateDiff / IP-Adapter orchestration that's hard to wire with raw diffusers

All adapters in this package communicate with ComfyUI via its WebSocket/HTTP API.
The WorkflowBuilder translates our clean domain model into ComfyUI's node-graph JSON.
"""

from app.adapters.comfyui.client import ComfyUIClient
from app.adapters.comfyui.image_to_image import ComfyUIImageToImageAdapter
from app.adapters.comfyui.llm import ComfyUILLMAdapter
from app.adapters.comfyui.model_manager import ComfyUIModelManager
from app.adapters.comfyui.text_to_image import ComfyUITextToImageAdapter
from app.adapters.comfyui.video import ComfyUIVideoAdapter
from app.adapters.comfyui.workflow_builder import WorkflowBuilder


__all__ = [
    "ComfyUIClient",
    "ComfyUIImageToImageAdapter",
    "ComfyUILLMAdapter",
    "ComfyUIModelManager",
    "ComfyUITextToImageAdapter",
    "ComfyUIVideoAdapter",
    "WorkflowBuilder",
]
