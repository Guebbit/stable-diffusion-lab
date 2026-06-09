"""
ComfyUI text-to-image adapter — submits txt2img workflows to ComfyUI server.

Implements TextToImageProvider protocol. Builds a workflow via WorkflowBuilder,
submits to ComfyUI, waits for completion via WebSocket, downloads output images.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.adapters.base import build_step_callback, save_artifacts_from_bytes
from app.adapters.comfyui.client import ComfyUIClient
from app.adapters.comfyui.workflow_builder import WorkflowBuilder
from app.domain.protocols import ProgressCallback
from app.domain.value_objects import ArtifactReference, GenerationParams


logger = logging.getLogger(__name__)


class ComfyUITextToImageAdapter:
    """Text-to-image generation via ComfyUI workflow execution."""

    def __init__(self, client: ComfyUIClient, workflow_builder: WorkflowBuilder) -> None:
        self._client = client
        self._builder = workflow_builder

    async def generate(
        self,
        params: GenerationParams,
        model_id: str,
        output_dir: Path,
        on_progress: ProgressCallback | None = None,
    ) -> list[ArtifactReference]:
        """
        Execute a text-to-image workflow on ComfyUI.

        1. Build workflow JSON from params
        2. Submit to ComfyUI queue
        3. Listen for progress via WebSocket
        4. Download output images on completion
        5. Save locally and return artifact references
        """
        # Build the workflow graph
        workflow = self._builder.build_text_to_image(params, model_id)

        # Submit to ComfyUI
        prompt_id = await self._client.queue_workflow(workflow)

        # Wait for completion with progress forwarding
        step_callback = build_step_callback(on_progress, params.num_inference_steps)
        output_data = await self._client.wait_for_completion(prompt_id, on_progress=step_callback)

        # Download image bytes from ComfyUI
        image_bytes_list: list[bytes] = []
        for node_output in output_data.values():
            images = node_output.get("images", []) if isinstance(node_output, dict) else []
            for image_info in images:
                filename = image_info.get("filename", "")
                subfolder = image_info.get("subfolder", "")
                image_bytes_list.append(await self._client.download_output(filename, subfolder))

        return save_artifacts_from_bytes(image_bytes_list, output_dir, params)