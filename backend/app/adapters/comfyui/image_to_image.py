"""
ComfyUI image-to-image adapter — submits img2img workflows to ComfyUI server.

Implements ImageToImageProvider protocol. Uploads the source image to ComfyUI,
builds an img2img workflow, executes, and downloads results.
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


class ComfyUIImageToImageAdapter:
    """Image-to-image generation via ComfyUI workflow execution."""

    def __init__(self, client: ComfyUIClient, workflow_builder: WorkflowBuilder) -> None:
        self._client = client
        self._builder = workflow_builder

    async def generate(
        self,
        params: GenerationParams,
        model_id: str,
        source_image_path: Path,
        output_dir: Path,
        strength: float = 0.75,
        on_progress: ProgressCallback | None = None,
    ) -> list[ArtifactReference]:
        """
        Execute an image-to-image workflow on ComfyUI.

        1. Upload source image to ComfyUI's input directory
        2. Build img2img workflow referencing the uploaded image
        3. Submit, wait, download, save
        """
        # Upload source image to ComfyUI
        uploaded_name = await self._client.upload_image(source_image_path)

        # Build workflow with reference to uploaded image
        workflow = self._builder.build_image_to_image(params, model_id, uploaded_name, strength)

        # Submit and wait
        prompt_id = await self._client.queue_workflow(workflow)

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