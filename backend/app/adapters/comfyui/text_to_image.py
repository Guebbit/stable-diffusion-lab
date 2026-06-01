"""
ComfyUI text-to-image adapter — submits txt2img workflows to ComfyUI server.

Implements TextToImageProvider protocol. Builds a workflow via WorkflowBuilder,
submits to ComfyUI, waits for completion via WebSocket, downloads output images.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from app.adapters.comfyui.client import ComfyUIClient
from app.adapters.comfyui.workflow_builder import WorkflowBuilder
from app.domain.protocols import ProgressCallback
from app.domain.value_objects import ArtifactReference, GenerationParams, JobProgress


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
        def progress_handler(current: int, total: int) -> None:
            if on_progress:
                on_progress(
                    JobProgress(
                        job_id=uuid.UUID(int=0),
                        status="running",
                        progress_percent=int((current / max(total, 1)) * 100),
                        current_step=current,
                        total_steps=total,
                    )
                )

        output_data = await self._client.wait_for_completion(
            prompt_id, on_progress=progress_handler
        )

        # Download and save output images
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts: list[ArtifactReference] = []

        # ComfyUI output format: {node_id: {"images": [{"filename": ..., "subfolder": ...}]}}
        for node_output in output_data.values():
            images = node_output.get("images", []) if isinstance(node_output, dict) else []
            for image_info in images:
                filename = image_info.get("filename", "")
                subfolder = image_info.get("subfolder", "")

                # Download image bytes from ComfyUI
                image_bytes = await self._client.download_output(filename, subfolder)

                # Save locally
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
