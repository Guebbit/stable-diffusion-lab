"""
ComfyUI video adapter — submits video generation workflows to ComfyUI server.

Implements VideoProvider protocol. Uses AnimateDiff or SVD workflow templates
for video generation via ComfyUI's node ecosystem.
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


class ComfyUIVideoAdapter:
    """Video generation via ComfyUI AnimateDiff/SVD workflow execution."""

    def __init__(self, client: ComfyUIClient, workflow_builder: WorkflowBuilder) -> None:
        self._client = client
        self._builder = workflow_builder

    async def generate(
        self,
        params: GenerationParams,
        model_id: str,
        output_dir: Path,
        source_image_path: Path | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> ArtifactReference:
        """
        Execute a video generation workflow on ComfyUI.

        Uses template-based workflows for video (AnimateDiff, SVD) since
        video workflows are too complex for programmatic construction.
        """
        # Select template based on whether we have a source image
        if source_image_path is not None:
            uploaded_name = await self._client.upload_image(source_image_path)
            template_name = "video_svd.json"
            overrides = {
                "__MODEL__": model_id,
                "__IMAGE__": uploaded_name,
                "__STEPS__": params.num_inference_steps,
                "__WIDTH__": params.width,
                "__HEIGHT__": params.height,
                "__SEED__": params.seed if params.seed is not None else -1,
            }
        else:
            template_name = "video_animatediff.json"
            overrides = {
                "__MODEL__": model_id,
                "__PROMPT__": params.prompt,
                "__NEGATIVE__": params.negative_prompt or "",
                "__STEPS__": params.num_inference_steps,
                "__CFG__": params.guidance_scale,
                "__WIDTH__": params.width,
                "__HEIGHT__": params.height,
                "__SEED__": params.seed if params.seed is not None else -1,
            }

        # Build workflow from template (falls back to basic if template missing)
        try:
            workflow = self._builder.build_from_template(template_name, overrides)
        except FileNotFoundError:
            logger.warning(
                "Video template %s not found, using basic txt2img as fallback",
                template_name,
            )
            workflow = self._builder.build_text_to_image(params, model_id)

        # Submit and wait
        prompt_id = await self._client.queue_workflow(workflow)

        def progress_handler(current: int, total: int) -> None:
            if on_progress:
                on_progress(
                    JobProgress(
                        job_id=uuid.UUID(int=0),
                        status="running",
                        progress_percent=int((current / max(total, 1)) * 100),
                        current_step=current,
                        total_steps=total,
                        message=f"Video frame generation (step {current}/{total})",
                    )
                )

        output_data = await self._client.wait_for_completion(
            prompt_id, on_progress=progress_handler, timeout=600.0
        )

        # Download video output
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_id = uuid.uuid4()
        filename = f"{artifact_id}.mp4"
        file_path = output_dir / filename

        # Find video output in ComfyUI response
        for node_output in output_data.values():
            if not isinstance(node_output, dict):
                continue
            # Video nodes output as "gifs" or "videos"
            videos = node_output.get("gifs", node_output.get("videos", []))
            if videos:
                video_info = videos[0]
                video_bytes = await self._client.download_output(
                    video_info.get("filename", ""),
                    video_info.get("subfolder", ""),
                )
                file_path.write_bytes(video_bytes)
                break

        return ArtifactReference(
            artifact_id=artifact_id,
            job_id=uuid.UUID(int=0),
            file_path=str(file_path),
            media_type="video/mp4",
            width=params.width,
            height=params.height,
            size_bytes=file_path.stat().st_size if file_path.exists() else 0,
        )
