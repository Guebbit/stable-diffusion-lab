"""
ComfyUI workflow builder — translates GenerationParams into ComfyUI node-graph JSON.

ComfyUI workflows are JSON objects where each key is a node ID and each value
describes the node type, inputs, and connections. This builder constructs
these workflow graphs programmatically from our clean domain model.

For complex workflows, pre-built templates (JSON files) can be loaded and
parameterized at runtime via build_from_template().
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.domain.value_objects import GenerationParams


logger = logging.getLogger(__name__)

# Directory containing pre-built workflow template JSON files
TEMPLATES_DIR = Path(__file__).parent.parent.parent.parent / "workflows" / "comfyui"


class WorkflowBuilder:
    """
    Constructs ComfyUI workflow JSON from GenerationParams.

    Two modes of operation:
    1. Programmatic: build_text_to_image() / build_image_to_image() construct
       minimal workflows from code (for simple cases).
    2. Template-based: build_from_template() loads a pre-built JSON and
       substitutes placeholder values (for complex multi-node workflows).

    Node ID conventions:
    - "1" = CheckpointLoader
    - "2" = CLIPTextEncode (positive prompt)
    - "3" = CLIPTextEncode (negative prompt)
    - "4" = KSampler
    - "5" = VAEDecode
    - "6" = SaveImage / output node
    """

    def build_text_to_image(
        self,
        params: GenerationParams,
        model_id: str,
    ) -> dict[str, Any]:
        """
        Build a basic text-to-image workflow (checkpoint → CLIP → sampler → VAE → save).

        Args:
            params: Generation parameters.
            model_id: Checkpoint filename as known by ComfyUI.

        Returns:
            ComfyUI workflow JSON ready to submit via /prompt.
        """
        seed = params.seed if params.seed is not None else -1  # -1 = random in ComfyUI

        return {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": model_id},
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": params.prompt,
                    "clip": ["1", 1],  # Connect to checkpoint CLIP output
                },
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": params.negative_prompt or "",
                    "clip": ["1", 1],
                },
            },
            "4": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],  # Connect to checkpoint model output
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["7", 0],
                    "seed": seed,
                    "steps": params.num_inference_steps,
                    "cfg": params.guidance_scale,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                },
            },
            "5": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["4", 0],
                    "vae": ["1", 2],  # Connect to checkpoint VAE output
                },
            },
            "6": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["5", 0],
                    "filename_prefix": "ailab",
                },
            },
            "7": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": params.width,
                    "height": params.height,
                    "batch_size": params.num_images,
                },
            },
        }

    def build_image_to_image(
        self,
        params: GenerationParams,
        model_id: str,
        source_image_name: str,
        strength: float = 0.75,
    ) -> dict[str, Any]:
        """
        Build a basic image-to-image workflow (load image → encode → sample → decode).

        Args:
            params: Generation parameters.
            model_id: Checkpoint filename.
            source_image_name: Filename of uploaded source image in ComfyUI.
            strength: Denoising strength (maps to denoise parameter).

        Returns:
            ComfyUI workflow JSON.
        """
        seed = params.seed if params.seed is not None else -1

        return {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": model_id},
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": params.prompt,
                    "clip": ["1", 1],
                },
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": params.negative_prompt or "",
                    "clip": ["1", 1],
                },
            },
            "4": {
                "class_type": "LoadImage",
                "inputs": {"image": source_image_name},
            },
            "5": {
                "class_type": "VAEEncode",
                "inputs": {
                    "pixels": ["4", 0],
                    "vae": ["1", 2],
                },
            },
            "6": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["5", 0],
                    "seed": seed,
                    "steps": params.num_inference_steps,
                    "cfg": params.guidance_scale,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": strength,
                },
            },
            "7": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["6", 0],
                    "vae": ["1", 2],
                },
            },
            "8": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["7", 0],
                    "filename_prefix": "ailab",
                },
            },
        }

    def build_from_template(
        self,
        template_name: str,
        overrides: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Load a pre-built workflow template and apply parameter overrides.

        Templates are JSON files with placeholder values (e.g., "__PROMPT__")
        that get replaced with actual values from the overrides dict.

        Args:
            template_name: Filename in the workflows/comfyui/ directory (e.g., "txt2img_lora.json").
            overrides: Dict of placeholder → actual value substitutions.

        Returns:
            Parameterized workflow JSON.

        Raises:
            FileNotFoundError: If the template file doesn't exist.
        """
        template_path = TEMPLATES_DIR / template_name
        if not template_path.exists():
            raise FileNotFoundError(f"Workflow template not found: {template_path}")

        # Load template as string for placeholder replacement
        template_str = template_path.read_text()

        # Replace placeholders
        for placeholder, value in overrides.items():
            if isinstance(value, str):
                template_str = template_str.replace(f'"{placeholder}"', json.dumps(value))
            else:
                template_str = template_str.replace(f'"{placeholder}"', json.dumps(value))

        return json.loads(template_str)
