"""Seed FLUX family — base model, LoRA, control adapters, IP-Adapter.

Revision ID: 002_seed_flux
Revises: 001_initial_schema
Create Date: 2026-06-15
"""

from __future__ import annotations

import uuid

from seed_helpers import seed_models, unseed_models

revision = "002_seed_flux"
down_revision = "001_initial_schema"
branch_labels = None
depends_on = None

MODELS = [
    # ── Base ──────────────────────────────────────────────────────────────
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000017"),
        "model_id": "black-forest-labs/FLUX.1-schnell",
        "name": "FLUX.1-schnell",
        "preferred_name": "FLUX.1 Schnell",
        "source": "huggingface",
        "family": "flux",
        "variant": "schnell",
        "model_type": "base_diffusion",
        "compatible_bases": [],
        "description": (
            "Distilled FLUX base model for fast text-to-image generation. "
            "Produces high-quality results in 4–8 steps, making it the best choice when speed matters "
            "or GPU resources are limited. Strong prompt fidelity and good typography with a fraction "
            "of the inference time of larger FLUX variants. "
            "Negative prompts have limited effect with FLUX and can usually be omitted."
        ),
        "short_description": "Fast distilled FLUX model for high-quality results in 4–8 steps",
        "tags": [
            "flux",
            "base",
            "text",
            "img2img",
            "fast",
            "distilled",
            "general",
            "prompt_fidelity",
            "realistic",
            "illustration",
            "typography",
        ],
        "source_url": "https://huggingface.co/black-forest-labs/FLUX.1-schnell",
        "version": "1.0",
        "capabilities": ["text_to_image", "image_to_image"],
        "total_size_bytes": 0,
        "download_size_bytes": 23800000000,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 8,
        "recommended_vram_max_gb": 16,
        "license": "apache-2.0",
        "base_model": "flux",
        "precision": "bf16",
        "requirements": {
            "recommended_resolution": "1024x1024",
            "recommended_steps": "4-8",
            "recommended_guidance_scale": "0.0",
        },
        "notes": (
            "Apache-2.0 license — no gating required. "
            "Use 4–8 steps with guidance_scale=0 for best results. "
            "Much faster than FLUX.1-dev with comparable quality on most prompts."
        ),
    },
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000018"),
        "model_id": "black-forest-labs/FLUX.1-dev",
        "name": "FLUX.1-dev",
        "preferred_name": "FLUX.1 Dev",
        "source": "huggingface",
        "family": "flux",
        "variant": "dev",
        "model_type": "base_diffusion",
        "compatible_bases": [],
        "description": (
            "Flagship FLUX base model for premium text-to-image and image-to-image generation. "
            "Best for high prompt fidelity, complex prompts, strong typography, realistic detail, "
            "and high-end general image synthesis. This is the quality-first FLUX model in the catalog. "
            "Use it when you want the best overall output quality and you have enough VRAM. "
            "Negative prompts have limited effect with FLUX and can usually be omitted."
        ),
        "short_description": "Premium all-purpose FLUX model for best prompt fidelity and image quality",
        "tags": [
            "flux",
            "base",
            "text",
            "img2img",
            "premium",
            "high_quality",
            "best_quality",
            "general",
            "prompt_fidelity",
            "realistic",
            "illustration",
            "typography",
            "flagship",
        ],
        "source_url": "https://huggingface.co/black-forest-labs/FLUX.1-dev",
        "version": "1.0",
        "capabilities": ["text_to_image", "image_to_image"],
        "total_size_bytes": 0,
        "download_size_bytes": 33800000000,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 12,
        "recommended_vram_max_gb": 24,
        "license": "flux-1-dev-non-commercial",
        "base_model": "flux",
        "precision": "bf16",
        "requirements": {
            "recommended_resolution": "1024x1024",
            "huggingface_gated": True,
        },
        "notes": (
            "Gated on HuggingFace — accept the FLUX.1-dev terms before downloading. "
            "Best used as the premium base model on high-end GPUs such as RTX 4090."
        ),
    },

    # ── LoRA ──────────────────────────────────────────────────────────────
    {
        "id": uuid.UUID("b0000000-0000-0000-0000-000000000001"),
        "model_id": "flux-dev-anime-lora",
        "name": "FLUX Anime Style LoRA",
        "preferred_name": "FLUX Anime LoRA",
        "source": "huggingface",
        "family": "lora",
        "variant": "flux-anime",
        "model_type": "lora",
        "compatible_bases": ["flux"],
        "description": (
            "Anime and manga style LoRA for FLUX. Use it to push FLUX outputs toward stylized "
            "characters, cleaner linework, cel shading, expressive faces, and vivid color palettes. "
            "Good for anime portraits, manga illustrations, stylized posters, and character art."
        ),
        "short_description": "Anime / manga style LoRA for FLUX character and illustration workflows",
        "tags": [
            "flux",
            "lora",
            "anime",
            "manga",
            "stylized",
            "illustration",
            "character",
            "lineart",
            "portrait",
            "style",
        ],
        "source_url": "https://huggingface.co/XLabs-AI/flux-lora-collection",
        "version": "1.0",
        "capabilities": ["text_to_image", "image_to_image"],
        "requirements": {
            "lora_strength_default": 0.6,
            "lora_strength_min": 0.4,
            "lora_strength_max": 0.8,
        },
        "notes": "Use when you want anime / manga stylization on top of FLUX.",
    },
    {
        "id": uuid.UUID("b0000000-0000-0000-0000-000000000002"),
        "model_id": "flux-dev-photoreal-lora",
        "name": "FLUX Photorealism LoRA",
        "preferred_name": "FLUX Realism LoRA",
        "source": "huggingface",
        "family": "lora",
        "variant": "flux-realism",
        "model_type": "lora",
        "compatible_bases": ["flux"],
        "description": (
            "Photorealism LoRA for FLUX. Use it to emphasize realistic skin texture, cinematic "
            "lighting, camera-like depth, natural materials, and polished portrait rendering. "
            "Best for portraits, editorial shots, lifestyle scenes, and realistic prompt work."
        ),
        "short_description": "Photorealism LoRA for FLUX portraits, cinematic scenes, and realism",
        "tags": [
            "flux",
            "lora",
            "realism",
            "photoreal",
            "portrait",
            "cinematic",
            "lighting",
            "photo",
            "style",
        ],
        "source_url": "https://huggingface.co/XLabs-AI/flux-RealismLora",
        "version": "1.0",
        "capabilities": ["text_to_image", "image_to_image"],
        "requirements": {
            "lora_strength_default": 0.5,
            "lora_strength_min": 0.3,
            "lora_strength_max": 0.8,
        },
        "notes": "Use for realistic and cinematic rendering on top of FLUX.",
    },
    {
        "id": uuid.UUID("b0000000-0000-0000-0000-000000000003"),
        "model_id": "flux-dev-lineart-lora",
        "name": "FLUX Lineart / Ink LoRA",
        "preferred_name": "FLUX Ink / Lineart",
        "source": "huggingface",
        "family": "lora",
        "variant": "flux-lineart",
        "model_type": "lora",
        "compatible_bases": ["flux"],
        "description": (
            "Lineart and ink style LoRA for FLUX. Best for comic-like rendering, cleaner contour "
            "definition, strong outlines, inked illustration, and stylized sketch polishing. "
            "Useful for inking sketches and turning rough forms into cleaner illustrated output."
        ),
        "short_description": "Ink / lineart LoRA for FLUX sketch, comic, and illustration workflows",
        "tags": [
            "flux",
            "lora",
            "lineart",
            "ink",
            "sketch",
            "sketch_to_ink",
            "comic",
            "illustration",
            "manga",
            "style",
        ],
        "source_url": "https://huggingface.co/search?q=flux+lineart+lora",
        "version": "1.0",
        "capabilities": ["text_to_image", "image_to_image", "sketch_to_ink"],
        "requirements": {
            "lora_strength_default": 0.7,
            "lora_strength_min": 0.5,
            "lora_strength_max": 0.9,
        },
        "notes": "Useful for sketch cleanup, comic-style rendering, and ink emphasis.",
    },
    {
        "id": uuid.UUID("b0000000-0000-0000-0000-000000000004"),
        "model_id": "flux-dev-nsfw-lora",
        "name": "FLUX NSFW Style LoRA",
        "preferred_name": "FLUX NSFW LoRA",
        "source": "huggingface",
        "family": "lora",
        "variant": "flux-nsfw",
        "model_type": "lora",
        "compatible_bases": ["flux"],
        "description": (
            "NSFW-oriented style LoRA for FLUX. Intended for adult-content workflows where the base "
            "model alone is too generic. Use it to steer anatomy, posing, lingerie / nude styling, "
            "and erotic composition depending on the paired checkpoint and prompt strategy."
        ),
        "short_description": "NSFW LoRA for adult-oriented FLUX prompt and character workflows",
        "tags": [
            "flux",
            "lora",
            "nsfw",
            "adult",
            "erotic",
            "character",
            "portrait",
            "style",
        ],
        "source_url": "https://huggingface.co/enhanceaiteam/Flux-Uncensored-V2",
        "version": "1.0",
        "capabilities": ["text_to_image", "image_to_image"],
        "requirements": {
            "lora_strength_default": 0.6,
            "lora_strength_min": 0.4,
            "lora_strength_max": 0.9,
        },
        "notes": "Catalog placeholder for adult-oriented FLUX styling workflows.",
    },

    # ── Control adapters ──────────────────────────────────────────────────
    {
        "id": uuid.UUID("b0000000-0000-0000-0000-000000000010"),
        "model_id": "flux-control-canny-adapter",
        "name": "FLUX Canny Control Adapter",
        "preferred_name": "FLUX Edge Control",
        "source": "huggingface",
        "family": "control_adapter",
        "variant": "canny",
        "model_type": "controlnet_like",
        "compatible_bases": ["flux"],
        "description": (
            "Canny / edge-guided control adapter for FLUX. Use it when you want strong shape, "
            "silhouette, and composition preservation from an existing image, sketch, or extracted "
            "edge map. Best for structure-preserving image-to-image edits."
        ),
        "short_description": "Canny / edge control for FLUX structure-preserving image editing",
        "tags": [
            "flux",
            "control",
            "canny",
            "edge",
            "structure",
            "composition",
            "img2img",
            "control_adapter",
        ],
        "source_url": "https://huggingface.co/search?q=flux+canny+control",
        "capabilities": ["image_to_image"],
        "notes": "Use when composition lock and edge adherence matter.",
    },
    {
        "id": uuid.UUID("b0000000-0000-0000-0000-000000000011"),
        "model_id": "flux-control-depth-adapter",
        "name": "FLUX Depth Control Adapter",
        "preferred_name": "FLUX Depth Control",
        "source": "huggingface",
        "family": "control_adapter",
        "variant": "depth",
        "model_type": "controlnet_like",
        "compatible_bases": ["flux"],
        "description": (
            "Depth-guided control adapter for FLUX. Use it to preserve perspective, foreground / "
            "background relationships, and 3D scene structure while still allowing style or content edits."
        ),
        "short_description": "Depth control for FLUX perspective-aware and structure-aware editing",
        "tags": [
            "flux",
            "control",
            "depth",
            "spatial",
            "perspective",
            "structure",
            "architecture",
            "landscape",
            "img2img",
            "control_adapter",
        ],
        "source_url": "https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Depth",
        "capabilities": ["image_to_image"],
        "notes": "Useful for interiors, architecture, landscapes, and spatial consistency.",
    },
    {
        "id": uuid.UUID("b0000000-0000-0000-0000-000000000012"),
        "model_id": "flux-control-openpose-adapter",
        "name": "FLUX OpenPose Control Adapter",
        "preferred_name": "FLUX Pose Control",
        "source": "huggingface",
        "family": "control_adapter",
        "variant": "openpose",
        "model_type": "controlnet_like",
        "compatible_bases": ["flux"],
        "description": (
            "Pose-guided control adapter for FLUX. Use it to keep or transfer human body pose from "
            "a reference image or pose skeleton. Best for character illustration, pose matching, and "
            "character consistency across scenes."
        ),
        "short_description": "Pose control for FLUX character posing and human-body guidance",
        "tags": [
            "flux",
            "control",
            "openpose",
            "pose",
            "character",
            "human",
            "body",
            "img2img",
            "control_adapter",
        ],
        "source_url": "https://huggingface.co/search?q=flux+openpose+control",
        "capabilities": ["image_to_image"],
        "notes": "Useful for pose transfer and character-focused workflows.",
    },
    {
        "id": uuid.UUID("b0000000-0000-0000-0000-000000000013"),
        "model_id": "flux-control-tile-adapter",
        "name": "FLUX Tile / Detail Control Adapter",
        "preferred_name": "FLUX Tile Control",
        "source": "huggingface",
        "family": "control_adapter",
        "variant": "tile",
        "model_type": "controlnet_like",
        "compatible_bases": ["flux"],
        "description": (
            "Tile / detail-preserving adapter for FLUX enhancement workflows. Useful for restoration, "
            "upscale-followup passes, and image enhancement where you want to recover surface detail "
            "while keeping the original scene recognizable."
        ),
        "short_description": "Tile/detail control for FLUX restoration and enhancement workflows",
        "tags": [
            "flux",
            "control",
            "tile",
            "detail",
            "restoration",
            "upscale",
            "enhancement",
            "img2img",
            "control_adapter",
        ],
        "source_url": "https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler",
        "capabilities": ["image_to_image", "upscale_image"],
        "notes": "Use after upscaling or for detail-preserving restoration passes.",
    },

    # ── IP-Adapter ────────────────────────────────────────────────────────
    {
        "id": uuid.UUID("b0000000-0000-0000-0000-000000000020"),
        "model_id": "flux-ip-adapter",
        "name": "FLUX IP-Adapter (Reference Image)",
        "preferred_name": "FLUX Reference Adapter",
        "source": "huggingface",
        "family": "ip_adapter",
        "variant": "flux-multi",
        "model_type": "ip_adapter",
        "compatible_bases": ["flux"],
        "description": (
            "Reference-image adapter for FLUX. Use it for style transfer, character consistency, "
            "palette borrowing, mood matching, and reference-driven composition. Best when you want "
            "the output to follow a reference image more closely than plain prompting allows."
        ),
        "short_description": "Reference-image adapter for FLUX style transfer and consistency",
        "tags": [
            "flux",
            "ip_adapter",
            "reference",
            "style_transfer",
            "consistency",
            "character",
            "image_prompt",
            "img2img",
        ],
        "source_url": "https://huggingface.co/search?q=flux+ip+adapter",
        "capabilities": ["image_to_image"],
        "notes": "Useful for style transfer, reference-driven generation, and consistency workflows.",
    },
]


def upgrade() -> None:
    seed_models(MODELS)


def downgrade() -> None:
    unseed_models(MODELS)