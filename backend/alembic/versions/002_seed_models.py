"""Seed the model catalog with all supported models.

Covers every model type: base diffusion, upscalers, face-restore, ControlNet,
T2I Adapter, LoRA, IP-Adapter, VAE, and vision-language models.

Tags are snake_case descriptors. Model type and base model are stored in the
dedicated model_type and base_model / compatible_bases columns, not in tags.

Revision ID: 002_seed_models
Revises: 001_initial_schema
Create Date: 2026-06-15
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import sqlalchemy as sa
from sqlalchemy import text
from alembic import op

revision = "002_seed_models"
down_revision = "001_initial_schema"
branch_labels = None
depends_on = None

# ---------------------------------------------------------------------------
# All models
# ---------------------------------------------------------------------------
MODELS = [

    # ── Base diffusion — FLUX ─────────────────────────────────────────────
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000017"),
        "model_id": "black-forest-labs/FLUX.1-dev",
        "name": "FLUX.1-dev",
        "preferred_name": "FLUX.1 Dev",
        "source": "huggingface",
        "family": "flux",
        "variant": "dev",
        "model_type": "base_diffusion",
        "compatible_bases": [],
        "description": (
            "Best for: flagship text-to-image generation and high-quality image-to-image work. "
            "Strengths: state-of-the-art prompt understanding via the T5-XXL text encoder, "
            "excellent coherence, fine detail, strong text rendering, and a significant step up "
            "from SDXL in following complex or long prompts. Use it when quality is the priority "
            "and VRAM budget allows. "
            "Note: negative prompts have very limited effect with this architecture — "
            "leave them empty or omit them. Not a helper model."
        ),
        "tags": [
            "flux", "diffusion", "all_arounder", "flagship", "high_quality",
            "transformer_architecture", "general_purpose",
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
            "Gated on HuggingFace — accept the FLUX.1-dev terms at the model page before "
            "downloading. Uses bf16 and a DiT transformer (no UNet); negative prompts have "
            "minimal effect and can be left empty. Minimum 12 GB VRAM; 16 GB+ recommended."
        ),
    },

    # ── Base diffusion — SDXL ─────────────────────────────────────────────
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000001"),
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "name": "SDXL 1.0",
        "preferred_name": "SDXL 1.0",
        "source": "huggingface",
        "family": "stable_diffusion_xl",
        "variant": "1.0",
        "model_type": "base_diffusion",
        "compatible_bases": [],
        "description": (
            "Best for: general text-to-image and generic image-to-image work. Strengths: "
            "balanced quality, good prompt following, and strong composition at 1024 resolution. "
            "Use it when you want one neutral all-around model. Not a helper model."
        ),
        "tags": [
            "diffusion", "all_arounder", "general_purpose", "neutral_style",
        ],
        "source_url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0",
        "version": "1.0",
        "capabilities": ["text_to_image", "image_to_image"],
        "total_size_bytes": 0,
        "download_size_bytes": 6930000000,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 8,
        "recommended_vram_max_gb": 16,
        "license": "openrail++",
        "base_model": "sdxl",
        "precision": "fp16",
        "requirements": {"recommended_resolution": "1024x1024"},
        "notes": "Base SDXL model; best default all-arounder.",
    },
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000004"),
        "model_id": "huggingshu/realvisxl-v5",
        "name": "RealVisXL V5",
        "preferred_name": "RealVisXL V5",
        "source": "huggingface",
        "family": "custom",
        "variant": "realvisxl-v5",
        "model_type": "base_diffusion",
        "compatible_bases": [],
        "description": (
            "Best for: realistic portraits, fashion shots, product images, and polished "
            "photorealistic scenes. Strengths: camera-like lighting, cleaner skin and texture, "
            "and strong realism without much prompt tuning. Not a helper model."
        ),
        "tags": [
            "diffusion", "realistic", "photorealistic", "portrait", "product_style",
        ],
        "source_url": "https://huggingface.co/huggingshu/realvisxl-v5",
        "version": "5.0",
        "capabilities": ["text_to_image"],
        "total_size_bytes": 0,
        "download_size_bytes": 6900000000,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 8,
        "recommended_vram_max_gb": 16,
        "license": "",
        "base_model": "sdxl",
        "precision": "fp16",
        "requirements": {"recommended_resolution": "1024x1024"},
        "notes": "Main realistic specialist for text-to-image.",
    },
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000005"),
        "model_id": "multimodalart/super-novelai-4.0-xl",
        "name": "Super NovelAI 4.0 XL",
        "preferred_name": "Super NovelAI 4.0 XL",
        "source": "huggingface",
        "family": "custom",
        "variant": "novelai-xl",
        "model_type": "base_diffusion",
        "compatible_bases": [],
        "description": (
            "Best for: anime characters, fantasy scenes, and colorful stylized illustrations. "
            "Strengths: strong illustrative bias, expressive character rendering, and vibrant "
            "color without much prompt tuning. Not a helper model."
        ),
        "tags": [
            "diffusion", "anime", "illustration", "anime_style", "stylized", "character_art",
        ],
        "source_url": "https://huggingface.co/multimodalart/super-novelai-4.0-xl",
        "version": "4.0",
        "capabilities": ["text_to_image", "image_to_image"],
        "total_size_bytes": 0,
        "download_size_bytes": 6900000000,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 8,
        "recommended_vram_max_gb": 16,
        "license": "",
        "base_model": "sdxl",
        "precision": "fp16",
        "requirements": {"recommended_resolution": "1024x1024"},
        "notes": "Main anime and stylized specialist.",
    },

    # ── Base diffusion — SD 2.x ───────────────────────────────────────────
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000002"),
        "model_id": "stabilityai/stable-diffusion-2-1",
        "name": "SD 2.1",
        "preferred_name": "SD 2.1",
        "source": "huggingface",
        "family": "stable_diffusion_2",
        "variant": "2.1",
        "model_type": "base_diffusion",
        "compatible_bases": [],
        "description": (
            "Best for: generic image-to-image and lighter general generation. Strengths: "
            "decent structure, lower resource needs than SDXL, and flexible use in broad "
            "workflows. Not a helper model."
        ),
        "tags": [
            "diffusion", "general_purpose", "generic_transform", "balanced",
        ],
        "source_url": "https://huggingface.co/stabilityai/stable-diffusion-2-1",
        "version": "2.1",
        "capabilities": ["text_to_image", "image_to_image"],
        "total_size_bytes": 0,
        "download_size_bytes": 5510000000,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 6,
        "recommended_vram_max_gb": 12,
        "license": "openrail++",
        "base_model": "sd2.1",
        "precision": "fp16",
        "requirements": {"recommended_resolution": "768x768"},
        "notes": "Useful generic model for transformations and experiments.",
    },

    # ── Base diffusion — SD 1.5 ───────────────────────────────────────────
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000003"),
        "model_id": "runwayml/stable-diffusion-v1-5",
        "name": "SD 1.5 Test Model",
        "preferred_name": "SD 1.5 Test Model",
        "source": "huggingface",
        "family": "stable_diffusion_1",
        "variant": "1.5",
        "model_type": "base_diffusion",
        "compatible_bases": [],
        "description": (
            "Best for: testing your platform, not final quality. Strengths: lighter weight, "
            "broad compatibility, and fast checks for routes, queues, prompts, and UI behavior. "
            "Not a helper model."
        ),
        "tags": [
            "diffusion", "test_model", "lightweight", "dev_testing",
        ],
        "source_url": "https://huggingface.co/runwayml/stable-diffusion-v1-5",
        "version": "1.5",
        "capabilities": ["text_to_image", "image_to_image"],
        "total_size_bytes": 0,
        "download_size_bytes": 4260000000,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 4,
        "recommended_vram_max_gb": 10,
        "license": "creativeml-openrail-m",
        "base_model": "sd1.5",
        "precision": "fp16",
        "requirements": {"recommended_resolution": "512x512"},
        "notes": "Designated test model only.",
    },
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000006"),
        "model_id": "lambdalabs/sd-image-variations-diffusers",
        "name": "Stable Image Variations",
        "preferred_name": "Stable Image Variations",
        "source": "huggingface",
        "family": "stable_diffusion_1",
        "variant": "image_variations",
        "model_type": "base_diffusion",
        "compatible_bases": [],
        "description": (
            "Best for: making alternate versions of an existing image. Strengths: keeps the "
            "overall idea and identity better than generic image-to-image while still producing "
            "new outputs. Not a helper model."
        ),
        "tags": [
            "diffusion", "variation", "identity_preserving", "concept_variation",
        ],
        "source_url": "https://huggingface.co/lambdalabs/sd-image-variations-diffusers",
        "version": "2.0",
        "capabilities": ["image_to_image"],
        "total_size_bytes": 0,
        "download_size_bytes": 4200000000,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 6,
        "recommended_vram_max_gb": 12,
        "license": "creativeml-openrail-m",
        "base_model": "sd1.5",
        "precision": "fp16",
        "requirements": {"input_type": "image"},
        "notes": "Best for variation-style image-to-image behavior.",
    },
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000007"),
        "model_id": "timbrooks/instruct-pix2pix",
        "name": "InstructPix2Pix",
        "preferred_name": "InstructPix2Pix",
        "source": "huggingface",
        "family": "stable_diffusion_edit",
        "variant": "pix2pix",
        "model_type": "base_diffusion",
        "compatible_bases": [],
        "description": (
            "Best for: prompt-based image edits like recoloring, style changes, mood changes, "
            "and object tweaks. Strengths: follows edit instructions directly. Not a helper model."
        ),
        "tags": [
            "diffusion", "instruction_editing", "edit_by_text", "style_transfer", "global_edit",
        ],
        "source_url": "https://huggingface.co/timbrooks/instruct-pix2pix",
        "version": "1.0",
        "capabilities": ["image_to_image", "recolor_image"],
        "total_size_bytes": 0,
        "download_size_bytes": 4900000000,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 6,
        "recommended_vram_max_gb": 12,
        "license": "openrail",
        "base_model": "sd1.5",
        "precision": "fp16",
        "requirements": {"input_type": "image"},
        "notes": "Primary recolor and instruction-based editing model.",
    },

    # ── Upscaler ──────────────────────────────────────────────────────────
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000008"),
        "model_id": "stabilityai/stable-diffusion-x4-upscaler",
        "name": "Stable Diffusion x4 Upscaler",
        "preferred_name": "SD x4 Upscaler",
        "source": "huggingface",
        "family": "stable_diffusion_upscaler",
        "variant": "x4",
        "model_type": "upscaler",
        "compatible_bases": ["sd2"],
        "description": (
            "Dedicated 4× upscaler based on SD 2.0. Takes a small low-res image and "
            "generates a high-resolution output with recovered texture and sharpness. "
            "Accepts an optional prompt to guide the detail synthesis. "
            "Appears only on the /image-upscale page — not a general generation model."
        ),
        "tags": [
            "dedicated_upscaler", "diffusion_upscaler", "resolution_enhancement",
            "detail_synthesis", "texture_recovery",
        ],
        "source_url": "https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler",
        "version": "x4",
        "capabilities": ["upscale_image"],
        "total_size_bytes": 0,
        "download_size_bytes": 5100000000,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 8,
        "recommended_vram_max_gb": 16,
        "license": "openrail++",
        "base_model": "sd2-upscaler",
        "precision": "fp16",
        "requirements": {"input_type": "image", "pipeline": "StableDiffusionUpscalePipeline"},
        "notes": "Primary dedicated upscaler. Supports tiled processing for large inputs.",
    },

    # ── Vision-language ───────────────────────────────────────────────────
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000009"),
        "model_id": "Salesforce/blip-image-captioning-large",
        "name": "BLIP Image Captioning Large",
        "preferred_name": "BLIP Caption Large",
        "source": "huggingface",
        "family": "vision_language",
        "variant": "caption-large",
        "model_type": "vision_language",
        "compatible_bases": [],
        "description": (
            "Best for: describing what is in an image in plain language. Strengths: solid "
            "caption quality and easy use for alt text, summaries, and scene descriptions. "
            "Not a helper model."
        ),
        "tags": [
            "vision_language", "captioning", "image_description", "scene_summary",
        ],
        "source_url": "https://huggingface.co/Salesforce/blip-image-captioning-large",
        "version": "large",
        "capabilities": ["image_description"],
        "total_size_bytes": 0,
        "download_size_bytes": 990000000,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 4,
        "recommended_vram_max_gb": 8,
        "license": "bsd-3-clause",
        "base_model": "blip",
        "precision": "fp16",
        "requirements": {"input_type": "image"},
        "notes": "Primary describe_image model.",
    },

    # ── ControlNet — SD 1.5 ───────────────────────────────────────────────
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000010"),
        "model_id": "lllyasviel/control_v11f1p_sd15_depth",
        "name": "ControlNet Depth SD1.5",
        "preferred_name": "ControlNet Depth SD1.5",
        "source": "huggingface",
        "family": "controlnet",
        "variant": "depth-sd15",
        "model_type": "controlnet",
        "compatible_bases": ["sd1.5"],
        "description": (
            "Best for: structure-aware edits where depth and spatial layout matter. "
            "This is a helper model. Strengths: preserves foreground-background relationships "
            "and scene geometry. Best paired with SD 1.5 Test Model."
        ),
        "tags": [
            "controlnet", "adapter", "helper",
            "depth_guided", "spatial_control", "structure_preserving",
        ],
        "source_url": "https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth",
        "version": "1.1",
        "capabilities": ["image_to_image", "recolor_image"],
        "total_size_bytes": 0,
        "download_size_bytes": 1450000000,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 6,
        "recommended_vram_max_gb": 12,
        "license": "openrail",
        "base_model": "sd1.5",
        "precision": "fp16",
        "requirements": {"requires_base_model": "runwayml/stable-diffusion-v1-5"},
        "notes": "Helper model; best paired with an SD1.5 base pipeline.",
    },
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000011"),
        "model_id": "lllyasviel/control_v11p_sd15_lineart",
        "name": "ControlNet Lineart SD1.5",
        "preferred_name": "ControlNet Lineart SD1.5",
        "source": "huggingface",
        "family": "controlnet",
        "variant": "lineart-sd15",
        "model_type": "controlnet",
        "compatible_bases": ["sd1.5"],
        "description": (
            "Best for: sketch-based and line-based workflows. This is a helper model. "
            "Strengths: follows outlines well and keeps drawing structure. "
            "Best paired with SD 1.5 Test Model for sketch_to_ink workflows."
        ),
        "tags": [
            "controlnet", "adapter", "helper",
            "lineart_guided", "illustration_control", "structure_preserving",
        ],
        "source_url": "https://huggingface.co/lllyasviel/control_v11p_sd15_lineart",
        "version": "1.0",
        "capabilities": ["image_to_image", "sketch_to_ink"],
        "total_size_bytes": 0,
        "download_size_bytes": 1450000000,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 6,
        "recommended_vram_max_gb": 12,
        "license": "openrail",
        "base_model": "sd1.5",
        "precision": "fp16",
        "requirements": {"requires_base_model": "runwayml/stable-diffusion-v1-5"},
        "notes": "Helper model; primary SD1.5 lineart/sketch ControlNet.",
    },
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000012"),
        "model_id": "lllyasviel/control_v11p_sd15_inpaint",
        "name": "ControlNet Inpaint SD1.5",
        "preferred_name": "ControlNet Inpaint SD1.5",
        "source": "huggingface",
        "family": "controlnet",
        "variant": "inpaint-sd15",
        "model_type": "controlnet",
        "compatible_bases": ["sd1.5"],
        "description": (
            "Best for: masked edits where only part of the image should change. "
            "This is a helper model. Strengths: localized control and safer partial recoloring. "
            "Best paired with SD 1.5 Test Model."
        ),
        "tags": [
            "controlnet", "adapter", "helper",
            "inpainting", "masked_edit", "localized_edit", "partial_recolor",
        ],
        "source_url": "https://huggingface.co/lllyasviel/control_v11p_sd15_inpaint",
        "version": "1.1",
        "capabilities": ["image_to_image", "recolor_image"],
        "total_size_bytes": 0,
        "download_size_bytes": 1450000000,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 6,
        "recommended_vram_max_gb": 12,
        "license": "openrail",
        "base_model": "sd1.5",
        "precision": "fp16",
        "requirements": {"requires_base_model": "runwayml/stable-diffusion-v1-5", "input_type": "image+mask"},
        "notes": "Helper model for localized edits and partial recoloring.",
    },
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000013"),
        "model_id": "lllyasviel/control_v11f1e_sd15_tile",
        "name": "ControlNet Tile SD1.5",
        "preferred_name": "ControlNet Tile SD1.5",
        "source": "huggingface",
        "family": "controlnet",
        "variant": "tile-sd15",
        "model_type": "controlnet",
        "compatible_bases": ["sd1.5"],
        "description": (
            "Best for: detail enhancement and tiled restoration workflows. "
            "This is a helper model. Strengths: helps recover texture and local detail "
            "while keeping the overall image recognizable."
        ),
        "tags": [
            "controlnet", "adapter", "helper",
            "detail_restoration", "texture_enhancement", "tile_control",
        ],
        "source_url": "https://huggingface.co/lllyasviel/control_v11f1e_sd15_tile",
        "version": "1.1",
        "capabilities": ["image_to_image"],
        "total_size_bytes": 0,
        "download_size_bytes": 1450000000,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 6,
        "recommended_vram_max_gb": 12,
        "license": "openrail",
        "base_model": "sd1.5",
        "precision": "fp16",
        "requirements": {"requires_base_model": "runwayml/stable-diffusion-v1-5"},
        "notes": "Helper model for enhancement and restoration-style upscale flows.",
    },
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000021"),
        "model_id": "lllyasviel/control_v11p_sd15_openpose",
        "name": "ControlNet OpenPose SD1.5",
        "preferred_name": "OpenPose ControlNet SD1.5",
        "source": "huggingface",
        "family": "controlnet",
        "variant": "openpose-sd15",
        "model_type": "controlnet",
        "compatible_bases": ["sd1.5"],
        "description": (
            "Best for: human pose transfer and character posing workflows. "
            "This is a helper model. Strengths: extracts skeleton/joint positions from a "
            "reference image and forces the output to match. Best paired with SD 1.5 Test Model."
        ),
        "tags": [
            "controlnet", "adapter", "helper",
            "pose_guided", "character_posing", "structure_preserving",
        ],
        "source_url": "https://huggingface.co/lllyasviel/control_v11p_sd15_openpose",
        "version": "1.1",
        "capabilities": ["image_to_image"],
        "total_size_bytes": 0,
        "download_size_bytes": 1450000000,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 6,
        "recommended_vram_max_gb": 12,
        "license": "openrail",
        "base_model": "sd1.5",
        "precision": "fp16",
        "requirements": {"requires_base_model": "runwayml/stable-diffusion-v1-5"},
        "notes": "Primary SD1.5 pose-guidance ControlNet.",
    },

    # ── ControlNet — SDXL ────────────────────────────────────────────────
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000020"),
        "model_id": "diffusers/controlnet-canny-sdxl-1.0",
        "name": "ControlNet Canny SDXL",
        "preferred_name": "Canny ControlNet SDXL",
        "source": "huggingface",
        "family": "controlnet",
        "variant": "canny-sdxl",
        "model_type": "controlnet",
        "compatible_bases": ["sdxl"],
        "description": (
            "Best for: edge-guided and structure-preserving generation at SDXL quality. "
            "This is a helper model — pair it with SDXL 1.0 or a compatible SDXL checkpoint. "
            "Strengths: tight edge adherence and strong composition lock."
        ),
        "tags": [
            "controlnet", "adapter", "helper",
            "edge_guided", "structure_preserving", "layout_control", "canny",
        ],
        "source_url": "https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0",
        "version": "1.0",
        "capabilities": ["image_to_image", "recolor_image"],
        "total_size_bytes": 0,
        "download_size_bytes": 2500000000,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 8,
        "recommended_vram_max_gb": 16,
        "license": "openrail",
        "base_model": "sdxl",
        "precision": "fp16",
        "requirements": {"requires_base_model": "stabilityai/stable-diffusion-xl-base-1.0"},
        "notes": "SDXL-native Canny ControlNet; best paired with SDXL 1.0.",
    },

    # ── T2I Adapters — SDXL ──────────────────────────────────────────────
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000014"),
        "model_id": "TencentARC/t2i-adapter-sketch-sdxl-1.0",
        "name": "T2I Adapter Sketch SDXL",
        "preferred_name": "Sketch SDXL Adapter",
        "source": "huggingface",
        "family": "t2i_adapter",
        "variant": "sketch-sdxl",
        "model_type": "t2i_adapter",
        "compatible_bases": ["sdxl"],
        "description": (
            "Best for: turning rough sketches into cleaner rendered images while keeping the "
            "sketch structure. This is a helper model. Strengths: strong sketch guidance and "
            "better shape retention than generic image-to-image. Best paired with SDXL 1.0."
        ),
        "tags": [
            "adapter", "helper",
            "sketch_guided", "shape_preserving", "illustration_control", "render_from_sketch",
        ],
        "source_url": "https://huggingface.co/TencentARC/t2i-adapter-sketch-sdxl-1.0",
        "version": "1.0",
        "capabilities": ["image_to_image", "sketch_to_ink"],
        "total_size_bytes": 0,
        "download_size_bytes": 300000000,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 8,
        "recommended_vram_max_gb": 16,
        "license": "apache-2.0",
        "base_model": "sdxl",
        "precision": "fp16",
        "requirements": {"requires_base_model": "stabilityai/stable-diffusion-xl-base-1.0", "input_type": "sketch"},
        "notes": "Primary SDXL helper for sketch-guided workflows.",
    },
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000015"),
        "model_id": "TencentARC/t2i-adapter-canny-sdxl-1.0",
        "name": "T2I Adapter Canny SDXL",
        "preferred_name": "Canny SDXL Adapter",
        "source": "huggingface",
        "family": "t2i_adapter",
        "variant": "canny-sdxl",
        "model_type": "t2i_adapter",
        "compatible_bases": ["sdxl"],
        "description": (
            "Best for: controlled generation from edge maps and strong shape guidance. "
            "This is a helper model. Strengths: follows object boundaries and layout closely. "
            "Best paired with SDXL 1.0."
        ),
        "tags": [
            "adapter", "helper",
            "edge_guided", "structure_preserving", "layout_control", "canny",
        ],
        "source_url": "https://huggingface.co/TencentARC/t2i-adapter-canny-sdxl-1.0",
        "version": "1.0",
        "capabilities": ["image_to_image", "recolor_image"],
        "total_size_bytes": 0,
        "download_size_bytes": 300000000,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 8,
        "recommended_vram_max_gb": 16,
        "license": "apache-2.0",
        "base_model": "sdxl",
        "precision": "fp16",
        "requirements": {"requires_base_model": "stabilityai/stable-diffusion-xl-base-1.0", "input_type": "edge-map"},
        "notes": "Primary SDXL helper for edge-guided controlled edits.",
    },

    # ── LoRA ─────────────────────────────────────────────────────────────
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000022"),
        "model_id": "nerijs/pixel-art-xl",
        "name": "Pixel Art XL LoRA",
        "preferred_name": "Pixel Art XL",
        "source": "huggingface",
        "family": "lora",
        "variant": "pixel-art-sdxl",
        "model_type": "lora",
        "compatible_bases": ["sdxl"],
        "description": (
            "SDXL LoRA that steers output toward pixel-art aesthetics: chunky pixels, "
            "limited palette, retro-game look. Load on top of SDXL 1.0 with strength 0.6–1.0. "
            "Lower strength blends pixel style with photorealism; higher goes full retro. "
            "No effect without a compatible base loaded."
        ),
        "tags": [
            "lora", "adapter", "helper",
            "style_pixel_art", "style_retro", "style_gaming",
        ],
        "source_url": "https://huggingface.co/nerijs/pixel-art-xl",
        "version": "1.0",
        "capabilities": ["text_to_image", "image_to_image"],
        "total_size_bytes": 0,
        "download_size_bytes": 150000000,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 8,
        "recommended_vram_max_gb": 16,
        "license": "creativeml-openrail-m",
        "base_model": "sdxl",
        "precision": "fp16",
        "requirements": {
            "requires_base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "lora_strength_default": 0.8,
        "lora_strength_min": 0.0,
        "lora_strength_max": 1.5,
        },
        "notes": "Style LoRA. Works best at strength 0.7–1.0. Trigger word: pixel art.",
    },
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000023"),
        "model_id": "latent-consistency/lcm-lora-sdxl",
        "name": "LCM LoRA SDXL",
        "preferred_name": "LCM LoRA SDXL",
        "source": "huggingface",
        "family": "lora",
        "variant": "lcm-sdxl",
        "model_type": "lora",
        "compatible_bases": ["sdxl"],
        "description": (
            "Speed LoRA for SDXL: 4–8 steps instead of 25–50 with the LCM scheduler. "
            "Does NOT change visual style — only inference speed. Use CFG 1–2. "
            "No effect without a compatible SDXL base loaded."
        ),
        "tags": [
            "lora", "adapter", "helper",
            "speed", "lcm", "few_step",
        ],
        "source_url": "https://huggingface.co/latent-consistency/lcm-lora-sdxl",
        "version": "1.0",
        "capabilities": ["text_to_image", "image_to_image"],
        "total_size_bytes": 0,
        "download_size_bytes": 200000000,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 8,
        "recommended_vram_max_gb": 16,
        "license": "mit",
        "base_model": "sdxl",
        "precision": "fp16",
        "requirements": {
            "requires_base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "scheduler": "LCMScheduler",
        "recommended_steps": "4-8",
        "recommended_guidance_scale": "1.0-2.0",
        "lora_strength_default": 1.0,
        },
        "notes": "Speed LoRA — use with LCMScheduler, CFG 1–2, 4–8 steps.",
    },
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000024"),
        "model_id": "latent-consistency/lcm-lora-sdv1-5",
        "name": "LCM LoRA SD1.5",
        "preferred_name": "LCM LoRA SD1.5",
        "source": "huggingface",
        "family": "lora",
        "variant": "lcm-sd15",
        "model_type": "lora",
        "compatible_bases": ["sd1.5"],
        "description": (
            "Speed LoRA for SD 1.5: 4–8 steps with the LCM scheduler and CFG 1–2. "
            "Great for fast previews before committing to a slower full-quality render. "
            "No effect without a compatible SD1.5 base loaded."
        ),
        "tags": [
            "lora", "adapter", "helper",
            "speed", "lcm", "few_step",
        ],
        "source_url": "https://huggingface.co/latent-consistency/lcm-lora-sdv1-5",
        "version": "1.0",
        "capabilities": ["text_to_image", "image_to_image"],
        "total_size_bytes": 0,
        "download_size_bytes": 200000000,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 4,
        "recommended_vram_max_gb": 10,
        "license": "mit",
        "base_model": "sd1.5",
        "precision": "fp16",
        "requirements": {
            "requires_base_model": "runwayml/stable-diffusion-v1-5",
        "scheduler": "LCMScheduler",
        "recommended_steps": "4-8",
        "recommended_guidance_scale": "1.0-2.0",
        "lora_strength_default": 1.0,
        },
        "notes": "Speed LoRA for SD1.5 — use with LCMScheduler, CFG 1–2, 4–8 steps.",
    },

    # ── IP-Adapter ───────────────────────────────────────────────────────
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000025"),
        "model_id": "h94/IP-Adapter",
        "name": "IP-Adapter",
        "preferred_name": "IP-Adapter",
        "source": "huggingface",
        "family": "ip_adapter",
        "variant": "multi",
        "model_type": "ip_adapter",
        "compatible_bases": ["sd1.5", "sdxl"],
        "description": (
            "Conditions generation on a reference image instead of (or alongside) a text prompt. "
            "The reference image drives composition, style, and color. "
            "Repo contains variants for both SD1.5 and SDXL. "
            "Use it when you want to 'paint in the style of' a reference."
        ),
        "tags": [
            "ip_adapter", "adapter", "helper",
            "image_prompt", "style_transfer", "reference_image",
        ],
        "source_url": "https://huggingface.co/h94/IP-Adapter",
        "version": "1.0",
        "capabilities": ["image_to_image"],
        "total_size_bytes": 0,
        "download_size_bytes": 2500000000,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 6,
        "recommended_vram_max_gb": 16,
        "license": "apache-2.0",
        "base_model": "sd1.5",
        "precision": "fp16",
        "requirements": {
            "input_type": "image",
        "variants": {
                "sd15": "models/ip-adapter_sd15.bin",
        "sd15_plus": "models/ip-adapter-plus_sd15.bin",
        "sdxl": "sdxl_models/ip-adapter_sdxl.bin",
        "sdxl_vit_h": "sdxl_models/ip-adapter_sdxl_vit-h.bin",
            },
        },
        "notes": "Load the correct variant file for the active base model family.",
    },

    # ── VAE ──────────────────────────────────────────────────────────────
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000026"),
        "model_id": "stabilityai/sd-vae-ft-mse",
        "name": "SD VAE ft-MSE",
        "preferred_name": "SD 1.5 VAE MSE",
        "source": "huggingface",
        "family": "vae",
        "variant": "ft-mse",
        "model_type": "vae",
        "compatible_bases": ["sd1.5"],
        "description": (
            "Fine-tuned VAE for SD 1.5. Reduces color bleeding and edge fringing compared to "
            "the default SD1.5 VAE, especially on skin tones and fine text. "
            "Drop-in replacement — swaps the decoder without changing the UNet."
        ),
        "tags": [
            "vae", "helper", "quality", "detail_recovery", "color_fix",
        ],
        "source_url": "https://huggingface.co/stabilityai/sd-vae-ft-mse",
        "version": "ft-mse",
        "capabilities": [],
        "total_size_bytes": 0,
        "download_size_bytes": 335000000,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 2,
        "recommended_vram_max_gb": 4,
        "license": "creativeml-openrail-m",
        "base_model": "sd1.5",
        "precision": "fp16",
        "requirements": {"requires_base_family": "sd1.5"},
        "notes": "Recommended drop-in VAE for all SD1.5 pipelines.",
    },
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000027"),
        "model_id": "madebyollin/sdxl-vae-fp16-fix",
        "name": "SDXL VAE fp16-fix",
        "preferred_name": "SDXL VAE fp16",
        "source": "huggingface",
        "family": "vae",
        "variant": "fp16-fix",
        "model_type": "vae",
        "compatible_bases": ["sdxl"],
        "description": (
            "Drop-in SDXL VAE that runs correctly in fp16 without producing NaN/black images. "
            "The official SDXL VAE requires fp32 to avoid numerical instability; this version "
            "has been fine-tuned to be fp16-safe, halving VAE VRAM usage."
        ),
        "tags": [
            "vae", "helper", "fp16", "nan_fix", "quality",
        ],
        "source_url": "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix",
        "version": "fp16-fix",
        "capabilities": [],
        "total_size_bytes": 0,
        "download_size_bytes": 335000000,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 2,
        "recommended_vram_max_gb": 4,
        "license": "mit",
        "base_model": "sdxl",
        "precision": "fp16",
        "requirements": {"requires_base_family": "sdxl"},
        "notes": "Must-have for fp16 SDXL — prevents NaN / black-frame output.",
    },

    # ── Face restore ─────────────────────────────────────────────────────
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000016"),
        "model_id": "TencentARC/GFPGAN/v1.3.4",
        "name": "GFPGAN v1.4",
        "preferred_name": "GFPGAN v1.4",
        "source": "github",
        "family": "gfpgan",
        "variant": "v1.4",
        "model_type": "face_restore",
        "compatible_bases": [],
        "description": (
            "GAN-based face restoration model from Tencent ARC. "
            "Detects all faces in the image, runs each through a generative prior "
            "trained on high-quality faces, then blends the result back. "
            "Fast and reliable. Best for: general face sharpening after upscaling."
        ),
        "tags": [
            "face_restore", "face_restoration", "face_enhancement", "gfpgan",
        ],
        "source_url": "https://github.com/TencentARC/GFPGAN",
        "version": "1.4",
        "capabilities": ["face_restore"],
        "total_size_bytes": 0,
        "download_size_bytes": 332000000,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 2,
        "recommended_vram_max_gb": 4,
        "license": "apache-2.0",
        "base_model": "gfpgan",
        "precision": "fp32",
        "requirements": {
            "pip_packages": ["gfpgan", "facexlib", "basicsr"],
            "input_type": "image",
        },
        "notes": "Weights downloaded from GitHub release TencentARC/GFPGAN v1.3.4.",
    },
]

_INSERT_SQL = text("""
    INSERT INTO models (
        id, model_id, name, preferred_name, source, family, variant,
        model_type, compatible_bases,
        description, tags, source_url, version, capabilities,
        total_size_bytes, download_size_bytes,
        status,
        recommended_vram_min_gb, recommended_vram_max_gb,
        license, base_model, precision, requirements, notes,
        created_at, updated_at
    ) VALUES (
        :id, :model_id, :name, :preferred_name, :source, :family, :variant,
        :model_type, CAST(:compatible_bases AS jsonb),
        :description, CAST(:tags AS jsonb), :source_url, :version,
        CAST(:capabilities AS jsonb),
        :total_size_bytes, :download_size_bytes,
        :status,
        :recommended_vram_min_gb, :recommended_vram_max_gb,
        :license, :base_model, :precision,
        CAST(:requirements AS jsonb), :notes,
        :created_at, :updated_at
    ) ON CONFLICT (model_id) DO NOTHING
""")


def upgrade() -> None:
    conn = op.get_bind()
    now = datetime.now(timezone.utc)

    for m in MODELS:
        record = m.copy()
        record["created_at"] = now
        record["updated_at"] = now
        record["tags"] = json.dumps(record["tags"])
        record["capabilities"] = json.dumps(record["capabilities"])
        record["requirements"] = json.dumps(record["requirements"])
        record["compatible_bases"] = json.dumps(record["compatible_bases"])
        conn.execute(_INSERT_SQL, record)


def downgrade() -> None:
    conn = op.get_bind()
    for m in MODELS:
        conn.execute(
            text("DELETE FROM models WHERE model_id = :id"),
            {"id": m["model_id"]},
        )
