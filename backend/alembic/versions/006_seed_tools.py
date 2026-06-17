"""Seed cross-family tools — upscalers, vision-language, face restore, IP-Adapter, ESRGAN.

Revision ID: 006_seed_tools
Revises: 005_seed_sd2
Create Date: 2026-06-15
"""

from __future__ import annotations

import uuid

from seed_helpers import seed_models, unseed_models

revision = "006_seed_tools"
down_revision = "005_seed_sd2"
branch_labels = None
depends_on = None

MODELS = [
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
            "Dedicated diffusion upscaler for 4× enlargement. Best for cases where you want not just bigger images "
            "but also synthetic detail recovery, texture synthesis, and polished upscale output."
        ),
        "short_description": "4× diffusion upscaler for synthetic detail recovery and polished enlargements",
        "tags": [
            "upscaler",
            "diffusion_upscaler",
            "upscale",
            "enhancement",
            "detail",
            "restoration",
            "image",
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
        "requirements": {
            "input_type": "image",
            "pipeline": "StableDiffusionUpscalePipeline",
        },
        "notes": "Best when you want generative upscale detail rather than strictly faithful restoration.",
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
            "Vision-language captioning model for describing the content of an image in natural language. "
            "Useful for alt text, prompt bootstrapping, reverse prompting, and building image summaries."
        ),
        "short_description": "Image captioning model for alt text, reverse prompting, and image summaries",
        "tags": [
            "vision_language",
            "caption",
            "describe",
            "reverse_prompt",
            "alt_text",
            "image_understanding",
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
        "notes": "Primary image description model.",
    },

    # ── Face restore ──────────────────────────────────────────────────────
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
            "Face restoration model for repairing blurry, damaged, low-resolution, or degraded faces in old photos "
            "and noisy images. Best used after general upscaling when human faces need targeted cleanup."
        ),
        "short_description": "Face restoration model for old, blurry, noisy, or low-resolution faces",
        "tags": [
            "face_restore",
            "face",
            "restoration",
            "old_photo",
            "noise",
            "portrait",
            "repair",
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
        "notes": "Excellent as a second-stage face repair tool after upscaling.",
    },

    # ── IP-Adapter — SD 1.5 / SDXL ───────────────────────────────────────
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
            "Reference-image adapter that lets generation follow a source image's style, color palette, mood, "
            "and composition more closely than text prompting alone. Works across SD1.5 and SDXL variants."
        ),
        "short_description": "Reference-image adapter for style transfer and reference-driven generation",
        "tags": [
            "ip_adapter",
            "reference",
            "style_transfer",
            "consistency",
            "image_prompt",
            "helper",
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
        "notes": "Use the variant matching the active base family.",
    },

    # ── ESRGAN upscalers (universal) ──────────────────────────────────────
    {
        "id": uuid.UUID("b0000000-0000-0000-0000-000000000030"),
        "model_id": "esrgan-4x-ultrasharp",
        "name": "ESRGAN 4x UltraSharp",
        "preferred_name": "UltraSharp Upscaler",
        "source": "huggingface",
        "family": "upscaler",
        "variant": "esrgan-4x",
        "model_type": "upscaler",
        "compatible_bases": ["flux", "sdxl", "sd1.5"],
        "description": (
            "High-detail ESRGAN upscaler focused on sharpness, edge definition, and visual punch. "
            "Best for anime, illustration, clean renders, and situations where crispness matters more than strict fidelity."
        ),
        "short_description": "Sharp 4× ESRGAN upscaler for detail-heavy anime, illustration, and clean renders",
        "tags": [
            "upscaler",
            "esrgan",
            "upscale",
            "sharp",
            "detail",
            "anime",
            "illustration",
            "enhancement",
        ],
        "source_url": "https://huggingface.co/Kim2091/UltraSharp",
        "capabilities": ["upscale_image"],
        "notes": "Excellent for detail-rich upscale output and stylized renders.",
    },
    {
        "id": uuid.UUID("b0000000-0000-0000-0000-000000000031"),
        "model_id": "real-esrgan-general-x4plus",
        "name": "RealESRGAN 4x+",
        "preferred_name": "RealESRGAN General",
        "source": "huggingface",
        "family": "upscaler",
        "variant": "realesrgan-x4",
        "model_type": "upscaler",
        "compatible_bases": ["flux", "sdxl", "sd1.5"],
        "description": (
            "General-purpose restoration and upscaling model for old, noisy, compressed, or degraded images. "
            "Best first-stage tool for faithful cleanup before more aggressive enhancement or face repair."
        ),
        "short_description": "General-purpose restoration upscaler for old, noisy, or degraded images",
        "tags": [
            "upscaler",
            "realesrgan",
            "upscale",
            "restoration",
            "old_photo",
            "noise",
            "degraded",
            "repair",
        ],
        "source_url": "https://github.com/xinntao/Real-ESRGAN",
        "capabilities": ["upscale_image"],
        "notes": "Best default first-stage upscaler for damaged or noisy images.",
    },
]


def upgrade() -> None:
    seed_models(MODELS)


def downgrade() -> None:
    unseed_models(MODELS)