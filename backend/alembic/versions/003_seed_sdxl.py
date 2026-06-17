"""Seed SDXL family — base models, ControlNet, T2I adapters, LoRA, VAE.

Revision ID: 003_seed_sdxl
Revises: 002_seed_flux
Create Date: 2026-06-15
"""

from __future__ import annotations

import uuid

from seed_helpers import seed_models, unseed_models

revision = "003_seed_sdxl"
down_revision = "002_seed_flux"
branch_labels = None
depends_on = None

MODELS = [
    # ── SDXL base checkpoints ─────────────────────────────────────────────
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
            "General-purpose SDXL base checkpoint for high-quality text-to-image and image-to-image generation. "
            "A strong neutral foundation for many workflows including prompting, sketch guidance, control adapters, "
            "LoRAs, and production-balanced pipelines."
        ),
        "short_description": "General-purpose SDXL base model for balanced quality and broad compatibility",
        "tags": [
            "sdxl",
            "base",
            "general",
            "text",
            "img2img",
            "balanced",
            "all_rounder",
            "production",
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
        "notes": "Main SDXL base model for broad workflows.",
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
            "Realistic SDXL checkpoint specialized for portraits, fashion, cinematic lighting, "
            "product-style imagery, and polished photorealistic scenes. Best SDXL option in this "
            "catalog for realism-focused prompt generation."
        ),
        "short_description": "Realistic SDXL specialist for portraits, fashion, product, and photo scenes",
        "tags": [
            "sdxl",
            "base",
            "realism",
            "photoreal",
            "portrait",
            "fashion",
            "product",
            "cinematic",
            "text",
            "img2img",
        ],
        "source_url": "https://huggingface.co/huggingshu/realvisxl-v5",
        "version": "5.0",
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
        "notes": "Primary realistic SDXL checkpoint.",
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
            "Stylized SDXL checkpoint specialized for anime, manga, fantasy scenes, vibrant color, "
            "illustration, and expressive character rendering. Best SDXL anime / stylized base in this catalog."
        ),
        "short_description": "Anime / manga / stylized SDXL specialist for vivid illustration and characters",
        "tags": [
            "sdxl",
            "base",
            "anime",
            "manga",
            "stylized",
            "illustration",
            "fantasy",
            "character",
            "text",
            "img2img",
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
        "notes": "Primary SDXL anime / manga / stylized checkpoint.",
    },
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000006"),
        "model_id": "RunDiffusion/Juggernaut-XL-v9",
        "name": "Juggernaut XL v9",
        "preferred_name": "Juggernaut XL",
        "source": "huggingface",
        "family": "custom",
        "variant": "juggernaut-xl-v9",
        "model_type": "base_diffusion",
        "compatible_bases": [],
        "description": (
            "Photorealistic SDXL checkpoint tuned for high-detail portraits, skin texture, natural "
            "lighting, and cinematic composition. One of the strongest community SDXL models for "
            "realistic human subjects, fashion, lifestyle, and editorial imagery. "
            "Handles complex prompts well and produces polished results with 20–30 steps."
        ),
        "short_description": "Top-tier realistic SDXL model for portraits, fashion, and cinematic shots",
        "tags": [
            "sdxl",
            "base",
            "realism",
            "photoreal",
            "portrait",
            "fashion",
            "cinematic",
            "lighting",
            "skin_detail",
            "general",
            "text",
            "img2img",
        ],
        "source_url": "https://huggingface.co/RunDiffusion/Juggernaut-XL-v9",
        "version": "9.0",
        "capabilities": ["text_to_image", "image_to_image"],
        "total_size_bytes": 0,
        "download_size_bytes": 6900000000,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 8,
        "recommended_vram_max_gb": 16,
        "license": "creativeml-openrail-m",
        "base_model": "sdxl",
        "precision": "fp16",
        "requirements": {
            "recommended_resolution": "1024x1024",
            "recommended_steps": "20-30",
            "recommended_guidance_scale": "7.0",
        },
        "notes": "Pair with negative prompts for best realism. Excellent for portrait and character work.",
    },
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000028"),
        "model_id": "sdxl-nsfw-specialist",
        "name": "SDXL NSFW Specialist",
        "preferred_name": "SDXL NSFW Specialist",
        "source": "huggingface",
        "family": "custom",
        "variant": "nsfw-sdxl",
        "model_type": "base_diffusion",
        "compatible_bases": [],
        "description": (
            "NSFW-oriented SDXL placeholder entry for adult-content workflows. Use this slot for a "
            "dedicated SDXL adult checkpoint if NSFW generation is important in your library. "
            "Intended for erotic, lingerie, nude, boudoir, or adult illustration prompting."
        ),
        "short_description": "Dedicated SDXL adult / NSFW model slot for erotic and mature prompting",
        "tags": [
            "sdxl",
            "base",
            "nsfw",
            "adult",
            "erotic",
            "lingerie",
            "nude",
            "mature",
            "text",
            "img2img",
        ],
        "source_url": "https://huggingface.co/search?q=sdxl+nsfw",
        "version": "1.0",
        "capabilities": ["text_to_image", "image_to_image"],
        "notes": "Catalog placeholder for a dedicated SDXL NSFW checkpoint.",
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
            "SDXL-native Canny ControlNet for structure-preserving image editing. Use it for edge-guided "
            "generation, composition lock, redraw workflows, and consistent shape preservation."
        ),
        "short_description": "Canny / edge-guided SDXL control model for structure-preserving edits",
        "tags": [
            "sdxl",
            "controlnet",
            "canny",
            "edge",
            "structure",
            "composition",
            "img2img",
            "helper",
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
        "notes": "Primary SDXL edge-guided control model.",
    },
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000029"),
        "model_id": "diffusers/controlnet-depth-sdxl-1.0",
        "name": "ControlNet Depth SDXL",
        "preferred_name": "Depth ControlNet SDXL",
        "source": "huggingface",
        "family": "controlnet",
        "variant": "depth-sdxl",
        "model_type": "controlnet",
        "compatible_bases": ["sdxl"],
        "description": (
            "Depth-guided SDXL ControlNet for preserving geometry, perspective, and spatial layout. "
            "Useful for architecture, interiors, landscapes, and structure-aware image editing."
        ),
        "short_description": "Depth-guided SDXL control model for geometry and perspective preservation",
        "tags": [
            "sdxl",
            "controlnet",
            "depth",
            "spatial",
            "perspective",
            "structure",
            "architecture",
            "landscape",
            "img2img",
            "helper",
        ],
        "source_url": "https://huggingface.co/search?q=controlnet+depth+sdxl",
        "capabilities": ["image_to_image"],
        "notes": "Useful for spatially coherent SDXL editing.",
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
            "Sketch-guided SDXL adapter for turning rough sketches into more polished rendered imagery "
            "while preserving the original structure. Best for sketch-to-image and sketch-to-ink workflows."
        ),
        "short_description": "Sketch-guided SDXL adapter for sketch-to-image and sketch-to-ink workflows",
        "tags": [
            "sdxl",
            "adapter",
            "sketch",
            "sketch_to_image",
            "sketch_to_ink",
            "lineart",
            "img2img",
            "helper",
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
        "requirements": {
            "requires_base_model": "stabilityai/stable-diffusion-xl-base-1.0",
            "input_type": "sketch",
        },
        "notes": "Primary SDXL sketch-guidance helper.",
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
            "Canny-guided SDXL adapter for controlled generation from edge maps. Good when you want "
            "lighter-weight shape guidance than a full ControlNet pipeline."
        ),
        "short_description": "Canny-guided SDXL adapter for edge-map controlled generation",
        "tags": [
            "sdxl",
            "adapter",
            "canny",
            "edge",
            "structure",
            "composition",
            "img2img",
            "helper",
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
        "requirements": {
            "requires_base_model": "stabilityai/stable-diffusion-xl-base-1.0",
            "input_type": "edge-map",
        },
        "notes": "Primary SDXL canny / edge adapter.",
    },

    # ── LoRA — SDXL ──────────────────────────────────────────────────────
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
            "Pixel-art LoRA for SDXL. Use it to push output toward retro-game aesthetics, visible pixel blocks, "
            "reduced color palettes, sprite-like rendering, and nostalgic 2D game visuals."
        ),
        "short_description": "Pixel-art LoRA for SDXL retro-game, sprite, and low-resolution aesthetics",
        "tags": [
            "sdxl",
            "lora",
            "pixel",
            "pixel_art",
            "retro",
            "game",
            "sprite",
            "stylized",
            "style",
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
        "notes": "Primary pixel-art style LoRA for SDXL.",
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
            "Speed LoRA for SDXL that allows low-step previews using the LCM scheduler. "
            "Ideal for fast iteration, rapid prompt testing, and quicker UI / pipeline validation."
        ),
        "short_description": "Speed LoRA for SDXL fast previews with low-step inference",
        "tags": [
            "sdxl",
            "lora",
            "speed",
            "lcm",
            "fast",
            "preview",
            "iteration",
            "helper",
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
        "notes": "Speed helper for SDXL low-step generation.",
    },
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000030"),
        "model_id": "sdxl-anime-detail-lora",
        "name": "SDXL Anime Detail LoRA",
        "preferred_name": "SDXL Anime Detail LoRA",
        "source": "huggingface",
        "family": "lora",
        "variant": "anime-detail-sdxl",
        "model_type": "lora",
        "compatible_bases": ["sdxl"],
        "description": (
            "Anime detail LoRA for SDXL to enhance expressive faces, cel-shaded surfaces, cleaner outlines, "
            "and stylized character rendering. Best paired with anime-oriented SDXL checkpoints."
        ),
        "short_description": "Anime detail LoRA for SDXL stylized character and manga illustration work",
        "tags": [
            "sdxl",
            "lora",
            "anime",
            "manga",
            "character",
            "stylized",
            "lineart",
            "illustration",
            "style",
        ],
        "source_url": "https://huggingface.co/search?q=sdxl+anime+lora",
        "capabilities": ["text_to_image", "image_to_image"],
        "notes": "Catalog placeholder for stronger SDXL anime styling.",
    },

    # ── VAE — SDXL ───────────────────────────────────────────────────────
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
            "FP16-safe SDXL VAE that prevents NaN / black-image issues and reduces VAE memory pressure. "
            "Recommended companion VAE for SDXL inference in fp16 pipelines."
        ),
        "short_description": "FP16-safe SDXL VAE that avoids NaN / black-frame problems",
        "tags": [
            "sdxl",
            "vae",
            "fp16",
            "stability",
            "nan_fix",
            "helper",
            "quality",
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
        "notes": "Recommended VAE for SDXL fp16 pipelines.",
    },
]


def upgrade() -> None:
    seed_models(MODELS)


def downgrade() -> None:
    unseed_models(MODELS)