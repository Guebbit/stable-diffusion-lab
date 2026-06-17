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
            "sdxl", "base", "general", "text", "img2img", "balanced", "all_rounder", "production",
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
        "model_id": "SG161222/RealVisXL_V5.0",
        "name": "RealVisXL V5.0",
        "preferred_name": "RealVisXL V5",
        "source": "huggingface",
        "family": "custom",
        "variant": "realvisxl-v5",
        "model_type": "base_diffusion",
        "compatible_bases": [],
        "description": (
            "Realistic SDXL checkpoint specialized for portraits, fashion, cinematic lighting, "
            "product-style imagery, and polished photorealistic scenes."
        ),
        "short_description": "Realistic SDXL specialist for portraits, fashion, product, and photo scenes",
        "tags": [
            "sdxl", "base", "realism", "photoreal", "portrait", "fashion",
            "product", "cinematic", "text", "img2img",
        ],
        "source_url": "https://huggingface.co/SG161222/RealVisXL_V5.0",
        "version": "5.0",
        "capabilities": ["text_to_image", "image_to_image"],
        "total_size_bytes": 0,
        "download_size_bytes": 6900000000,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 8,
        "recommended_vram_max_gb": 16,
        "license": "openrail++",
        "base_model": "sdxl",
        "precision": "fp16",
        "requirements": {"recommended_resolution": "1024x1024"},
        "notes": "Primary realistic SDXL checkpoint.",
    },
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000005"),
        "model_id": "cagliostrolab/animagine-xl-4.0",
        "name": "Animagine XL 4.0",
        "preferred_name": "Animagine XL 4.0",
        "source": "huggingface",
        "family": "custom",
        "variant": "animagine-xl-4.0",
        "model_type": "base_diffusion",
        "compatible_bases": [],
        "description": (
            "Stylized SDXL checkpoint specialized for anime, manga, fantasy scenes, vibrant color, "
            "illustration, and expressive character rendering."
        ),
        "short_description": "Anime / manga / stylized SDXL specialist for vivid illustration and characters",
        "tags": [
            "sdxl", "base", "anime", "manga", "stylized", "illustration",
            "fantasy", "character", "text", "img2img",
        ],
        "source_url": "https://huggingface.co/cagliostrolab/animagine-xl-4.0",
        "version": "4.0",
        "capabilities": ["text_to_image", "image_to_image"],
        "total_size_bytes": 0,
        "download_size_bytes": 6900000000,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 8,
        "recommended_vram_max_gb": 16,
        "license": "fair-ai-public-license-1.0-sd",
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
            "lighting, and cinematic composition. Strong for realistic human subjects and editorial imagery."
        ),
        "short_description": "Top-tier realistic SDXL model for portraits, fashion, and cinematic shots",
        "tags": [
            "sdxl", "base", "realism", "photoreal", "portrait", "fashion",
            "cinematic", "lighting", "skin_detail", "general", "text", "img2img",
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
        "notes": "Excellent for portrait and character work.",
    },
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000028"),
        "model_id": "LyliaEngine/Pony_Diffusion_V6_XL",
        "name": "Pony Diffusion V6 XL",
        "preferred_name": "Pony Diffusion V6 XL",
        "source": "huggingface",
        "family": "custom",
        "variant": "pony-diffusion-v6-xl",
        "model_type": "base_diffusion",
        "compatible_bases": [],
        "description": (
            "SDXL finetune for stylized SFW and NSFW image generation with strong support for "
            "anthro, furry, feral, humanoid, and drawn character art. Best fit in this catalog "
            "for explicit illustrated workflows, character-heavy prompting, and fandom-style art."
        ),
        "short_description": "SDXL NSFW and furry-focused illustrated model for anthro, character, and drawn art",
        "tags": [
            "sdxl",
            "base",
            "nsfw",
            "adult",
            "furry",
            "anthro",
            "feral",
            "drawn",
            "stylized",
            "character",
            "illustration",
            "anime",
            "text",
            "img2img",
        ],
        "source_url": "https://huggingface.co/LyliaEngine/Pony_Diffusion_V6_XL",
        "version": "6.0",
        "capabilities": ["text_to_image", "image_to_image"],
        "total_size_bytes": 0,
        "download_size_bytes": 6900000000,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 8,
        "recommended_vram_max_gb": 16,
        "license": "unknown",
        "base_model": "sdxl",
        "precision": "fp16",
        "requirements": {"recommended_resolution": "1024x1024"},
        "notes": "Primary NSFW / furry / anthro illustrated SDXL checkpoint.",
    },
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000036"),
        "model_id": "SeaArtLab/SeaArt-Furry-XL-1.0",
        "name": "SeaArt Furry XL 1.0",
        "preferred_name": "SeaArt Furry XL",
        "source": "huggingface",
        "family": "custom",
        "variant": "seaart-furry-xl-1.0",
        "model_type": "base_diffusion",
        "compatible_bases": [],
        "description": (
            "SDXL-based furry art model specialized for high-quality anthro and furry illustration. "
            "Useful as a secondary specialist when prompts are strongly animal-character focused."
        ),
        "short_description": "Furry-focused SDXL model for anthro and animal-character illustration",
        "tags": [
            "sdxl",
            "base",
            "furry",
            "anthro",
            "animal_character",
            "stylized",
            "illustration",
            "drawn",
            "nsfw",
            "text",
            "img2img",
        ],
        "source_url": "https://huggingface.co/SeaArtLab/SeaArt-Furry-XL-1.0",
        "version": "1.0",
        "capabilities": ["text_to_image", "image_to_image"],
        "total_size_bytes": 0,
        "download_size_bytes": 6900000000,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 8,
        "recommended_vram_max_gb": 16,
        "license": "fair-ai-public-license-1.0-sd",
        "base_model": "sdxl",
        "precision": "fp16",
        "requirements": {"recommended_resolution": "1024x1024"},
        "notes": "Secondary furry-specialist SDXL checkpoint.",
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
            "sdxl", "controlnet", "canny", "edge", "structure", "composition", "img2img", "helper",
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
            "sdxl", "controlnet", "depth", "spatial", "perspective", "structure",
            "architecture", "landscape", "img2img", "helper",
        ],
        "source_url": "https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0",
        "version": "1.0",
        "capabilities": ["image_to_image"],
        "total_size_bytes": 0,
        "download_size_bytes": 2500000000,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 8,
        "recommended_vram_max_gb": 16,
        "license": "openrail",
        "base_model": "sdxl",
        "precision": "fp16",
        "requirements": {"requires_base_model": "stabilityai/stable-diffusion-xl-base-1.0"},
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
            "while preserving the original structure."
        ),
        "short_description": "Sketch-guided SDXL adapter for sketch-to-image and sketch-to-ink workflows",
        "tags": [
            "sdxl", "adapter", "sketch", "sketch_to_image", "sketch_to_ink", "lineart", "img2img", "helper",
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
            "sdxl", "adapter", "canny", "edge", "structure", "composition", "img2img", "helper",
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
            "sdxl", "lora", "pixel", "pixel_art", "retro", "game", "sprite", "stylized", "style",
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
            "Ideal for fast iteration, rapid prompt testing, and quicker UI or pipeline validation."
        ),
        "short_description": "Speed LoRA for SDXL fast previews with low-step inference",
        "tags": [
            "sdxl", "lora", "speed", "lcm", "fast", "preview", "iteration", "helper",
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
            "sdxl", "vae", "fp16", "stability", "nan_fix", "helper", "quality",
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

    # ── Upscaling / restoration / quality improvement ────────────────────
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000031"),
        "model_id": "ai-forever/Real-ESRGAN",
        "name": "Real-ESRGAN General Upscaler",
        "preferred_name": "Real-ESRGAN",
        "source": "huggingface",
        "family": "upscaler",
        "variant": "general-x4",
        "model_type": "upscaler",
        "compatible_bases": [],
        "description": (
            "General-purpose image super-resolution helper for enlarging images while preserving detail. "
            "Useful for photos, AI outputs, and cleanup before or after generative passes."
        ),
        "short_description": "General x4 image upscaler for photos and AI-generated images",
        "tags": [
            "upscale", "super_resolution", "enhance", "detail", "photo", "img2img", "quality",
        ],
        "source_url": "https://huggingface.co/ai-forever/Real-ESRGAN",
        "version": "x4",
        "capabilities": ["upscale_image", "improve_quality"],
        "total_size_bytes": 0,
        "download_size_bytes": 67000000,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 4,
        "recommended_vram_max_gb": 8,
        "license": "bsd-3-clause",
        "base_model": "",
        "precision": "fp16",
        "requirements": {"recommended_scale": "4x"},
        "notes": "Primary general-purpose upscaler.",
    },
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000032"),
        "model_id": "skytnt/animegan2",
        "name": "Anime Upscale / Cleanup Helper",
        "preferred_name": "Anime Cleanup Helper",
        "source": "huggingface",
        "family": "upscaler",
        "variant": "anime-helper",
        "model_type": "upscaler",
        "compatible_bases": [],
        "description": (
            "Anime-oriented enhancement helper for stylized imagery, line cleanup, and flatter shaded art. "
            "Useful when your inputs are illustrations, manga, sprites, or anime renders."
        ),
        "short_description": "Stylized image cleanup helper for anime, manga, and illustration workflows",
        "tags": [
            "anime", "manga", "stylized", "cleanup", "enhance", "lineart", "quality",
        ],
        "source_url": "https://huggingface.co/skytnt/animegan2",
        "version": "2.0",
        "capabilities": ["improve_quality", "denoise_image"],
        "total_size_bytes": 0,
        "download_size_bytes": 0,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 4,
        "recommended_vram_max_gb": 8,
        "license": "unknown",
        "base_model": "",
        "precision": "fp16",
        "requirements": {},
        "notes": "Use for anime / illustration cleanup rather than photoreal restoration.",
    },
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000033"),
        "model_id": "tencentarc/gfpgan",
        "name": "GFPGAN Face Restoration",
        "preferred_name": "GFPGAN",
        "source": "huggingface",
        "family": "face_restoration",
        "variant": "face-restore-v1",
        "model_type": "face_restoration",
        "compatible_bases": [],
        "description": (
            "Face restoration helper for repairing blurred, compressed, low-resolution, or old portrait faces."
        ),
        "short_description": "Face restoration helper for portraits and degraded faces",
        "tags": [
            "face", "portrait", "restoration", "repair", "old_photo", "quality", "denoise",
        ],
        "source_url": "https://huggingface.co/tencentarc/gfpgan",
        "version": "1.0",
        "capabilities": ["restore_faces", "improve_quality"],
        "total_size_bytes": 0,
        "download_size_bytes": 350000000,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 4,
        "recommended_vram_max_gb": 8,
        "license": "apache-2.0",
        "base_model": "",
        "precision": "fp16",
        "requirements": {},
        "notes": "Best used on portrait-focused restoration passes.",
    },
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000034"),
        "model_id": "sczhou/codeformer",
        "name": "CodeFormer Face Restoration",
        "preferred_name": "CodeFormer",
        "source": "huggingface",
        "family": "face_restoration",
        "variant": "codeformer",
        "model_type": "face_restoration",
        "compatible_bases": [],
        "description": (
            "Face restoration model for old, noisy, or compressed portraits. Often useful as an alternative "
            "to GFPGAN when identity preservation matters."
        ),
        "short_description": "Alternative portrait restoration helper for old and noisy faces",
        "tags": [
            "face", "portrait", "restoration", "repair", "old_photo", "identity", "quality",
        ],
        "source_url": "https://huggingface.co/sczhou/codeformer",
        "version": "1.0",
        "capabilities": ["restore_faces", "improve_quality"],
        "total_size_bytes": 0,
        "download_size_bytes": 160000000,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 4,
        "recommended_vram_max_gb": 8,
        "license": "bsd-3-clause",
        "base_model": "",
        "precision": "fp16",
        "requirements": {},
        "notes": "Useful alternative to GFPGAN for portrait repair.",
    },
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000035"),
        "model_id": "microsoft/bringing-old-photos-back-to-life",
        "name": "Old Photo Restoration",
        "preferred_name": "Old Photo Restore",
        "source": "huggingface",
        "family": "restoration",
        "variant": "old-photo",
        "model_type": "restoration",
        "compatible_bases": [],
        "description": (
            "Restoration helper for old or damaged photographs, including faded, noisy, scratched, or degraded images."
        ),
        "short_description": "Old-photo restoration helper for faded, scratched, or degraded images",
        "tags": [
            "old_photo", "restoration", "repair", "scratches", "denoise", "degradation", "quality",
        ],
        "source_url": "https://huggingface.co/microsoft/bringing-old-photos-back-to-life",
        "version": "1.0",
        "capabilities": ["restore_old_photos", "denoise_image", "improve_quality"],
        "total_size_bytes": 0,
        "download_size_bytes": 0,
        "status": "not_downloaded",
        "recommended_vram_min_gb": 4,
        "recommended_vram_max_gb": 8,
        "license": "unknown",
        "base_model": "",
        "precision": "fp16",
        "requirements": {},
        "notes": "Useful for archival and family-photo repair workflows.",
    },
]


def upgrade() -> None:
    seed_models(MODELS)


def downgrade() -> None:
    unseed_models(MODELS)