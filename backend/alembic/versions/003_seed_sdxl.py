"""Seed SDXL family — base models, ControlNet, T2I adapters, LoRA, VAE.

Revision ID: 003_seed_sdxl
Revises: 002_seed_flux
Create Date: 2026-06-15
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import text
from alembic import op

revision = "003_seed_sdxl"
down_revision = "002_seed_flux"
branch_labels = None
depends_on = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "preferred_name": "",
    "variant": "",
    "compatible_bases": [],
    "description": "",
    "short_description": "",
    "tags": [],
    "source_url": "",
    "version": "",
    "capabilities": [],
    "total_size_bytes": 0,
    "download_size_bytes": None,
    "status": "not_downloaded",
    "recommended_vram_min_gb": None,
    "recommended_vram_max_gb": None,
    "license": "",
    "base_model": "",
    "precision": "",
    "requirements": {},
    "notes": "",
}

_INSERT_SQL = text("""
    INSERT INTO models (
        id, model_id, name, preferred_name, source, family, variant,
        model_type, compatible_bases,
        description, short_description, tags, source_url, version, capabilities,
        total_size_bytes, download_size_bytes,
        status,
        recommended_vram_min_gb, recommended_vram_max_gb,
        license, base_model, precision, requirements, notes,
        created_at, updated_at
    ) VALUES (
        :id, :model_id, :name, :preferred_name, :source, :family, :variant,
        :model_type, CAST(:compatible_bases AS jsonb),
        :description, :short_description, CAST(:tags AS jsonb), :source_url, :version,
        CAST(:capabilities AS jsonb),
        :total_size_bytes, :download_size_bytes,
        :status,
        :recommended_vram_min_gb, :recommended_vram_max_gb,
        :license, :base_model, :precision,
        CAST(:requirements AS jsonb), :notes,
        :created_at, :updated_at
    ) ON CONFLICT (model_id) DO NOTHING
""")


def _prep(m: dict, now) -> dict:
    rec = {**_DEFAULTS, **m}
    if "recommended_strength" in rec:
        req = dict(rec.get("requirements") or {})
        req["recommended_strength"] = rec.pop("recommended_strength")
        rec["requirements"] = req
    rec["created_at"] = now
    rec["updated_at"] = now
    rec["tags"] = json.dumps(rec["tags"])
    rec["capabilities"] = json.dumps(rec["capabilities"])
    rec["requirements"] = json.dumps(rec["requirements"])
    rec["compatible_bases"] = json.dumps(rec["compatible_bases"])
    return rec


# ---------------------------------------------------------------------------
# SDXL base checkpoints
# ---------------------------------------------------------------------------
MODELS = [
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
        "short_description": "General-purpose text-to-image and img2img all-arounder at 1024px",
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
        "short_description": "Photorealistic portraits, fashion, and product shots (SDXL)",
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
        "short_description": "Anime characters and vivid stylized illustration (SDXL)",
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
        "short_description": "Edge-guided structure-preserving control for SDXL (helper)",
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
        "short_description": "Converts rough sketches into rendered images — SDXL helper",
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
        "requirements": {
            "requires_base_model": "stabilityai/stable-diffusion-xl-base-1.0",
            "input_type": "sketch",
        },
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
        "short_description": "Edge-map guided controlled generation — SDXL helper",
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
        "requirements": {
            "requires_base_model": "stabilityai/stable-diffusion-xl-base-1.0",
            "input_type": "edge-map",
        },
        "notes": "Primary SDXL helper for edge-guided controlled edits.",
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
            "SDXL LoRA that steers output toward pixel-art aesthetics: chunky pixels, "
            "limited palette, retro-game look. Load on top of SDXL 1.0 with strength 0.6–1.0. "
            "Lower strength blends pixel style with photorealism; higher goes full retro. "
            "No effect without a compatible base loaded."
        ),
        "short_description": "Style LoRA — steers SDXL output toward pixel-art aesthetics",
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
        "short_description": "Speed LoRA — reduces SDXL inference to 4–8 steps",
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
            "Drop-in SDXL VAE that runs correctly in fp16 without producing NaN/black images. "
            "The official SDXL VAE requires fp32 to avoid numerical instability; this version "
            "has been fine-tuned to be fp16-safe, halving VAE VRAM usage."
        ),
        "short_description": "fp16-safe SDXL VAE — prevents NaN errors and black frames",
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
]


def upgrade() -> None:
    conn = op.get_bind()
    now = datetime.now(timezone.utc)
    for m in MODELS:
        conn.execute(_INSERT_SQL, _prep(m, now))


def downgrade() -> None:
    conn = op.get_bind()
    for m in MODELS:
        conn.execute(
            text("DELETE FROM models WHERE model_id = :id"),
            {"id": m["model_id"]},
        )
