"""Seed cross-family tools — upscalers, vision-language, face restore, IP-Adapter.

Revision ID: 006_seed_tools
Revises: 005_seed_sd2
Create Date: 2026-06-15
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import text
from alembic import op

revision = "006_seed_tools"
down_revision = "005_seed_sd2"
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
# Cross-family tools
# ---------------------------------------------------------------------------
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
            "Dedicated 4× upscaler based on SD 2.0. Takes a small low-res image and "
            "generates a high-resolution output with recovered texture and sharpness. "
            "Accepts an optional prompt to guide the detail synthesis. "
            "Appears only on the /image-upscale page — not a general generation model."
        ),
        "short_description": "Diffusion 4× upscaler with texture and detail synthesis",
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
        "requirements": {
            "input_type": "image",
            "pipeline": "StableDiffusionUpscalePipeline",
        },
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
        "short_description": "Generates a natural-language description of an image",
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
            "GAN-based face restoration model from Tencent ARC. "
            "Detects all faces in the image, runs each through a generative prior "
            "trained on high-quality faces, then blends the result back. "
            "Fast and reliable. Best for: general face sharpening after upscaling."
        ),
        "short_description": "GAN model — detects and restores all faces in an image",
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
            "Conditions generation on a reference image instead of (or alongside) a text prompt. "
            "The reference image drives composition, style, and color. "
            "Repo contains variants for both SD1.5 and SDXL. "
            "Use it when you want to 'paint in the style of' a reference."
        ),
        "short_description": "Conditions generation on a reference image rather than text",
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
