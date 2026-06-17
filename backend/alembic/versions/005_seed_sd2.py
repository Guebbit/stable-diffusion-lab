"""Seed SD 2.x family — base models and helpers.

Revision ID: 005_seed_sd2
Revises: 004_seed_sd15
Create Date: 2026-06-15
"""

from __future__ import annotations

import uuid

from seed_helpers import seed_models, unseed_models

revision = "005_seed_sd2"
down_revision = "004_seed_sd15"
branch_labels = None
depends_on = None

MODELS = [
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
            "General-purpose SD 2.1 base model for lighter-weight prompt generation and image editing. "
            "Useful as a middle-ground option between SD1.5 speed and SDXL complexity, though not the primary focus."
        ),
        "short_description": "General-purpose SD2.1 model for lighter prompt generation and image editing",
        "tags": [
            "sd2",
            "base",
            "general",
            "text",
            "img2img",
            "balanced",
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
        "notes": "Useful generic SD2.1 model, but usually secondary to SD1.5 / SDXL / FLUX.",
    },
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000036"),
        "model_id": "sd21-depth-helper",
        "name": "SD2.1 Depth Helper",
        "preferred_name": "SD2.1 Depth Helper",
        "source": "huggingface",
        "family": "controlnet",
        "variant": "depth-sd21",
        "model_type": "controlnet",
        "compatible_bases": ["sd2.1"],
        "description": (
            "Depth-oriented helper placeholder for SD2.1 structure-aware editing workflows. "
            "Use this slot if you later want SD2.1-specific spatial control or depth-conditioned editing."
        ),
        "short_description": "Placeholder depth helper for SD2.1 structure-aware editing",
        "tags": [
            "sd2",
            "controlnet",
            "depth",
            "spatial",
            "structure",
            "img2img",
            "helper",
        ],
        "source_url": "https://huggingface.co/search?q=sd2.1+depth+control",
        "capabilities": ["image_to_image"],
        "notes": "Catalog placeholder for SD2.1 depth guidance.",
    },
]


def upgrade() -> None:
    seed_models(MODELS)


def downgrade() -> None:
    unseed_models(MODELS)