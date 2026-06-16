"""Add short_description column to models and populate seeded rows.

short_description is a max-200-char summary intended for compact UI contexts
(select dropdowns, card subtitles). The existing description field keeps its
longer, richer content.

Revision ID: 003_add_short_description
Revises: 002_seed_models
Create Date: 2026-06-16
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import text

revision = "003_add_short_description"
down_revision = "002_seed_models"
branch_labels = None
depends_on = None

# Short descriptions keyed by model_id (the unique string identifier).
# Kept in the same order as 002_seed_models for easy cross-referencing.
SHORT_DESCRIPTIONS: dict[str, str] = {
    # Base diffusion — FLUX
    "black-forest-labs/FLUX.1-dev": "Best quality text-to-image, excellent prompt fidelity — 12+ GB VRAM",
    # Base diffusion — SDXL
    "stabilityai/stable-diffusion-xl-base-1.0": "General-purpose text-to-image and img2img all-arounder at 1024px",
    "huggingshu/realvisxl-v5": "Photorealistic portraits, fashion, and product shots (SDXL)",
    "multimodalart/super-novelai-4.0-xl": "Anime characters and vivid stylized illustration (SDXL)",
    # Base diffusion — SD 2.x
    "stabilityai/stable-diffusion-2-1": "Lightweight generic model for img2img and general experiments",
    # Base diffusion — SD 1.5
    "runwayml/stable-diffusion-v1-5": "Lightweight test model — for dev and integration checks only",
    "lambdalabs/sd-image-variations-diffusers": "Creates alternate variations of an existing image",
    "timbrooks/instruct-pix2pix": "Edits images by text instructions — recolor, style transfer, mood",
    # Upscaler
    "stabilityai/stable-diffusion-x4-upscaler": "Diffusion 4× upscaler with texture and detail synthesis",
    # Vision-language
    "Salesforce/blip-image-captioning-large": "Generates a natural-language description of an image",
    # ControlNet — SD 1.5
    "lllyasviel/control_v11f1p_sd15_depth": "Depth-guided structural control for SD1.5 pipelines (helper)",
    "lllyasviel/control_v11p_sd15_lineart": "Lineart and sketch-guided control for SD1.5 pipelines (helper)",
    "lllyasviel/control_v11p_sd15_inpaint": "Masked partial editing control for SD1.5 pipelines (helper)",
    "lllyasviel/control_v11f1e_sd15_tile": "Tile-based detail and texture restoration for SD1.5 (helper)",
    "lllyasviel/control_v11p_sd15_openpose": "Human pose-transfer control for SD1.5 pipelines (helper)",
    # ControlNet — SDXL
    "diffusers/controlnet-canny-sdxl-1.0": "Edge-guided structure-preserving control for SDXL (helper)",
    # T2I Adapters — SDXL
    "TencentARC/t2i-adapter-sketch-sdxl-1.0": "Converts rough sketches into rendered images — SDXL helper",
    "TencentARC/t2i-adapter-canny-sdxl-1.0": "Edge-map guided controlled generation — SDXL helper",
    # LoRA
    "nerijs/pixel-art-xl": "Style LoRA — steers SDXL output toward pixel-art aesthetics",
    "latent-consistency/lcm-lora-sdxl": "Speed LoRA — reduces SDXL inference to 4–8 steps",
    "latent-consistency/lcm-lora-sdv1-5": "Speed LoRA — reduces SD1.5 inference to 4–8 steps",
    # IP-Adapter
    "h94/IP-Adapter": "Conditions generation on a reference image rather than text",
    # VAE
    "stabilityai/sd-vae-ft-mse": "Fine-tuned SD1.5 VAE — reduces color bleeding and fringing",
    "madebyollin/sdxl-vae-fp16-fix": "fp16-safe SDXL VAE — prevents NaN errors and black frames",
    # Face restore
    "TencentARC/GFPGAN/v1.3.4": "GAN model — detects and restores all faces in an image",
}


def upgrade() -> None:
    # Add column to existing tables (no-op on fresh installs that already have it).
    op.execute(
        text(
            "ALTER TABLE models"
            " ADD COLUMN IF NOT EXISTS short_description VARCHAR(200) NOT NULL DEFAULT ''"
        )
    )

    # Populate short descriptions for all seeded models.
    conn = op.get_bind()
    for model_id, short_desc in SHORT_DESCRIPTIONS.items():
        conn.execute(
            text(
                "UPDATE models SET short_description = :sd WHERE model_id = :mid"
            ),
            {"sd": short_desc, "mid": model_id},
        )


def downgrade() -> None:
    op.drop_column("models", "short_description")
