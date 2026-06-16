"""Rework model tags to snake_case; drop type:* and base:* prefixed tags.

Model type and base model are stored in dedicated columns (model_type,
base_model, compatible_bases). Tags are now plain snake_case descriptors.

Revision ID: 004_clean_tags
Revises: 003_add_short_description
Create Date: 2026-06-16
"""

from __future__ import annotations

import json

from alembic import op
from sqlalchemy import text

revision = "004_clean_tags"
down_revision = "003_add_short_description"
branch_labels = None
depends_on = None

# New canonical tags keyed by model_id.
NEW_TAGS: dict[str, list[str]] = {
    "black-forest-labs/FLUX.1-dev": [
        "flux", "diffusion", "all_arounder", "flagship", "high_quality",
        "transformer_architecture", "general_purpose",
    ],
    "stabilityai/stable-diffusion-xl-base-1.0": [
        "diffusion", "all_arounder", "general_purpose", "neutral_style",
    ],
    "huggingshu/realvisxl-v5": [
        "diffusion", "realistic", "photorealistic", "portrait", "product_style",
    ],
    "multimodalart/super-novelai-4.0-xl": [
        "diffusion", "anime", "illustration", "anime_style", "stylized", "character_art",
    ],
    "stabilityai/stable-diffusion-2-1": [
        "diffusion", "general_purpose", "generic_transform", "balanced",
    ],
    "runwayml/stable-diffusion-v1-5": [
        "diffusion", "test_model", "lightweight", "dev_testing",
    ],
    "lambdalabs/sd-image-variations-diffusers": [
        "diffusion", "variation", "identity_preserving", "concept_variation",
    ],
    "timbrooks/instruct-pix2pix": [
        "diffusion", "instruction_editing", "edit_by_text", "style_transfer", "global_edit",
    ],
    "stabilityai/stable-diffusion-x4-upscaler": [
        "dedicated_upscaler", "diffusion_upscaler", "resolution_enhancement",
        "detail_synthesis", "texture_recovery",
    ],
    "Salesforce/blip-image-captioning-large": [
        "vision_language", "captioning", "image_description", "scene_summary",
    ],
    "lllyasviel/control_v11f1p_sd15_depth": [
        "controlnet", "adapter", "helper",
        "depth_guided", "spatial_control", "structure_preserving",
    ],
    "lllyasviel/control_v11p_sd15_lineart": [
        "controlnet", "adapter", "helper",
        "lineart_guided", "illustration_control", "structure_preserving",
    ],
    "lllyasviel/control_v11p_sd15_inpaint": [
        "controlnet", "adapter", "helper",
        "inpainting", "masked_edit", "localized_edit", "partial_recolor",
    ],
    "lllyasviel/control_v11f1e_sd15_tile": [
        "controlnet", "adapter", "helper",
        "detail_restoration", "texture_enhancement", "tile_control",
    ],
    "lllyasviel/control_v11p_sd15_openpose": [
        "controlnet", "adapter", "helper",
        "pose_guided", "character_posing", "structure_preserving",
    ],
    "diffusers/controlnet-canny-sdxl-1.0": [
        "controlnet", "adapter", "helper",
        "edge_guided", "structure_preserving", "layout_control", "canny",
    ],
    "TencentARC/t2i-adapter-sketch-sdxl-1.0": [
        "adapter", "helper",
        "sketch_guided", "shape_preserving", "illustration_control", "render_from_sketch",
    ],
    "TencentARC/t2i-adapter-canny-sdxl-1.0": [
        "adapter", "helper",
        "edge_guided", "structure_preserving", "layout_control", "canny",
    ],
    "nerijs/pixel-art-xl": [
        "lora", "adapter", "helper",
        "style_pixel_art", "style_retro", "style_gaming",
    ],
    "latent-consistency/lcm-lora-sdxl": [
        "lora", "adapter", "helper",
        "speed", "lcm", "few_step",
    ],
    "latent-consistency/lcm-lora-sdv1-5": [
        "lora", "adapter", "helper",
        "speed", "lcm", "few_step",
    ],
    "h94/IP-Adapter": [
        "ip_adapter", "adapter", "helper",
        "image_prompt", "style_transfer", "reference_image",
    ],
    "stabilityai/sd-vae-ft-mse": [
        "vae", "helper", "quality", "detail_recovery", "color_fix",
    ],
    "madebyollin/sdxl-vae-fp16-fix": [
        "vae", "helper", "fp16", "nan_fix", "quality",
    ],
    "TencentARC/GFPGAN/v1.3.4": [
        "face_restore", "face_restoration", "face_enhancement", "gfpgan",
    ],
}


def upgrade() -> None:
    conn = op.get_bind()
    for model_id, tags in NEW_TAGS.items():
        conn.execute(
            text("UPDATE models SET tags = CAST(:tags AS jsonb) WHERE model_id = :mid"),
            {"tags": json.dumps(tags), "mid": model_id},
        )


def downgrade() -> None:
    pass
