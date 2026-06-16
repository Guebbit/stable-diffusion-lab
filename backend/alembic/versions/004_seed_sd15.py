"""Seed SD 1.5 family — base models, ControlNet, LoRA, VAE.

Revision ID: 004_seed_sd15
Revises: 003_seed_sdxl
Create Date: 2026-06-15
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import text
from alembic import op

revision = "004_seed_sd15"
down_revision = "003_seed_sdxl"
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
# SD 1.5 base checkpoints
# ---------------------------------------------------------------------------
MODELS = [
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
        "short_description": "Lightweight test model — for dev and integration checks only",
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
        "short_description": "Creates alternate variations of an existing image",
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
        "short_description": "Edits images by text instructions — recolor, style transfer, mood",
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
        "short_description": "Depth-guided structural control for SD1.5 pipelines (helper)",
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
        "short_description": "Lineart and sketch-guided control for SD1.5 pipelines (helper)",
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
        "short_description": "Masked partial editing control for SD1.5 pipelines (helper)",
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
        "requirements": {
            "requires_base_model": "runwayml/stable-diffusion-v1-5",
            "input_type": "image+mask",
        },
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
        "short_description": "Tile-based detail and texture restoration for SD1.5 (helper)",
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
        "short_description": "Human pose-transfer control for SD1.5 pipelines (helper)",
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

    # ── LoRA — SD 1.5 ────────────────────────────────────────────────────
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
        "short_description": "Speed LoRA — reduces SD1.5 inference to 4–8 steps",
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

    # ── VAE — SD 1.5 ─────────────────────────────────────────────────────
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
        "short_description": "Fine-tuned SD1.5 VAE — reduces color bleeding and fringing",
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
