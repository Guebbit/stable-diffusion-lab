"""Seed FLUX.1 Dev add-ons — LoRA, control adapters, IP-Adapter, ESRGAN upscalers.

Revision ID: 007_seed_flux_addons
Revises: 006_seed_tools
Create Date: 2026-06-15
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import text
from alembic import op

revision = "007_seed_flux_addons"
down_revision = "006_seed_tools"
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
# FLUX.1 Dev — LoRA (style)
# ---------------------------------------------------------------------------
MODELS = [
    {
        "id": uuid.UUID("b0000000-0000-0000-0000-000000000001"),
        "model_id": "flux-dev-anime-lora",
        "name": "FLUX Anime Style LoRA",
        "preferred_name": "FLUX Anime LoRA",
        "source": "huggingface",
        "family": "lora",
        "variant": "flux-anime",
        "model_type": "lora",
        "compatible_bases": ["flux"],
        "description": (
            "Best for: anime characters, stylized illustration, manga-like rendering. "
            "Strengths: pushes FLUX toward clean linework, expressive faces, and vibrant colors."
        ),
        "short_description": "Anime / manga stylization for FLUX.1 Dev",
        "tags": ["lora", "flux", "anime", "illustration", "stylized"],
        "source_url": "https://huggingface.co/search?q=flux+anime+lora",
        "version": "1.0",
        "capabilities": ["text_to_image", "image_to_image"],
        "recommended_strength": 0.6,
        "notes": "Use strength 0.4–0.8 depending on how stylized you want it.",
    },
    {
        "id": uuid.UUID("b0000000-0000-0000-0000-000000000002"),
        "model_id": "flux-dev-photoreal-lora",
        "name": "FLUX Photorealism LoRA",
        "preferred_name": "FLUX Realism LoRA",
        "source": "huggingface",
        "family": "lora",
        "variant": "flux-realism",
        "model_type": "lora",
        "compatible_bases": ["flux"],
        "description": (
            "Best for: cinematic photography, portraits, realistic lighting. "
            "Strengths: improves skin texture, depth, and camera-like rendering."
        ),
        "short_description": "Photorealism enhancement for FLUX",
        "tags": ["lora", "flux", "realism", "photo", "cinematic"],
        "source_url": "https://huggingface.co/search?q=flux+realism+lora",
        "version": "1.0",
        "capabilities": ["text_to_image"],
        "recommended_strength": 0.5,
        "notes": "Pairs well with portrait / lighting prompts.",
    },
    {
        "id": uuid.UUID("b0000000-0000-0000-0000-000000000003"),
        "model_id": "flux-dev-lineart-lora",
        "name": "FLUX Lineart / Ink LoRA",
        "preferred_name": "FLUX Ink / Lineart",
        "source": "huggingface",
        "family": "lora",
        "variant": "flux-lineart",
        "model_type": "lora",
        "compatible_bases": ["flux"],
        "description": (
            "Best for: ink sketches, comic linework, clean outlines. "
            "Strengths: improves contour fidelity and reduces messy diffusion edges."
        ),
        "short_description": "Ink and lineart control for FLUX",
        "tags": ["lora", "flux", "lineart", "ink", "sketch"],
        "source_url": "https://huggingface.co/search?q=flux+lineart+lora",
        "version": "1.0",
        "capabilities": ["text_to_image", "image_to_image"],
        "recommended_strength": 0.7,
        "notes": "Good base for sketch-to-final workflows.",
    },

    # ── FLUX.1 Dev — Structure control (ControlNet-like adapters) ─────────
    {
        "id": uuid.UUID("b0000000-0000-0000-0000-000000000010"),
        "model_id": "flux-control-canny-adapter",
        "name": "FLUX Canny Control Adapter",
        "preferred_name": "FLUX Edge Control",
        "source": "huggingface",
        "family": "control_adapter",
        "variant": "canny",
        "model_type": "controlnet_like",
        "compatible_bases": ["flux"],
        "description": (
            "Best for: edge-guided generation, maintaining structure from sketches or photos. "
            "Strengths: preserves silhouettes and composition tightly."
        ),
        "short_description": "Edge / canny structure control for FLUX",
        "tags": ["flux", "control", "canny", "structure", "edge_guided"],
        "source_url": "https://huggingface.co/search?q=flux+canny+control",
        "capabilities": ["image_to_image"],
        "notes": "Use when you need strict shape preservation.",
    },
    {
        "id": uuid.UUID("b0000000-0000-0000-0000-000000000011"),
        "model_id": "flux-control-depth-adapter",
        "name": "FLUX Depth Control Adapter",
        "preferred_name": "FLUX Depth Control",
        "source": "huggingface",
        "family": "control_adapter",
        "variant": "depth",
        "model_type": "controlnet_like",
        "compatible_bases": ["flux"],
        "description": (
            "Best for: 3D structure, perspective control, spatial consistency. "
            "Strengths: preserves geometry and depth ordering."
        ),
        "short_description": "Depth-guided structure control for FLUX",
        "tags": ["flux", "control", "depth", "perspective", "spatial"],
        "source_url": "https://huggingface.co/search?q=flux+depth+control",
        "capabilities": ["image_to_image"],
        "notes": "Good for architecture, interiors, landscapes.",
    },

    # ── FLUX.1 Dev — Reference / IP style control ─────────────────────────
    {
        "id": uuid.UUID("b0000000-0000-0000-0000-000000000020"),
        "model_id": "flux-ip-adapter",
        "name": "FLUX IP-Adapter (Reference Image)",
        "preferred_name": "FLUX Reference Adapter",
        "source": "huggingface",
        "family": "ip_adapter",
        "variant": "flux-multi",
        "model_type": "ip_adapter",
        "compatible_bases": ["flux"],
        "description": (
            "Best for: style transfer, character consistency, reference-driven generation. "
            "Strengths: keeps identity, palette, and composition from reference images."
        ),
        "short_description": "Reference-image conditioning for FLUX",
        "tags": ["flux", "ip_adapter", "reference", "style_transfer", "consistency"],
        "source_url": "https://huggingface.co/search?q=flux+ip+adapter",
        "capabilities": ["image_to_image"],
        "notes": "Essential for consistent characters.",
    },

    # ── Upscaling / Restoration (universal — compatible with flux/sdxl/sd1.5) ──
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
            "Best for: high-quality upscaling with strong edge detail recovery. "
            "Strengths: sharp textures, good for anime + photo + illustration."
        ),
        "short_description": "High-quality 4× ESRGAN upscaler",
        "tags": ["upscaler", "esrgan", "sharp", "detail_recovery"],
        "source_url": "https://huggingface.co/search?q=4x+ultrasharp+esrgan",
        "capabilities": ["upscale_image"],
        "notes": "Use after FLUX generation for maximum detail.",
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
            "Best for: general-purpose restoration and upscaling. "
            "Strengths: stable results, fewer artifacts, good for old images."
        ),
        "short_description": "General-purpose restoration upscaler",
        "tags": ["upscaler", "realesrgan", "restoration", "old_image_fix"],
        "source_url": "https://huggingface.co/search?q=realesrgan+x4plus",
        "capabilities": ["upscale_image"],
        "notes": "Better than SD upscalers for degraded images.",
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
