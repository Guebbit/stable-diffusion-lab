"""Seed SD 2.x family — base models.

Revision ID: 005_seed_sd2
Revises: 004_seed_sd15
Create Date: 2026-06-15
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import text
from alembic import op

revision = "005_seed_sd2"
down_revision = "004_seed_sd15"
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
# SD 2.x base models
# ---------------------------------------------------------------------------
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
            "Best for: generic image-to-image and lighter general generation. Strengths: "
            "decent structure, lower resource needs than SDXL, and flexible use in broad "
            "workflows. Not a helper model."
        ),
        "short_description": "Lightweight generic model for img2img and general experiments",
        "tags": [
            "diffusion", "general_purpose", "generic_transform", "balanced",
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
        "notes": "Useful generic model for transformations and experiments.",
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
