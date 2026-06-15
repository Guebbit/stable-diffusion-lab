"""Seed upscale-specific models (dedicated upscalers for the /image-upscale workflow).

Revision ID: 003_seed_upscale_models
Revises: 002_seed_models
Create Date: 2026-06-15
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import sqlalchemy as sa
from sqlalchemy import text
from alembic import op

revision = "003_seed_upscale_models"
down_revision = "002_seed_models"
branch_labels = None
depends_on = None

# ---------------------------------------------------------------------------
# Models seeded here appear exclusively in the /image-upscale workflow.
# All have capability "upscale_image" and do NOT appear on the /image page.
# ---------------------------------------------------------------------------
UPSCALE_MODELS = [
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000008"),
        "model_id": "stabilityai/stable-diffusion-x4-upscaler",
        "name": "Stable Diffusion x4 Upscaler",
        "preferred_name": "SD x4 Upscaler",
        "source": "huggingface",
        "family": "stable_diffusion_upscaler",
        "variant": "x4",
        "description": (
            "Dedicated 4× upscaler based on SD 2.0. Takes a small low-res image and "
            "generates a high-resolution output with recovered texture and sharpness. "
            "Accepts an optional prompt to guide the detail synthesis. "
            "Best for: enlarging clean source images with maximum quality. "
            "Appears only on the /image-upscale page — not a general generation model."
        ),
        "tags": [
            "upscale_image",
            "dedicated-upscaler",
            "diffusion-upscaler",
            "resolution-enhancement",
            "detail-synthesis",
            "texture-recovery",
            "page:image-upscale",
        ],
        "source_url": "https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler",
        "version": "x4",
        "capabilities": ["upscale_image"],
        "total_size_bytes": 0,
        "disk_size_bytes": 0,
        "download_size_bytes": 5100000000,
        "file_path": "",
        "status": "not_downloaded",
        "checksum": "",
        "is_verified": False,
        "recommended_vram_min_gb": 8,
        "recommended_vram_max_gb": 16,
        "license": "openrail++",
        "base_model": "sd2-upscaler",
        "precision": "fp16",
        "requirements": {"input_type": "image", "pipeline": "StableDiffusionUpscalePipeline"},
        "notes": "Primary dedicated upscaler. Supports tiled processing for large inputs.",
    },
]


def upgrade() -> None:
    conn = op.get_bind()
    meta = sa.MetaData()
    models_table = sa.Table(
        "models",
        meta,
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True)),
        sa.Column("model_id", sa.String(512)),
        sa.Column("name", sa.String(255)),
        sa.Column("preferred_name", sa.String(255)),
        sa.Column("source", sa.String(50)),
        sa.Column("family", sa.String(50)),
        sa.Column("variant", sa.String(100)),
        sa.Column("description", sa.Text),
        sa.Column("tags", sa.dialects.postgresql.JSONB),
        sa.Column("source_url", sa.String(1024)),
        sa.Column("version", sa.String(100)),
        sa.Column("capabilities", sa.dialects.postgresql.JSONB),
        sa.Column("total_size_bytes", sa.BigInteger),
        sa.Column("disk_size_bytes", sa.BigInteger),
        sa.Column("download_size_bytes", sa.BigInteger),
        sa.Column("file_path", sa.String(1024)),
        sa.Column("status", sa.String(50)),
        sa.Column("checksum", sa.String(512)),
        sa.Column("is_verified", sa.Boolean),
        sa.Column("recommended_vram_min_gb", sa.Integer),
        sa.Column("recommended_vram_max_gb", sa.Integer),
        sa.Column("license", sa.String(200)),
        sa.Column("base_model", sa.String(100)),
        sa.Column("precision", sa.String(20)),
        sa.Column("requirements", sa.dialects.postgresql.JSONB),
        sa.Column("notes", sa.Text),
        sa.Column("created_at", sa.DateTime),
        sa.Column("updated_at", sa.DateTime),
    )

    now = datetime.now(timezone.utc)

    for model_data in UPSCALE_MODELS:
        record = model_data.copy()
        record["created_at"] = now
        record["updated_at"] = now
        record["tags"] = json.dumps(record["tags"])
        record["capabilities"] = json.dumps(record["capabilities"])
        record["requirements"] = json.dumps(record["requirements"])
        # ON CONFLICT DO UPDATE so re-running this migration overwrites stale 002 data
        conn.execute(
            text(
                "INSERT INTO models (id, model_id, name, preferred_name, source, family, variant, "
                "description, tags, source_url, version, capabilities, total_size_bytes, "
                "disk_size_bytes, download_size_bytes, file_path, status, checksum, is_verified, "
                "recommended_vram_min_gb, recommended_vram_max_gb, license, base_model, precision, "
                "requirements, notes, created_at, updated_at) "
                "VALUES (:id, :model_id, :name, :preferred_name, :source, :family, :variant, "
                ":description, CAST(:tags AS jsonb), :source_url, :version, CAST(:capabilities AS jsonb), "
                ":total_size_bytes, :disk_size_bytes, :download_size_bytes, :file_path, :status, "
                ":checksum, :is_verified, :recommended_vram_min_gb, :recommended_vram_max_gb, "
                ":license, :base_model, :precision, CAST(:requirements AS jsonb), :notes, "
                ":created_at, :updated_at) "
                "ON CONFLICT (model_id) DO UPDATE SET "
                "  name=EXCLUDED.name, preferred_name=EXCLUDED.preferred_name, "
                "  description=EXCLUDED.description, tags=EXCLUDED.tags, "
                "  capabilities=EXCLUDED.capabilities, requirements=EXCLUDED.requirements, "
                "  notes=EXCLUDED.notes, updated_at=EXCLUDED.updated_at"
            ),
            record,
        )


def downgrade() -> None:
    conn = op.get_bind()
    meta = sa.MetaData()
    models_table = sa.Table("models", meta, sa.Column("model_id", sa.String(512)))
    model_ids = [m["model_id"] for m in UPSCALE_MODELS]
    conn.execute(models_table.delete().where(models_table.c.model_id.in_(model_ids)))
