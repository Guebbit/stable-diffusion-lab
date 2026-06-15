"""Seed face-restoration models for the /image-upscale pipeline step 3.

These models are used exclusively by the face-restore step of the upscale pipeline.
They detect faces in the output image and sharpen/repair them after upscaling.

Revision ID: 004_seed_face_restore_models
Revises: 003_seed_upscale_models
Create Date: 2026-06-15
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import sqlalchemy as sa
from sqlalchemy import text
from alembic import op

revision = "004_seed_face_restore_models"
down_revision = "003_seed_upscale_models"
branch_labels = None
depends_on = None

# ---------------------------------------------------------------------------
# All models here have capability "face_restore" and appear only in the
# face-restore step selector on the /image-upscale page.
# ---------------------------------------------------------------------------
FACE_RESTORE_MODELS = [
    {
        "id": uuid.UUID("a0000000-0000-0000-0000-000000000016"),
        "model_id": "TencentARC/GFPGANv1.4",
        "name": "GFPGAN v1.4",
        "preferred_name": "GFPGAN v1.4",
        "source": "huggingface",
        "family": "gfpgan",
        "variant": "v1.4",
        "description": (
            "GAN-based face restoration model from Tencent ARC. "
            "Detects all faces in the image, runs each through a generative prior "
            "trained on high-quality faces, then blends the result back. "
            "Fast and reliable. Best for: general face sharpening after upscaling. "
            "The 'fidelity' slider controls the balance between restoration quality "
            "and identity preservation — high fidelity keeps the original face closer."
        ),
        "tags": [
            "face_restore",
            "face-restoration",
            "face-enhancement",
            "gfpgan",
            "page:image-upscale",
        ],
        "source_url": "https://github.com/TencentARC/GFPGAN",
        "version": "1.4",
        "capabilities": ["face_restore"],
        "total_size_bytes": 0,
        "disk_size_bytes": 0,
        "download_size_bytes": 332000000,
        "file_path": "",
        "status": "not_downloaded",
        "checksum": "",
        "is_verified": False,
        "recommended_vram_min_gb": 2,
        "recommended_vram_max_gb": 4,
        "license": "apache-2.0",
        "base_model": "gfpgan",
        "precision": "fp32",
        "requirements": {
            "pip_packages": ["gfpgan", "facexlib", "basicsr"],
            "input_type": "image",
        },
        "notes": "Auto-downloads weights on first use via the gfpgan pip package.",
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

    for model_data in FACE_RESTORE_MODELS:
        record = model_data.copy()
        record["created_at"] = now
        record["updated_at"] = now
        record["tags"] = json.dumps(record["tags"])
        record["capabilities"] = json.dumps(record["capabilities"])
        record["requirements"] = json.dumps(record["requirements"])
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
                "  capabilities=EXCLUDED.capabilities, tags=EXCLUDED.tags, "
                "  description=EXCLUDED.description, notes=EXCLUDED.notes, "
                "  updated_at=EXCLUDED.updated_at"
            ),
            record,
        )


def downgrade() -> None:
    conn = op.get_bind()
    meta = sa.MetaData()
    models_table = sa.Table("models", meta, sa.Column("model_id", sa.String(512)))
    model_ids = [m["model_id"] for m in FACE_RESTORE_MODELS]
    conn.execute(models_table.delete().where(models_table.c.model_id.in_(model_ids)))
