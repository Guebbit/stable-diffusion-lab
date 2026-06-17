"""Shared helpers for seed migrations — ORM-based insert with ON CONFLICT DO NOTHING."""

from __future__ import annotations

from datetime import datetime, timezone

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import insert as pg_insert

_DEFAULTS: dict = {
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

_models_table = sa.Table(
    "models",
    sa.MetaData(),
    sa.Column("id", postgresql.UUID(as_uuid=True)),
    sa.Column("model_id", sa.String),
    sa.Column("name", sa.String),
    sa.Column("preferred_name", sa.String),
    sa.Column("source", sa.String),
    sa.Column("family", sa.String),
    sa.Column("variant", sa.String),
    sa.Column("model_type", sa.String),
    sa.Column("compatible_bases", postgresql.JSONB),
    sa.Column("description", sa.Text),
    sa.Column("short_description", sa.String),
    sa.Column("tags", postgresql.JSONB),
    sa.Column("source_url", sa.String),
    sa.Column("version", sa.String),
    sa.Column("capabilities", postgresql.JSONB),
    sa.Column("total_size_bytes", sa.BigInteger),
    sa.Column("download_size_bytes", sa.BigInteger),
    sa.Column("status", sa.String),
    sa.Column("recommended_vram_min_gb", sa.Integer),
    sa.Column("recommended_vram_max_gb", sa.Integer),
    sa.Column("license", sa.String),
    sa.Column("base_model", sa.String),
    sa.Column("precision", sa.String),
    sa.Column("requirements", postgresql.JSONB),
    sa.Column("notes", sa.Text),
    sa.Column("created_at", sa.DateTime(timezone=True)),
    sa.Column("updated_at", sa.DateTime(timezone=True)),
)


def _prep(m: dict, now: datetime) -> dict:
    rec = {**_DEFAULTS, **m}
    if "recommended_strength" in rec:
        req = dict(rec.get("requirements") or {})
        req["recommended_strength"] = rec.pop("recommended_strength")
        rec["requirements"] = req
    rec["created_at"] = now
    rec["updated_at"] = now
    return rec


def seed_models(models: list[dict]) -> None:
    now = datetime.now(timezone.utc)
    rows = [_prep(m, now) for m in models]
    conn = op.get_bind()
    stmt = pg_insert(_models_table).values(rows).on_conflict_do_nothing()
    conn.execute(stmt)


def unseed_models(models: list[dict]) -> None:
    ids = [m["model_id"] for m in models]
    conn = op.get_bind()
    conn.execute(_models_table.delete().where(_models_table.c.model_id.in_(ids)))
