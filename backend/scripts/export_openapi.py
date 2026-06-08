"""
Export OpenAPI schema without running the server.

Generates both openapi.json and openapi.yaml at the project root.
Usage: python -m backend.scripts.export_openapi
   or:  python backend/scripts/export_openapi.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

# Ensure the backend package is importable when run as script
_project_root = Path(__file__).resolve().parent.parent.parent
_backend_dir = _project_root / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from app.main import create_app  # noqa: E402


def export(openapi: dict, project_root: Path) -> None:
    """Write openapi.json and openapi.yaml to *project_root*."""
    (project_root / "openapi.json").write_text(
        json.dumps(openapi, indent=2) + "\n", encoding="utf-8"
    )
    (project_root / "openapi.yaml").write_text(
        yaml.dump(openapi, default_flow_style=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    app = create_app()
    schema = app.openapi()
    export(schema, _project_root)
    print(f"OpenAPI schema exported to {_project_root}/openapi.json and openapi.yaml")


if __name__ == "__main__":
    main()