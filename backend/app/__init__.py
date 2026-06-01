"""
AI Lab Backend — Root application package.

Architecture layers (dependency flows downward only):

    API (routers, schemas, websocket)
     ↓
    Services (business logic orchestration)
     ↓
    Orchestrator (job queue, event bus)
     ↓
    Adapters (direct/bentoml/comfyui inference)
     ↓
    Infrastructure (config, database, storage)
     ↓
    Domain (enums, value objects, protocols — no dependencies)

Entry point: app.main:create_app (FastAPI factory)
See docs/architecture/step-2-project-structure.md for full design.
"""
