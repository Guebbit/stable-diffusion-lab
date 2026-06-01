# Step 2 — Project Structure, Module Boundaries, and Naming Rules

> **Purpose**: Define the complete Python package tree, naming conventions, layer dependency rules, and maintainability guidelines for a solo developer managing this codebase long-term.

---

## 1. Final Package/Module Tree

```
backend/
├── pyproject.toml              # Project metadata, dependencies, tool config
├── requirements.txt            # Pinned production deps (used by Dockerfile)
├── alembic.ini                 # Database migration config
├── Dockerfile                  # Production container build
│
├── alembic/                    # Database migrations
│   ├── env.py                  # Migration environment (async engine)
│   ├── script.py.mako          # Migration file template
│   └── versions/               # Auto-generated migration files
│
├── app/                        # Main application package
│   ├── __init__.py             # Package root + architecture docstring
│   ├── main.py                 # FastAPI factory (composition root)
│   │
│   ├── domain/                 # Layer 1: Pure domain logic (no I/O, no deps)
│   │   ├── __init__.py
│   │   ├── enums.py            # All StrEnum types (single source of truth)
│   │   ├── value_objects.py    # Immutable @dataclass containers
│   │   └── protocols.py        # Protocol interfaces for adapters
│   │
│   ├── infrastructure/         # Layer 2: External systems (DB, filesystem, config)
│   │   ├── __init__.py
│   │   ├── config/
│   │   │   ├── __init__.py
│   │   │   └── settings.py     # Pydantic BaseSettings (env-based config)
│   │   ├── database/
│   │   │   ├── __init__.py
│   │   │   ├── base.py         # SQLAlchemy Base, mixins
│   │   │   ├── session.py      # Async session factory
│   │   │   ├── models.py       # ORM model definitions
│   │   │   └── repositories/   # Data access layer
│   │   │       ├── __init__.py
│   │   │       ├── model_repository.py
│   │   │       ├── job_repository.py
│   │   │       └── artifact_repository.py
│   │   └── storage/
│   │       ├── __init__.py
│   │       └── storage_manager.py  # Filesystem path management
│   │
│   ├── adapters/               # Layer 3: Inference execution implementations
│   │   ├── __init__.py
│   │   ├── resource_coordinator.py  # GPU mutex/semaphore
│   │   ├── direct/             # Direct Python inference (diffusers/transformers)
│   │   │   ├── __init__.py
│   │   │   ├── text_to_image.py
│   │   │   ├── image_to_image.py
│   │   │   ├── vision.py
│   │   │   ├── video.py
│   │   │   └── llm.py
│   │   ├── bentoml/            # BentoML-backed inference (future)
│   │   │   └── __init__.py
│   │   └── comfyui/            # ComfyUI workflow execution (future)
│   │       └── __init__.py
│   │
│   ├── services/               # Layer 4: Business logic orchestration
│   │   ├── __init__.py
│   │   ├── model_service.py    # Model catalog & lifecycle
│   │   └── generation_service.py  # Job creation & status
│   │
│   ├── orchestrator/           # Layer 5: Job queue & worker management
│   │   ├── __init__.py
│   │   ├── event_bus.py        # In-process pub/sub for progress
│   │   └── worker.py           # Background job poll loop
│   │
│   └── api/                    # Layer 6: HTTP/WS interface (entry point)
│       ├── __init__.py
│       ├── schemas/            # Pydantic request/response models
│       │   ├── __init__.py     # Re-exports all schemas
│       │   ├── generation.py   # Generation endpoint schemas
│       │   ├── models.py       # Model registry schemas
│       │   └── system.py       # System status schemas
│       ├── routers/            # FastAPI route handlers
│       │   ├── __init__.py
│       │   ├── generation.py   # POST /generation/*, GET /generation/jobs/*
│       │   ├── models.py       # CRUD /models/*
│       │   └── system.py       # GET /system/status
│       └── websocket/
│           ├── __init__.py
│           └── hub.py          # WebSocket connection manager
│
└── tests/                      # Test suite (mirrors app/ structure)
    ├── __init__.py
    ├── conftest.py             # Shared fixtures (app, client, mocks)
    ├── unit/
    │   ├── __init__.py
    │   ├── domain/             # Enum tests, value object tests
    │   │   └── __init__.py
    │   ├── services/           # Service logic tests (mocked repos)
    │   │   └── __init__.py
    │   └── adapters/           # Adapter unit tests (mocked models)
    │       └── __init__.py
    └── integration/
        ├── __init__.py
        ├── api/                # Full endpoint integration tests
        │   └── __init__.py
        └── database/           # Repository tests against real DB
            └── __init__.py
```

---

## 2. Naming Conventions

### Files and Folders

| Element | Convention | Example |
|---------|-----------|---------|
| Python modules | `snake_case.py` | `text_to_image.py` |
| Package dirs | `snake_case/` | `direct/`, `database/` |
| Test files | `test_<module>.py` | `test_model_service.py` |
| Migration files | Auto-generated by Alembic | `001_initial_schema.py` |

### Python Identifiers

| Element | Convention | Example |
|---------|-----------|---------|
| Classes | `PascalCase` | `ModelService`, `DirectTextToImageAdapter` |
| Functions/methods | `snake_case` | `submit_text_to_image`, `get_by_id` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_VRAM_MB`, `DEFAULT_STEPS` |
| Private helpers | `_snake_case` | `_load_pipeline`, `_run_inference` |
| Type aliases | `PascalCase` | `ProgressSubscriber` |
| Enum members | `UPPER_SNAKE_CASE` | `ModelSource.HUGGINGFACE` |
| Protocol classes | `PascalCase` + descriptive noun | `TextToImageProvider` |

### Class Naming by Role

| Role | Suffix | Example |
|------|--------|---------|
| FastAPI router | (none, module name is enough) | `routers/generation.py` |
| Service class | `Service` | `ModelService`, `GenerationService` |
| Repository class | `Repository` | `ModelRepository`, `JobRepository` |
| Adapter class | Adapter | `DirectTextToImageAdapter` |
| Protocol/interface | `Provider` / `Manager` | `TextToImageProvider`, `ModelManager` |
| Pydantic schema (request) | `Request` | `TextToImageRequest` |
| Pydantic schema (response) | `Response` | `JobStatusResponse` |
| ORM model | `Record` | `ModelRecord`, `JobRecord` |
| Value object | (descriptive name) | `GenerationParams`, `JobProgress` |
| Enum class | (descriptive name) | `ModelSource`, `JobStatus` |
| Settings | `Settings` | `Settings` |

### Database Naming

| Element | Convention | Example |
|---------|-----------|---------|
| Table names | `plural_snake_case` | `models`, `jobs`, `artifacts` |
| Column names | `snake_case` | `model_id`, `created_at` |
| Foreign keys | `<referenced_table_singular>_id` | `model_id`, `job_id` |
| Indexes | `ix_<table>_<column>` | `ix_models_model_id` |
| Unique constraints | `uq_<table>_<column>` | `uq_models_model_id` |

---

## 3. Layer Dependency Rules

```
┌─────────────────────────────────────────────────┐
│  API (routers, schemas, websocket)              │  ← Entry point
├─────────────────────────────────────────────────┤
│  Services (business logic)                      │
├─────────────────────────────────────────────────┤
│  Orchestrator (worker, event bus)               │
├─────────────────────────────────────────────────┤
│  Adapters (direct, bentoml, comfyui)            │
├─────────────────────────────────────────────────┤
│  Infrastructure (config, database, storage)     │
├─────────────────────────────────────────────────┤
│  Domain (enums, value objects, protocols)        │  ← Pure, no dependencies
└─────────────────────────────────────────────────┘
```

### Allowed imports (downward only)

- **API** → Services, Schemas, Infrastructure (for DI), Domain
- **Services** → Orchestrator, Infrastructure, Domain
- **Orchestrator** → Adapters, Infrastructure, Domain
- **Adapters** → Infrastructure, Domain
- **Infrastructure** → Domain
- **Domain** → standard library only

### Forbidden dependency directions

| FROM | CANNOT IMPORT | WHY |
|------|---------------|-----|
| Domain | Any other layer | Must remain pure and portable |
| Infrastructure | Adapters, Services, API | Would create circular deps |
| Adapters | Services, Orchestrator, API | Adapters are called BY upper layers |
| Services | API | Services must be usable without HTTP |
| Orchestrator | API | Worker runs independently of request cycle |

### Cross-cutting exception

- `app.main` (composition root) can import from ALL layers to wire them together
- This is the ONLY file allowed to break layering rules

---

## 4. Module Size and Complexity Rules

### Hard limits (enforced by Ruff)

| Rule | Limit | Rationale |
|------|-------|-----------|
| Max function complexity (McCabe) | 10 | Forces decomposition |
| Max function arguments | 7 | Suggests missing abstraction |
| Max function return statements | 6 | Avoids spaghetti logic |
| Max function branches | 12 | Keeps control flow readable |
| Max function statements | 50 | Prevents mega-functions |
| Line length | 100 chars | Readable without scrolling |

### Soft limits (enforced by discipline)

| Rule | Limit | Action when exceeded |
|------|-------|---------------------|
| Module size | ~300 lines | Split into sub-modules |
| Class size | ~200 lines | Extract helper class or mixin |
| Single file responsibility | 1 concept | Create a new file |
| Router file | ~10 endpoints | Split into sub-routers |
| Test file | matches source file | One `test_*.py` per source module |

---

## 5. Import Organization

Imports are organized by Ruff (isort rules) into these groups, in this order:

```python
# 1. Standard library
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from uuid import UUID

# 2. Third-party packages
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

# 3. First-party (app)
from app.domain.enums import JobStatus, JobType
from app.domain.value_objects import GenerationParams
from app.infrastructure.database.session import get_async_session
```

Rules:
- Always use `from __future__ import annotations` (enables modern type syntax)
- Never use `import *`
- Never use relative imports across package boundaries
- Relative imports allowed ONLY within the same sub-package (e.g., `from .base import Base`)

---

## 6. Solo Developer Maintainability Rules

### Keep it searchable
- Every class/function/enum has exactly ONE canonical location
- No duplicate string literals — use enums from `domain/enums.py`
- No magic numbers — use named constants

### Keep it flat
- Maximum 3 levels of nesting inside `app/`
- No deeply nested sub-sub-sub-packages
- If a directory would have only one file, don't make it a directory

### Keep it obvious
- File names describe their content completely
- No generic names like `utils.py`, `helpers.py`, `misc.py`
- If a utility is needed, it belongs in the layer that owns its domain concept

### Keep it small
- Every commit should leave the project in a working state
- Every module should be readable in under 5 minutes
- Every class should have a single sentence describing what it does

### Keep it typed
- All public APIs must have complete type annotations
- Internal helpers should have annotations too (enforced by mypy strict mode)
- Use `Protocol` instead of `ABC` for interface definitions

---

## 7. Where Things Live (Decision Guide)

| I need to... | It goes in... |
|--------------|---------------|
| Define a new enum value | `domain/enums.py` |
| Add a data transfer shape | `domain/value_objects.py` |
| Define a new adapter contract | `domain/protocols.py` |
| Read environment variables | `infrastructure/config/settings.py` |
| Add a database table | `infrastructure/database/models.py` |
| Query the database | `infrastructure/database/repositories/` |
| Manage filesystem paths | `infrastructure/storage/storage_manager.py` |
| Call torch/diffusers/transformers | `adapters/direct/` |
| Coordinate GPU access | `adapters/resource_coordinator.py` |
| Combine repos + adapters into a workflow | `services/` |
| Run background jobs | `orchestrator/worker.py` |
| Broadcast events | `orchestrator/event_bus.py` |
| Handle HTTP requests | `api/routers/` |
| Define request/response shapes | `api/schemas/` |
| Push real-time updates | `api/websocket/hub.py` |
| Create database migrations | `alembic/versions/` |
| Write tests | `tests/` (mirroring `app/` structure) |
