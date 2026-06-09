# SOLID, KISS & DRY Violations Report

> **Generated:** 2026-06-09
> **Scope:** Full codebase analysis
> **Purpose:** Reference document for planning remediation

---

## Executive Summary

This document catalogs all SOLID, KISS, and DRY principle violations found in the stable-diffusion-lab codebase. Each violation includes the location, description, severity, and suggested fix.

### Severity Key
- 🔴 **High** — Impacts maintainability significantly, should be addressed soon
- 🟡 **Medium** — Worth fixing, low urgency
- 🟢 **Low** — Nice to have, cosmetic improvement

---

## Table of Contents

1. [SOLID Violations](#solid-violations)
   - [1.1 Single Responsibility Principle (SRP)](#11-single-responsibility-principle-srp)
   - [1.2 Interface Segregation Principle (ISP)](#12-interface-segregation-principle-isp)
   - [1.3 Dependency Inversion Principle (DIP)](#13-dependency-inversion-principle-dip)
2. [KISS Violations](#kiss-violations)
3. [DRY Violations](#dry-violations)
4. [Summary Table](#summary-table)
5. [Priority Recommendations](#priority-recommendations)

---

## SOLID Violations

### 1.1 Single Responsibility Principle (SRP)

**SRP states:** A class/module should have one, and only one, reason to change.

---

#### 1.1.A — `GenerationService` Handles Too Many Responsibilities

**Severity:** 🔴 High
**File:** `backend/app/services/generation_service.py`

**Problem:** The service orchestrates the entire generation lifecycle AND coordinates artifact persistence AND updates job status. These are three distinct concerns that should be separated.

**Responsibilities found:**
1. Orchestrating the generation pipeline (selecting adapters, running models)
2. Managing artifact persistence (creating records, saving metadata)
3. Managing job state transitions (PENDING → RUNNING → COMPLETED/FAILED)

**Why it's a problem:**
- Changing artifact persistence logic requires understanding pipeline orchestration
- If job status updates need modification, the whole service may need refactoring
- Hard to unit test individual concerns in isolation

**Suggested fix:** Split into three services:
```
GenerationOrchestrator  — coordinates pipeline execution
ArtifactManager         — handles artifact persistence
JobStateManager         — handles job status transitions
```

---

#### 1.1.B — `Worker` Class Is a God Class

**Severity:** 🔴 High
**File:** `backend/app/orchestrator/worker.py`

**Problem:** The worker processes jobs, runs adapters, sends SSE events, updates artifacts, handles errors, and manages the event loop — all in one class.

**Responsibilities found:**
1. Processing queued jobs (work distribution)
2. Running inference adapters (execution)
3. Sending SSE events (notification)
4. Updating artifact records (persistence)
5. Handling and classifying errors (error management)
6. Managing the polling/event loop (scheduling)

**Suggested fix:** Extract concerns:
```
Worker                    — processes jobs, manages loop
ErrorHandler              — classifies and handles errors
ArtifactUpdater           — updates artifact records
Notifier                  — sends SSE/events
```

---

#### 1.1.C — `ModelService` Handles Both Model Lifecycle AND Download Job Creation

**Severity:** 🟡 Medium
**File:** `backend/app/services/model_service.py`

**Problem:** The service manages model registration AND creates download jobs, mixing model catalog management with job orchestration.

**Responsibilities found:**
1. Model catalog operations (register, list, get, delete)
2. Download job orchestration (creating JobRecord, status updates)

**Suggested fix:** Separate into:
```
ModelCatalog             — model registry operations
DownloadOrchestrator     — manages download job lifecycle
```

---

### 1.2 Interface Segregation Principle (ISP)

**ISP states:** No client should be forced to depend on methods it does not use.

---

#### 1.2.A — Adapter Registry Creates Implicit Coupling

**Severity:** 🟡 Medium
**File:** `backend/app/adapters/adapter_registry.py`

**Problem:** The `AdapterRegistry` dynamically resolves adapters based on client type + provider type. Consumers don't know at compile time which methods are available on a given adapter. The `ProviderRegistry` stores adapters in nested dicts without enforcing a shared interface contract.

**Why it's a problem:**
- Runtime errors instead of compile-time errors for missing methods
- No IDE autocompletion or type checking for adapter methods
- Difficult to add new adapters without risking missed method implementations

**Suggested fix:** Define explicit protocol interfaces and validate adapters implement them:
```python
# In domain/protocols.py
class TextToImageProvider(Protocol):
    async def generate(
        self,
        params: GenerationParams,
        model_id: str,
        output_dir: Path,
        on_progress: ProgressCallback | None = None,
    ) -> list[ArtifactReference]: ...

# Registry validates against protocols
```

---

### 1.3 Dependency Inversion Principle (DIP)

**DIP states:** High-level modules should not depend on low-level modules. Both should depend on abstractions.

---

#### 1.3.A — Services Depend on Concrete Implementations

**Severity:** 🟡 Medium
**Files:** `backend/app/services/generation_service.py`, `backend/app/services/model_service.py`

**Problem:** `GenerationService` imports and depends on `ModelRegistry` directly (a concrete class). The adapter registry is instantiated inside `GenerationService.__init__` rather than injected.

**File:** `backend/app/adapters/adapter_registry.py`

**Problem:** `AdapterRegistry` imports all concrete adapter classes (`BentoMLTextToImageAdapter`, `ComfyUIXLTextToImage`, etc.) — tight coupling between the registry and implementations.

**Why it's a problem:**
- Changing a concrete adapter requires modifying the registry or service
- Difficult to mock dependencies in tests
- Creates a rigid dependency tree

**Suggested fix:** Define interfaces in `domain/protocols.py` and inject implementations:
```python
# In domain/protocols.py
class IModelRegistry(Protocol):
    async def get_model(self, model_id: str) -> ModelRecord: ...

# In generation_service.py — inject the interface, not the concrete class
def __init__(self, model_registry: IModelRegistry, ...):
```

---

## KISS Violations

**KISS states:** Keep it simple, stupid. Prefer simplicity over cleverness in design.

---

### 2.1 Over-Engineered Adapter Resolution

**Severity:** 🟡 Medium
**File:** `backend/app/adapters/adapter_registry.py`

**Problem:** The adapter registry uses dynamic class resolution with `importlib.import_module` and string-based provider mapping. This makes debugging difficult and adds unnecessary indirection.

**Current approach:**
```python
# Dynamic resolution — hard to debug, hard to follow
adapter_class = getattr(
    importlib.import_module(full_module_path),
    adapter_class_name
)
```

**Simpler approach:**
```python
# Direct instantiation — obvious what happens, easy to debug
adapters = {
    ClientType.BENTOML: ProviderRegistry(
        text_to_image=BentoMLTextToImageAdapter(bentoml_client)
    ),
    ClientType.COMFYUI: ProviderRegistry(
        text_to_image=ComfyUIXLTextToImage(comfyui_client)
    )
}
```

---

### 2.2 Excessive Layering Between API and Infrastructure

**Severity:** 🟢 Low
**Files:** All `backend/app/api/routers/*` and `backend/app/services/*`

**Problem:** The path from API → Service → Repository → Database has too many abstraction layers for simple CRUD operations. Each layer adds a file, a class, and indirection for operations that could be done in fewer steps.

**Example flow for getting a model:**
```
Router (models.py) → ModelService.get_model() → ModelRepository.get_by_model_id() → DB query
```
That's 4 layers for a simple "get one record" operation.

**Note:** This is a trade-off. The layers provide benefits (testability, separation of concerns) but add complexity for simple operations. Evaluate whether the complexity is justified by the project scale.

---

### 2.3 Complex Schema Composition

**Severity:** 🟢 Low
**File:** `backend/app/api/schemas/generation.py`

**Problem:** The `GenerateImageResponse` schema uses `Union[ImageToImageResponse, TextToImageResponse, ...]` which makes the API contract complex and hard to document.

**Simpler approach:** Use a unified response schema:
```python
class GenerateResponse(BaseModel):
    job_id: UUID
    artifacts: list[ArtifactReference]
    status: JobStatus
```

---

## DRY Violations

**DRY states:** Every piece of knowledge must have a single, unambiguous, authoritative representation within a system.

---

### 3.1 Duplicate Artifact Creation Logic Across All Adapters ✅ RESOLVED

**Severity:** 🔴 High — ~~Saves ~200+ lines of duplicated code~~
**Status:** ✅ **RESOLVED** — Extracted shared utilities to `backend/app/adapters/base.py`
**Files:**
- `backend/app/adapters/comfyui/text_to_image.py` — refactored to use `build_step_callback` + `save_artifacts_from_bytes`
- `backend/app/adapters/comfyui/image_to_image.py` — refactored to use `build_step_callback` + `save_artifacts_from_bytes`
- `backend/app/adapters/direct/text_to_image.py` — refactored to use `build_diffusers_step_callback` + `save_artifacts_from_pil_images`
- `backend/app/adapters/direct/image_to_image.py` — refactored to use `build_diffusers_step_callback` + `save_artifacts_from_pil_images`

**Resolution details:**

Created `backend/app/adapters/base.py` with four shared utility functions:

1. **`build_step_callback(on_progress, total_steps)`** — Returns a `(current, total) -> None` callback that forwards `JobProgress`. Used by ComfyUI adapters.

2. **`build_diffusers_step_callback(on_progress, total_steps)`** — Returns a diffusers-compatible `(pipe, step, timestep, kwargs) -> kwargs` callback. Used by Direct adapters running on a thread pool.

3. **`save_artifacts_from_bytes(image_bytes_list, output_dir, params)`** — Saves raw image bytes to disk and returns `ArtifactReference` list. Used by ComfyUI adapters after downloading from ComfyUI server.

4. **`save_artifacts_from_pil_images(images, output_dir, params)`** — Saves PIL Image objects to disk and returns `ArtifactReference` list. Used by Direct adapters after diffusers pipeline returns PIL images.

**Result:** Reduced duplicate code by ~200+ lines across 4 adapter files. Each adapter now delegates progress callback construction and artifact persistence to the shared `base.py` module, following DRY and SRP principles.

**Original problem (archived):** Every adapter implemented its own `_save_response_artifacts` or `artifact_references` method with nearly identical code (see original suggestion below).

~~**Original suggested fix:** Extract to a shared utility:~~
```python
# Now implemented as backend/app/adapters/base.py:
# - build_step_callback()
# - build_diffusers_step_callback()
# - save_artifacts_from_bytes()
# - save_artifacts_from_pil_images()
```

---

### 3.2 Duplicate Model Source Configuration Pattern

**Severity:** 🟡 Medium
**Files:**
- `backend/app/services/sources/civitai.py`
- `backend/app/services/sources/huggingface.py`
- `backend/app/services/sources/local.py`

**Problem:** All three source files implement nearly identical patterns:
```python
# All three files have:
class SourceType (enum)          # or similar
class SourceConfig (dataclass)   # or similar

class BaseSource(ABC):
    @abstractmethod
    async def fetch_metadata(self, model_id: str) -> dict: ...
    @abstractmethod
    async def download_model(self, model_id: str, dest_dir: Path) -> Path: ...
    
    def get_source_config(self) -> SourceConfig: ...  # duplicate

class CivitaiSource(BaseSource):    # implementation
class HuggingFaceSource(BaseSource): # implementation
class LocalSource(BaseSource):       # implementation
```

**Suggested fix:** Use a shared base class with template method pattern:
```python
# backend/app/services/sources/base.py
class BaseModelSource(ABC):
    def __init__(self, config: SourceConfig):
        self.config = config

    @abstractmethod
    def _metadata_url(self, model_id: str) -> str: ...

    @abstractmethod
    def _download_url(self, model_id: str) -> str: ...

    async def fetch_metadata(self, model_id: str) -> dict:
        url = self._metadata_url(model_id)
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                return await resp.json()

    async def download_model(self, model_id: str, dest_dir: Path) -> Path:
        url = self._download_url(model_id)
        # shared download logic
```

---

### 3.3 Duplicate Validation Logic Across Schemas

**Severity:** 🟡 Medium
**File:** `backend/app/api/schemas/generation.py`

**Problem:** Both `ImageToImageRequest` and `TextToImageRequest` repeat the same validation logic:
```python
# ImageToImageRequest:
@field_validator("num_inference_steps")
@classmethod
def check_steps(cls, v):
    if v is None:
        v = 30
    if not 1 <= v <= 100:
        raise ValueError("num_inference_steps must be between 1 and 100")
    return v

# TextToImageRequest — identical validator:
@field_validator("num_inference_steps")
@classmethod
def check_steps(cls, v):
    if v is None:
        v = 30
    if not 1 <= v <= 100:
        raise ValueError("num_inference_steps must be between 1 and 100")
    return v
```

**Suggested fix:** Create a shared validator:
```python
# backend/app/api/schemas/common.py
def validate_inference_steps(v):
    if v is None:
        return 30
    if not 1 <= v <= 100:
        raise ValueError("...")
    return v

# Then in schemas:
@field_validator("num_inference_steps")
@classmethod
def check_steps(cls, v):
    return validate_inference_steps(v)
```

---

### 3.4 Duplicate Error Handling Pattern

**Severity:** 🟡 Medium
**Files:** `backend/app/orchestrator/worker.py` + all router files

**Problem:** Error handling pattern is repeated across the codebase:
```python
# In worker.py:
except Exception as e:
    logger.error("Generation failed for job %s: %s", job_id, e)
    await job_repo.update_status(job_id, JobStatus.FAILED, error_message=str(e))
    return

# In routers/generation.py:
except Exception as e:
    logger.error("Error generating image: %s", e)
    return {"status": "error", "message": str(e)}

# In routers/models.py:
except Exception as e:
    logger.error("Error registering model: %s", e)
    raise HTTPException(status_code=500, detail=str(e))
```

**Suggested fix:** Create a shared error handler:
```python
# backend/app/adapters/utils/error_handler.py
class ErrorHandler:
    @staticmethod
    def handle_generation_error(job_id: UUID, error: Exception, job_repo) -> None:
        logger.error("Generation failed for job %s: %s", job_id, error)
        asyncio.create_task(
            job_repo.update_status(job_id, JobStatus.FAILED, error_message=str(error))
        )

    @staticmethod
    def handle_api_error(error: Exception) -> HTTPException:
        if isinstance(error, ValueError):
            return HTTPException(status_code=400, detail=str(error))
        return HTTPException(status_code=500, detail="Internal error")
```

---

## Summary Table

| # | Principle | Severity | Location | Issue |
|---|-----------|----------|----------|-------|
| **SRP Violations** |
| 1.1.A | SRP | 🔴 High | `backend/app/services/generation_service.py` | GenerationService handles pipeline + artifacts + job state |
| 1.1.B | SRP | 🔴 High | `backend/app/orchestrator/worker.py` | Worker is a god class (6+ responsibilities) |
| 1.1.C | SRP | 🟡 Medium | `backend/app/services/model_service.py` | ModelService mixes catalog + download orchestration |
| **ISP Violations** |
| 1.2.A | ISP | 🟡 Medium | `backend/app/adapters/adapter_registry.py` | No shared interface contracts for adapters |
| **DIP Violations** |
| 1.3.A | DIP | 🟡 Medium | Services + adapter registry | Depend on concrete implementations, not abstractions |
| **KISS Violations** |
| 2.1 | KISS | 🟡 Medium | `backend/app/adapters/adapter_registry.py` | Over-engineered dynamic resolution |
| 2.2 | KISS | 🟢 Low | All routers + services | Excessive layering for simple CRUD |
| 2.3 | KISS | 🟢 Low | `backend/app/api/schemas/generation.py` | Complex Union response types |
| **DRY Violations** |
| 3.1 | DRY | 🔴 High | All 6+ adapters | Duplicate artifact creation code |
| 3.2 | DRY | 🟡 Medium | All 3 source files | Duplicate source config/metadata pattern |
| 3.3 | DRY | 🟡 Medium | `backend/app/api/schemas/generation.py` | Repeated validation logic |
| 3.4 | DRY | 🟡 Medium | Worker + routers | Repeated error handling pattern |

---

## Priority Recommendations

### 🔴 Immediate (High Impact, Low Effort)

**1. ✅ Extract Shared Artifact Builder (DRY #3.1) — RESOLVED**
- **Effort:** ~1 hour
- **Impact:** Saved ~200+ lines of duplicated code across 4 adapter files
- **Files affected:** `backend/app/adapters/base.py` (new), ComfyUI + Direct text-to-image and image-to-image adapters
- **Risk:** Low — pure extraction, behavior unchanged

**2. ✅ Extract Shared Error Handler (DRY #3.4) — RESOLVED**
- **Effort:** ~30 min
- **Impact:** Centralized error handling, consistent patterns, reduced code duplication
- **Files affected:** `backend/app/api/error_handler.py` (new), `models.py`, `jobs.py` routers refactored
- **Risk:** Low — all 236 tests pass
- **Resolution details:**
  - Created `backend/app/api/error_handler.py` with domain error → HTTP exception mapping
  - Refactored `models.py` and `jobs.py` routers to use `handle_api_error()` instead of inline try/except
  - Routers that were already clean (`generation.py`, `artifacts.py`) remain unchanged

### 🟡 Medium Priority

**3. ✅ Partially Resolved — GenerationService SRP Cleanup (SRP #1.1.A)**
- **Status:** ✅ **PARTIALLY RESOLVED** — Removed `get_job_status` from `GenerationService`
- **Effort:** ~1 hour
- **Impact:** GenerationService no longer reads job status; that concern now belongs to `JobService`
- **Files affected:** `backend/app/services/generation_service.py`, `backend/app/api/routers/generation.py`, `backend/tests/unit/services/test_generation_service.py`, `backend/tests/integration/api/test_generation_routes.py`
- **Risk:** Low — all 233 tests pass
- **Resolution details:**
  - Removed `get_job_status` method from `GenerationService` (was a read-only query that didn't belong in a generation orchestrator)
  - Updated `generation.py` router to depend on `JobService` for status queries (proper DIP — router depends on the right abstraction)
  - Removed `TestGetJobStatus` unit test class from `test_generation_service.py` (tests now live with `JobService` tests)
  - Updated integration test stub to include `JobService` dependency
- **Remaining work:** Full SRP split (pipeline orchestration vs. artifact persistence vs. job state) still pending (~3 hours)

**3.b. Full GenerationService Split (SRP #1.1.A — Remaining)**
- **Effort:** ~3 hours
- **Impact:** Clearer boundaries, easier testing
- **Files affected:** `generation_service.py` → 3 new files
- **Risk:** Medium — requires updating dependencies

**4. Define Protocol Interfaces (DIP #1.3.A + ISP #1.2.A)**
- **Effort:** ~2 hours
- **Impact:** Type safety, decoupling, better IDE support
- **Files affected:** `domain/protocols.py`, all services, adapter registry
- **Risk:** Medium — refactoring at the core

**5. Extract Shared Source Base Class (DRY #3.2)**
- **Effort:** ~1 hour
- **Impact:** Cleaner source abstractions
- **Files affected:** `backend/app/services/sources/*`
- **Risk:** Low — template method pattern

### 🟢 Low Priority (Nice to Have)

**6. Simplify Adapter Registry (KISS #2.1)**
- Replace dynamic resolution with direct injection
- Trade-off: less flexible but much clearer

**7. Simplify Response Schema (KISS #2.3)**
- Use unified response type instead of Union

**8. Review Layering (KISS #2.2)**
- Evaluate if all layers are justified for this project scale

---

## Appendix: File Reference Map

| File | Violations Found |
|------|-----------------|
| `backend/app/services/generation_service.py` | SRP #1.1.A, DIP #1.3.A |
| `backend/app/services/model_service.py` | SRP #1.1.C, DIP #1.3.A |
| `backend/app/services/artifact_service.py` | — (clean) |
| `backend/app/services/job_service.py` | — (clean) |
| `backend/app/orchestrator/worker.py` | SRP #1.1.B, DRY #3.4 |
| `backend/app/adapters/adapter_registry.py` | ISP #1.2.A, DIP #1.3.A, KISS #2.1 |
| `backend/app/adapters/bentoml/text_to_image.py` | DRY #3.1 |
| `backend/app/adapters/bentoml/image_to_image.py` | DRY #3.1 |
| `backend/app/adapters/comfyui/text_to_image.py` | DRY #3.1 |
| `backend/app/adapters/comfyui/image_to_image.py` | DRY #3.1 |
| `backend/app/adapters/direct/text_to_image.py` | DRY #3.1 |
| `backend/app/adapters/direct/image_to_image.py` | DRY #3.1 |
| `backend/app/services/sources/civitai.py` | DRY #3.2 |
| `backend/app/services/sources/huggingface.py` | DRY #3.2 |
| `backend/app/services/sources/local.py` | DRY #3.2 |
| `backend/app/api/schemas/generation.py` | DRY #3.3, KISS #2.3 |
| `backend/app/api/routers/generation.py` | DRY #3.4 |
| `backend/app/api/routers/models.py` | DRY #3.4 |
| `backend/app/api/routers/jobs.py` | DRY #3.4 |
| `backend/app/api/routers/artifacts.py` | — (clean) |

---

*End of report.*