# Step 4 — API, Job Orchestration, and Model Lifecycle Design

> **Purpose**: Define the complete HTTP/WebSocket API surface, endpoint conventions, async job orchestration, state machines, progress/cancellation/failure handling, and the full model lifecycle pipeline (download, install, load, unload, delete, verify).

---

## 1. Route Groups and URL Structure

All API routes are versioned under `/api/v1` to allow future breaking changes without disruption.

```
/api/v1/
├── /generation/              # Image, video, LLM generation endpoints
│   ├── POST /text-to-image   # Submit text-to-image job
│   ├── POST /image-to-image  # Submit image-to-image job
│   ├── POST /video           # Submit video generation job
│   ├── POST /caption         # Submit image captioning job
│   └── POST /llm/chat        # Submit LLM chat completion job
│
├── /jobs/                    # Job queue management
│   ├── GET  /                # List jobs (filterable: status, type, model)
│   ├── GET  /{job_id}        # Get job details + progress
│   ├── POST /{job_id}/cancel # Request cancellation
│   ├── POST /{job_id}/retry  # Retry a failed job
│   └── GET  /{job_id}/events # Get job audit trail (state transitions)
│
├── /models/                  # Model registry and lifecycle
│   ├── GET  /                # List models (filterable: source, family, status, capability)
│   ├── POST /                # Register a model (metadata only)
│   ├── GET  /{model_id}      # Get model details
│   ├── DELETE /{model_id}    # Delete model (files + catalog entry)
│   ├── POST /{model_id}/download  # Trigger download
│   ├── POST /{model_id}/load      # Load into GPU/CPU memory
│   ├── POST /{model_id}/unload    # Unload from memory
│   ├── POST /{model_id}/verify    # Run integrity check
│   └── GET  /{model_id}/files     # List tracked files for this model
│
├── /artifacts/               # Generated output gallery
│   ├── GET  /                # List artifacts (filterable, paginated)
│   ├── GET  /{artifact_id}   # Get artifact metadata
│   ├── GET  /{artifact_id}/file       # Serve the actual file
│   ├── GET  /{artifact_id}/thumbnail  # Serve the thumbnail
│   ├── PATCH /{artifact_id}           # Update gallery metadata (favorite, rating, notes)
│   └── DELETE /{artifact_id}          # Delete artifact (file + record)
│
├── /system/                  # System health and runtime info
│   ├── GET  /status          # Health check + resource usage
│   └── GET  /config          # Current runtime configuration (non-sensitive)
│
└── /ws                       # WebSocket endpoint
    └── WS /events            # Real-time job progress + system events
```

---

## 2. Endpoint Naming Conventions

| Convention | Rule | Example |
|-----------|------|---------|
| **HTTP method** | Matches the action semantics | `GET` for read, `POST` for action/create, `PATCH` for partial update, `DELETE` for removal |
| **Noun-based paths** | Resources are nouns, not verbs | `/models/`, not `/get-models` |
| **Action sub-paths** | Non-CRUD actions use `POST /{id}/{action}` | `POST /models/{id}/download` |
| **Plural collections** | Collection endpoints always plural | `/models/`, `/jobs/`, `/artifacts/` |
| **Consistent IDs** | UUIDs for internal IDs, model_id strings for model references | `GET /jobs/{uuid}`, `POST /models/{model_id_string}/download` |
| **Query params for filtering** | List endpoints use query params | `GET /models/?source=huggingface&status=downloaded` |
| **Standard pagination** | Offset-based with `limit` and `offset` params | `GET /artifacts/?limit=50&offset=100` |

---

## 3. Request/Response Design

### 3.1 Request Patterns

All requests use JSON bodies for `POST`/`PATCH`. Query parameters for filtering and pagination on `GET`.

```
Requests:
├── Generation requests → contain model_id + generation parameters
├── Lifecycle actions  → mostly path-only (POST /models/{id}/load)
├── Registration       → full metadata body (ModelRegisterRequest)
└── Updates            → partial fields only (PATCH semantics)
```

### 3.2 Response Envelope

All responses use a **flat structure** (no wrapping envelope). Errors use standard HTTP status codes with a consistent error body:

```json
{
  "detail": "Model not found",
  "error_code": "MODEL_NOT_FOUND",
  "context": {"model_id": "stabilityai/sdxl-base-1.0"}
}
```

### 3.3 Async Job Responses (202 Accepted)

All generation and long-running endpoints return `202 Accepted` with a job reference:

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Job submitted"
}
```

The client then either:
1. **Polls** `GET /jobs/{job_id}` for status
2. **Subscribes** to `WS /events` for real-time progress

### 3.4 Pagination Response

List endpoints include pagination metadata:

```json
{
  "items": [...],
  "total": 142,
  "limit": 50,
  "offset": 0,
  "has_more": true
}
```

---

## 4. Stable API Strategy

### Versioning

- Prefix: `/api/v1/`
- Increment only for breaking changes (field removals, type changes)
- Adding optional fields or new endpoints does NOT require version bump

### Stability guarantees

| Layer | Stability | Change rules |
|-------|-----------|--------------|
| URL paths | Stable | Never rename or remove without version bump |
| Response fields | Additive-only | New fields can be added; existing fields never removed |
| Request fields | Backward-compatible | New optional fields allowed; required fields never added |
| Error codes | Stable | New codes allowed; existing codes never change meaning |
| WebSocket events | Additive-only | New event types allowed; existing shapes never change |

### Schema versioning

Pydantic schemas in `api/schemas/` are the **single source of truth** for the public API contract. Internal changes to ORM models, services, or adapters MUST NOT leak to the API schemas without deliberate decision.

---

## 5. Sync vs Async Endpoint Behavior

| Endpoint type | Response behavior | Rationale |
|---------------|-------------------|-----------|
| `GET /models/` | **Synchronous** — returns immediately | DB query, fast |
| `POST /models/` | **Synchronous** — returns immediately | Metadata insert, fast |
| `GET /jobs/{id}` | **Synchronous** — returns immediately | DB query, fast |
| `POST /generation/*` | **Async** — returns 202 + job_id | GPU inference takes 5-120 seconds |
| `POST /models/{id}/download` | **Async** — returns 202 + job_id | Downloads take minutes to hours |
| `POST /models/{id}/load` | **Async** — returns 202 + job_id | Loading to GPU takes 5-30 seconds |
| `POST /models/{id}/unload` | **Synchronous** — returns 200 | Memory release is near-instant |
| `POST /models/{id}/verify` | **Async** — returns 202 + job_id | Hashing large files takes time |
| `DELETE /models/{id}` | **Synchronous** — returns 204 | File deletion is fast |
| `POST /jobs/{id}/cancel` | **Synchronous** — returns 200 | Sets flag, worker checks it |

### Rule of thumb

> If an operation takes **less than 500ms** → synchronous.
> If it **might** take longer → create a job, return 202.

---

## 6. Job State Machine

### 6.1 States

```
┌─────────┐
│ PENDING │ ← Job created, waiting in queue
└────┬────┘
     │ Worker claims job (atomic DB update)
     ▼
┌─────────┐
│ RUNNING │ ← Actively executing (GPU locked)
└────┬────┘
     │
     ├──────────────────────┐──────────────────────┐
     ▼                      ▼                      ▼
┌───────────┐        ┌──────────┐          ┌───────────┐
│ COMPLETED │        │  FAILED  │          │ CANCELLED │
└───────────┘        └────┬─────┘          └───────────┘
                          │
                          │ retry (if attempts < max_attempts)
                          ▼
                     ┌─────────┐
                     │ PENDING │ (attempt incremented)
                     └─────────┘
```

### 6.2 Transition Rules

| From | To | Trigger | Who |
|------|----|---------|-----|
| — | PENDING | Job created | Service layer |
| PENDING | RUNNING | Worker claims job | Worker (atomic `FOR UPDATE SKIP LOCKED`) |
| RUNNING | COMPLETED | Execution succeeded | Worker |
| RUNNING | FAILED | Exception during execution | Worker |
| RUNNING | CANCELLED | Cancellation flag checked mid-execution | Worker |
| PENDING | CANCELLED | Cancellation before pickup | Worker (or service) |
| FAILED | PENDING | Retry triggered (manual or auto) | Service layer |

### 6.3 Invalid Transitions (rejected)

- COMPLETED → anything
- CANCELLED → anything
- PENDING → COMPLETED (must go through RUNNING)
- PENDING → FAILED (must go through RUNNING)

---

## 7. Queue Behavior

### 7.1 Priority-based FIFO

Jobs are ordered by:
1. `priority` (higher value = higher priority, DESC)
2. `created_at` (older = first, ASC)

This allows urgent operations (model load) to jump ahead of batch generation jobs.

### 7.2 Claim Query

```sql
SELECT * FROM jobs
WHERE status = 'pending'
ORDER BY priority DESC, created_at ASC
LIMIT 1
FOR UPDATE SKIP LOCKED
```

`SKIP LOCKED` ensures that if multiple worker instances ever exist, they don't block each other.

### 7.3 Job Type Priorities (defaults)

| Job type | Default priority | Rationale |
|----------|-----------------|-----------|
| `model_load` | 10 | User is actively waiting |
| `model_download` | 5 | User initiated but tolerant of delay |
| `text_to_image` | 3 | Standard generation |
| `image_to_image` | 3 | Standard generation |
| `video_generation` | 2 | Takes longer, lower urgency |
| `llm_inference` | 3 | Interactive, but queued |
| `image_captioning` | 2 | Background task |

### 7.4 Concurrency Model

The worker runs **one job at a time** (GPU is the bottleneck). The `ResourceCoordinator` enforces this via an `asyncio.Semaphore(1)`.

Future scaling options:
- Multiple workers for CPU-only tasks (downloads, verification)
- Separate GPU and CPU queues
- These are additive changes — no architecture modification needed

---

## 8. Retries

### 8.1 Retry Policy

| Condition | Retry behavior |
|-----------|---------------|
| OOM (out of memory) | No auto-retry. User must reduce params. |
| Network timeout (download) | Auto-retry up to `max_attempts` with exponential backoff |
| GPU driver crash | Auto-retry once after 5-second cooldown |
| User code error (invalid params) | No retry. Mark FAILED with clear error. |
| Unrecognized exception | No auto-retry. Log full traceback. |

### 8.2 Retry Mechanism

```python
# In service layer (retry logic)
async def retry_job(self, job_id: UUID) -> UUID:
    """Create a new PENDING job from a FAILED job's params."""
    original = await self._job_repo.get_by_id(job_id)
    if original.status != "failed":
        raise InvalidStateError("Only failed jobs can be retried")
    if original.attempt >= original.max_attempts:
        raise RetryLimitExceeded(f"Job reached max attempts ({original.max_attempts})")

    # Update attempt counter and reset to pending
    await self._job_repo.reset_to_pending(job_id, attempt=original.attempt + 1)
    return job_id
```

### 8.3 Backoff for Downloads

Download jobs use exponential backoff with jitter:

```
delay = min(base_delay * 2^(attempt - 1) + random(0, 1), max_delay)
base_delay = 5 seconds
max_delay = 300 seconds (5 minutes)
```

---

## 9. Progress Reporting

### 9.1 Mechanism

Progress flows through two channels simultaneously:

```
Adapter → ProgressCallback → Worker → EventBus → WebSocket Hub → Client
                                    → JobRepository (DB persist)
```

### 9.2 Progress Data Shape

```python
@dataclass(frozen=True)
class JobProgress:
    """Real-time progress snapshot."""
    job_id: UUID
    status: str
    progress_percent: int = 0
    current_step: int = 0
    total_steps: int = 0
    message: str = ""
    eta_seconds: float | None = None
```

### 9.3 WebSocket Event Format

```json
{
  "event": "job.progress",
  "data": {
    "job_id": "550e8400-...",
    "status": "running",
    "progress_percent": 45,
    "current_step": 9,
    "total_steps": 20,
    "message": "Denoising step 9/20",
    "eta_seconds": 6.2
  }
}
```

### 9.4 Other WebSocket Events

| Event | Trigger |
|-------|---------|
| `job.created` | New job enters queue |
| `job.started` | Worker claims job |
| `job.progress` | Step completed / download chunk |
| `job.completed` | Job finished successfully |
| `job.failed` | Job encountered an error |
| `job.cancelled` | Job was cancelled |
| `model.status_changed` | Model lifecycle change (downloading → downloaded, etc.) |
| `system.gpu_status` | GPU memory/utilization update (periodic) |

### 9.5 Progress Throttling

- Generation steps: emit on **every step** (typically 20-50 events per job)
- Download progress: emit at most **once per second** (avoids flooding on fast connections)
- DB persistence: write progress to `jobs` table at most **every 5 seconds** or on state change

---

## 10. Cancellation

### 10.1 Cancellation Flow

```
Client                     API                  Worker
  │                         │                     │
  ├── POST /jobs/{id}/cancel ──▶ │                │
  │                         │ set flag in memory  │
  │                         │ ◀── 200 OK ────────│
  │                         │                     │
  │                         │    (next step)      │
  │                         │                     ├── check flag
  │                         │                     ├── raise CancellationError
  │                         │                     ├── mark CANCELLED in DB
  │                         │                     └── broadcast event
```

### 10.2 Cancellation Granularity

- **PENDING jobs**: Cancelled immediately (before worker picks them up)
- **RUNNING jobs**: Cancelled at the next checkpoint (between inference steps). NOT interrupted mid-step to avoid GPU state corruption.
- **COMPLETED/FAILED jobs**: Cannot be cancelled (returns 409 Conflict)

### 10.3 Implementation

The worker holds an in-memory `Set[UUID]` of jobs requested for cancellation. During execution, adapters receive a `is_cancelled: Callable[[], bool]` callback that they check between steps:

```python
# Inside adapter
for step in range(total_steps):
    if is_cancelled():
        raise CancellationError(f"Job cancelled at step {step}/{total_steps}")
    # ... execute step ...
```

---

## 11. Failure Handling

### 11.1 Error Categories

| Category | HTTP response | Job result | User action |
|----------|--------------|------------|-------------|
| Validation error | 422 Unprocessable | N/A (no job created) | Fix request params |
| Model not found | 404 Not Found | N/A (no job created) | Register/download model first |
| Model not ready | 409 Conflict | N/A (no job created) | Wait for download/load |
| OOM during inference | — (async) | FAILED + error msg | Reduce image size or batch |
| Download network error | — (async) | FAILED + retry info | Auto-retry or manual retry |
| GPU driver error | — (async) | FAILED + traceback | Check system, retry |
| Internal error | 500 Internal | N/A | Bug report |

### 11.2 Error Response Schema

```json
{
  "detail": "Human-readable error message",
  "error_code": "MACHINE_READABLE_CODE",
  "context": {
    "field": "width",
    "value": 9999,
    "max_allowed": 2048
  }
}
```

### 11.3 Timeout Handling

Jobs have an optional `timeout_at` field. The worker checks this:
- If `datetime.now() > job.timeout_at`: mark as FAILED with timeout error
- Default timeouts:
  - Generation jobs: 10 minutes
  - Download jobs: 4 hours
  - Model load: 5 minutes
  - Verification: 30 minutes

### 11.4 Stale Job Recovery

On worker startup, the worker scans for jobs stuck in `RUNNING` status (from a previous crash):
- If `started_at` is older than 2× the timeout → mark as FAILED with "Worker crash recovery"
- This prevents zombie jobs blocking the queue

---

## 12. Model Source Abstraction

### 12.1 Source Provider Interface

```python
@runtime_checkable
class ModelSourceProvider(Protocol):
    """Handles model discovery and download from a specific source."""

    async def resolve_model_info(self, model_id: str) -> ModelMetadata:
        """Fetch remote metadata (file list, sizes, checksums)."""
        ...

    async def download_file(
        self,
        model_id: str,
        file_path: str,
        destination: Path,
        on_progress: DownloadProgressCallback | None = None,
        resume_from_byte: int = 0,
    ) -> FileIntegrity:
        """Download a single file with resume support."""
        ...

    def supports_resume(self) -> bool:
        """Whether this source supports byte-range resume."""
        ...
```

### 12.2 Concrete Implementations

| Source | Provider class | Location |
|--------|----------------|----------|
| Hugging Face | `HuggingFaceSourceProvider` | `app/services/sources/huggingface.py` |
| Civitai | `CivitaiSourceProvider` | `app/services/sources/civitai.py` |
| Local import | `LocalSourceProvider` | `app/services/sources/local.py` |

### 12.3 Source Selection

The service resolves which provider to use based on `model.source`:

```python
def _get_source_provider(self, source: str) -> ModelSourceProvider:
    """Route to the correct source provider."""
    providers = {
        ModelSource.HUGGINGFACE: self._hf_provider,
        ModelSource.CIVITAI: self._civitai_provider,
        ModelSource.LOCAL: self._local_provider,
    }
    return providers[source]
```

---

## 13. Download / Install Pipeline

### 13.1 Pipeline Steps

```
Register → Resolve → Download → Verify → Ready
   │          │          │          │        │
   │    Fetch remote     │   Check SHA256   Model can
   │    file manifest    │   per file       be loaded
   │                     │
   │            Per-file download with
   │            resume support and
   │            progress tracking
   │
   Metadata only (fast, no disk I/O)
```

### 13.2 Download State Machine (per model)

```
NOT_DOWNLOADED → DOWNLOADING → DOWNLOADED → (LOADING → LOADED)
                     │                │
                     │ pause/error     │ verify
                     ▼                ▼
              DOWNLOAD_PAUSED    (is_verified = true)
                     │
                     │ resume
                     ▼
               DOWNLOADING
```

### 13.3 Multi-File Download Orchestration

For HuggingFace models with many files:

1. **Resolve phase**: Fetch file manifest from HF API → create `model_files` rows
2. **Download phase**: Process files sequentially (or in limited parallel):
   - For each file: check if partially downloaded → resume from byte offset
   - Update `model_files.downloaded_bytes` and `model_files.status` per file
   - Update `models.download_progress` (aggregate percentage)
3. **Completion**: All files downloaded → model status = `DOWNLOADED`

### 13.4 Resume After Interruption

```python
# On download job start, check manifest state:
pending_files = await self._model_file_repo.get_incomplete_files(model_uuid)
for file_record in pending_files:
    resume_byte = file_record.downloaded_bytes  # Where we left off
    await source_provider.download_file(
        model_id=model.model_id,
        file_path=file_record.relative_path,
        destination=target_path,
        resume_from_byte=resume_byte,
        on_progress=progress_callback,
    )
    file_record.status = "complete"
    file_record.downloaded_bytes = file_record.size_bytes
```

### 13.5 Download Manifest (.download_manifest)

A JSON file written alongside model files on disk:

```json
{
  "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
  "source": "huggingface",
  "started_at": "2025-03-15T10:30:00Z",
  "files": [
    {
      "relative_path": "unet/diffusion_pytorch_model.safetensors",
      "expected_size_bytes": 5135149056,
      "downloaded_bytes": 5135149056,
      "sha256": "a1b2c3...",
      "status": "complete"
    }
  ],
  "overall_status": "complete",
  "total_expected_bytes": 6800000000,
  "total_downloaded_bytes": 6800000000
}
```

This manifest provides **disaster recovery**: if the database is lost, the backend can reconstruct download state from the manifest files on disk.

---

## 14. Load / Unload Pipeline

### 14.1 Load Flow

```
POST /models/{model_id}/load
         │
         ▼
┌─────────────────────────────────────────┐
│ Service: validate model state            │
│ - Is status == DOWNLOADED or LOADED?     │
│ - Is another load job running for this?  │
└────────────────────┬────────────────────┘
                     │
                     ▼ (creates MODEL_LOAD job)
┌─────────────────────────────────────────┐
│ Worker: acquire GPU lock                 │
│ Worker: call ModelManager.load_model()   │
│ Worker: update model status → LOADED     │
└─────────────────────────────────────────┘
```

### 14.2 Unload Flow

Unload is **synchronous** (no job needed — freeing memory is fast):

```
POST /models/{model_id}/unload
         │
         ▼
┌─────────────────────────────────────────┐
│ Service: validate model state            │
│ - Is status == LOADED?                   │
│ Service: call ModelManager.unload_model()│
│ Service: update model status → DOWNLOADED│
│ Return: 200 OK                           │
└─────────────────────────────────────────┘
```

### 14.3 Constraints

1. **Only one model load at a time** — GPU memory is the constraint
2. **Cannot delete a loaded model** — must unload first (returns 409)
3. **Cannot load an unverified model** — configurable (setting: `require_verification_before_load`)
4. **Auto-unload** — when loading model B, if model A is loaded and incompatible, auto-unload A first

---

## 15. Delete Pipeline

### 15.1 Delete Flow

```
DELETE /models/{model_id}
         │
         ▼
┌─────────────────────────────────────────┐
│ Service: validate model state            │
│ - Is status == LOADED? → 409 Conflict    │
│ - Is download in progress? → cancel it   │
│                                          │
│ Service: delete files from disk          │
│ Service: delete model_files rows (CASCADE)│
│ Service: delete model row                │
│ Return: 204 No Content                   │
└─────────────────────────────────────────┘
```

### 15.2 Orphan Cleanup

A periodic background task (not a job — runs on a schedule):
- Scans `{storage_root}/models/` for directories not tracked in DB
- Reports them in system status (does NOT auto-delete — too dangerous)
- Admin can trigger cleanup via `POST /system/cleanup` (future)

---

## 16. Integrity Verification

### 16.1 Verification Flow

```
POST /models/{model_id}/verify
         │
         ▼ (creates verification job)
┌─────────────────────────────────────────┐
│ Worker: for each model_file row:         │
│ - Read file from disk                    │
│ - Compute SHA256                         │
│ - Compare with model_files.sha256        │
│ - Update model_files.is_verified         │
│                                          │
│ If all verified:                         │
│ - Set models.is_verified = TRUE          │
│ - Set models.last_verified_at = now()    │
│                                          │
│ If any mismatch:                         │
│ - Job fails with list of bad files       │
│ - Model status → ERROR                   │
└─────────────────────────────────────────┘
```

### 16.2 When Verification Runs

| Trigger | Automatic? |
|---------|-----------|
| After download completes | Yes (always) |
| User request (POST /verify) | Manual |
| Before first load (if `require_verification_before_load`) | Configurable |

---

## 17. Partial Download Tracking

### 17.1 Database-Level Tracking

Each file in a multi-file model has its own row in `model_files`:

```
model_files:
├── relative_path: "unet/diffusion_pytorch_model.safetensors"
│   ├── size_bytes: 5_135_149_056
│   ├── downloaded_bytes: 2_567_574_528  ← partial
│   └── status: "downloading"
│
├── relative_path: "vae/diffusion_pytorch_model.safetensors"
│   ├── size_bytes: 334_643_268
│   ├── downloaded_bytes: 334_643_268  ← complete
│   └── status: "complete"
```

### 17.2 Aggregate Progress Calculation

```python
def calculate_download_progress(model_files: list[ModelFileRecord]) -> int:
    """Calculate overall download percentage from file-level tracking."""
    total_expected = sum(f.size_bytes for f in model_files)
    total_downloaded = sum(f.downloaded_bytes for f in model_files)
    if total_expected == 0:
        return 0
    return int((total_downloaded / total_expected) * 100)
```

### 17.3 Disk-Level Tracking

Partial files are written with a `.part` suffix:
```
models/huggingface/stabilityai--sdxl-base-1.0/
├── unet/diffusion_pytorch_model.safetensors.part   ← incomplete
├── vae/diffusion_pytorch_model.safetensors         ← complete
└── .download_manifest                               ← tracks state
```

On resume, the `.part` file is appended to. On completion, it's renamed (removing `.part`).

---

## 18. Module Placement for Step 4 Components

Following the layer architecture from Step 2:

```
app/
├── api/
│   ├── routers/
│   │   ├── generation.py      # POST /generation/*
│   │   ├── jobs.py            # GET/POST /jobs/*
│   │   ├── models.py          # GET/POST/DELETE /models/*
│   │   ├── artifacts.py       # GET/PATCH/DELETE /artifacts/*
│   │   └── system.py          # GET /system/*
│   ├── schemas/
│   │   ├── generation.py      # Generation request/response schemas
│   │   ├── jobs.py            # Job query/response schemas
│   │   ├── models.py          # Model registry schemas
│   │   ├── artifacts.py       # Artifact/gallery schemas
│   │   ├── system.py          # System status schemas
│   │   └── common.py          # PaginatedResponse, ErrorResponse
│   └── websocket/
│       └── hub.py             # WebSocket manager + event broadcast
│
├── services/
│   ├── model_service.py       # Model register/download/load/unload/delete
│   ├── generation_service.py  # Job submission for all generation types
│   ├── job_service.py         # Job status, cancel, retry, listing
│   ├── artifact_service.py    # Artifact CRUD, gallery operations
│   └── sources/               # Model source provider implementations
│       ├── __init__.py
│       ├── huggingface.py     # HuggingFaceSourceProvider
│       ├── civitai.py         # CivitaiSourceProvider
│       └── local.py           # LocalSourceProvider
│
├── orchestrator/
│   ├── worker.py              # Job execution loop + dispatch
│   └── event_bus.py           # In-process pub/sub
│
├── domain/
│   ├── enums.py               # All enum values
│   ├── value_objects.py       # Immutable data containers
│   ├── protocols.py           # Adapter contracts
│   └── errors.py              # Domain-specific exception hierarchy
│
└── infrastructure/
    ├── database/
    │   ├── models.py          # ORM records (5 tables)
    │   └── repositories/
    │       ├── model_repository.py
    │       ├── model_file_repository.py   # NEW
    │       ├── job_repository.py
    │       ├── job_event_repository.py    # NEW
    │       └── artifact_repository.py
    └── storage/
        └── storage_manager.py
```

---

## 19. Error Hierarchy

Domain-specific exceptions (in `app/domain/errors.py`):

```python
class AILabError(Exception):
    """Base exception for all domain errors."""
    error_code: str = "INTERNAL_ERROR"

class ModelNotFoundError(AILabError):
    """Raised when a model_id doesn't exist in the registry."""
    error_code = "MODEL_NOT_FOUND"

class ModelNotReadyError(AILabError):
    """Raised when a model isn't in the required state for an operation."""
    error_code = "MODEL_NOT_READY"

class InvalidStateTransitionError(AILabError):
    """Raised when a job/model state change violates the state machine."""
    error_code = "INVALID_STATE_TRANSITION"

class JobNotFoundError(AILabError):
    """Raised when a job_id doesn't exist."""
    error_code = "JOB_NOT_FOUND"

class RetryLimitExceededError(AILabError):
    """Raised when a job has exhausted its retry budget."""
    error_code = "RETRY_LIMIT_EXCEEDED"

class CancellationError(AILabError):
    """Raised by adapters when cancellation is detected."""
    error_code = "JOB_CANCELLED"

class IntegrityVerificationError(AILabError):
    """Raised when file integrity check fails."""
    error_code = "INTEGRITY_CHECK_FAILED"
```

These map to HTTP status codes via a FastAPI exception handler:

```python
@app.exception_handler(AILabError)
async def ailab_error_handler(request, exc):
    status_map = {
        "MODEL_NOT_FOUND": 404,
        "JOB_NOT_FOUND": 404,
        "MODEL_NOT_READY": 409,
        "INVALID_STATE_TRANSITION": 409,
        "RETRY_LIMIT_EXCEEDED": 409,
        ...
    }
    return JSONResponse(
        status_code=status_map.get(exc.error_code, 500),
        content={"detail": str(exc), "error_code": exc.error_code},
    )
```

---

## 20. Summary of Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| All long operations return 202 + job_id | Uniform async pattern, never block the API |
| Single worker, single GPU lock | Simple, correct for local single-GPU setup |
| DB-based queue with `SKIP LOCKED` | No external queue dependency (Redis/RabbitMQ) |
| Priority column on jobs | Critical operations (load) skip ahead of batch gen |
| File-level download tracking | Resume interrupted downloads without re-downloading |
| WebSocket for progress | Real-time UX without polling overhead |
| Cancellation via flag (cooperative) | Safe GPU state — never kill mid-computation |
| Source providers as Protocol | Add new model sources without touching existing code |
| Domain exceptions with error_codes | Consistent error responses across all endpoints |
| Pagination on all list endpoints | Prevent unbounded responses as catalog grows |
| Artifact gallery metadata (favorite/rating) | Enable local curation without external tools |
