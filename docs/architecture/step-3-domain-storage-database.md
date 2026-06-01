# Step 3 — Domain Model, Storage, and Database Design

> **Purpose**: Define the complete data model, ownership boundaries between PostgreSQL and the filesystem, concrete database schema, directory layouts, and the indexing/constraint strategy for the AI Lab backend.

---

## 1. Domain Entity Map

The system has **four core aggregates** and several supporting value objects:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DOMAIN ENTITIES                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    1:N     ┌──────────────┐    1:N     ┌──────────────┐
│  │    Model     │───────────▶│     Job      │───────────▶│   Artifact   │
│  │  (catalog)   │            │  (work unit) │            │  (output)    │
│  └──────────────┘            └──────────────┘            └──────────────┘
│         │                          │                           │
│         │ 1:N                      │ 1:N                      │
│         ▼                          ▼                          │
│  ┌──────────────┐           ┌──────────────┐                 │
│  │  ModelFile   │           │   JobEvent   │                 │
│  │  (on-disk)   │           │  (history)   │                 │
│  └──────────────┘           └──────────────┘                 │
│                                                               │
└─────────────────────────────────────────────────────────────────────┘
```

### Entity Definitions

| Entity | Role | Identity | Owner |
|--------|------|----------|-------|
| **Model** | A registered model in the catalog (metadata + lifecycle status) | UUID (`id`) + unique `model_id` string | PostgreSQL |
| **ModelFile** | A tracked file belonging to a model on disk (weight shard, config, tokenizer) | UUID (`id`), belongs to one Model | PostgreSQL + Filesystem |
| **Job** | A unit of work (generation, download, model load/unload) | UUID (`id`) | PostgreSQL |
| **JobEvent** | An immutable audit entry for job state transitions | UUID (`id`), belongs to one Job | PostgreSQL |
| **Artifact** | A generated output file (image, video, text) | UUID (`id`), belongs to one Job | PostgreSQL + Filesystem |

### Value Objects (domain layer, no persistence)

| Value Object | Purpose |
|--------------|---------|
| `GenerationParams` | Immutable parameter set for generation jobs |
| `ModelIdentifier` | Source-agnostic model reference (source + id + family + variant) |
| `JobProgress` | Real-time progress snapshot emitted via WebSocket |
| `ArtifactReference` | Pointer to artifact file with metadata |
| `DownloadProgress` | Download state snapshot for progress reporting |
| `FileIntegrity` | Checksum + expected size for verification |

---

## 2. What Belongs Where

### PostgreSQL owns:

- **Identity and relationships** — who owns what, foreign keys, cascades
- **Lifecycle state** — model status, job status, download progress percentages
- **Metadata for search and display** — prompt text, tags, model names, sizes, dimensions
- **Audit history** — job state transitions with timestamps
- **Integrity tracking** — checksums, expected sizes, verification status
- **Configuration snapshots** — generation parameters used (immutable after job creation)

### Filesystem owns:

- **Binary content** — model weights, generated images/videos, thumbnails
- **Intermediate files** — temp processing files, partial downloads
- **User uploads** — imported source images for img2img
- **Logs** — rotating application logs (optional)

### The golden rule:

> PostgreSQL stores **what** exists and **where** it is.
> The filesystem stores **the actual bytes**.
> If a file disappears from disk, the database row remains as a record
> (status can be updated to reflect the loss).

---

## 3. Filesystem Directory Structure

```
{storage_root}/                      # Default: /app/storage
│
├── models/                          # Model weights and configs
│   ├── huggingface/                 # Mirrors HF repo structure
│   │   └── {org}--{repo}/           # e.g., stabilityai--stable-diffusion-xl-base-1.0
│   │       ├── model_index.json
│   │       ├── unet/
│   │       ├── vae/
│   │       ├── text_encoder/
│   │       ├── tokenizer/
│   │       └── .download_manifest   # Tracks partial downloads
│   │
│   ├── civitai/                     # Single-file checkpoints
│   │   └── {model_id}/             # e.g., 12345
│   │       ├── model.safetensors
│   │       └── .download_manifest
│   │
│   └── local/                       # Manually imported models
│       └── {user_given_name}/
│           └── *.safetensors / *.ckpt / *.gguf
│
├── artifacts/                       # Generated outputs
│   └── {job_id}/                    # UUID directory per job
│       ├── 0001.png                 # Sequential numbering within a job
│       ├── 0002.png
│       ├── metadata.json            # Denormalized gen params (portable, self-contained)
│       └── *.mp4                    # Video outputs
│
├── thumbnails/                      # Pre-generated gallery thumbnails
│   └── {artifact_id}.webp           # WebP for size efficiency, UUID-named
│
├── temp/                            # Auto-cleaned intermediate files
│   ├── upload_{uuid}.tmp            # In-progress uploads
│   └── processing_{uuid}/           # Per-job temp workspace
│
├── imports/                         # User-uploaded source images
│   └── {uuid}_{original_name}       # Preserved original filename with UUID prefix
│
└── logs/                            # Optional file-based logs
    └── ailab_{date}.log
```

### Key design decisions:

1. **Model directories use `--` as path separator** — slashes in HF repo names (`org/model`) become `org--model` for filesystem safety.
2. **Artifacts are grouped by job_id** — makes cleanup trivial (delete one directory), and each job is self-contained.
3. **Thumbnails are separate from artifacts** — they are regeneratable (can be rebuilt from source), and a flat structure allows fast direct access.
4. **`.download_manifest` files** — JSON files tracking per-file download state, expected checksums, and resume offsets. Stored alongside the model so they survive database resets.
5. **`metadata.json` per job** — ensures generated outputs are self-documenting even without the database (portability).

---

## 4. Download Manifest Format

Each model directory contains a `.download_manifest` JSON file:

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
    },
    {
      "relative_path": "vae/diffusion_pytorch_model.safetensors",
      "expected_size_bytes": 334643268,
      "downloaded_bytes": 167321634,
      "sha256": null,
      "status": "partial"
    }
  ],
  "overall_status": "partial",
  "total_expected_bytes": 6800000000,
  "total_downloaded_bytes": 5302470690
}
```

This manifest enables:
- **Resume after crash** — knows exactly which files are incomplete and at what byte offset
- **Integrity verification** — expected SHA256 per file for post-download validation
- **Status recovery** — if the DB loses download state, the manifest is the source of truth on disk

---

## 5. Concrete Database Schema

### 5.1 `models` table

```sql
CREATE TABLE models (
    -- Identity
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id        VARCHAR(512) NOT NULL UNIQUE,   -- e.g., "stabilityai/stable-diffusion-xl-base-1.0"
    name            VARCHAR(255) NOT NULL,           -- Human-friendly display name
    source          VARCHAR(50) NOT NULL,            -- "huggingface" | "civitai" | "local"
    family          VARCHAR(50) NOT NULL DEFAULT 'custom', -- "sd15" | "sdxl" | "flux" | "custom"
    variant         VARCHAR(100) NOT NULL DEFAULT '',  -- "fp16", "fp32", "gguf-q4"

    -- Metadata
    description     TEXT NOT NULL DEFAULT '',
    tags            JSONB NOT NULL DEFAULT '[]'::jsonb,
    source_url      VARCHAR(1024) NOT NULL DEFAULT '',
    version         VARCHAR(100) NOT NULL DEFAULT '',  -- Model version string

    -- Size and storage
    total_size_bytes BIGINT NOT NULL DEFAULT 0,       -- Total expected size of all files
    disk_size_bytes  BIGINT NOT NULL DEFAULT 0,       -- Actual bytes on disk currently
    file_path       VARCHAR(1024) NOT NULL DEFAULT '', -- Root directory path on disk

    -- Lifecycle status
    status          VARCHAR(50) NOT NULL DEFAULT 'not_downloaded',
    download_progress INTEGER NOT NULL DEFAULT 0,     -- 0-100 percentage

    -- Integrity
    checksum        VARCHAR(128) NOT NULL DEFAULT '',  -- Overall model checksum (if available)
    is_verified     BOOLEAN NOT NULL DEFAULT FALSE,    -- Passed integrity check
    last_verified_at TIMESTAMPTZ,                      -- When integrity was last confirmed

    -- Capabilities (what this model can do)
    capabilities    JSONB NOT NULL DEFAULT '[]'::jsonb,  -- ["text_to_image", "image_to_image"]

    -- Timestamps
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes
CREATE INDEX ix_models_source_status ON models (source, status);
CREATE INDEX ix_models_family ON models (family);
CREATE INDEX ix_models_status ON models (status);
CREATE INDEX ix_models_tags ON models USING GIN (tags);
CREATE INDEX ix_models_capabilities ON models USING GIN (capabilities);
```

### 5.2 `model_files` table

```sql
CREATE TABLE model_files (
    -- Identity
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id        UUID NOT NULL REFERENCES models(id) ON DELETE CASCADE,

    -- File info
    relative_path   VARCHAR(1024) NOT NULL,   -- Path relative to model root directory
    size_bytes      BIGINT NOT NULL DEFAULT 0, -- Expected file size
    downloaded_bytes BIGINT NOT NULL DEFAULT 0, -- How much has been downloaded
    status          VARCHAR(50) NOT NULL DEFAULT 'pending',  -- pending | downloading | complete | failed
    sha256          VARCHAR(64) NOT NULL DEFAULT '',  -- Expected hash for verification
    is_verified     BOOLEAN NOT NULL DEFAULT FALSE,

    -- Timestamps
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Constraints
    CONSTRAINT uq_model_files_path UNIQUE (model_id, relative_path)
);

-- Indexes
CREATE INDEX ix_model_files_model_id ON model_files (model_id);
CREATE INDEX ix_model_files_status ON model_files (status);
```

### 5.3 `jobs` table

```sql
CREATE TABLE jobs (
    -- Identity
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Classification
    job_type        VARCHAR(50) NOT NULL,     -- "text_to_image" | "model_download" | etc.
    status          VARCHAR(50) NOT NULL DEFAULT 'pending',

    -- Progress tracking
    progress_percent INTEGER NOT NULL DEFAULT 0,
    current_step    INTEGER NOT NULL DEFAULT 0,
    total_steps     INTEGER NOT NULL DEFAULT 0,
    message         TEXT NOT NULL DEFAULT '',

    -- Priority and ordering
    priority        INTEGER NOT NULL DEFAULT 0,  -- Higher = processed first

    -- Timing
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    timeout_at      TIMESTAMPTZ,             -- When this job should be force-cancelled

    -- Parameters and results (flexible JSON for any job type)
    params          JSONB NOT NULL DEFAULT '{}'::jsonb,
    result          JSONB NOT NULL DEFAULT '{}'::jsonb,
    error           TEXT NOT NULL DEFAULT '',

    -- Retry tracking
    attempt         INTEGER NOT NULL DEFAULT 1,
    max_attempts    INTEGER NOT NULL DEFAULT 1,

    -- Relations
    model_id        UUID REFERENCES models(id) ON DELETE SET NULL,

    -- Timestamps
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes
CREATE INDEX ix_jobs_status ON jobs (status);
CREATE INDEX ix_jobs_type_status ON jobs (job_type, status);
CREATE INDEX ix_jobs_created_at ON jobs (created_at);
CREATE INDEX ix_jobs_priority_created ON jobs (status, priority DESC, created_at ASC)
    WHERE status = 'pending';  -- Partial index: only for the queue polling query
CREATE INDEX ix_jobs_model_id ON jobs (model_id);
```

### 5.4 `job_events` table

```sql
CREATE TABLE job_events (
    -- Identity
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id          UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,

    -- Event data
    from_status     VARCHAR(50) NOT NULL,
    to_status       VARCHAR(50) NOT NULL,
    message         TEXT NOT NULL DEFAULT '',
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,  -- Extra context (error details, etc.)

    -- Timestamp (immutable — append-only table)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes
CREATE INDEX ix_job_events_job_id ON job_events (job_id);
CREATE INDEX ix_job_events_created_at ON job_events (created_at);
```

### 5.5 `artifacts` table

```sql
CREATE TABLE artifacts (
    -- Identity
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- File info
    file_path       VARCHAR(1024) NOT NULL,
    thumbnail_path  VARCHAR(1024) NOT NULL DEFAULT '',
    media_type      VARCHAR(100) NOT NULL DEFAULT 'image/png',
    size_bytes      INTEGER NOT NULL DEFAULT 0,
    width           INTEGER NOT NULL DEFAULT 0,
    height          INTEGER NOT NULL DEFAULT 0,
    duration_seconds FLOAT,                  -- For video artifacts

    -- Generation metadata (denormalized for fast gallery queries)
    prompt          TEXT NOT NULL DEFAULT '',
    negative_prompt TEXT NOT NULL DEFAULT '',
    seed            BIGINT NOT NULL DEFAULT 0,
    model_name      VARCHAR(255) NOT NULL DEFAULT '',
    model_id_ref    VARCHAR(512) NOT NULL DEFAULT '',  -- model_id string (not FK, for display)
    generation_params JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- Relations
    job_id          UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,

    -- Gallery organization
    is_favorite     BOOLEAN NOT NULL DEFAULT FALSE,
    rating          INTEGER NOT NULL DEFAULT 0,   -- 0-5 stars
    notes           TEXT NOT NULL DEFAULT '',       -- User annotation

    -- Timestamps
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes
CREATE INDEX ix_artifacts_job_id ON artifacts (job_id);
CREATE INDEX ix_artifacts_created_at ON artifacts (created_at);
CREATE INDEX ix_artifacts_model_name ON artifacts (model_name);
CREATE INDEX ix_artifacts_is_favorite ON artifacts (is_favorite) WHERE is_favorite = TRUE;
CREATE INDEX ix_artifacts_generation_params ON artifacts USING GIN (generation_params);
```

---

## 6. Entity Relationships Diagram

```
┌─────────────────────┐
│       models        │
├─────────────────────┤
│ id (PK)             │
│ model_id (UNIQUE)   │──────┐
│ source              │      │
│ family              │      │
│ status              │      │
│ ...                 │      │
└─────────────────────┘      │
       │                     │
       │ 1:N (CASCADE)       │ 0:N (SET NULL)
       ▼                     ▼
┌─────────────────────┐  ┌─────────────────────┐
│    model_files      │  │       jobs          │
├─────────────────────┤  ├─────────────────────┤
│ id (PK)             │  │ id (PK)             │
│ model_id (FK)       │  │ model_id (FK, null) │
│ relative_path       │  │ job_type            │
│ size_bytes          │  │ status              │
│ downloaded_bytes    │  │ priority            │
│ status              │  │ params (JSONB)      │
│ sha256              │  │ ...                 │
└─────────────────────┘  └─────────────────────┘
                                │            │
                                │ 1:N        │ 1:N
                                │ (CASCADE)  │ (CASCADE)
                                ▼            ▼
                         ┌──────────────┐ ┌──────────────┐
                         │  artifacts   │ │  job_events  │
                         ├──────────────┤ ├──────────────┤
                         │ id (PK)      │ │ id (PK)      │
                         │ job_id (FK)  │ │ job_id (FK)  │
                         │ file_path    │ │ from_status  │
                         │ prompt       │ │ to_status    │
                         │ seed         │ │ message      │
                         │ ...          │ │ ...          │
                         └──────────────┘ └──────────────┘
```

---

## 7. Cascade and Deletion Strategy

| Parent | Child | ON DELETE | Rationale |
|--------|-------|-----------|-----------|
| `models` | `model_files` | CASCADE | Files are part of the model aggregate |
| `models` | `jobs` | SET NULL | Job history is preserved even if model is removed |
| `jobs` | `artifacts` | CASCADE | Artifacts have no meaning without their job |
| `jobs` | `job_events` | CASCADE | Events are internal audit for the job |

### Soft-delete consideration:

The system does **not** use soft-deletes. Rationale:
- Local-first application — no compliance/legal requirement to retain deleted records
- Simplifies queries (no `WHERE deleted_at IS NULL` everywhere)
- Model deletion triggers filesystem cleanup via the service layer
- Job/artifact deletion is rare (gallery cleanup is manual)

If future analytics are needed, `job_events` serves as the immutable history.

---

## 8. Audit and History Design

### Job Events (append-only log)

Every job state transition produces a `job_events` row:

```
Job created:     NULL → pending    (message: "Job queued")
Job started:     pending → running (message: "Worker picked up job")
Progress update: running → running (metadata: {step: 15, total: 50})
Job completed:   running → completed (message: "Generated 4 images")
Job failed:      running → failed  (message: "OOM error", metadata: {traceback: "..."})
Job cancelled:   running → cancelled (message: "User requested cancellation")
Job retried:     failed → pending  (metadata: {attempt: 2, reason: "auto_retry"})
```

This gives full observability into job lifecycle without polluting the main `jobs` table.

### Model history

Model lifecycle changes are tracked implicitly through jobs:
- Download → `MODEL_DOWNLOAD` job with events
- Load/unload → `MODEL_LOAD` job with events
- The `models.updated_at` timestamp shows the last modification

No separate model audit table is needed — the job system already captures the full history.

---

## 9. Important Constraints and Invariants

### Database-level constraints

```sql
-- Model status must be a known value
ALTER TABLE models ADD CONSTRAINT ck_models_status
    CHECK (status IN ('not_downloaded', 'downloading', 'download_paused',
                      'downloaded', 'loading', 'loaded', 'error'));

-- Job status must follow state machine
ALTER TABLE jobs ADD CONSTRAINT ck_jobs_status
    CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled'));

-- Download progress is 0-100
ALTER TABLE models ADD CONSTRAINT ck_models_download_progress
    CHECK (download_progress >= 0 AND download_progress <= 100);

-- Job progress is 0-100
ALTER TABLE jobs ADD CONSTRAINT ck_jobs_progress
    CHECK (progress_percent >= 0 AND progress_percent <= 100);

-- Artifact rating is 0-5
ALTER TABLE artifacts ADD CONSTRAINT ck_artifacts_rating
    CHECK (rating >= 0 AND rating <= 5);

-- Model file status must be valid
ALTER TABLE model_files ADD CONSTRAINT ck_model_files_status
    CHECK (status IN ('pending', 'downloading', 'complete', 'failed'));

-- Job priority is non-negative
ALTER TABLE jobs ADD CONSTRAINT ck_jobs_priority
    CHECK (priority >= 0);
```

### Application-level invariants (enforced by services)

1. **A model cannot be deleted while `status = 'loading'` or `status = 'loaded'`** — must unload first.
2. **A model cannot be loaded if `status != 'downloaded'` and `is_verified = FALSE`** — must download and verify first (configurable: can skip verification).
3. **Only one `MODEL_LOAD` job per model can be `running` at a time** — enforced by the orchestrator.
4. **A job can only transition forward** (except `failed → pending` for retries) — enforced by the service layer with state machine validation.
5. **Artifact files must exist on disk before the `artifacts` row is inserted** — write file first, then record.

---

## 10. JSONB Column Schemas

### `models.tags`
```json
["landscape", "anime", "photorealistic", "sdxl"]
```

### `models.capabilities`
```json
["text_to_image", "image_to_image"]
```

### `jobs.params` (varies by job_type)

**For `text_to_image`:**
```json
{
  "prompt": "a cat in space",
  "negative_prompt": "blurry",
  "width": 1024,
  "height": 1024,
  "num_inference_steps": 30,
  "guidance_scale": 7.5,
  "seed": 42,
  "num_images": 4,
  "scheduler": "euler_a"
}
```

**For `model_download`:**
```json
{
  "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
  "source": "huggingface",
  "variant": "fp16",
  "resume": true
}
```

**For `llm_inference`:**
```json
{
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 512,
  "temperature": 0.7
}
```

### `jobs.result` (varies by job_type)

**For generation jobs:**
```json
{
  "artifact_ids": ["uuid1", "uuid2"],
  "total_time_seconds": 12.5
}
```

**For download jobs:**
```json
{
  "total_bytes": 5135149056,
  "files_downloaded": 12
}
```

### `job_events.metadata`
```json
{
  "traceback": "...",
  "attempt": 2,
  "reason": "auto_retry",
  "gpu_memory_used_mb": 8192
}
```

---

## 11. Index Strategy Rationale

| Index | Query it optimizes |
|-------|--------------------|
| `ix_models_source_status` | "Show all HuggingFace models that are downloaded" |
| `ix_models_family` | "Show all SDXL models" |
| `ix_models_status` | "Show all models currently loading" |
| `ix_models_tags` (GIN) | "Find models tagged 'anime'" |
| `ix_models_capabilities` (GIN) | "Find models that support video generation" |
| `ix_model_files_model_id` | "Get all files for this model" |
| `ix_model_files_status` | "Find all incomplete file downloads" |
| `ix_jobs_status` | Worker polling: "Get next pending job" |
| `ix_jobs_type_status` | "Show all running generation jobs" |
| `ix_jobs_priority_created` (partial) | Queue ordering: priority DESC, then FIFO — only indexes pending jobs |
| `ix_jobs_model_id` | "Show all jobs for this model" |
| `ix_jobs_created_at` | "Show recent jobs" (pagination) |
| `ix_job_events_job_id` | "Get full history of this job" |
| `ix_artifacts_job_id` | "Get all outputs from this job" |
| `ix_artifacts_created_at` | Gallery pagination (newest first) |
| `ix_artifacts_model_name` | "Show all images generated with this model" |
| `ix_artifacts_is_favorite` (partial) | "Show favorites" — only indexes `TRUE` rows |
| `ix_artifacts_generation_params` (GIN) | Advanced search by generation parameters |

---

## 12. Model Cache Layout — HuggingFace vs Civitai vs Local

### HuggingFace models

HuggingFace models are multi-file repositories. The storage layout mirrors HF's structure:

```
models/huggingface/stabilityai--stable-diffusion-xl-base-1.0/
├── model_index.json          # Pipeline configuration
├── scheduler/
│   └── scheduler_config.json
├── text_encoder/
│   ├── config.json
│   └── model.safetensors
├── text_encoder_2/
│   ├── config.json
│   └── model.safetensors
├── tokenizer/
│   └── tokenizer_config.json
├── unet/
│   ├── config.json
│   └── diffusion_pytorch_model.fp16.safetensors
├── vae/
│   ├── config.json
│   └── diffusion_pytorch_model.fp16.safetensors
└── .download_manifest
```

This layout means `diffusers.from_pretrained()` can load directly from the directory — no conversion needed.

### Civitai models

Civitai models are typically single-file checkpoints:

```
models/civitai/12345/
├── dreamshaper_8.safetensors    # The checkpoint file
├── preview.png                  # Optional preview image
└── .download_manifest
```

### Local models

Locally imported models have a flexible structure:

```
models/local/my-custom-lora/
├── lora_weights.safetensors
└── .download_manifest           # Stores import metadata (no download tracking)
```

---

## 13. Artifact Portability — `metadata.json`

Each job output directory contains a self-describing metadata file:

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "job_type": "text_to_image",
  "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
  "model_family": "sdxl",
  "created_at": "2025-03-15T14:30:00Z",
  "params": {
    "prompt": "a cat wearing a spacesuit on the moon",
    "negative_prompt": "blurry, low quality",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "seed": 42,
    "scheduler": "euler_a"
  },
  "outputs": [
    {
      "filename": "0001.png",
      "media_type": "image/png",
      "width": 1024,
      "height": 1024,
      "size_bytes": 1245678,
      "seed": 42
    },
    {
      "filename": "0002.png",
      "media_type": "image/png",
      "width": 1024,
      "height": 1024,
      "size_bytes": 1198432,
      "seed": 43
    }
  ]
}
```

This makes artifact directories **portable and self-contained** — they can be:
- Backed up independently
- Shared between users
- Understood without database access
- Re-imported into a fresh installation

---

## 14. Updated ORM Models (reflecting the full schema)

The existing `models.py` should be expanded to include:

### New: `ModelFileRecord`

```python
class ModelFileRecord(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """
    A tracked file belonging to a model on disk.

    Used for multi-file models (HuggingFace repos) to track individual file
    download progress and integrity. Enables resumable downloads and per-file
    verification.
    """

    __tablename__ = "model_files"

    model_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("models.id", ondelete="CASCADE"), nullable=False
    )
    relative_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    downloaded_bytes: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    sha256: Mapped[str] = mapped_column(String(64), default="")
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    model: Mapped["ModelRecord"] = relationship(back_populates="files")

    __table_args__ = (
        Index("ix_model_files_model_id", "model_id"),
        Index("ix_model_files_status", "status"),
        # Unique path per model
        {"unique_together": ("model_id", "relative_path")},
    )
```

### New: `JobEventRecord`

```python
class JobEventRecord(Base, UUIDPrimaryKeyMixin):
    """
    Immutable audit entry for job state transitions.

    Append-only — never updated or deleted (except via CASCADE from parent job).
    Provides full observability into job lifecycle without cluttering the jobs table.
    """

    __tablename__ = "job_events"

    job_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
    )
    from_status: Mapped[str] = mapped_column(String(50), nullable=False)
    to_status: Mapped[str] = mapped_column(String(50), nullable=False)
    message: Mapped[str] = mapped_column(Text, default="")
    metadata: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    job: Mapped["JobRecord"] = relationship(back_populates="events")

    __table_args__ = (
        Index("ix_job_events_job_id", "job_id"),
        Index("ix_job_events_created_at", "created_at"),
    )
```

### Additions to existing `ModelRecord`

```python
# New columns to add:
variant: Mapped[str] = mapped_column(String(100), default="")
version: Mapped[str] = mapped_column(String(100), default="")
total_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
disk_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
last_verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
capabilities: Mapped[list] = mapped_column(JSONB, default=list)

# New relationship:
files: Mapped[list["ModelFileRecord"]] = relationship(back_populates="model", lazy="selectin")
```

### Additions to existing `JobRecord`

```python
# New columns to add:
priority: Mapped[int] = mapped_column(Integer, default=0)
timeout_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

# New relationship:
events: Mapped[list["JobEventRecord"]] = relationship(back_populates="job", lazy="selectin")
```

### Additions to existing `ArtifactRecord`

```python
# New columns to add:
duration_seconds: Mapped[float | None] = mapped_column(Float, default=None)
model_id_ref: Mapped[str] = mapped_column(String(512), default="")
is_favorite: Mapped[bool] = mapped_column(Boolean, default=False)
rating: Mapped[int] = mapped_column(Integer, default=0)
notes: Mapped[str] = mapped_column(Text, default="")
```

---

## 15. Query Patterns (what the repositories will optimize for)

| Repository | Key query | Implementation |
|------------|-----------|----------------|
| `ModelRepository` | List models with filters (source, family, status, tag) | Dynamic WHERE + GIN index on tags |
| `ModelRepository` | Get model with all files | Eager load `files` relationship |
| `ModelRepository` | Find incomplete downloads | `WHERE status = 'downloading'` |
| `ModelFileRepository` | Get incomplete files for a model | `WHERE model_id = ? AND status != 'complete'` |
| `JobRepository` | Poll next job for worker | `WHERE status = 'pending' ORDER BY priority DESC, created_at ASC LIMIT 1 FOR UPDATE SKIP LOCKED` |
| `JobRepository` | Get job with events and artifacts | Eager load both relationships |
| `JobRepository` | List recent jobs (paginated) | `ORDER BY created_at DESC LIMIT ? OFFSET ?` |
| `ArtifactRepository` | Gallery browse (paginated, newest first) | `ORDER BY created_at DESC` with cursor pagination |
| `ArtifactRepository` | Filter by model | `WHERE model_name = ?` |
| `ArtifactRepository` | Get favorites | Partial index on `is_favorite = TRUE` |
| `ArtifactRepository` | Search by prompt text | `WHERE prompt ILIKE '%term%'` (upgrade to FTS if needed) |

### Worker polling query (critical path)

```sql
-- Atomic job claim: picks next pending job and marks it running in one query
UPDATE jobs
SET status = 'running', started_at = now(), updated_at = now()
WHERE id = (
    SELECT id FROM jobs
    WHERE status = 'pending'
    ORDER BY priority DESC, created_at ASC
    LIMIT 1
    FOR UPDATE SKIP LOCKED
)
RETURNING *;
```

`FOR UPDATE SKIP LOCKED` ensures multiple workers (if scaled later) don't contend on the same row.

---

## 16. Data Lifecycle Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA LIFECYCLE FLOWS                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  MODEL DOWNLOAD:                                                    │
│  API request → Job created (pending) → Worker claims job →          │
│  Download starts → model_files rows track progress →                │
│  .download_manifest updated on disk → All files complete →          │
│  Integrity verified → Model status = "downloaded" →                 │
│  Job status = "completed"                                           │
│                                                                     │
│  GENERATION:                                                        │
│  API request → Job created (pending) → Worker claims job →          │
│  Model loaded (if not already) → Inference runs →                   │
│  Artifacts saved to disk → metadata.json written →                  │
│  Thumbnails generated → Artifact rows inserted →                    │
│  Job status = "completed"                                           │
│                                                                     │
│  MODEL DELETION:                                                    │
│  API request → Verify model is not loaded → Delete model_files →    │
│  Delete filesystem directory → Delete model row (cascades files) →  │
│  Associated jobs retain history (model_id set NULL)                 │
│                                                                     │
│  ARTIFACT DELETION:                                                 │
│  API request → Delete files from disk → Delete thumbnail →          │
│  Delete artifact row → Job retains metadata                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 17. Design Decisions Log

| Decision | Chosen | Alternatives considered | Reason |
|----------|--------|------------------------|--------|
| Primary keys | UUID v4 | Auto-increment, ULID | Client-side generation, no coordination needed, standard |
| JSON columns | JSONB | Separate tables | Flexible for varying params across job types; GIN-indexable |
| Model file tracking | Separate table | Single JSON column | Enables per-file resume, per-file verification, proper indexing |
| Job event history | Separate append-only table | Columns on job table | Unbounded history without bloating the hot jobs table |
| Deletion strategy | Hard delete | Soft delete | Local app, no compliance needs, simpler queries |
| Artifact metadata | Denormalized on artifact | Join to job.params | Gallery queries are the hot path; avoid joins |
| Download manifest | Both DB + disk file | DB only | Disk file survives DB reset, enables portable recovery |
| Thumbnail storage | Flat directory | Nested by date | Simple, fast UUID-based lookup, no date logic |
| BIGINT for sizes | Yes | INTEGER | Model files regularly exceed 2GB (INT max = 2.1GB) |
| Partial indexes | For pending jobs, favorites | Full indexes | Smaller index size, faster queries on the hot subset |
