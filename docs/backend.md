# Backend Documentation

## Stack

- **Runtime**: Python 3.12, FastAPI (async), Uvicorn
- **Database**: PostgreSQL 16 via SQLAlchemy (asyncpg driver), Alembic migrations
- **Storage**: Local filesystem (`/app/storage`)
- **Inference**: PyTorch/Diffusers (direct), optional BentoML, optional ComfyUI
- **Real-time**: Server-Sent Events (SSE)

---

## Configuration (`.env` / environment variables)

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://ailab:ailab_local@localhost:5432/ailab` | PostgreSQL connection |
| `STORAGE_ROOT` | `/app/storage` | Root for all local file storage |
| `INFERENCE_DEVICE` | `auto` | `cuda`, `cpu`, or `auto` |
| `INFERENCE_BACKEND` | `direct_python` | Default backend: `direct_python`, `bentoml`, `comfyui` |
| `MAX_VRAM_MB` | `0` (unlimited) | VRAM budget for the GPU lock |
| `MAX_CACHED_PIPELINES` | `1` | How many pipelines to keep hot in memory |
| `MAX_WORKERS` | `1` | Max concurrent GPU jobs (serialize = 1) |
| `JOB_POLL_INTERVAL` | `1.0` | Seconds between worker queue polls |
| `BENTOML_ENABLED` | `false` | Enable BentoML adapter |
| `BENTOML_URL` | `http://localhost:3000` | BentoML service URL |
| `COMFYUI_ENABLED` | `false` | Enable ComfyUI adapter |
| `COMFYUI_URL` | `http://localhost:8188` | ComfyUI service URL |
| `HUGGINGFACE_TOKEN` | `` | HF token (for gated/private models) |
| `CIVITAI_TOKEN` | `` | Civitai API key |
| `BACKEND_PORT` | `8000` | HTTP port |
| `CORS_ORIGINS` | `["http://localhost:5173","http://localhost:3000"]` | Allowed origins |

Storage sub-directories are resolved under `STORAGE_ROOT`:
- `models/` — downloaded model weights
- `artifacts/` — generated output files
- `thumbnails/` — thumbnail cache
- `temp/` — temporary files
- `imports/` — imported source files

---

## Database Schema

Five tables, managed by Alembic migrations.

### `models`
Model catalog. One row per registered model version.

| Column | Type | Notes |
|---|---|---|
| `id` | UUID PK | Internal identifier |
| `model_id` | String(512) UNIQUE | External ID (e.g. HF repo `org/name`) |
| `name` | String(255) | Human-readable name |
| `preferred_name` | String(255) | Display name override |
| `source` | String(50) | `huggingface`, `civitai`, `local` |
| `family` | String(50) | `sd15`, `sdxl`, `flux`, `custom` |
| `variant` | String(100) | `fp16`, `fp32`, `gguf-q4`, etc. |
| `capabilities` | JSONB `[]` | e.g. `["text_to_image","image_to_image"]` |
| `status` | String(50) | See Model Status below |
| `download_progress` | Integer | 0–100 |
| `local_path` | String | Absolute path on disk after download |
| `file_count` | Integer | Number of files in the model |
| `total_size_bytes` | BigInteger | Total size on disk |
| `download_size_bytes` | BigInteger | Estimated download size |
| `recommended_vram_min_gb` | Integer | Minimum VRAM recommendation |
| `recommended_vram_max_gb` | Integer | Maximum VRAM recommendation |
| `is_verified` | Boolean | Integrity check passed |

### `model_files`
Per-file tracking for multi-file models (HuggingFace repos). Enables resume.

| Column | Type | Notes |
|---|---|---|
| `model_id` | UUID FK → `models.id` CASCADE | Parent model |
| `relative_path` | String(1024) | Path within the model directory |
| `size_bytes` | BigInteger | Total file size |
| `downloaded_bytes` | BigInteger | Bytes downloaded so far |
| `status` | String(50) | `pending`, `downloading`, `done` |
| `sha256` | String(64) | Integrity hash |

### `jobs`
Background job queue. All async work is a job.

| Column | Type | Notes |
|---|---|---|
| `id` | UUID PK | |
| `job_type` | String(50) | See Job Types below |
| `status` | String(50) | See Job Status below |
| `progress_percent` | Integer | 0–100 |
| `current_step` | Integer | e.g. current diffusion step |
| `total_steps` | Integer | e.g. total diffusion steps |
| `message` | Text | Human-readable progress message |
| `priority` | Integer | Higher = picked first |
| `params` | JSONB | Serialized job parameters |
| `result` | JSONB | Output references on completion |
| `error` | Text | Error message/traceback on failure |
| `attempt` | Integer | Current attempt number |
| `max_attempts` | Integer | Max retries allowed |
| `model_id` | UUID FK → `models.id` SET NULL | Associated model |
| `started_at` | DateTime | When execution began |
| `completed_at` | DateTime | When execution ended |
| `timeout_at` | DateTime | Expiry deadline |

### `job_events`
Immutable audit log. Append-only; never updated.

| Column | Type | Notes |
|---|---|---|
| `job_id` | UUID FK → `jobs.id` CASCADE | Parent job |
| `from_status` | String(50) | Previous status |
| `to_status` | String(50) | New status |
| `message` | Text | Transition note |
| `metadata` | JSONB | Additional transition data |
| `created_at` | DateTime | Immutable timestamp |

### `artifacts`
Generated outputs (images, video, text). File metadata is denormalized for fast gallery queries.

| Column | Type | Notes |
|---|---|---|
| `job_id` | UUID FK → `jobs.id` CASCADE | Producing job |
| `file_path` | String(1024) | Absolute path on disk |
| `thumbnail_path` | String(1024) | Thumbnail path |
| `media_type` | String(100) | `image/png`, `video/mp4`, etc. |
| `size_bytes` | Integer | File size |
| `width` / `height` | Integer | Image dimensions |
| `duration_seconds` | Float | Video length |
| `prompt` | Text | Generation prompt (denormalized) |
| `negative_prompt` | Text | |
| `seed` | BigInteger | RNG seed used |
| `model_name` | String(255) | Model used (denormalized) |
| `generation_params` | JSONB | Full params snapshot |
| `is_favorite` | Boolean | Gallery flag |
| `rating` | Integer | 0–5 stars |
| `notes` | Text | User annotation |

---

## Enumerations

### ModelSource
`huggingface` | `civitai` | `local`

### ModelFamily
`sd15` | `sdxl` | `flux` | `custom`

### ModelStatus
`not_downloaded` → `downloading` → `downloaded` → `loading` → `loaded` | `error`
Also: `download_paused`

### JobStatus (state machine)
`pending` → `running` → `completed`
                     → `failed`
                     → `cancelled`

### JobType
| Value | Description |
|---|---|
| `text_to_image` | Inference: generate image from prompt |
| `image_to_image` | Inference: transform image with prompt |
| `image_captioning` | Inference: describe an image |
| `video_generation` | Inference: generate video |
| `llm_inference` | Inference: LLM chat completion |
| `model_download` | Model lifecycle: download files |
| `model_delete` | Model lifecycle: delete files |
| `model_refresh` | Model lifecycle: refresh metadata |

### InferenceBackend
`direct_python` | `bentoml` | `comfyui`

---

## API Endpoints

All routes are prefixed `/api/v1`. Interactive docs at `/docs` (Swagger) and `/redoc`.

### Generation — `/api/v1/generation`

#### `POST /api/v1/generation/text-to-image` → `202`
Submit a text-to-image job. Returns immediately with a `job_id`.

Request body (`application/json`):
```json
{
  "prompt": "string (required)",
  "negative_prompt": "",
  "model_id": "string (required)",
  "width": 512,
  "height": 512,
  "num_inference_steps": 20,
  "guidance_scale": 7.5,
  "seed": null,
  "num_images": 1
}
```

Constraints: `width`/`height` 64–2048, multiple of 8. `num_inference_steps` 1–150. `guidance_scale` 1.0–30.0. `num_images` 1–8.

Optional request header: `X-Correlation-ID` — echoed back in response and all related SSE events.

Response:
```json
{ "job_id": "uuid", "status": "pending", "message": "Job submitted", "correlation_id": null }
```

---

#### `POST /api/v1/generation/describe` → `202`
Submit an image captioning job. Accepts `multipart/form-data`.

Fields:
- `model_id` (string, required)
- `image` (file, required) — any image format

Response: same `JobResponse` as above.

---

#### `GET /api/v1/generation/jobs/{job_id}`
Poll status of a generation job. Also returns artifact references when `status == "completed"`.

Response:
```json
{
  "id": "uuid",
  "status": "pending|running|completed|failed|cancelled",
  "job_type": "text_to_image",
  "created_at": "ISO8601",
  "updated_at": "ISO8601",
  "progress_percent": 0,
  "error": "",
  "model_id": "uuid",
  "params": {}
}
```

---

### Jobs — `/api/v1/jobs`

#### `GET /api/v1/jobs/`
List jobs with optional filtering and pagination.

Query params:
- `status` — filter by status value
- `job_type` — filter by job type value
- `limit` (1–100, default 50)
- `offset` (default 0)

Response: paginated `{ items: [...], total, limit, offset, has_more }` with full `JobDetailResponse` objects including `current_step`, `total_steps`, `started_at`, `completed_at`, `timeout_at`, `attempt`, `max_attempts`.

---

#### `GET /api/v1/jobs/{job_id}`
Full detail for a single job.

---

#### `POST /api/v1/jobs/{job_id}/cancel`
Request cancellation. If the job is pending it will be cancelled before execution. If running, the worker marks it cancelled after the current step.

Response: `{ job_id, status, message: "Cancellation requested" }`

---

#### `POST /api/v1/jobs/{job_id}/retry`
Retry a failed job. Resets status to `pending` with `attempt` incremented.

Response: full `JobDetailResponse`.

---

#### `GET /api/v1/jobs/{job_id}/events`
Audit trail — all state transitions for a job.

Response: list of `{ id, job_id, from_status, to_status, message, metadata, created_at }`.

---

#### `GET /api/v1/jobs/{job_id}/timeline`
In-memory typed observability events for a specific job (from the SSE event ring buffer, not DB).

Query params: `limit` (1–2000, default 500)

Response: `{ job_id, events: [TypedEvent] }`.

---

### Models — `/api/v1/models`

#### `GET /api/v1/models/`
List all registered models.

Query params:
- `source` — filter by source (`huggingface`, `civitai`, `local`)
- `capabilities` — comma-separated OR filter (e.g. `txt2img,img2img`)
- `limit` (1–100, default 50)
- `offset` (default 0)

Response: list of full `ModelRegistryResponse` objects (see schema above).

---

#### `GET /api/v1/models/{model_id}`
Get details for a specific model. `model_id` is the external identifier (e.g. `stabilityai/sdxl-base-1.0`). Uses `:path` matcher so slashes are allowed.

---

#### `POST /api/v1/models/` → `201`
Register a new model in the catalog (metadata only; no download triggered).

Request body:
```json
{
  "model_id": "org/repo",
  "name": "Display Name",
  "source": "huggingface",
  "family": "sdxl",
  "variant": "fp16",
  "description": "",
  "tags": [],
  "source_url": "",
  "preferred_name": "",
  "capabilities": ["text_to_image"]
}
```

If the `model_id` is already registered, returns the existing record (idempotent).

---

#### `POST /api/v1/models/{model_id}/download` → `202`
Trigger download of an already-registered model. Creates a `model_download` job and returns a `job_id`. Progress tracked via SSE or job polling.

Response: `{ job_id, status: "pending", message: "Download queued" }`.

---

#### `DELETE /api/v1/models/{model_id}` → `204`
Delete a model's files from disk and remove the catalog entry.

---

### Artifacts — `/api/v1/artifacts`

#### `GET /api/v1/artifacts/`
List generated artifacts (gallery).

Query params:
- `model_name` — filter by model name string
- `media_type` — filter by MIME type (e.g. `image/png`, `video/mp4`)
- `is_favorite` — filter by favorite flag (`true`/`false`)
- `limit` (1–100, default 50)
- `offset` (default 0)

Response: paginated list of `ArtifactResponse`.

---

#### `GET /api/v1/artifacts/{artifact_id}`
Get metadata for one artifact.

---

#### `PATCH /api/v1/artifacts/{artifact_id}`
Update gallery metadata (partial update).

Request body (all fields optional):
```json
{
  "is_favorite": true,
  "rating": 4,
  "notes": "nice composition"
}
```

`rating` must be 0–5.

---

#### `GET /api/v1/artifacts/{artifact_id}/file`
Serve the raw file. Returns `FileResponse` with the correct `media_type`.

**This is the URL the FE should use to display images/videos.** The `file_path` field in `ArtifactResponse` is already set to this URL (`/api/v1/artifacts/{id}/file`).

---

#### `DELETE /api/v1/artifacts/{artifact_id}` → `204`
Delete the artifact record and file from disk.

---

### System — `/api/v1/system`

#### `GET /api/v1/system/health`
Simple health check.

Response: `{ status: "healthy", warnings: [], blockers: [], recommendations: [] }`

Status values: `ok` | `busy` | `degraded` | `error`

Warnings emitted when:
- More than 10 jobs pending → `queue_backlog_high`
- VRAM usage > 85% of budget → `high_vram_pressure`
- OOM event in last 15 min → `oom_detected`
- Last model load failed → `model_load_failed`

Blockers (→ `error` status):
- Jobs pending but worker not running → `queue_stalled`

---

#### `GET /api/v1/system/status`
Model count statistics.

Response: `{ total_models: 42, models_by_family: { "sdxl": 10, "sd15": 32 } }`

---

### SSE — `/sse/observability`

#### `GET /sse/observability`
Server-Sent Events stream. The client receives a continuous stream of JSON-encoded typed events.

Optional query param: `subscribe=job,model` — comma-separated event category filter. With no filter the client receives all events.

Categories match the `event_type` prefix:
- `job` — `job.enqueued`, `job.started`, `job.progress`, `job.completed`, `job.failed`, `job.cancelled`
- `model` — `model.downloaded`, `model.download_failed`
- `resource` — `resource.lock_acquired`, `resource.lock_released`, `resource.oom`
- `artifact` — `job.artifact_saved`
- `system` — generic system events

Each SSE frame:
```
data: {"event_id":"...", "event_type":"job.progress", "timestamp":"...", "correlation_id":null, "component":"job", "level":"info", "job_id":"...", "message":"Downloading model.safetensors: 45%", "payload":{...}}

```

Keep-alive frames are sent every ~1 second when idle: `: keepalive`

---

## Inference Backends

Three backends are available. The active backend per job type is resolved in this order:
1. Backend specified in the job's `params.backend` field
2. Configured default (`INFERENCE_BACKEND`)
3. Fallback to `direct_python`

### Direct Python (`direct_python`)
Always available. Runs diffusers/transformers pipelines in-process. Supports:
- text-to-image, image-to-image (Stable Diffusion via diffusers)
- image-captioning / vision (BLIP/Florence)
- video generation
- LLM inference

Pipelines are cached in `PipelineCache` (LRU, size controlled by `MAX_CACHED_PIPELINES`).

### BentoML (`bentoml`)
Optional. Delegates to an external BentoML service (`BENTOML_URL`). Enabled with `BENTOML_ENABLED=true`. Supports all five job types via HTTP calls to the BentoML service.

### ComfyUI (`comfyui`)
Optional. Delegates to a ComfyUI instance (`COMFYUI_URL`). Enabled with `COMFYUI_ENABLED=true`. Supports text-to-image, image-to-image, video. Does **not** support image captioning or LLM.

---

## Model Sources

### HuggingFace
- Resolves the full file manifest via the HF Hub API
- Downloads each file with `hf_hub_download` (handles LFS, caching, token auth)
- Skips files already present on disk
- Stores under `storage/models/huggingface/{org}/{repo}/`

### Civitai
- Fetches model versions via `https://civitai.com/api/v1/models/{id}`
- `model_id` format: `12345` (numeric ID) or `12345/version-id`
- Downloads only `type: "Model"` files (skips metadata/preview)
- Streaming download with progress and resume support
- Stores under `storage/models/civitai/{model_id}/`

### Local
- For models already present on the filesystem
- No download step needed

---

## Job Worker

A single background asyncio task (`JobWorker`) polls the database every `JOB_POLL_INTERVAL` seconds for `PENDING` jobs and processes them one at a time (configurable via `MAX_WORKERS`).

### Dispatch logic
```
PENDING job claimed
    ├─ job_type in [text_to_image, image_to_image, image_captioning, video_generation, llm_inference]
    │   └─ acquire GPU lock (ResourceCoordinator semaphore)
    │       └─ PipelineExecutor → AdapterRegistry → concrete adapter
    │           └─ release GPU lock
    └─ job_type in [model_download, model_delete, model_refresh]
        └─ ModelOperationHandler (no GPU lock)
```

### GPU lock
`ResourceCoordinator` uses an `asyncio.Semaphore(MAX_WORKERS)` (default 1). Inference jobs wait in FIFO order. VRAM usage is tracked for monitoring but does not gate admission.

---

## Key Workflows

### Text-to-Image Generation

1. Client `POST /api/v1/generation/text-to-image` with prompt + model_id
2. `GenerationService` → `ModelResolver` resolves `model_id` to a filesystem path
3. `JobCreator` writes a `PENDING` job to DB; publishes `job.enqueued` SSE event
4. Response → client gets `job_id` (202)
5. Background: `JobWorker` claims the job, acquires GPU lock
6. `PipelineExecutor` → correct adapter's `generate()` method runs inference
7. Output images saved to `storage/artifacts/{job_id}/{uuid}.png`
8. `ArtifactRecord` rows written to DB
9. Job marked `COMPLETED`; `job.completed` and `job.artifact_saved` SSE events published
10. Client polls `GET /api/v1/jobs/{job_id}` or reads SSE to detect completion
11. Client fetches `GET /api/v1/artifacts/` or `GET /api/v1/artifacts/{id}/file` to display results

### Model Download

1. Model must already be registered (`POST /api/v1/models/`)
2. Client `POST /api/v1/models/{model_id}/download`
3. `JobCreator` writes a `PENDING` `model_download` job; publishes `job.enqueued`
4. Response → client gets `job_id` (202)
5. Background: `JobWorker` claims the job (no GPU lock needed)
6. `ModelOperationHandler` picks the source provider (HuggingFace or Civitai)
7. Provider resolves the file manifest; job `total_steps` set to file count
8. Each file downloads sequentially; DB updated every 1% progress; `job.progress` SSE events fired
9. After all files done: model status → `downloaded`, `local_path` recorded
10. `model.downloaded` SSE event published; job marked `COMPLETED`

### Image Describe (Captioning)

1. Client `POST /api/v1/generation/describe` with `multipart/form-data` (image + model_id)
2. Image saved to `/tmp/{uuid}.ext`
3. `JobCreator` writes `PENDING` `image_captioning` job with `image_path`
4. Worker → GPU lock → `VisionProvider.caption()` → text result
5. Job marked `COMPLETED` (no artifact saved currently — caption is in job result/logs)

---

## Real-time Event Reference

| `event_type` | `component` | `level` | When |
|---|---|---|---|
| `job.enqueued` | `job` | `info` | Job written to DB |
| `job.started` | `job` | `info` | Worker picked up the job |
| `job.progress` | `job` | `info` | Download/inference step progress |
| `job.completed` | `job` | `info` | Job finished successfully |
| `job.failed` | `job` | `error` | Job threw an exception |
| `job.cancelled` | `job` | `info` | Job cancelled before or during execution |
| `job.artifact_saved` | `artifact` | `info` | Artifacts written to disk after completion |
| `model.downloaded` | `model` | `info` | Model download finished |
| `model.download_failed` | `model` | `error` | Model download failed |
| `resource.lock_acquired` | `resource` | `info` | GPU lock acquired; payload has `wait_seconds` |
| `resource.lock_released` | `resource` | `info` | GPU lock released |
| `resource.oom` | `resource` | `error` | CUDA out-of-memory detected |

All events carry: `event_id`, `event_type`, `timestamp`, `correlation_id`, `component`, `level`, `job_id`, `message`, `payload`.

---

## Observability

- **Ring buffer**: last 1000 typed events kept in memory; accessible via `GET /api/v1/jobs/{job_id}/timeline`
- **Metrics**: in-memory counters/gauges updated on each event (`jobs_submitted`, `jobs_failed`, `cache_hits`, `queue_depth`, etc.)
- **Health**: `GET /api/v1/system/health` aggregates GPU state, queue depth, OOM history, and cache state into a single health signal
- **Structured logs**: every event is also emitted as a JSON log line by the `EventBus`

---

## Error Handling

All routers use a common `from_exception()` helper that maps domain errors to HTTP status codes:

| Exception | HTTP |
|---|---|
| `JobNotFoundError` | 404 |
| `ValueError` (model not found, etc.) | 404 |
| Other | 500 |

Validation errors return FastAPI's default 422 with field-level detail.

---

## Deployment

```
docker-compose up
```

Services:
- `db` — PostgreSQL 16 on port 5432
- `backend` — FastAPI on port 8000
- `frontend` — Vite dev server on port 5173

Storage is persisted in the `storage_data` Docker volume. Models are stored under `storage/models/`, artifacts under `storage/artifacts/`.

Alembic migrations run automatically on startup (`alembic upgrade head` in the Dockerfile entrypoint).
