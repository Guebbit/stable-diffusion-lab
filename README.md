# AI Lab

A **local-first, open-source AI lab** for image generation, video generation, and local model lifecycle management. Built with **Vue 3 + TypeScript** frontend and **Python FastAPI** backend with PostgreSQL.

---

## Architecture

```
┌─────────────────────────────┐    REST/WS     ┌──────────────────────────────────┐
│  Frontend  (Vue 3/Vuetify)  │ ◄────────────► │  Backend  (FastAPI + Workers)    │
│  http://localhost:5173      │                 │  http://localhost:8000           │
└─────────────────────────────┘                 └──────────┬───────────────────────┘
                                                           │
                                                ┌──────────┴──────────┐
                                                │  PostgreSQL 16      │
                                                │  Local filesystem   │
                                                │  GPU (CUDA/CPU/MPS) │
                                                └─────────────────────┘
```

### Backend Layers

```
API (routers, schemas, websocket)
 ↓
Services (business logic)
 ↓
Orchestrator (job queue, event bus)
 ↓
Adapters (direct python / bentoml / comfyui)
 ↓
Infrastructure (config, database, storage)
 ↓
Domain (enums, value objects, protocols)
```

## Features

- **Text-to-image** — generate images from text prompts
- **Image-to-image** — transform existing images with guidance
- **Vision/captioning** — describe images using vision-language models
- **Video generation** — create video from text or image inputs
- **Local LLM** — chat completions with local language models
- **Model management** — download, install, load, unload, delete models from HuggingFace and CivitAI
- **Async job queue** — all heavy inference runs in background workers
- **Real-time progress** — WebSocket updates for job status
- **Multiple inference backends** — direct Python, BentoML, ComfyUI (pluggable)

---

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) ≥ 24
- [Docker Compose](https://docs.docker.com/compose/) v2
- *(Optional)* NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### Run

```bash
git clone https://github.com/Guebbit/stable-diffusion-lab.git
cd stable-diffusion-lab

# (Optional) provide API keys for gated models
cp .env-example .env
# Edit .env with your HUGGINGFACE_TOKEN and CIVITAI_TOKEN

docker compose up --build
```

Open **http://localhost:5173** in your browser.

> **GPU support**: Uncomment the `deploy` section in `docker-compose.yml` to pass the NVIDIA GPU into the backend container.

---

## Development Setup

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Start PostgreSQL (via Docker)
docker compose up db -d

# Run the app
uvicorn app.main:create_app --factory --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Linting & Type Checking

```bash
cd backend
ruff check app/            # Lint
ruff format app/           # Format
mypy app/                  # Type check
pytest                     # Run tests
```

---

## API Reference (v1)

All endpoints are prefixed with `/api/v1`.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/generation/text-to-image` | Submit text-to-image job (202) |
| `GET` | `/jobs/` | List jobs with filtering and pagination |
| `GET` | `/jobs/{job_id}` | Get job detail |
| `POST` | `/jobs/{job_id}/cancel` | Cancel a running job |
| `POST` | `/jobs/{job_id}/retry` | Retry a failed job |
| `GET` | `/jobs/{job_id}/events` | Get job events log |
| `GET` | `/jobs/{job_id}/timeline` | Get job timeline |
| `GET` | `/models/` | List registered models |
| `GET` | `/models/{model_id}` | Get model detail |
| `POST` | `/models/` | Register a new model |
| `POST` | `/models/{model_id}/download` | Trigger model download |
| `DELETE` | `/models/{model_id}` | Delete model from catalog and disk |
| `GET` | `/artifacts/` | List generated artifacts |
| `DELETE` | `/artifacts/{artifact_id}` | Delete an artifact |
| `GET` | `/system/health` | Basic health check |
| `GET` | `/system/status` | System status, device info, loaded models |
| `GET` | `/sse/observability` | Real-time event streaming (SSE) |

Full interactive docs at **http://localhost:8000/docs** (Swagger UI).

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL connection string |
| `STORAGE_ROOT` | `/app/storage` | Root directory for models, artifacts, temp files |
| `HUGGINGFACE_TOKEN` | *(empty)* | HuggingFace token for gated models |
| `CIVITAI_TOKEN` | *(empty)* | CivitAI API token for downloads |
| `INFERENCE_DEVICE` | `auto` | Force device: `cuda`, `cpu`, `mps`, or `auto` |
| `INFERENCE_BACKEND` | `direct_python` | Default backend: `direct_python`, `bentoml`, `comfyui` |
| `MAX_WORKERS` | `1` | Concurrent job worker count |
| `DEBUG` | `false` | Enable debug mode |

---

## Documentation

See [`docs/`](./docs/) for:
- [Architecture design](./docs/architecture/) — system design, module boundaries, naming rules

---

## OpenAPI Contract & Code Generation

The OpenAPI specification at the project root (`openapi.yaml` / `openapi.json`) is the **single source of truth** for REST API contracts between frontend and backend.

### Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Backend:  make export-openapi                               │
│     → writes openapi.json + openapi.yaml to project root        │
│                                                                 │
│  2. Frontend: npm run gen:api                                   │
│     → orval generates TypeScript types + API client              │
│     → output: frontend/src/api/gen/                             │
└─────────────────────────────────────────────────────────────────┘
```

### Commands

```bash
# Export OpenAPI schema from FastAPI (no server needed)
cd backend && make export-openapi

# Generate frontend API client from OpenAPI schema
cd frontend && npm run gen:api
```

### Orval Configuration

Orval is configured in `frontend/orval.config.ts` to:
- Read from `../openapi.yaml` (project root)
- Generate TypeScript types and axios-based API clients
- Output to `frontend/src/api/gen/`
- Organize output by API tags

---

## Event-Driven Events (SSE)

The backend emits typed observability events via Server-Sent Events at `GET /sse/observability`. Clients can filter events using the `subscribe` query parameter (comma-separated event types).

| Event Type | Component | Fields in `payload` | Description |
|---|---|---|---|
| `system.event` | `system` | `status`, `devices`, `uptime` | System health, device availability, startup/shutdown |
| `job.event` | `job` | `job_id`, `status`, `progress`, `model_name`, `backend` | Job lifecycle: created → running → completed/failed/cancelled |
| `model.event` | `model` | `model_id`, `model_name`, `status`, `progress`, `backend` | Model lifecycle: registered → downloading → ready/failed |
| `resource.event` | `resource` | `gpu_id`, `memory_used`, `memory_total`, `locks` | GPU memory allocation/deallocation, resource lock events |
| `artifact.event` | `artifact` | `artifact_id`, `file_path`, `job_id`, `mime_type` | Generated artifact saved and available |

### Event Schema

All events share this common structure:

```jsonc
{
  "event_id": "uuid",           // Unique event identifier
  "event_type": "job.event",    // Event category
  "timestamp": "ISO-8601",      // Event creation time
  "correlation_id": "uuid|null", // Links events across components
  "component": "job",           // Event source component
  "level": "info",              // info, warning, error
  "job_id": "uuid|null",        // Associated job identifier
  "payload": {}                 // Event-specific data
}
```

### Usage Example

```typescript
const eventSource = new EventSource('http://localhost:8000/sse/observability?subscribe=job.event,artifact.event');

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`[${data.event_type}] ${JSON.stringify(data.payload)}`);
};
```

---

## License

This project is licensed under the MIT License.