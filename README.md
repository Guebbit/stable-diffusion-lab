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
# Edit .env with your HF_TOKEN and CIVITAI_TOKEN

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
| `GET` | `/system/status` | Full runtime/GPU/model/job snapshot with health warnings |
| `GET` | `/system/events` | Recent structured observability events |
| `GET` | `/system/jobs/{job_id}/timeline` | Per-job observability timeline |
| `GET` | `/system/metrics` | In-memory counters/gauges/histograms snapshot |
| `GET` | `/models/` | List registered models |
| `POST` | `/models/` | Register a new model |
| `POST` | `/models/{model_id}/download` | Trigger model download |
| `DELETE` | `/models/{model_id}` | Delete model from catalog and disk |
| `POST` | `/generation/text-to-image` | Submit text-to-image job (202) |
| `GET` | `/generation/jobs/{job_id}` | Get job status and progress |
| `WS` | `/ws/progress` | Real-time job progress updates |
| `WS` | `/ws/observability` | Real-time structured observability event stream |

Full interactive docs at **http://localhost:8000/docs** (Swagger UI).

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL connection string |
| `STORAGE_ROOT` | `/app/storage` | Root directory for models, artifacts, temp files |
| `HF_TOKEN` | *(empty)* | HuggingFace token for gated models |
| `CIVITAI_TOKEN` | *(empty)* | CivitAI API token for downloads |
| `INFERENCE_DEVICE` | `auto` | Force device: `cuda`, `cpu`, `mps`, or `auto` |
| `INFERENCE_BACKEND` | `direct_python` | Default backend: `direct_python`, `bentoml`, `comfyui` |
| `MAX_WORKERS` | `1` | Concurrent job worker count |
| `DEBUG` | `false` | Enable debug mode |

---

## Documentation

See [`docs/`](./docs/) for:
- [Architecture design](./docs/architecture/) — system design, module boundaries, naming rules
