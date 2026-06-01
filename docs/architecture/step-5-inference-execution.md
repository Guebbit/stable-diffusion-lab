# Step 5 — Inference Execution Architecture

> **Purpose**: Define the complete inference adapter layer — how the backend executes AI workloads across three backends (direct Python, BentoML, ComfyUI), including pipeline caching, resource coordination, and backend selection.

---

## 1. Architecture Overview: The Three-Backend Strategy

The inference layer implements a **Strategy pattern** via Python Protocols. The service layer never knows (or cares) which backend runs the actual inference — it only interacts with the protocols defined in `app/domain/protocols.py`.

```
┌────────────────────────────────────────────────────────────────┐
│                     SERVICE LAYER                               │
│  GenerationService / ModelService / JobWorker                   │
│  Depends ONLY on: TextToImageProvider, VideoProvider, etc.      │
└────────────────────────────┬───────────────────────────────────┘
                             │ Protocol dispatch
           ┌─────────────────┼──────────────────┐
           ▼                 ▼                   ▼
┌──────────────────┐ ┌───────────────────┐ ┌──────────────────┐
│  direct/         │ │  bentoml/         │ │  comfyui/        │
│  In-process      │ │  HTTP client to   │ │  HTTP client to  │
│  diffusers/      │ │  BentoML runner   │ │  ComfyUI server  │
│  transformers/   │ │                   │ │                  │
│  torch           │ │                   │ │                  │
└──────────────────┘ └───────────────────┘ └──────────────────┘
         │                    │                      │
         ▼                    ▼                      ▼
   [Local GPU/CPU]     [BentoML process]     [ComfyUI process]
```

**Key principle**: Heavy ML imports (`torch`, `diffusers`, `transformers`) only ever appear inside `app/adapters/`. They are never imported at module load time — always deferred to first-use within method bodies.

---

## 2. Module Layout

```
app/adapters/
├── __init__.py
├── adapter_registry.py        # Maps (job_type, backend) → adapter instance
├── resource_coordinator.py    # GPU mutex + VRAM tracking
├── direct/
│   ├── __init__.py            # Re-exports all adapter classes
│   ├── pipeline_cache.py      # LRU pipeline cache with VRAM budget
│   ├── model_manager.py       # DirectModelManager (ModelManager protocol)
│   ├── text_to_image.py       # DirectTextToImageAdapter
│   ├── image_to_image.py      # DirectImageToImageAdapter
│   ├── vision.py              # DirectVisionAdapter
│   ├── video.py               # DirectVideoAdapter
│   └── llm.py                 # DirectLLMAdapter
├── bentoml/
│   ├── __init__.py
│   ├── client.py              # BentoMLClient (HTTP + retry)
│   ├── text_to_image.py       # BentoMLTextToImageAdapter
│   ├── image_to_image.py      # BentoMLImageToImageAdapter
│   ├── vision.py              # BentoMLVisionAdapter
│   ├── video.py               # BentoMLVideoAdapter
│   ├── llm.py                 # BentoMLLLMAdapter
│   └── model_manager.py       # BentoMLModelManager
└── comfyui/
    ├── __init__.py
    ├── client.py              # ComfyUIClient (WebSocket + REST)
    ├── workflow_builder.py    # Builds ComfyUI workflow JSON from params
    ├── text_to_image.py       # ComfyUITextToImageAdapter
    ├── image_to_image.py      # ComfyUIImageToImageAdapter
    ├── video.py               # ComfyUIVideoAdapter
    ├── llm.py                 # Stub (ComfyUI doesn't support LLM)
    └── model_manager.py       # ComfyUIModelManager
```

---

## 3. PipelineCache Design

The PipelineCache is the heart of direct inference. It prevents redundant model loading while respecting VRAM limits.

**Key features:**
- LRU eviction (OrderedDict-based)
- Configurable capacity (`max_cached`, default: 1)
- Optional VRAM budget enforcement
- Async-safe (asyncio.Lock for load/evict serialization)
- Shared across all direct adapters

**Cache flow:**
```
adapter.generate() → cache.get_or_load(model_id, loader) → pipeline object
                                 │
                         Cache hit? → Return immediately
                         Cache miss? → Evict LRU → Call loader → Store → Return
```

---

## 4. AdapterRegistry — Backend Selection

```python
# Resolution order:
1. If request specifies a backend → use that
2. If not, use configured default (settings.inference_backend)
3. If configured backend unavailable → fall back to DIRECT_PYTHON
```

Wired at startup in `app/main.py`:
- Direct adapters are always registered
- BentoML adapters registered only if `settings.bentoml_enabled`
- ComfyUI adapters registered only if `settings.comfyui_enabled`

---

## 5. ResourceCoordinator

Manages GPU as an exclusive resource:
- `asyncio.Semaphore(max_concurrent)` — default 1 = mutex
- VRAM usage tracking (estimated, from adapter reports)
- Device status reporting for the system/status API
- Job counter for monitoring

---

## 6. Execution Flow (Text-to-Image Example)

```
1. API receives POST /api/v1/generation/text-to-image
2. GenerationService creates JobRecord (PENDING) in DB
3. JobWorker picks up PENDING job in poll loop
4. Worker claims job (atomic → RUNNING)
5. Worker calls ResourceCoordinator.acquire(job_id)
6. Worker asks AdapterRegistry for TEXT_TO_IMAGE provider
7. Worker calls adapter.generate(params, model_id, output_dir, on_progress)
8. Adapter → PipelineCache.get_or_load() → inference → save artifacts
9. Worker marks job COMPLETED
10. Worker releases GPU lock
11. EventBus broadcasts completion → WebSocket → frontend
```

---

## 7. Configuration (settings.py)

| Setting | Default | Description |
|---------|---------|-------------|
| `inference_device` | `"auto"` | Target device (cuda/cpu/auto) |
| `inference_backend` | `"direct_python"` | Default backend |
| `max_vram_mb` | `0` | VRAM budget (0 = unlimited) |
| `max_cached_pipelines` | `1` | Max pipelines in cache |
| `max_workers` | `1` | Concurrent GPU jobs |
| `bentoml_enabled` | `false` | Enable BentoML backend |
| `bentoml_url` | `http://localhost:3000` | BentoML service URL |
| `comfyui_enabled` | `false` | Enable ComfyUI backend |
| `comfyui_url` | `http://localhost:8188` | ComfyUI server URL |

---

## 8. Error Handling Per Backend

| Backend | Failure Mode | Recovery |
|---------|-------------|----------|
| Direct | CUDA OOM | Catch RuntimeError, empty_cache(), mark FAILED |
| Direct | Model corrupted | Catch during from_pretrained, mark ERROR |
| BentoML | Service unreachable | Retry 3× with backoff, then FAILED |
| BentoML | Timeout | Configurable per job type |
| ComfyUI | Server disconnected | Reconnect WebSocket, retry |
| ComfyUI | Node execution error | Parse error JSON, mark FAILED |

---

## 9. Boundary Rule

> The main backend NEVER cedes control of job state, metadata, or artifact storage to an external process. BentoML and ComfyUI are treated as **execution engines** — the backend is always the **orchestrator**.
