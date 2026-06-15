"""
FastAPI application factory — creates and configures the app instance.

This is the composition root where all layers are wired together:
- Adapters are instantiated and registered in the AdapterRegistry
- Routers are registered
- Middleware is added
- Lifecycle hooks (startup/shutdown) manage worker and cache lifecycle
- The SSE hub is connected to the event bus for real-time event streaming
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

# Configure root logger before any third-party code runs.
# Uvicorn's dictConfig only sets up its own namespace loggers and leaves the
# root handler empty, so app.* INFO messages are silently dropped by default.
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s - %(message)s",
    force=True,
)
for _noisy_lib in ("sqlalchemy", "huggingface_hub", "httpx", "httpcore", "filelock"):
    logging.getLogger(_noisy_lib).setLevel(logging.WARNING)
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.adapters.adapter_registry import AdapterRegistry
from app.adapters.direct import (
    DirectImageToImageAdapter,
    DirectLLMAdapter,
    DirectModelManager,
    DirectSketchToInkAdapter,
    DirectTextToImageAdapter,
    DirectUpscaleAdapter,
    DirectVideoAdapter,
    DirectVisionAdapter,
    PipelineCache,
)
from app.adapters.resource_coordinator import ResourceCoordinator
from app.api.routers import artifacts, generation, jobs, models, system
from app.api.sse.hub import sse_hub
from app.domain.enums import InferenceBackend, JobType
from app.infrastructure.config.settings import get_settings
from app.orchestrator.event_bus import event_bus
from app.orchestrator.metrics import MetricsRegistry
from app.orchestrator.observability import ObservabilityService
from app.orchestrator.worker import JobWorker


logger = logging.getLogger(__name__)


def _build_adapter_registry(
    pipeline_cache: PipelineCache,
    settings: object,
) -> AdapterRegistry:
    """
    Build and populate the adapter registry with all available backends.

    Direct adapters are always registered. BentoML and ComfyUI adapters
    are registered only if their respective services are enabled in config.
    """
    registry = AdapterRegistry(
        default_backend=InferenceBackend(settings.inference_backend)  # type: ignore[attr-defined]
    )

    # --- Direct adapters (always available) ---
    direct_txt2img = DirectTextToImageAdapter(pipeline_cache)
    direct_img2img = DirectImageToImageAdapter(pipeline_cache)
    direct_vision = DirectVisionAdapter(pipeline_cache)
    direct_upscale = DirectUpscaleAdapter(pipeline_cache)
    direct_sketch = DirectSketchToInkAdapter(pipeline_cache)
    direct_video = DirectVideoAdapter(pipeline_cache)
    direct_llm = DirectLLMAdapter(pipeline_cache)

    registry.register(JobType.TEXT_TO_IMAGE, InferenceBackend.DIRECT_PYTHON, direct_txt2img)
    registry.register(JobType.IMAGE_TO_IMAGE, InferenceBackend.DIRECT_PYTHON, direct_img2img)
    registry.register(JobType.IMAGE_ANALYSIS, InferenceBackend.DIRECT_PYTHON, direct_vision)
    registry.register(JobType.UPSCALE, InferenceBackend.DIRECT_PYTHON, direct_upscale)
    registry.register(JobType.RECOLOR, InferenceBackend.DIRECT_PYTHON, direct_img2img, validate_protocol=False)
    registry.register(JobType.SKETCH_TO_INK, InferenceBackend.DIRECT_PYTHON, direct_sketch, validate_protocol=False)
    registry.register(JobType.VIDEO_GENERATION, InferenceBackend.DIRECT_PYTHON, direct_video)
    registry.register(JobType.LLM_INFERENCE, InferenceBackend.DIRECT_PYTHON, direct_llm)

    # --- BentoML adapters (optional) ---
    if settings.bentoml_enabled:  # type: ignore[attr-defined]
        from app.adapters.bentoml import (
            BentoMLClient,
            BentoMLImageToImageAdapter,
            BentoMLLLMAdapter,
            BentoMLTextToImageAdapter,
            BentoMLVideoAdapter,
            BentoMLVisionAdapter,
        )

        bento_client = BentoMLClient(base_url=settings.bentoml_url)  # type: ignore[attr-defined]

        registry.register(
            JobType.TEXT_TO_IMAGE,
            InferenceBackend.BENTOML,
            BentoMLTextToImageAdapter(bento_client),
        )
        registry.register(
            JobType.IMAGE_TO_IMAGE,
            InferenceBackend.BENTOML,
            BentoMLImageToImageAdapter(bento_client),
        )
        registry.register(
            JobType.VIDEO_GENERATION,
            InferenceBackend.BENTOML,
            BentoMLVideoAdapter(bento_client),
        )
        registry.register(
            JobType.LLM_INFERENCE,
            InferenceBackend.BENTOML,
            BentoMLLLMAdapter(bento_client),
        )
        logger.info("BentoML adapters registered (url: %s)", settings.bentoml_url)  # type: ignore[attr-defined]

    # --- ComfyUI adapters (optional) ---
    if settings.comfyui_enabled:  # type: ignore[attr-defined]
        from app.adapters.comfyui import (
            ComfyUIClient,
            ComfyUIImageToImageAdapter,
            ComfyUITextToImageAdapter,
            ComfyUIVideoAdapter,
            WorkflowBuilder,
        )

        comfyui_client = ComfyUIClient(base_url=settings.comfyui_url)  # type: ignore[attr-defined]
        workflow_builder = WorkflowBuilder()

        registry.register(
            JobType.TEXT_TO_IMAGE,
            InferenceBackend.COMFYUI,
            ComfyUITextToImageAdapter(comfyui_client, workflow_builder),
        )
        registry.register(
            JobType.IMAGE_TO_IMAGE,
            InferenceBackend.COMFYUI,
            ComfyUIImageToImageAdapter(comfyui_client, workflow_builder),
        )
        registry.register(
            JobType.VIDEO_GENERATION,
            InferenceBackend.COMFYUI,
            ComfyUIVideoAdapter(comfyui_client, workflow_builder),
        )
        logger.info("ComfyUI adapters registered (url: %s)", settings.comfyui_url)  # type: ignore[attr-defined]

    return registry


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifecycle manager.

    Startup: build adapters, wire event bus → SSE hub, start job worker.
    Shutdown: stop job worker, evict pipeline cache, clean up resources.
    """
    settings = get_settings()
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)

    # --- Build infrastructure ---
    pipeline_cache = PipelineCache(
        max_cached=settings.max_cached_pipelines,
        vram_budget_mb=settings.max_vram_mb,
    )
    resource_coordinator = ResourceCoordinator(
        max_concurrent=settings.max_workers,
        vram_budget_mb=settings.max_vram_mb,
    )

    # --- Build adapter registry ---
    adapter_registry = _build_adapter_registry(pipeline_cache, settings)

    # --- Build model manager ---
    model_manager = DirectModelManager(pipeline_cache)

    # --- Connect event bus to SSE hub ---
    event_bus.subscribe_typed(sse_hub.broadcast_typed)

    # --- Build metrics + observability ---
    from app.infrastructure.database.session import get_session_factory

    session_factory = get_session_factory()
    metrics_registry = MetricsRegistry()
    event_bus.subscribe_typed(metrics_registry.record_event)
    observability = ObservabilityService(
        session_factory=session_factory,
        resource_coordinator=resource_coordinator,
        pipeline_cache=pipeline_cache,
        event_bus=event_bus,
        metrics_registry=metrics_registry,
    )

    # --- Start job worker ---
    worker = JobWorker(
        session_factory=session_factory,
        resource_coordinator=resource_coordinator,
        adapter_registry=adapter_registry,
    )
    await worker.start()

    # Store references on app.state for access in route handlers
    app.state.pipeline_cache = pipeline_cache
    app.state.resource_coordinator = resource_coordinator
    app.state.adapter_registry = adapter_registry
    app.state.model_manager = model_manager
    app.state.worker = worker
    app.state.metrics_registry = metrics_registry
    app.state.observability = observability

    yield

    # --- Shutdown ---
    await worker.stop()
    await pipeline_cache.evict_all()
    event_bus.unsubscribe_typed(sse_hub.broadcast_typed)
    event_bus.unsubscribe_typed(metrics_registry.record_event)
    logger.info("Application shutdown complete")


def create_app() -> FastAPI:
    """
    Application factory — builds and returns the configured FastAPI app.

    Call this from the entry point (main.py) or tests.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # --- CORS middleware (local dev: allow frontend origins) ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Register API routers under /api/v1 ---
    api_prefix = "/api/v1"
    app.include_router(generation.router, prefix=api_prefix)
    app.include_router(jobs.router, prefix=api_prefix)
    app.include_router(models.router, prefix=api_prefix)
    app.include_router(artifacts.router, prefix=api_prefix)
    app.include_router(system.router, prefix=api_prefix)

    # --- SSE endpoint for observability event streaming ---
    @app.get("/sse/observability")
    async def sse_observability(request: Request) -> StreamingResponse:
        """SSE endpoint for typed observability event streaming."""
        subscribe = request.query_params.get("subscribe", "")
        filters = {item.strip() for item in subscribe.split(",") if item.strip()} if subscribe else None
        return await sse_hub.create_stream(request, subscribe=filters)

    return app