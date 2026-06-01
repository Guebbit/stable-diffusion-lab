"""
FastAPI application factory — creates and configures the app instance.

This is the composition root where all layers are wired together:
- Routers are registered
- Middleware is added
- Lifecycle hooks (startup/shutdown) are defined
- The WebSocket hub is connected to the event bus
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import artifacts, generation, jobs, models, system
from app.api.websocket.hub import ws_hub
from app.infrastructure.config.settings import get_settings
from app.orchestrator.event_bus import event_bus

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifecycle manager.

    Startup: wire event bus → WebSocket hub, start job worker.
    Shutdown: stop job worker, clean up resources.
    """
    settings = get_settings()
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)

    # Connect event bus to WebSocket hub (progress events → clients)
    event_bus.subscribe(ws_hub.broadcast)

    # TODO: Start job worker once fully wired
    # worker = JobWorker(...)
    # await worker.start()

    yield

    # Shutdown
    # await worker.stop()
    event_bus.unsubscribe(ws_hub.broadcast)
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

    # --- WebSocket endpoint ---
    @app.websocket("/ws/progress")
    async def websocket_progress(websocket: WebSocket) -> None:
        """WebSocket endpoint for real-time job progress events."""
        await ws_hub.connect(websocket)
        try:
            # Keep connection alive — client sends pings, we just listen
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            ws_hub.disconnect(websocket)

    return app
