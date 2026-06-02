"""Observability service for system snapshots, health and timelines."""

from __future__ import annotations

import json
import logging
import platform
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import func as sa_func
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.adapters.direct.pipeline_cache import PipelineCache
from app.adapters.resource_coordinator import ResourceCoordinator
from app.domain.events import TypedEvent
from app.infrastructure.config.settings import get_settings
from app.infrastructure.database.models import ArtifactRecord
from app.infrastructure.database.repositories import JobRepository
from app.orchestrator.event_bus import EventBus
from app.orchestrator.metrics import MetricsRegistry


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    """Runtime metadata."""

    version: str
    uptime_seconds: float
    python_version: str
    torch_version: str | None
    cuda_available: bool


@dataclass(frozen=True, slots=True)
class GPUStatus:
    """GPU/resource status."""

    busy: bool
    current_job_id: str | None
    device_name: str
    estimated_vram_used_mb: int
    vram_budget_mb: int
    cuda_allocated_mb: float
    cuda_reserved_mb: float
    last_activity: datetime | None


@dataclass(frozen=True, slots=True)
class ModelStatus:
    """Model cache and loaded model status."""

    loaded_models: list[str]
    cache_hits: int
    cache_misses: int
    warm: bool
    cache_size: int
    last_load_at: datetime | None
    last_unload_at: datetime | None
    last_load_failed: bool


@dataclass(frozen=True, slots=True)
class JobQueueStatus:
    """Job queue state."""

    pending: int
    running: int
    completed: int
    failed: int
    cancelled: int
    queue_depth: int
    oldest_pending_age_seconds: float | None
    current_job_id: str | None


@dataclass(frozen=True, slots=True)
class ArtifactStatus:
    """Artifact storage state."""

    recent_saves_count: int
    storage_health: str
    disk_free_mb: float


@dataclass(frozen=True, slots=True)
class HealthStatus:
    """System health semantics."""

    status: str
    warnings: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SystemStateSnapshot:
    """Comprehensive system observability snapshot."""

    runtime: RuntimeStatus
    gpu: GPUStatus
    models: ModelStatus
    jobs: JobQueueStatus
    artifacts: ArtifactStatus
    health: HealthStatus
    correlation_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize snapshot to JSON-compatible structure."""
        data = asdict(self)
        self._serialize_datetimes(data)
        return data

    @classmethod
    def _serialize_datetimes(cls, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, dict):
            for key, child in list(value.items()):
                value[key] = cls._serialize_datetimes(child)
        if isinstance(value, list):
            for index, child in enumerate(value):
                value[index] = cls._serialize_datetimes(child)
        return value


class ObservabilityService:
    """Aggregates observability state from live runtime components."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        resource_coordinator: ResourceCoordinator,
        pipeline_cache: PipelineCache,
        event_bus: EventBus,
        metrics_registry: MetricsRegistry,
    ) -> None:
        self._started_at = datetime.now(timezone.utc)
        self._session_factory = session_factory
        self._resource_coordinator = resource_coordinator
        self._pipeline_cache = pipeline_cache
        self._event_bus = event_bus
        self._metrics = metrics_registry

    async def get_system_snapshot(self, correlation_id: str | None = None) -> SystemStateSnapshot:
        """Build and return a full state snapshot."""
        runtime = self._runtime_status()
        gpu = self._gpu_status()
        models = self._model_status()
        jobs = await self._job_status()
        artifacts = await self._artifact_status()
        health = self._health_status(gpu, models, jobs)

        self._metrics.set_gauge("queue_depth", float(jobs.queue_depth))
        self._metrics.set_gauge("active_jobs", float(jobs.running))
        self._metrics.set_gauge("cached_pipelines", float(models.cache_size))
        self._metrics.set_gauge("cuda_allocated_mb", gpu.cuda_allocated_mb)
        self._metrics.set_gauge("cuda_reserved_mb", gpu.cuda_reserved_mb)
        self._metrics.set_gauge("disk_free_mb", artifacts.disk_free_mb)
        self._metrics.set_counter("cache_hits", models.cache_hits)
        self._metrics.set_counter("cache_misses", models.cache_misses)

        return SystemStateSnapshot(
            runtime=runtime,
            gpu=gpu,
            models=models,
            jobs=jobs,
            artifacts=artifacts,
            health=health,
            correlation_id=correlation_id,
        )

    def get_recent_events(self, limit: int = 200) -> list[dict[str, Any]]:
        """Return recent event stream payloads."""
        return [event.to_dict() for event in self._event_bus.get_recent_events(limit)]

    def get_job_timeline(self, job_id: str, limit: int = 500) -> list[dict[str, Any]]:
        """Return per-job timeline from event history."""
        return [event.to_dict() for event in self._event_bus.get_job_events(job_id, limit)]

    def get_metrics_snapshot(self) -> dict[str, dict[str, object]]:
        """Return current metrics snapshot."""
        return self._metrics.snapshot()

    async def publish_event(self, event: TypedEvent) -> None:
        """Publish event and emit structured JSON log."""
        await self._event_bus.publish_event(event)
        logger.info(json.dumps({"event": event.to_dict()}, ensure_ascii=False, default=str))

    def _runtime_status(self) -> RuntimeStatus:
        settings = get_settings()
        torch_version: str | None = None
        cuda_available = False
        try:
            import torch

            torch_version = torch.__version__
            cuda_available = torch.cuda.is_available()
        except Exception:
            torch_version = None
            cuda_available = False

        uptime = datetime.now(timezone.utc) - self._started_at
        return RuntimeStatus(
            version=settings.app_version,
            uptime_seconds=uptime.total_seconds(),
            python_version=platform.python_version(),
            torch_version=torch_version,
            cuda_available=cuda_available,
        )

    def _gpu_status(self) -> GPUStatus:
        status = self._resource_coordinator.get_device_status()
        allocated_mb = 0.0
        reserved_mb = 0.0
        try:
            import torch

            if torch.cuda.is_available():
                allocated_mb = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2)
                reserved_mb = round(torch.cuda.memory_reserved(0) / (1024 * 1024), 2)
        except Exception:
            allocated_mb = 0.0
            reserved_mb = 0.0

        return GPUStatus(
            busy=status.is_busy,
            current_job_id=status.current_job_id,
            device_name=status.device_name,
            estimated_vram_used_mb=status.estimated_vram_used_mb,
            vram_budget_mb=status.vram_budget_mb,
            cuda_allocated_mb=allocated_mb,
            cuda_reserved_mb=reserved_mb,
            last_activity=status.last_activity,
        )

    def _model_status(self) -> ModelStatus:
        cache_stats = self._pipeline_cache.get_cache_stats()
        return ModelStatus(
            loaded_models=self._pipeline_cache.get_loaded_ids(),
            cache_hits=int(cache_stats["hits"]),
            cache_misses=int(cache_stats["misses"]),
            warm=bool(cache_stats["warm"]),
            cache_size=self._pipeline_cache.size,
            last_load_at=cache_stats["last_load_at"],
            last_unload_at=cache_stats["last_unload_at"],
            last_load_failed=bool(cache_stats["last_load_failed"]),
        )

    async def _job_status(self) -> JobQueueStatus:
        async with self._session_factory() as session:
            repo = JobRepository(session)
            counts = await repo.get_queue_counts()
            oldest_pending_age_seconds = await repo.get_oldest_pending_age_seconds()
            current_job_id = await repo.get_current_running_job_id()

        return JobQueueStatus(
            pending=counts["pending"],
            running=counts["running"],
            completed=counts["completed"],
            failed=counts["failed"],
            cancelled=counts["cancelled"],
            queue_depth=counts["queue_depth"],
            oldest_pending_age_seconds=oldest_pending_age_seconds,
            current_job_id=current_job_id,
        )

    async def _artifact_status(self) -> ArtifactStatus:
        settings = get_settings()
        artifacts_path: Path = settings.artifacts_path
        storage_health = "ok"
        disk_free_mb = 0.0
        try:
            usage = shutil.disk_usage(artifacts_path)
            disk_free_mb = round(usage.free / (1024 * 1024), 2)
            if disk_free_mb < 512:
                storage_health = "low_space"
        except FileNotFoundError:
            storage_health = "path_missing"

        since = datetime.now(timezone.utc) - timedelta(minutes=30)
        async with self._session_factory() as session:
            stmt = (
                select(sa_func.count())
                .select_from(ArtifactRecord)
                .where(ArtifactRecord.created_at >= since)
            )
            result = await session.execute(stmt)
            recent_saves_count = int(result.scalar_one() or 0)

        return ArtifactStatus(
            recent_saves_count=recent_saves_count,
            storage_health=storage_health,
            disk_free_mb=disk_free_mb,
        )

    def _health_status(
        self,
        gpu: GPUStatus,
        models: ModelStatus,
        jobs: JobQueueStatus,
    ) -> HealthStatus:
        warnings: list[str] = []
        blockers: list[str] = []
        recommendations: list[str] = []

        if jobs.pending > 10:
            warnings.append("queue_backlog_high")
            recommendations.append("Increase workers or reduce queue inflow.")

        if gpu.vram_budget_mb > 0 and gpu.estimated_vram_used_mb > int(gpu.vram_budget_mb * 0.85):
            warnings.append("high_vram_pressure")
            recommendations.append("Reduce resolution/batch size or unload inactive models.")

        if self._metrics.counters.get("oom_errors", 0) > 0:
            warnings.append("oom_detected")

        if models.last_load_failed:
            warnings.append("model_load_failed")

        if jobs.pending > 0 and jobs.running == 0:
            blockers.append("queue_stalled")
            recommendations.append("Inspect worker logs and resource lock holder.")

        status = "ok"
        if blockers:
            status = "error"
        elif warnings:
            status = "degraded"
        elif gpu.busy:
            status = "busy"

        return HealthStatus(
            status=status,
            warnings=warnings,
            blockers=blockers,
            recommendations=recommendations,
        )
