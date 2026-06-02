"""
Observability service — centralizes system snapshot, events, timelines, and metrics.

Aggregates state from runtime settings, resource coordinator, pipeline cache,
job queue records, and the in-memory observability bus. Produces:
- truthful system snapshots for /system/status
- recent event lists and per-job timelines
- metric snapshots and health status/warning heuristics
"""

from __future__ import annotations

import platform
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.system import (
    GpuSnapshot,
    HealthSnapshot,
    JobQueueSnapshot,
    MetricsSnapshotResponse,
    ModelCacheSnapshot,
    RuntimeSnapshot,
    SystemEventResponse,
    SystemStatusResponse,
)
from app.domain.observability import MetricPoint, ObservabilityEvent
from app.infrastructure.config.settings import get_settings
from app.infrastructure.database.models import JobRecord
from app.orchestrator.observability_bus import observability_bus


class ObservabilityService:
    """
    Build consolidated observability views from real backend state owners.

    Health status is derived from lightweight heuristics (queue backlog, GPU
    pressure, warming/cache state, OOM/load failure counters, lock duration,
    and repeated error events) to classify system state as ok/busy/degraded/
    warming_up/error with machine-readable reasons.
    """

    def __init__(self, app: FastAPI, session: AsyncSession) -> None:
        self._app = app
        self._session = session

    async def get_system_status(self) -> SystemStatusResponse:
        settings = get_settings()
        start_time: datetime | None = getattr(self._app.state, "started_at", None)
        uptime_seconds = (
            int((datetime.now(timezone.utc) - start_time).total_seconds())
            if start_time
            else 0
        )

        resource = self._app.state.resource_coordinator
        cache = self._app.state.pipeline_cache
        device_status = resource.get_device_status()
        loaded_model_ids = cache.get_loaded_ids()
        gpu = GpuSnapshot(
            available=self._torch_cuda_available(),
            device="cuda" if self._torch_cuda_available() else "cpu",
            device_name=device_status.device_name,
            busy=device_status.is_busy,
            active_job_id=device_status.current_job_id,
            estimated_vram_used_mb=device_status.estimated_vram_used_mb,
            vram_budget_mb=device_status.vram_budget_mb,
            cuda_allocated_mb=self._cuda_allocated_mb(),
            cuda_reserved_mb=self._cuda_reserved_mb(),
        )

        counts = await self._status_counts()
        active_job_id = await self._active_job_id()
        oldest_pending_age = await self._oldest_pending_age_seconds()
        jobs = JobQueueSnapshot(
            pending=counts.get("pending", 0),
            running=counts.get("running", 0),
            completed=counts.get("completed", 0),
            failed=counts.get("failed", 0),
            cancelled=counts.get("cancelled", 0),
            active_job_id=active_job_id,
            oldest_pending_age_seconds=oldest_pending_age,
        )
        models = ModelCacheSnapshot(
            loaded_models=loaded_model_ids,
            active_model=loaded_model_ids[-1] if loaded_model_ids else None,
            cache_size=cache.size,
            max_cached=cache.max_cached,
            estimated_vram_usage_mb=cache.current_vram_usage_mb,
        )

        observability_bus.metrics.set_gauge("jobs.queue_depth", float(jobs.pending))
        observability_bus.metrics.set_gauge("jobs.active", float(jobs.running))
        observability_bus.metrics.set_gauge("models.cached", float(models.cache_size))
        observability_bus.metrics.set_gauge("models.loaded", float(len(models.loaded_models)))
        observability_bus.metrics.set_gauge("gpu.cuda_allocated_mb", gpu.cuda_allocated_mb)
        observability_bus.metrics.set_gauge("gpu.cuda_reserved_mb", gpu.cuda_reserved_mb)

        health = self._health_snapshot(gpu=gpu, jobs=jobs, models=models)
        runtime = RuntimeSnapshot(
            app_name=settings.app_name,
            version=settings.app_version,
            uptime_seconds=uptime_seconds,
            python_version=platform.python_version(),
            inference_backend=settings.inference_backend,
        )

        return SystemStatusResponse(
            status=health.status,
            version=settings.app_version,
            device=gpu.device,
            gpu_busy=gpu.busy,
            loaded_models=models.loaded_models,
            pending_jobs=jobs.pending,
            runtime=runtime,
            gpu=gpu,
            models=models,
            jobs=jobs,
            health=health,
        )

    def get_recent_events(self, limit: int = 100) -> list[SystemEventResponse]:
        return [self._event_to_response(event) for event in observability_bus.recent_events(limit)]

    def get_job_timeline(self, job_id: str, limit: int = 200) -> list[SystemEventResponse]:
        return [
            self._event_to_response(event)
            for event in observability_bus.job_timeline(job_id=job_id, limit=limit)
        ]

    def get_metrics(self) -> MetricsSnapshotResponse:
        snapshot = observability_bus.metrics.snapshot()
        return MetricsSnapshotResponse(
            counters=[self._metric_to_response(metric) for metric in snapshot["counters"]],
            gauges=[self._metric_to_response(metric) for metric in snapshot["gauges"]],
            histograms=[self._metric_to_response(metric) for metric in snapshot["histograms"]],
        )

    async def _status_counts(self) -> dict[str, int]:
        stmt = select(JobRecord.status, func.count(JobRecord.id)).group_by(JobRecord.status)
        result = await self._session.execute(stmt)
        return {status: int(count) for status, count in result.all()}

    async def _active_job_id(self) -> str | None:
        stmt = (
            select(JobRecord.id)
            .where(JobRecord.status == "running")
            .order_by(JobRecord.started_at.desc().nullslast())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        value = result.scalar_one_or_none()
        return str(value) if value else None

    async def _oldest_pending_age_seconds(self) -> int | None:
        stmt = (
            select(JobRecord.created_at)
            .where(JobRecord.status == "pending")
            .order_by(JobRecord.created_at.asc())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        created_at = result.scalar_one_or_none()
        if created_at is None:
            return None
        return int((datetime.now(timezone.utc) - created_at).total_seconds())

    def _health_snapshot(
        self,
        gpu: GpuSnapshot,
        jobs: JobQueueSnapshot,
        models: ModelCacheSnapshot,
    ) -> HealthSnapshot:
        warnings: list[str] = []
        reasons: list[str] = []

        if not gpu.available:
            warnings.append("CUDA unavailable; running on CPU.")
            reasons.append("gpu_unavailable")
        if jobs.pending >= 5:
            warnings.append("Queue backlog detected.")
            reasons.append("queue_backlog")
        if gpu.cuda_reserved_mb > 0 and gpu.cuda_allocated_mb > 0.9 * gpu.cuda_reserved_mb:
            warnings.append("High GPU memory pressure.")
            reasons.append("gpu_memory_pressure")
        if models.cache_size == 0 and jobs.running > 0:
            warnings.append("Pipeline cache warming up.")
            reasons.append("pipeline_warming_up")
        if observability_bus.metrics.get_counter("gpu.oom_errors") >= 1:
            warnings.append("OOM errors detected recently.")
            reasons.append("repeated_oom")
        if observability_bus.metrics.get_counter("model.load_failures") >= 1:
            warnings.append("Model load failures detected.")
            reasons.append("model_load_failures")
        if observability_bus.metrics.get_counter("cache.misses") >= 10:
            warnings.append("High cache miss count detected.")
            reasons.append("cache_miss_spike")
        if observability_bus.metrics.get_hist_avg("resource.lock_held_seconds") > 120:
            warnings.append("Resource lock held too long.")
            reasons.append("long_held_lock")
        if observability_bus.metrics.get_counter("observability.error_events") >= 5:
            warnings.append("Multiple error events reported.")
            reasons.append("repeated_failures")

        status = "ok"
        if reasons:
            if "repeated_oom" in reasons or "model_load_failures" in reasons:
                status = "error"
            elif "pipeline_warming_up" in reasons:
                status = "warming_up"
            elif "queue_backlog" in reasons or "gpu_memory_pressure" in reasons:
                status = "degraded"
            else:
                status = "busy" if jobs.running > 0 else "degraded"
        elif jobs.running > 0 or gpu.busy:
            status = "busy"

        return HealthSnapshot(status=status, warnings=warnings, reasons=reasons)

    @staticmethod
    def _event_to_response(event: ObservabilityEvent) -> SystemEventResponse:
        return SystemEventResponse(
            event_id=event.event_id,
            event_type=event.event_type,
            component=event.component,
            level=event.level,
            message=event.message,
            timestamp=event.timestamp,
            job_id=event.job_id,
            correlation_id=event.correlation_id,
            payload=event.payload,
        )

    @staticmethod
    def _metric_to_response(metric: MetricPoint) -> dict[str, Any]:
        return {
            "name": metric.name,
            "kind": metric.kind,
            "value": metric.value,
            "unit": metric.unit,
            "updated_at": metric.updated_at,
        }

    @staticmethod
    def _torch_cuda_available() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    @staticmethod
    def _cuda_allocated_mb() -> float:
        try:
            import torch

            if torch.cuda.is_available():
                return float(torch.cuda.memory_allocated(0) / (1024 * 1024))
        except Exception:
            return 0.0
        return 0.0

    @staticmethod
    def _cuda_reserved_mb() -> float:
        try:
            import torch

            if torch.cuda.is_available():
                return float(torch.cuda.memory_reserved(0) / (1024 * 1024))
        except Exception:
            return 0.0
        return 0.0
