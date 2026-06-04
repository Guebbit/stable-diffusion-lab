"""
Application settings — single source of truth for all configuration.

Uses Pydantic BaseSettings to load values from environment variables and .env files.
Every configurable value lives here — no magic strings scattered in the codebase.

Usage:
    from app.infrastructure.config.settings import get_settings
    settings = get_settings()
    print(settings.database_url)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All application configuration in one place.

    Values are loaded in priority order:
    1. Environment variables (highest priority, for Docker/production)
    2. .env file (for local development)
    3. Default values defined here (fallback)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- Application ---
    app_name: str = "AI Lab Backend"
    app_version: str = "0.1.0"
    debug: bool = False

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = Field(8000, alias="BACKEND_PORT")

    # --- Database (PostgreSQL) ---
    database_url: str = "postgresql+asyncpg://ailab:ailab_local@localhost:5432/ailab"

    # --- Filesystem paths ---
    # Base directory for all local storage (models, artifacts, temp files)
    storage_root: Path = Path("/app/storage")

    # Sub-directories (relative to storage_root, resolved at runtime)
    models_dir: str = "models"
    artifacts_dir: str = "artifacts"
    thumbnails_dir: str = "thumbnails"
    temp_dir: str = "temp"
    imports_dir: str = "imports"
    logs_dir: str = "logs"

    # --- External API tokens (optional, for model downloads) ---
    huggingface_token: str = Field("", alias="HUGGINGFACE_TOKEN")
    civitai_token: str = Field("", alias="CIVITAI_TOKEN")

    # --- Inference ---
    # Default device for inference ("cuda", "cpu", or "auto")
    inference_device: str = "auto"
    # Default inference backend
    inference_backend: str = "direct_python"
    # Maximum VRAM budget in MB (0 = no limit, use all available)
    max_vram_mb: int = 0
    # Maximum pipelines to keep loaded in PipelineCache simultaneously
    max_cached_pipelines: int = 1

    # --- Job orchestrator ---
    # Maximum concurrent inference workers (1 = serialize GPU work)
    max_workers: int = 1
    # How often the worker loop polls for new jobs (seconds)
    job_poll_interval: float = 1.0
    # Default job timeout in seconds (0 = no timeout)
    job_timeout_seconds: int = 0

    # --- BentoML (optional backend) ---
    bentoml_enabled: bool = False
    bentoml_url: str = "http://localhost:3000"

    # --- ComfyUI (optional backend) ---
    comfyui_enabled: bool = False
    comfyui_url: str = "http://localhost:8188"

    # --- CORS ---
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # --- Resolved paths (computed properties) ---
    @property
    def models_path(self) -> Path:
        """Absolute path to model storage directory."""
        return self.storage_root / self.models_dir

    @property
    def artifacts_path(self) -> Path:
        """Absolute path to generated artifacts directory."""
        return self.storage_root / self.artifacts_dir

    @property
    def thumbnails_path(self) -> Path:
        """Absolute path to thumbnail cache directory."""
        return self.storage_root / self.thumbnails_dir

    @property
    def temp_path(self) -> Path:
        """Absolute path to temporary files directory."""
        return self.storage_root / self.temp_dir

    @property
    def imports_path(self) -> Path:
        """Absolute path to imported source files directory."""
        return self.storage_root / self.imports_dir

    @property
    def logs_path(self) -> Path:
        """Absolute path to log files directory."""
        return self.storage_root / self.logs_dir


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.

    Cached so the .env file is only read once. Call get_settings.cache_clear()
    in tests to reset between test cases.
    """
    return Settings()
