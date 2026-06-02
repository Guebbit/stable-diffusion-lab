"""
API routers sub-package — each file defines routes for one domain.
"""

from app.api.routers import artifacts, generation, jobs, legacy, models, system

__all__ = ["artifacts", "generation", "jobs", "legacy", "models", "system"]
