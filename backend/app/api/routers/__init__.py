"""
API routers sub-package — each file defines routes for one domain.
"""

from app.api.routers import artifacts, generation, jobs, models, system, upscale

__all__ = ["artifacts", "generation", "jobs", "models", "system", "upscale"]
