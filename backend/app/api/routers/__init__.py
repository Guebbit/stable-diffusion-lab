"""
API routers sub-package — each file defines routes for one domain.
"""

from app.api.routers import generation, models, system

__all__ = ["generation", "models", "system"]
