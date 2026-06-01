"""
Repository interfaces and implementations for database access.

Repositories are the ONLY way services interact with the database.
Each repository handles one aggregate root (Model, Job, Artifact).
"""

from app.infrastructure.database.repositories.model_repository import ModelRepository
from app.infrastructure.database.repositories.job_repository import JobRepository
from app.infrastructure.database.repositories.artifact_repository import ArtifactRepository

__all__ = ["ModelRepository", "JobRepository", "ArtifactRepository"]
