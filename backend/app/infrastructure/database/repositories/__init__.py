"""
Repository interfaces and implementations for database access.

Repositories are the ONLY way services interact with the database.
Each repository handles one aggregate root (Model, Job, Artifact) or
its child entities (ModelFile, JobEvent).
"""

from app.infrastructure.database.repositories.artifact_repository import ArtifactRepository
from app.infrastructure.database.repositories.job_event_repository import JobEventRepository
from app.infrastructure.database.repositories.job_repository import JobRepository
from app.infrastructure.database.repositories.model_file_repository import ModelFileRepository
from app.infrastructure.database.repositories.model_repository import ModelRepository

__all__ = [
    "ArtifactRepository",
    "JobEventRepository",
    "JobRepository",
    "ModelFileRepository",
    "ModelRepository",
]
