"""
Service layer — domain logic and orchestration.

Services implement the "what should happen" for each use case.
They coordinate repositories, adapters, and the job orchestrator
but never access the database directly or import ML libraries.
"""
