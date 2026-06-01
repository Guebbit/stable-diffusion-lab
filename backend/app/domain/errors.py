"""
Domain-specific exception hierarchy.

All custom exceptions inherit from AILabError. Each has a stable error_code
that maps to HTTP status codes in the API layer's exception handler.

This module lives in the domain layer because errors represent business rules
violations — they are NOT tied to HTTP or infrastructure concerns.
"""

from __future__ import annotations


class AILabError(Exception):
    """
    Base exception for all domain errors.

    Subclasses define an error_code that the API layer maps to HTTP status codes.
    """

    error_code: str = "INTERNAL_ERROR"

    def __init__(self, message: str = "", **context: object) -> None:
        super().__init__(message)
        self.message = message
        self.context = context


class ModelNotFoundError(AILabError):
    """Raised when a model_id doesn't exist in the registry."""

    error_code = "MODEL_NOT_FOUND"


class ModelNotReadyError(AILabError):
    """Raised when a model isn't in the required state for an operation."""

    error_code = "MODEL_NOT_READY"


class InvalidStateTransitionError(AILabError):
    """Raised when a job/model state change violates the state machine."""

    error_code = "INVALID_STATE_TRANSITION"


class JobNotFoundError(AILabError):
    """Raised when a job_id doesn't exist."""

    error_code = "JOB_NOT_FOUND"


class RetryLimitExceededError(AILabError):
    """Raised when a job has exhausted its retry budget."""

    error_code = "RETRY_LIMIT_EXCEEDED"


class CancellationError(AILabError):
    """Raised by adapters when cancellation is detected mid-execution."""

    error_code = "JOB_CANCELLED"


class IntegrityVerificationError(AILabError):
    """Raised when file integrity check fails (SHA256 mismatch)."""

    error_code = "INTEGRITY_CHECK_FAILED"


class DownloadError(AILabError):
    """Raised when a model download fails (network, auth, disk space)."""

    error_code = "DOWNLOAD_FAILED"
