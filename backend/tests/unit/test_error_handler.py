"""
Unit tests for the centralised error handler.

Verifies that every AILabError subclass, ValueError, and unexpected
exception maps to the correct HTTP status code and detail payload.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.api.error_handler import from_exception
from app.domain.errors import (
    AILabError,
    DownloadError,
    IntegrityVerificationError,
    InvalidStateTransitionError,
    JobNotFoundError,
    ModelNotFoundError,
    ModelNotReadyError,
    RetryLimitExceededError,
)


class TestDomainErrorToHTTPMapping:
    """Every AILabError subclass maps to a documented HTTP status."""

    def test_model_not_found_maps_to_404(self) -> None:
        result = from_exception(ModelNotFoundError("Model xyz not found"))
        assert result.status_code == 404

    def test_model_not_ready_maps_to_404(self) -> None:
        result = from_exception(ModelNotReadyError("Model not ready"))
        assert result.status_code == 404

    def test_job_not_found_maps_to_404(self) -> None:
        result = from_exception(JobNotFoundError("Job 123 not found"))
        assert result.status_code == 404

    def test_invalid_state_transition_maps_to_409(self) -> None:
        result = from_exception(InvalidStateTransitionError("Cannot cancel completed job"))
        assert result.status_code == 409

    def test_retry_limit_exceeded_maps_to_409(self) -> None:
        result = from_exception(RetryLimitExceededError("Max attempts reached"))
        assert result.status_code == 409

    def test_download_error_maps_to_400(self) -> None:
        result = from_exception(DownloadError("Network failure"))
        assert result.status_code == 400

    def test_integrity_check_failed_maps_to_400(self) -> None:
        result = from_exception(IntegrityVerificationError("SHA256 mismatch"))
        assert result.status_code == 400

    def test_detail_preserves_full_exception_message(self) -> None:
        msg = "Job abc-123 not found"
        result = from_exception(JobNotFoundError(msg))
        assert result.detail == msg

    def test_returns_http_exception_instance(self) -> None:
        result = from_exception(JobNotFoundError("not found"))
        assert isinstance(result, HTTPException)

    def test_unknown_error_code_falls_back_to_500(self) -> None:
        class _Mystery(AILabError):
            error_code = "TOTALLY_UNKNOWN_CODE"

        result = from_exception(_Mystery("unexpected domain error"))
        assert result.status_code == 500

    def test_all_404_errors_carry_original_message(self) -> None:
        for exc_cls in (ModelNotFoundError, ModelNotReadyError, JobNotFoundError):
            exc = exc_cls("specific message")
            result = from_exception(exc)
            assert "specific message" in result.detail

    def test_all_conflict_errors_carry_original_message(self) -> None:
        for exc_cls in (InvalidStateTransitionError, RetryLimitExceededError):
            exc = exc_cls("specific conflict message")
            result = from_exception(exc)
            assert "specific conflict message" in result.detail


class TestValueErrorMapping:
    """ValueError is treated as a 404 (model / resource-not-found idiom)."""

    def test_value_error_maps_to_404(self) -> None:
        result = from_exception(ValueError("Model not found"))
        assert result.status_code == 404

    def test_value_error_detail_echoes_message(self) -> None:
        result = from_exception(ValueError("test/model-id not found in registry"))
        assert "test/model-id" in result.detail

    def test_empty_value_error_maps_to_404(self) -> None:
        result = from_exception(ValueError())
        assert result.status_code == 404


class TestGenericExceptionMapping:
    """Unrecognised exceptions produce a 500 with a safe, opaque message."""

    def test_runtime_error_maps_to_500(self) -> None:
        result = from_exception(RuntimeError("unexpected crash"))
        assert result.status_code == 500

    def test_generic_detail_does_not_leak_internals(self) -> None:
        result = from_exception(RuntimeError("sensitive internal stacktrace info"))
        assert result.detail == "Internal server error"
        assert "sensitive" not in result.detail

    def test_key_error_maps_to_500(self) -> None:
        result = from_exception(KeyError("missing_key"))
        assert result.status_code == 500

    def test_type_error_maps_to_500(self) -> None:
        result = from_exception(TypeError("unexpected type"))
        assert result.status_code == 500

    def test_os_error_maps_to_500(self) -> None:
        result = from_exception(OSError("disk full"))
        assert result.status_code == 500
