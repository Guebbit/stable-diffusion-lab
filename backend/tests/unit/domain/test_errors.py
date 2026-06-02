from __future__ import annotations

from app.domain.errors import (
    AILabError,
    CancellationError,
    DownloadError,
    IntegrityVerificationError,
    InvalidStateTransitionError,
    JobNotFoundError,
    ModelNotFoundError,
    ModelNotReadyError,
    RetryLimitExceededError,
)


def test_error_codes_match_expected_values() -> None:
    assert ModelNotFoundError.error_code == "MODEL_NOT_FOUND"
    assert ModelNotReadyError.error_code == "MODEL_NOT_READY"
    assert InvalidStateTransitionError.error_code == "INVALID_STATE_TRANSITION"
    assert JobNotFoundError.error_code == "JOB_NOT_FOUND"
    assert RetryLimitExceededError.error_code == "RETRY_LIMIT_EXCEEDED"
    assert CancellationError.error_code == "JOB_CANCELLED"
    assert IntegrityVerificationError.error_code == "INTEGRITY_CHECK_FAILED"
    assert DownloadError.error_code == "DOWNLOAD_FAILED"


def test_error_message_and_context_are_preserved() -> None:
    error = AILabError("something broke", job_id="abc", step=2)

    assert str(error) == "something broke"
    assert error.message == "something broke"
    assert error.context == {"job_id": "abc", "step": 2}


def test_all_domain_errors_extend_ailab_error() -> None:
    error_types = [
        ModelNotFoundError,
        ModelNotReadyError,
        InvalidStateTransitionError,
        JobNotFoundError,
        RetryLimitExceededError,
        CancellationError,
        IntegrityVerificationError,
        DownloadError,
    ]

    for error_type in error_types:
        assert issubclass(error_type, AILabError)
        error = error_type("test")
        assert isinstance(error, AILabError)
