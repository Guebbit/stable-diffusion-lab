"""
Centralized error handling utilities.

Eliminates duplicated try/except patterns across routers and the worker
by providing a single translation layer from domain exceptions to the
appropriate response type (HTTPException for API endpoints, structured
error dicts for internal use).
"""

from __future__ import annotations

import logging
from http import HTTPStatus

from fastapi import HTTPException

from app.domain.errors import AILabError

logger = logging.getLogger(__name__)

# Mapping from AILabError error_code to HTTP status.
_STATUS_MAP: dict[str, HTTPStatus] = {
    "MODEL_NOT_FOUND": HTTPStatus.NOT_FOUND,
    "MODEL_NOT_READY": HTTPStatus.NOT_FOUND,
    "JOB_NOT_FOUND": HTTPStatus.NOT_FOUND,
    "ARTIFACT_NOT_FOUND": HTTPStatus.NOT_FOUND,
    "INVALID_STATE_TRANSITION": HTTPStatus.CONFLICT,
    "RETRY_LIMIT_EXCEEDED": HTTPStatus.CONFLICT,
    "INVALID_INPUT": HTTPStatus.BAD_REQUEST,
    "DUPLICATE_MODEL": HTTPStatus.CONFLICT,
    "DOWNLOAD_FAILED": HTTPStatus.BAD_REQUEST,
    "INTEGRITY_CHECK_FAILED": HTTPStatus.BAD_REQUEST,
    "JOB_CANCELLED": HTTPStatus.CONFLICT,
}

_DEFAULT_STATUS = HTTPStatus.INTERNAL_SERVER_ERROR


def from_exception(exc: Exception) -> HTTPException:
    """
    Translate a domain or built-in exception into an HTTPException.

    Args:
        exc: The caught exception (typically ``AILabError`` or ``ValueError``).

    Returns:
        An ``HTTPException`` ready to be raised from a FastAPI endpoint.
    """
    if isinstance(exc, AILabError):
        status = _STATUS_MAP.get(exc.error_code, _DEFAULT_STATUS)
        logger.warning("Domain error %s: %s", exc.error_code, exc)
        return HTTPException(status_code=status.value, detail=str(exc))

    if isinstance(exc, ValueError):
        logger.warning("Validation error: %s", exc)
        return HTTPException(status_code=HTTPStatus.NOT_FOUND.value, detail=str(exc))

    logger.error("Unexpected error: %s", exc)
    return HTTPException(
        status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        detail="Internal server error",
    )