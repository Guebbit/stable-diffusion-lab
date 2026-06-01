"""
Common schemas shared across multiple route groups.

Contains generic response wrappers, pagination models, and error responses
that are reused across the entire API surface.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    """Standard paginated response wrapper for all list endpoints."""

    items: list[T]
    total: int = Field(description="Total number of items matching the query")
    limit: int = Field(description="Maximum items per page")
    offset: int = Field(description="Number of items skipped")
    has_more: bool = Field(description="Whether more items exist beyond this page")


class ErrorResponse(BaseModel):
    """Standard error response body for all non-2xx responses."""

    detail: str = Field(description="Human-readable error message")
    error_code: str = Field(description="Machine-readable error code for programmatic handling")
    context: dict[str, Any] = Field(
        default_factory=dict, description="Additional context about the error"
    )
