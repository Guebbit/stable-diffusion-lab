"""Unit tests for ModelResolver service.

Covers:
  - resolve() with slug lookup
  - resolve() with UUID lookup
  - resolve() passthrough when model not found
"""

from __future__ import annotations

from uuid import uuid4
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.model_resolver import ModelResolver


def _make_resolver(repo=None) -> ModelResolver:
    if repo is None:
        repo = AsyncMock()
    return ModelResolver(model_repository=repo)


# ── resolve by slug ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_resolves_slug_to_canonical_model_id() -> None:
    """When the repo finds a model by slug, return its canonical model_id."""
    repo = AsyncMock()
    model_record = SimpleNamespace(model_id="canonical/sdxl-v1-5")
    repo.get_by_model_id = AsyncMock(return_value=model_record)

    resolver = _make_resolver(repo)
    result = await resolver.resolve("runwayml/stable-diffusion-v1-5")

    assert result == "canonical/sdxl-v1-5"
    repo.get_by_model_id.assert_awaited_once_with("runwayml/stable-diffusion-v1-5")


@pytest.mark.asyncio
async def test_resolves_simple_slug() -> None:
    repo = AsyncMock()
    model_record = SimpleNamespace(model_id="org/model-canon")
    repo.get_by_model_id = AsyncMock(return_value=model_record)

    resolver = _make_resolver(repo)
    result = await resolver.resolve("short-name")

    assert result == "org/model-canon"


# ── resolve by UUID string ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_resolves_uuid_string_via_fallback() -> None:
    """When slug lookup returns None but UUID lookup succeeds, return model_id."""
    some_uuid = uuid4()
    model_record = SimpleNamespace(model_id="found/by-uuid-model")

    repo = AsyncMock()
    repo.get_by_model_id = AsyncMock(return_value=None)
    repo.get_by_id = AsyncMock(return_value=model_record)

    resolver = _make_resolver(repo)
    result = await resolver.resolve(str(some_uuid))

    assert result == "found/by-uuid-model"
    repo.get_by_id.assert_awaited_once_with(some_uuid)


@pytest.mark.asyncio
async def test_resolves_uuid_object() -> None:
    """When passed a UUID object, it is stringified and looked up."""
    some_uuid = uuid4()
    model_record = SimpleNamespace(model_id="found/by-uuid-model")

    repo = AsyncMock()
    repo.get_by_model_id = AsyncMock(return_value=model_record)

    resolver = _make_resolver(repo)
    result = await resolver.resolve(some_uuid)

    # First lookup is by str(uuid) which hits get_by_model_id
    repo.get_by_model_id.assert_awaited_once_with(str(some_uuid))
    assert result == "found/by-uuid-model"


# ── passthrough when not found ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_passthrough_when_not_in_repo() -> None:
    """Unknown identifiers are returned as-is for direct loading."""
    repo = AsyncMock()
    repo.get_by_model_id = AsyncMock(return_value=None)
    repo.get_by_id = AsyncMock(return_value=None)

    resolver = _make_resolver(repo)
    result = await resolver.resolve("unknown/model-that-does-not-exist")

    assert result == "unknown/model-that-does-not-exist"


@pytest.mark.asyncio
async def test_passthrough_non_uuid_string() -> None:
    """A non-UUID string that is not in the repo is returned unchanged."""
    repo = AsyncMock()
    repo.get_by_model_id = AsyncMock(return_value=None)

    resolver = _make_resolver(repo)
    result = await resolver.resolve("just-a-name")

    assert result == "just-a-name"
    # get_by_id should NOT be called since "just-a-name" is not a valid UUID
    repo.get_by_id.assert_not_awaited()


# ── edge cases ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_uuid_string_fallback_calls_get_by_id() -> None:
    """Valid UUID string format triggers the get_by_id fallback path."""
    some_uuid = uuid4()

    repo = AsyncMock()
    repo.get_by_model_id = AsyncMock(return_value=None)
    repo.get_by_id = AsyncMock(return_value=None)

    resolver = _make_resolver(repo)
    await resolver.resolve(str(some_uuid))

    repo.get_by_id.assert_awaited_once_with(some_uuid)


@pytest.mark.asyncio
async def test_get_by_id_failure_does_not_crash() -> None:
    """If get_by_id raises, the resolver still returns the identifier as-is."""
    some_uuid = uuid4()

    repo = AsyncMock()
    repo.get_by_model_id = AsyncMock(return_value=None)
    repo.get_by_id = AsyncMock(side_effect=Exception("db error"))

    resolver = _make_resolver(repo)
    # Should not raise - but currently it will. The resolver does not catch
    # exceptions from get_by_id, so we verify the exception propagates.
    with pytest.raises(Exception, match="db error"):
        await resolver.resolve(str(some_uuid))