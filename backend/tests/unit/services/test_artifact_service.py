"""
Unit tests for ArtifactService.

Tests artifact CRUD: get, list, update_metadata, delete.
Uses stubbed repositories and mocked filesystem.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from pytest_mock import MockerFixture

from app.infrastructure.database.models import ArtifactRecord
from app.services.artifact_service import ArtifactService


# ── Stubs ────────────────────────────────────────────────────────────────

class _ArtifactRepoStub:
    """Stub ArtifactRepository with in-memory storage."""

    def __init__(self) -> None:
        self._store: dict[str, ArtifactRecord] = {}

    async def get_by_id(self, artifact_id: object) -> ArtifactRecord | None:
        return self._store.get(str(artifact_id))

    async def create(self, artifact: ArtifactRecord) -> ArtifactRecord:
        self._store[str(artifact.id)] = artifact
        return artifact

    async def update(self, artifact_id: object, **kwargs: object) -> None:
        key = str(artifact_id)
        if key in self._store:
            for k, v in kwargs.items():
                setattr(self._store[key], k, v)

    async def delete(self, artifact_id: object) -> None:
        self._store.pop(str(artifact_id), None)

    async def list_filtered(
        self,
        model_name: str | None = None,
        media_type: str | None = None,
        is_favorite: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[ArtifactRecord], int]:
        artifacts = list(self._store.values())

        if model_name is not None:
            artifacts = [a for a in artifacts if a.model_name == model_name]
        if media_type is not None:
            artifacts = [a for a in artifacts if a.media_type == media_type]
        if is_favorite is not None:
            artifacts = [a for a in artifacts if a.is_favorite == is_favorite]

        total = len(artifacts)
        page = artifacts[offset: offset + limit]
        return page, total

    def add_artifact(self, artifact: ArtifactRecord) -> None:
        """Helper to seed test data."""
        self._store[str(artifact.id)] = artifact


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def artifact_repo() -> _ArtifactRepoStub:
    return _ArtifactRepoStub()


@pytest.fixture
def service(artifact_repo: _ArtifactRepoStub) -> ArtifactService:
    return ArtifactService(artifact_repo)


@pytest.fixture
def sample_artifact() -> ArtifactRecord:
    """Create a sample artifact for testing."""
    from app.infrastructure.database.models import JobRecord

    # Create artifact with required fields
    return ArtifactRecord(
        id=uuid4(),
        job_id=uuid4(),
        file_path="/tmp/artifacts/test_image.png",
        thumbnail_path="/tmp/artifacts/thumbs/test_image_thumb.png",
        media_type="image/png",
        width=512,
        height=512,
        size_bytes=102400,
        model_name="runwayml/stable-diffusion-v1-5",
        is_favorite=False,
        rating=0,
        notes="",
    )


# ── Tests: get_artifact ──────────────────────────────────────────────────

class TestGetArtifact:
    """Tests for ArtifactService.get_artifact()."""

    @pytest.mark.asyncio
    async def test_returns_artifact_when_exists(
        self,
        service: ArtifactService,
        artifact_repo: _ArtifactRepoStub,
        sample_artifact: ArtifactRecord,
    ) -> None:
        artifact_repo.add_artifact(sample_artifact)
        found = await service.get_artifact(sample_artifact.id)
        assert found is not None
        assert found.id == sample_artifact.id
        assert found.media_type == "image/png"

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self, service: ArtifactService) -> None:
        missing_id = uuid4()
        found = await service.get_artifact(missing_id)
        assert found is None

    @pytest.mark.asyncio
    async def test_preserves_all_fields(
        self,
        service: ArtifactService,
        artifact_repo: _ArtifactRepoStub,
        sample_artifact: ArtifactRecord,
    ) -> None:
        artifact_repo.add_artifact(sample_artifact)
        found = await service.get_artifact(sample_artifact.id)
        assert found is not None
        assert found.width == 512
        assert found.height == 512
        assert found.size_bytes == 102400
        assert found.model_name == "runwayml/stable-diffusion-v1-5"


# ── Tests: list_artifacts ────────────────────────────────────────────────

class TestListArtifacts:
    """Tests for ArtifactService.list_artifacts()."""

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_artifacts(
        self, service: ArtifactService
    ) -> None:
        artifacts, total = await service.list_artifacts()
        assert artifacts == []
        assert total == 0

    @pytest.mark.asyncio
    async def test_returns_all_artifacts(
        self,
        service: ArtifactService,
        artifact_repo: _ArtifactRepoStub,
        sample_artifact: ArtifactRecord,
    ) -> None:
        artifact_repo.add_artifact(sample_artifact)
        artifacts, total = await service.list_artifacts()
        assert len(artifacts) == 1
        assert total == 1

    @pytest.mark.asyncio
    async def test_filters_by_model_name(
        self,
        service: ArtifactService,
        artifact_repo: _ArtifactRepoStub,
        sample_artifact: ArtifactRecord,
    ) -> None:
        artifact_repo.add_artifact(sample_artifact)
        # Add another artifact with a different model
        other = ArtifactRecord(
            id=uuid4(),
            job_id=uuid4(),
            file_path="/tmp/other.png",
            media_type="image/png",
            model_name="stabilityai/sdxl",
            is_favorite=False,
        )
        artifact_repo.add_artifact(other)

        artifacts, total = await service.list_artifacts(model_name="runwayml/stable-diffusion-v1-5")
        assert total == 1
        assert artifacts[0].model_name == "runwayml/stable-diffusion-v1-5"

    @pytest.mark.asyncio
    async def test_filters_by_media_type(
        self,
        service: ArtifactService,
        artifact_repo: _ArtifactRepoStub,
        sample_artifact: ArtifactRecord,
    ) -> None:
        artifact_repo.add_artifact(sample_artifact)
        video_artifact = ArtifactRecord(
            id=uuid4(),
            job_id=uuid4(),
            file_path="/tmp/video.mp4",
            media_type="video/mp4",
            model_name="test/model",
            is_favorite=False,
        )
        artifact_repo.add_artifact(video_artifact)

        artifacts, total = await service.list_artifacts(media_type="image/png")
        assert total == 1
        assert artifacts[0].media_type == "image/png"

    @pytest.mark.asyncio
    async def test_filters_by_is_favorite(
        self,
        service: ArtifactService,
        artifact_repo: _ArtifactRepoStub,
        sample_artifact: ArtifactRecord,
    ) -> None:
        # Make one favorite
        sample_artifact.is_favorite = True
        artifact_repo.add_artifact(sample_artifact)

        not_favorite = ArtifactRecord(
            id=uuid4(),
            job_id=uuid4(),
            file_path="/tmp/not_fav.png",
            media_type="image/png",
            model_name="test/model",
            is_favorite=False,
        )
        artifact_repo.add_artifact(not_favorite)

        artifacts, total = await service.list_artifacts(is_favorite=True)
        assert total == 1
        assert artifacts[0].is_favorite is True

    @pytest.mark.asyncio
    async def test_pagination_limit(
        self,
        service: ArtifactService,
        artifact_repo: _ArtifactRepoStub,
    ) -> None:
        # Add 5 artifacts
        for _ in range(5):
            art = ArtifactRecord(
                id=uuid4(),
                job_id=uuid4(),
                file_path="/tmp/img.png",
                media_type="image/png",
                model_name="test/model",
                is_favorite=False,
            )
            artifact_repo.add_artifact(art)

        artifacts, total = await service.list_artifacts(limit=2)
        assert total == 5
        assert len(artifacts) == 2

    @pytest.mark.asyncio
    async def test_pagination_offset(
        self,
        service: ArtifactService,
        artifact_repo: _ArtifactRepoStub,
    ) -> None:
        for _ in range(5):
            art = ArtifactRecord(
                id=uuid4(),
                job_id=uuid4(),
                file_path="/tmp/img.png",
                media_type="image/png",
                model_name="test/model",
                is_favorite=False,
            )
            artifact_repo.add_artifact(art)

        artifacts, total = await service.list_artifacts(limit=2, offset=2)
        assert total == 5
        assert len(artifacts) == 2

    @pytest.mark.asyncio
    async def test_combined_filters(
        self,
        service: ArtifactService,
        artifact_repo: _ArtifactRepoStub,
        sample_artifact: ArtifactRecord,
    ) -> None:
        artifact_repo.add_artifact(sample_artifact)

        # Add artifact matching model but different media type
        other_media = ArtifactRecord(
            id=uuid4(),
            job_id=uuid4(),
            file_path="/tmp/video.mp4",
            media_type="video/mp4",
            model_name="runwayml/stable-diffusion-v1-5",
            is_favorite=True,
        )
        artifact_repo.add_artifact(other_media)

        artifacts, total = await service.list_artifacts(
            model_name="runwayml/stable-diffusion-v1-5",
            media_type="image/png",
        )
        assert total == 1
        assert artifacts[0].media_type == "image/png"


# ── Tests: update_metadata ───────────────────────────────────────────────

class TestUpdateMetadata:
    """Tests for ArtifactService.update_metadata()."""

    @pytest.mark.asyncio
    async def test_updates_favorite_flag(
        self,
        service: ArtifactService,
        artifact_repo: _ArtifactRepoStub,
        sample_artifact: ArtifactRecord,
    ) -> None:
        artifact_repo.add_artifact(sample_artifact)
        result = await service.update_metadata(sample_artifact.id, is_favorite=True)
        assert result is not None
        assert result.is_favorite is True

    @pytest.mark.asyncio
    async def test_updates_rating(
        self,
        service: ArtifactService,
        artifact_repo: _ArtifactRepoStub,
        sample_artifact: ArtifactRecord,
    ) -> None:
        artifact_repo.add_artifact(sample_artifact)
        result = await service.update_metadata(sample_artifact.id, rating=5)
        assert result is not None
        assert result.rating == 5

    @pytest.mark.asyncio
    async def test_updates_notes(
        self,
        service: ArtifactService,
        artifact_repo: _ArtifactRepoStub,
        sample_artifact: ArtifactRecord,
    ) -> None:
        artifact_repo.add_artifact(sample_artifact)
        result = await service.update_metadata(sample_artifact.id, notes="Beautiful image")
        assert result is not None
        assert result.notes == "Beautiful image"

    @pytest.mark.asyncio
    async def test_updates_multiple_fields(
        self,
        service: ArtifactService,
        artifact_repo: _ArtifactRepoStub,
        sample_artifact: ArtifactRecord,
    ) -> None:
        artifact_repo.add_artifact(sample_artifact)
        result = await service.update_metadata(
            sample_artifact.id,
            is_favorite=True,
            rating=5,
            notes="Amazing!",
        )
        assert result is not None
        assert result.is_favorite is True
        assert result.rating == 5
        assert result.notes == "Amazing!"

    @pytest.mark.asyncio
    async def test_partial_update_preserves_other_fields(
        self,
        service: ArtifactService,
        artifact_repo: _ArtifactRepoStub,
        sample_artifact: ArtifactRecord,
    ) -> None:
        sample_artifact.rating = 3
        artifact_repo.add_artifact(sample_artifact)

        result = await service.update_metadata(sample_artifact.id, notes="Updated notes")
        assert result is not None
        assert result.notes == "Updated notes"
        assert result.rating == 3  # Unchanged

    @pytest.mark.asyncio
    async def test_returns_none_for_missing_artifact(self, service: ArtifactService) -> None:
        missing_id = uuid4()
        result = await service.update_metadata(missing_id, is_favorite=True)
        assert result is None

    @pytest.mark.asyncio
    async def test_no_updates_does_not_modify(
        self,
        service: ArtifactService,
        artifact_repo: _ArtifactRepoStub,
        sample_artifact: ArtifactRecord,
    ) -> None:
        artifact_repo.add_artifact(sample_artifact)
        result = await service.update_metadata(sample_artifact.id)
        assert result is not None
        assert result.is_favorite == sample_artifact.is_favorite
        assert result.rating == sample_artifact.rating

    @pytest.mark.asyncio
    async def test_none_values_treated_as_no_change(
        self,
        service: ArtifactService,
        artifact_repo: _ArtifactRepoStub,
        sample_artifact: ArtifactRecord,
    ) -> None:
        sample_artifact.rating = 4
        artifact_repo.add_artifact(sample_artifact)

        result = await service.update_metadata(
            sample_artifact.id,
            is_favorite=None,
            rating=None,
            notes=None,
        )
        assert result is not None
        assert result.rating == 4  # Unchanged


# ── Tests: delete_artifact ───────────────────────────────────────────────

class TestDeleteArtifact:
    """Tests for ArtifactService.delete_artifact()."""

    @pytest.mark.asyncio
    async def test_deletes_artifact_from_repo(
        self,
        service: ArtifactService,
        artifact_repo: _ArtifactRepoStub,
        sample_artifact: ArtifactRecord,
        mocker: MockerFixture,
    ) -> None:
        artifact_repo.add_artifact(sample_artifact)
        # Mock file path to not actually delete
        mocker.patch.object(Path, "exists", return_value=False)

        result = await service.delete_artifact(sample_artifact.id)
        assert result is True
        assert await artifact_repo.get_by_id(sample_artifact.id) is None

    @pytest.mark.asyncio
    async def test_returns_false_for_missing_artifact(self, service: ArtifactService) -> None:
        missing_id = uuid4()
        result = await service.delete_artifact(missing_id)
        assert result is False

    @pytest.mark.asyncio
    async def test_deletes_file_when_exists(
        self,
        service: ArtifactService,
        artifact_repo: _ArtifactRepoStub,
        sample_artifact: ArtifactRecord,
        mocker: MockerFixture,
    ) -> None:
        artifact_repo.add_artifact(sample_artifact)

        mock_path = mocker.MagicMock(spec=Path)
        mock_path.exists.return_value = True
        mocker.patch("app.services.artifact_service.Path", return_value=mock_path)

        await service.delete_artifact(sample_artifact.id)
        # Called twice: once for file_path, once for thumbnail_path
        assert mock_path.unlink.call_count == 2

    @pytest.mark.asyncio
    async def test_skips_file_deletion_when_not_found(
        self,
        service: ArtifactService,
        artifact_repo: _ArtifactRepoStub,
        sample_artifact: ArtifactRecord,
        mocker: MockerFixture,
    ) -> None:
        artifact_repo.add_artifact(sample_artifact)

        mock_path = mocker.MagicMock(spec=Path)
        mock_path.exists.return_value = False
        mocker.patch("app.services.artifact_service.Path", return_value=mock_path)

        result = await service.delete_artifact(sample_artifact.id)
        assert result is True

    @pytest.mark.asyncio
    async def test_deletes_thumbnail_when_exists(
        self,
        service: ArtifactService,
        artifact_repo: _ArtifactRepoStub,
        sample_artifact: ArtifactRecord,
        mocker: MockerFixture,
    ) -> None:
        artifact_repo.add_artifact(sample_artifact)

        mock_main_path = mocker.MagicMock(spec=Path)
        mock_main_path.exists.return_value = True

        mock_thumb_path = mocker.MagicMock(spec=Path)
        mock_thumb_path.exists.return_value = True

        def path_side_effect(path_str: str) -> Path:
            if "thumb" in str(path_str):
                return mock_thumb_path
            return mock_main_path

        mocker.patch("app.services.artifact_service.Path", side_effect=path_side_effect)

        await service.delete_artifact(sample_artifact.id)
        mock_thumb_path.unlink.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_artifact_without_thumbnail(
        self,
        service: ArtifactService,
        artifact_repo: _ArtifactRepoStub,
        mocker: MockerFixture,
    ) -> None:
        artifact = ArtifactRecord(
            id=uuid4(),
            job_id=uuid4(),
            file_path="/tmp/no_thumb.png",
            thumbnail_path="",
            media_type="image/png",
            model_name="test/model",
            is_favorite=False,
        )
        artifact_repo.add_artifact(artifact)

        mock_path = mocker.MagicMock(spec=Path)
        mock_path.exists.return_value = False
        mocker.patch("app.services.artifact_service.Path", return_value=mock_path)

        result = await service.delete_artifact(artifact.id)
        assert result is True

    @pytest.mark.asyncio
    async def test_removes_from_db_after_file_cleanup(
        self,
        service: ArtifactService,
        artifact_repo: _ArtifactRepoStub,
        sample_artifact: ArtifactRecord,
        mocker: MockerFixture,
    ) -> None:
        artifact_repo.add_artifact(sample_artifact)
        mocker.patch.object(Path, "exists", return_value=False)

        await service.delete_artifact(sample_artifact.id)
        # Verify artifact was removed from the repository
        remaining = await artifact_repo.get_by_id(sample_artifact.id)
        assert remaining is None