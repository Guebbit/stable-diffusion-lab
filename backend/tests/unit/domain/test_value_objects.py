"""Unit tests for domain value objects.

Covers:
  - GenerationParams defaults, frozen behavior, equality
  - ModelIdentifier fields and equality
  - JobProgress timestamp default, frozen behavior
  - ArtifactReference fields
  - DownloadProgress percentage calculation
"""

from __future__ import annotations

from datetime import datetime, UTC
from uuid import uuid4

import pytest

from app.domain.value_objects import (
    ArtifactReference,
    DownloadProgress,
    GenerationParams,
    JobProgress,
    ModelIdentifier,
)


# ── GenerationParams ─────────────────────────────────


class TestGenerationParams:
    def test_default_values(self) -> None:
        """Only prompt is required; all other fields have sensible defaults."""
        params = GenerationParams(prompt="a cat")

        assert params.prompt == "a cat"
        assert params.negative_prompt == ""
        assert params.width == 512
        assert params.height == 512
        assert params.num_inference_steps == 20
        assert params.guidance_scale == 7.5
        assert params.seed is None
        assert params.num_images == 1
        assert params.extra == {}

    def test_custom_values(self) -> None:
        """All fields can be overridden."""
        params = GenerationParams(
            prompt="sunset",
            negative_prompt="blurry",
            width=768,
            height=512,
            num_inference_steps=40,
            guidance_scale=9.0,
            seed=42,
            num_images=3,
            extra={"custom": True},
        )

        assert params.prompt == "sunset"
        assert params.negative_prompt == "blurry"
        assert params.width == 768
        assert params.height == 512
        assert params.num_inference_steps == 40
        assert params.guidance_scale == 9.0
        assert params.seed == 42
        assert params.num_images == 3
        assert params.extra == {"custom": True}

    def test_frozen_cannot_mutate(self) -> None:
        """GenerationParams is frozen; mutation raises AttributeError or FrozenInstanceError."""
        params = GenerationParams(prompt="test")

        with pytest.raises((AttributeError, Exception)):
            params.prompt = "changed"  # type: ignore

    def test_equality_by_value(self) -> None:
        """Two instances with the same field values are equal."""
        a = GenerationParams(prompt="same", width=640)
        b = GenerationParams(prompt="same", width=640)

        assert a == b

    def test_inequality_when_fields_differ(self) -> None:
        a = GenerationParams(prompt="a")
        b = GenerationParams(prompt="b")

        assert a != b


# ── ModelIdentifier ─────────────────────────


class TestModelIdentifier:
    def test_basic_fields(self) -> None:
        ident = ModelIdentifier(model_id="org/model", source="civitai")

        assert ident.model_id == "org/model"
        assert ident.source == "civitai"
        assert ident.family == ""
        assert ident.variant == ""

    def test_with_family_and_variant(self) -> None:
        ident = ModelIdentifier(
            model_id="org/model",
            source="huggingface",
            family="stable-diffusion",
            variant="v1-5",
        )

        assert ident.family == "stable-diffusion"
        assert ident.variant == "v1-5"

    def test_equality_by_value(self) -> None:
        a = ModelIdentifier(model_id="x", source="hf")
        b = ModelIdentifier(model_id="x", source="hf")

        assert a == b

    def test_frozen_cannot_mutate(self) -> None:
        ident = ModelIdentifier(model_id="x", source="hf")

        with pytest.raises((AttributeError, Exception)):
            ident.model_id = "y"  # type: ignore


# ── JobProgress ─────────────────────────


class TestJobProgress:
    def test_default_values(self) -> None:
        job_id = uuid4()
        progress = JobProgress(job_id=job_id, status="running")

        assert progress.job_id == job_id
        assert progress.status == "running"
        assert progress.progress_percent == 0
        assert progress.current_step == 0
        assert progress.total_steps == 0
        assert progress.message == ""
        assert isinstance(progress.timestamp, datetime)

    def test_custom_progress(self) -> None:
        job_id = uuid4()
        progress = JobProgress(
            job_id=job_id,
            status="running",
            progress_percent=50,
            current_step=5,
            total_steps=10,
            message="generating",
        )

        assert progress.progress_percent == 50
        assert progress.current_step == 5
        assert progress.total_steps == 10
        assert progress.message == "generating"

    def test_frozen_cannot_mutate(self) -> None:
        progress = JobProgress(job_id=uuid4(), status="done")

        with pytest.raises((AttributeError, Exception)):
            progress.status = "failed"  # type: ignore


# ── ArtifactReference ─────────────────────────


class TestArtifactReference:
    def test_basic_fields(self) -> None:
        ref = ArtifactReference(
            artifact_id=uuid4(),
            job_id=uuid4(),
            file_path="/data/output.png",
        )

        assert ref.thumbnail_path == ""
        assert ref.media_type == "image/png"
        assert ref.width == 0
        assert ref.height == 0
        assert ref.size_bytes == 0

    def test_full_fields(self) -> None:
        aid = uuid4()
        jid = uuid4()
        ref = ArtifactReference(
            artifact_id=aid,
            job_id=jid,
            file_path="/data/out.jpg",
            thumbnail_path="/data/thumb.jpg",
            media_type="image/jpeg",
            width=1024,
            height=768,
            size_bytes=50000,
        )

        assert ref.thumbnail_path == "/data/thumb.jpg"
        assert ref.media_type == "image/jpeg"
        assert ref.width == 1024
        assert ref.height == 768
        assert ref.size_bytes == 50000

    def test_frozen_cannot_mutate(self) -> None:
        ref = ArtifactReference(
            artifact_id=uuid4(),
            job_id=uuid4(),
            file_path="/x.png",
        )

        with pytest.raises((AttributeError, Exception)):
            ref.file_path = "/y.png"  # type: ignore


# ── DownloadProgress ─────────────────────────


class TestDownloadProgress:
    def test_default_values(self) -> None:
        dp = DownloadProgress(model_id="org/model")

        assert dp.downloaded_bytes == 0
        assert dp.total_bytes is None
        assert dp.percentage == 0
        assert dp.speed_bytes_per_sec == 0.0
        assert dp.is_complete is False
        assert dp.error == ""

    def test_custom_progress(self) -> None:
        dp = DownloadProgress(
            model_id="org/model",
            downloaded_bytes=500,
            total_bytes=1000,
            percentage=50,
            speed_bytes_per_sec=100.0,
        )

        assert dp.percentage == 50
        assert dp.speed_bytes_per_sec == 100.0
        assert dp.is_complete is False

    def test_complete_download(self) -> None:
        dp = DownloadProgress(
            model_id="org/model",
            downloaded_bytes=1000,
            total_bytes=1000,
            percentage=100,
            is_complete=True,
        )

        assert dp.is_complete is True

    def test_error_state(self) -> None:
        dp = DownloadProgress(
            model_id="org/model",
            error="network timeout",
        )

        assert dp.error == "network timeout"

    def test_frozen_cannot_mutate(self) -> None:
        dp = DownloadProgress(model_id="x")

        with pytest.raises((AttributeError, Exception)):
            dp.model_id = "y"  # type: ignore