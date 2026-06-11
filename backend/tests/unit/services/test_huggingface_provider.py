"""
Unit tests for HuggingFaceSourceProvider.download_file path resolution.

These tests guard against the bug where hf_hub_download places a file at
local_dir/file_path (preserving subdirectory structure) but the code searched
for it at local_dir/basename, causing FileNotFoundError for nested paths.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from app.services.sources.huggingface import HuggingFaceSourceProvider


def _fake_hf_download(repo_id, filename, token, local_dir, **kwargs):
    """
    Simulate hf_hub_download: place the file at local_dir/filename,
    preserving subdirectory structure exactly as the real library does.
    """
    out = Path(local_dir) / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"weights")
    return str(out)


@pytest.fixture()
def provider() -> HuggingFaceSourceProvider:
    return HuggingFaceSourceProvider(token=None)


# ── Path resolution ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_flat_filename_downloads_to_destination(provider, tmp_path) -> None:
    """A top-level file (no subdirs) lands at the given destination."""
    destination = tmp_path / "sd15" / "model.safetensors"

    with (
        patch("app.services.sources.huggingface.hf_hub_download", side_effect=_fake_hf_download),
        patch("app.services.sources.huggingface.compute_file_sha256", return_value="aabbcc"),
    ):
        result = await provider.download_file(
            model_id="org/model",
            file_path="model.safetensors",
            destination=destination,
        )

    assert destination.exists(), "file must exist at the declared destination"
    assert result["sha256"] == "aabbcc"
    assert result["size_bytes"] == len(b"weights")


@pytest.mark.asyncio
async def test_nested_one_level_downloads_to_destination(provider, tmp_path) -> None:
    """
    Regression: feature_extractor/preprocessor_config.json must land at
    destination, not at destination.parent/preprocessor_config.json.

    The old bug: code used Path(file_path).name to locate the downloaded file,
    stripping subdirectory prefixes, so hf_hub_download's output was never found.
    """
    destination = tmp_path / "sd15" / "feature_extractor" / "preprocessor_config.json"

    with (
        patch("app.services.sources.huggingface.hf_hub_download", side_effect=_fake_hf_download),
        patch("app.services.sources.huggingface.compute_file_sha256", return_value="deadbeef"),
    ):
        result = await provider.download_file(
            model_id="runwayml/stable-diffusion-v1-5",
            file_path="feature_extractor/preprocessor_config.json",
            destination=destination,
        )

    assert destination.exists()
    assert result["sha256"] == "deadbeef"


@pytest.mark.asyncio
async def test_nested_two_levels_downloads_to_destination(provider, tmp_path) -> None:
    """Deep path a/b/file.bin resolves correctly."""
    destination = tmp_path / "model" / "a" / "b" / "file.bin"

    with (
        patch("app.services.sources.huggingface.hf_hub_download", side_effect=_fake_hf_download),
        patch("app.services.sources.huggingface.compute_file_sha256", return_value="ff00"),
    ):
        result = await provider.download_file(
            model_id="org/repo",
            file_path="a/b/file.bin",
            destination=destination,
        )

    assert destination.exists()
    assert result["size_bytes"] > 0


@pytest.mark.asyncio
async def test_same_basename_different_subdirs_no_collision(provider, tmp_path) -> None:
    """
    text_encoder/config.json and unet/config.json must not overwrite each other.
    Both must exist after download.
    """
    dest_text_encoder = tmp_path / "sd15" / "text_encoder" / "config.json"
    dest_unet = tmp_path / "sd15" / "unet" / "config.json"

    with (
        patch("app.services.sources.huggingface.hf_hub_download", side_effect=_fake_hf_download),
        patch("app.services.sources.huggingface.compute_file_sha256", return_value="00"),
    ):
        await provider.download_file(
            model_id="org/model",
            file_path="text_encoder/config.json",
            destination=dest_text_encoder,
        )
        await provider.download_file(
            model_id="org/model",
            file_path="unet/config.json",
            destination=dest_unet,
        )

    assert dest_text_encoder.exists()
    assert dest_unet.exists()
    assert dest_text_encoder != dest_unet


# ── hf_hub_download is called with the correct base_dir ──────────────────────


@pytest.mark.asyncio
async def test_hf_hub_download_receives_correct_base_dir(provider, tmp_path) -> None:
    """
    The local_dir passed to hf_hub_download must equal destination minus
    file_path components — not destination.parent (which would double-nest subdirs).
    """
    destination = tmp_path / "sd15" / "scheduler" / "scheduler_config.json"
    file_path = "scheduler/scheduler_config.json"
    expected_base_dir = tmp_path / "sd15"  # destination minus file_path parts

    captured: dict[str, str] = {}

    def capturing_download(repo_id, filename, token, local_dir, **kwargs):
        captured["local_dir"] = local_dir
        out = Path(local_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"x")
        return str(out)

    with (
        patch("app.services.sources.huggingface.hf_hub_download", side_effect=capturing_download),
        patch("app.services.sources.huggingface.compute_file_sha256", return_value="00"),
    ):
        await provider.download_file(
            model_id="org/model",
            file_path=file_path,
            destination=destination,
        )

    assert Path(captured["local_dir"]) == expected_base_dir


# ── Skip when file already exists ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_skips_download_when_file_exists(provider, tmp_path) -> None:
    """If the destination file already exists with content, hf_hub_download is not called."""
    destination = tmp_path / "model.safetensors"
    destination.write_bytes(b"existing weights")

    called = False

    def should_not_be_called(*args, **kwargs):
        nonlocal called
        called = True

    with (
        patch("app.services.sources.huggingface.hf_hub_download", side_effect=should_not_be_called),
        patch("app.services.sources.huggingface.compute_file_sha256", return_value="existing"),
    ):
        result = await provider.download_file(
            model_id="org/model",
            file_path="model.safetensors",
            destination=destination,
        )

    assert not called
    assert result["sha256"] == "existing"


# ── Progress callback ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sync_progress_callback_called_at_100(provider, tmp_path) -> None:
    """Sync on_progress is called with percent=100 after download completes."""
    destination = tmp_path / "model.bin"
    calls: list[dict] = []

    with (
        patch("app.services.sources.huggingface.hf_hub_download", side_effect=_fake_hf_download),
        patch("app.services.sources.huggingface.compute_file_sha256", return_value="00"),
    ):
        await provider.download_file(
            model_id="org/model",
            file_path="model.bin",
            destination=destination,
            on_progress=calls.append,
        )

    assert len(calls) == 1
    assert calls[0]["percent"] == 100.0


@pytest.mark.asyncio
async def test_async_progress_callback_called_at_100(provider, tmp_path) -> None:
    """Async on_progress is awaited with percent=100 after download completes."""
    destination = tmp_path / "model.bin"
    calls: list[dict] = []

    async def async_callback(info: dict) -> None:
        calls.append(info)

    with (
        patch("app.services.sources.huggingface.hf_hub_download", side_effect=_fake_hf_download),
        patch("app.services.sources.huggingface.compute_file_sha256", return_value="00"),
    ):
        await provider.download_file(
            model_id="org/model",
            file_path="model.bin",
            destination=destination,
            on_progress=async_callback,
        )

    assert len(calls) == 1
    assert calls[0]["percent"] == 100.0
