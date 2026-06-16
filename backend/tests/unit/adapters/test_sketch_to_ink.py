"""
Unit tests for DirectSketchToInkAdapter._run_inference.

Key regression: StableDiffusionXLAdapterPipeline does NOT support
callback_on_step_end (added in newer diffusers pipelines). The T2I
adapter path must use the legacy callback= / callback_steps= interface
instead.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.adapters.direct.sketch_to_ink import (
    DirectSketchToInkAdapter,
    _IMG2IMG,
    _T2I_ADAPTER,
)
from app.domain.value_objects import GenerationParams


def _make_params(**overrides) -> GenerationParams:
    defaults = dict(
        prompt="test sketch",
        negative_prompt="",
        width=512,
        height=512,
        num_inference_steps=4,
        guidance_scale=7.5,
        seed=42,
    )
    defaults.update(overrides)
    return GenerationParams(**defaults)


def _fake_result() -> SimpleNamespace:
    img = MagicMock()
    img.save = MagicMock()
    return SimpleNamespace(images=[img])


@contextmanager
def _mock_torch_and_pil(mock_pipeline):
    """
    Patch torch and PIL for the local imports inside _run_inference.

    _run_inference does `import torch` and `from PIL import Image` at call
    time, so we patch sys.modules rather than module-level attributes.
    """
    fake_torch = MagicMock()
    fake_torch.Generator.return_value.manual_seed.return_value = MagicMock()

    fake_image_obj = MagicMock()
    fake_image_obj.convert.return_value.resize.return_value = MagicMock()
    fake_Image_module = MagicMock()
    fake_Image_module.open.return_value = fake_image_obj
    fake_Image_module.LANCZOS = 1

    fake_PIL = MagicMock()
    fake_PIL.Image = fake_Image_module

    mock_pipeline.device = "cpu"

    with (
        patch.dict(sys.modules, {"torch": fake_torch, "PIL": fake_PIL, "PIL.Image": fake_Image_module}),
        patch("app.adapters.direct.sketch_to_ink.save_artifacts_from_pil_images", return_value=[]),
    ):
        yield


class TestT2IAdapterCallbackInterface:
    """
    Regression: StableDiffusionXLAdapterPipeline uses callback= not callback_on_step_end=.

    If this breaks, the T2I adapter path will raise TypeError at inference time:
      'StableDiffusionXLAdapterPipeline.__call__() got an unexpected keyword argument
      callback_on_step_end'
    """

    def test_t2i_path_uses_callback_not_callback_on_step_end(self, tmp_path: Path) -> None:
        mock_pipeline = MagicMock(return_value=_fake_result())

        with _mock_torch_and_pil(mock_pipeline):
            DirectSketchToInkAdapter._run_inference(
                pipeline_tuple=(_T2I_ADAPTER, mock_pipeline),
                params=_make_params(),
                source_image_path=tmp_path / "sketch.png",
                output_dir=tmp_path,
                strength=0.9,
                adapter_conditioning_scale=0.9,
                on_progress=None,
            )

        kwargs = mock_pipeline.call_args.kwargs
        assert "callback" in kwargs, (
            "T2I adapter path must use 'callback=' (legacy API). "
            "StableDiffusionXLAdapterPipeline does not support callback_on_step_end."
        )
        assert "callback_steps" in kwargs, (
            "T2I adapter path must set callback_steps= alongside callback=."
        )
        assert "callback_on_step_end" not in kwargs, (
            "T2I adapter path must NOT pass callback_on_step_end= — "
            "StableDiffusionXLAdapterPipeline raises TypeError for that argument."
        )

    def test_img2img_path_uses_callback_on_step_end(self, tmp_path: Path) -> None:
        mock_pipeline = MagicMock(return_value=_fake_result())

        with _mock_torch_and_pil(mock_pipeline):
            DirectSketchToInkAdapter._run_inference(
                pipeline_tuple=(_IMG2IMG, mock_pipeline),
                params=_make_params(),
                source_image_path=tmp_path / "sketch.png",
                output_dir=tmp_path,
                strength=0.9,
                adapter_conditioning_scale=0.9,
                on_progress=None,
            )

        kwargs = mock_pipeline.call_args.kwargs
        assert "callback_on_step_end" in kwargs, (
            "img2img fallback path must use callback_on_step_end= (new API)."
        )
        assert "callback" not in kwargs, (
            "img2img fallback path must not use the legacy callback= parameter."
        )

    def test_t2i_path_passes_adapter_conditioning_scale(self, tmp_path: Path) -> None:
        mock_pipeline = MagicMock(return_value=_fake_result())

        with _mock_torch_and_pil(mock_pipeline):
            DirectSketchToInkAdapter._run_inference(
                pipeline_tuple=(_T2I_ADAPTER, mock_pipeline),
                params=_make_params(),
                source_image_path=tmp_path / "sketch.png",
                output_dir=tmp_path,
                strength=0.9,
                adapter_conditioning_scale=0.75,
                on_progress=None,
            )

        assert mock_pipeline.call_args.kwargs.get("adapter_conditioning_scale") == 0.75

    def test_img2img_path_passes_strength(self, tmp_path: Path) -> None:
        mock_pipeline = MagicMock(return_value=_fake_result())

        with _mock_torch_and_pil(mock_pipeline):
            DirectSketchToInkAdapter._run_inference(
                pipeline_tuple=(_IMG2IMG, mock_pipeline),
                params=_make_params(),
                source_image_path=tmp_path / "sketch.png",
                output_dir=tmp_path,
                strength=0.65,
                adapter_conditioning_scale=0.9,
                on_progress=None,
            )

        assert mock_pipeline.call_args.kwargs.get("strength") == 0.65
