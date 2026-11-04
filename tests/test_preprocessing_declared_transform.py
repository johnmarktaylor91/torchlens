"""Explicit and timm-style declarations build an executable transform."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.preprocessing as pp
from torchlens.preprocessing._declared_transform import transform_from_declared
from torchlens.preprocessing._records import DeclaredPreprocessing


class _ImageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.conv(x).mean(dim=(2, 3))


def _pil_image(size: tuple[int, int] = (40, 64)) -> Any:
    from PIL import Image

    return Image.new("RGB", size, color=(120, 30, 200))


_EXPLICIT = {"resize_size": 36, "crop_size": 32, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}


def test_explicit_mapping_builds_a_transform_matching_its_fields() -> None:
    """Resize, center crop, to-tensor and normalize follow the declaration exactly."""

    pytest.importorskip("torchvision.transforms")
    resolution = pp.resolve(_EXPLICIT)
    assert resolution.preprocessing.source == "explicit_declaration"
    assert resolution.input_transform is not None

    tensor = resolution.input_transform(_pil_image())
    assert tuple(tensor.shape) == (3, 32, 32)
    # (120, 30, 200)/255 normalised with mean .5 / std .5
    expected = (torch.tensor([120.0, 30.0, 200.0]) / 255 - 0.5) / 0.5
    assert torch.allclose(tensor[:, 0, 0], expected, atol=1e-6)


def test_attached_explicit_mapping_traces_a_pil_image() -> None:
    """``attach_authority(model, mapping)`` then ``tl.trace(model, image)`` works end to end."""

    pytest.importorskip("torchvision.transforms")
    model = pp.attach_authority(_ImageModel(), dict(_EXPLICIT))
    trace = tl.trace(model, _pil_image(), save="none")
    record = trace.provenance.input_preprocessor
    assert record is not None
    assert record.source == "explicit_declaration"
    assert len(trace.input_layers) == 1


def test_attached_timm_style_mapping_traces_a_pil_image() -> None:
    """A timm data-config mapping (crop_pct, input_size) also yields a runnable transform."""

    pytest.importorskip("torchvision.transforms")
    mapping = {
        "input_size": (3, 32, 32),
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "crop_pct": 0.875,
        "interpolation": "bicubic",
    }
    resolution = pp.resolve(mapping)
    assert resolution.input_transform is not None
    assert tuple(resolution.input_transform(_pil_image()).shape) == (3, 32, 32)

    model = pp.attach_authority(_ImageModel(), mapping)
    trace = tl.trace(model, _pil_image(), save="none")
    assert trace.provenance.input_preprocessor.source == "timm"


def test_unexecutable_declarations_build_no_transform() -> None:
    """Declarations outside the executable subset stay transform-less (never guessed)."""

    pytest.importorskip("torchvision.transforms")
    assert transform_from_declared(DeclaredPreprocessing()) is None
    bgr = DeclaredPreprocessing(resize_size=32, channel_order="bgr")
    assert transform_from_declared(bgr) is None
    odd_interp = DeclaredPreprocessing(resize_size=32, interpolation="area")
    assert transform_from_declared(odd_interp) is None
    mean_only = DeclaredPreprocessing(mean=(0.5, 0.5, 0.5))
    assert transform_from_declared(mean_only) is None
