"""Tests for the menagerie validator ``--input-scale`` spatial down-scaling."""

from __future__ import annotations

import pytest
import torch

from menagerie.validate_menagerie import (
    MIN_SCALED_SPATIAL_DIM,
    _scale_example_input,
    build_parser,
)


def test_input_scale_half_halves_4d_spatial_dims() -> None:
    """``--input-scale 0.5`` halves H and W of a 4D image input, keeping N and C."""

    example = torch.randn(2, 3, 224, 224)
    scaled = _scale_example_input(example, 0.5)

    assert scaled.shape == (2, 3, 112, 112)
    # Batch and channel dims are preserved.
    assert scaled.shape[0] == example.shape[0]
    assert scaled.shape[1] == example.shape[1]
    # dtype and device carry over unchanged.
    assert scaled.dtype == example.dtype
    assert scaled.device == example.device


def test_input_scale_clamps_to_minimum_spatial_dim() -> None:
    """Scaling never shrinks a spatial dim below the safe minimum."""

    example = torch.randn(1, 3, 40, 40)
    scaled = _scale_example_input(example, 0.1)

    # 40 * 0.1 = 4 -> clamped up to MIN_SCALED_SPATIAL_DIM.
    assert scaled.shape[2] == MIN_SCALED_SPATIAL_DIM
    assert scaled.shape[3] == MIN_SCALED_SPATIAL_DIM


def test_input_scale_leaves_low_rank_input_untouched() -> None:
    """Inputs with fewer than 3 dims (no spatial axes) are returned unchanged."""

    example = torch.randn(4, 16)
    scaled = _scale_example_input(example, 0.5)

    assert scaled is example


def test_input_scale_scales_nested_container_inputs() -> None:
    """Tuple/list/dict input trees are scaled element-wise."""

    example = (torch.randn(1, 3, 64, 64), {"aux": torch.randn(1, 8, 128, 96)}, [7])
    scaled = _scale_example_input(example, 0.5)

    assert scaled[0].shape == (1, 3, 32, 32)
    # 128*0.5=64, 96*0.5=48, both above the clamp minimum.
    assert scaled[1]["aux"].shape == (1, 8, 64, 48)
    # Non-tensor leaves are passed through unchanged.
    assert scaled[2] == [7]


def test_input_scale_preserves_requires_grad_for_float_inputs() -> None:
    """A float input that carried requires_grad keeps it after scaling."""

    example = torch.randn(1, 3, 64, 64, requires_grad=True)
    scaled = _scale_example_input(example, 0.5)

    assert scaled.requires_grad is True


def test_parser_accepts_and_bounds_input_scale() -> None:
    """The parser parses ``--input-scale`` and rejects out-of-range values."""

    parser = build_parser()

    assert parser.parse_args(["--input-scale", "0.5"]).input_scale == pytest.approx(0.5)
    # Default is full resolution.
    assert parser.parse_args([]).input_scale == pytest.approx(1.0)

    for bad in ("0", "-0.5", "1.5", "nan-ish"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--input-scale", bad])
