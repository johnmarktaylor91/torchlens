"""Tests for public formatting helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from torch import nn

from torchlens.utils import format_flops, format_size
from torchlens.visualization._label_format import (
    format_module_kwargs,
    format_module_path,
    format_param_list,
    format_shape,
)


def test_format_size_binary_units() -> None:
    """Byte formatter uses binary units and keeps bytes integral."""

    assert format_size(0) == "0 B"
    assert format_size(1023) == "1023 B"
    assert format_size(1024) == "1.0 KB"
    assert format_size(1024**2 * 1.5) == "1.5 MB"


def test_format_size_rejects_negative_values() -> None:
    """Negative byte counts are rejected."""

    with pytest.raises(ValueError):
        format_size(-1)


def test_format_flops_si_units() -> None:
    """FLOP formatter uses SI units."""

    assert format_flops(0) == "0 FLOPs"
    assert format_flops(999) == "999 FLOPs"
    assert format_flops(1000) == "1.0 KFLOPs"
    assert format_flops(3_400_000_000) == "3.4 GFLOPs"


def test_format_flops_accepts_convention_marker() -> None:
    """The FMA convention flag is accepted by the public formatter."""

    assert format_flops(2000, count_fma_as_two=True) == "2.0 KFLOPs"


def test_format_flops_rejects_negative_values() -> None:
    """Negative FLOP counts are rejected."""

    with pytest.raises(ValueError):
        format_flops(-1)


def test_visualization_format_shape_uses_python_tuple_notation() -> None:
    """Visualization shape formatter uses Python tuple syntax."""

    assert format_shape((1, 13, 3072)) == "(1, 13, 3072)"
    assert format_shape([3072]) == "(3072,)"
    assert format_shape(()) == "()"


def test_visualization_format_module_kwargs_uses_actual_names_and_order() -> None:
    """Visualization kwargs use actual module attribute names in declaration order."""

    module = nn.Linear(in_features=16, out_features=32)
    module.layer_type = "linear"  # type: ignore[attr-defined]
    module.func_config = {"out_features": 32, "in_features": 16}  # type: ignore[attr-defined]

    assert format_module_kwargs(module) == "in_features=16, out_features=32"


def test_visualization_format_param_list_uses_middle_dot() -> None:
    """Visualization param lists use tuple shapes and the locked separator."""

    params = [
        SimpleNamespace(name="weight", shape=(3072, 768)),
        SimpleNamespace(name="bias", shape=(3072,)),
    ]

    assert format_param_list(params) == "params: weight (3072, 768) · bias (3072,)"


def test_visualization_format_module_path_inserts_space_after_at() -> None:
    """Visualization module paths use a single space after the at sign."""

    assert format_module_path("<br/>@transformer.layer.5.ffn.lin1") == (
        "@ transformer.layer.5.ffn.lin1"
    )
