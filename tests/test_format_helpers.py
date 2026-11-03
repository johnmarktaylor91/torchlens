"""Tests for public formatting helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from torchlens._utils.display import format_flops, format_size
from torchlens._visualization._label_format import (
    format_param_list,
)


def test_format_size_rejects_negative_values() -> None:
    """Negative byte counts are rejected."""

    with pytest.raises(ValueError):
        format_size(-1)


def test_format_flops_rejects_negative_values() -> None:
    """Negative FLOP counts are rejected."""

    with pytest.raises(ValueError):
        format_flops(-1)


def test_visualization_format_param_list_handles_mixed_and_anonymous_params() -> None:
    """Visualization param lists support mixed trainability and anonymous params."""

    params = [
        SimpleNamespace(name="weight", shape=(3072, 768), is_trainable=True),
        SimpleNamespace(shape=(3072,), is_trainable=False),
        SimpleNamespace(shape=(3072,), is_trainable=True),
    ]

    assert format_param_list(params) == "params: weight (3072, 768) · [3072,] · (3072,)"
