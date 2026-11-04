"""Tests for public formatting helpers."""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._utils.display import format_flops, format_size
from torchlens._visualization._label_format import (
    format_output_shape,
    format_param_list,
    format_shape,
    scalar_value_line,
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


def test_output_shape_reads_scalar_for_a_zero_dim_tensor_only() -> None:
    """A 0-d output reads ``scalar``; ``(1,)`` and the raw tuple helper stay as they were."""

    assert format_output_shape(()) == "scalar" and format_output_shape((1,)) == "(1,)"
    assert format_output_shape(torch.Size([2, 3])) == "(2, 3)"
    assert format_shape(()) == "()"  # kwarg values such as dim=() keep the literal


def _scalar_pass(value: Any, pass_number: int = 1, saved: bool = True) -> SimpleNamespace:
    """Return a fake pass holding a saved 0-d activation."""

    out = torch.tensor(value) if saved else None
    return SimpleNamespace(pass_number=pass_number, has_saved_activation=saved, out=out)


def _rolled(*values: Any) -> SimpleNamespace:
    """Return a fake rolled Layer whose passes hold ``values`` (dict order shuffled)."""

    passes = [_scalar_pass(value, index + 1) for index, value in enumerate(values)]
    return SimpleNamespace(
        shape=torch.Size([]), is_terminal_bool=False, ops=dict(enumerate(reversed(passes)))
    )


def test_scalar_value_line_prints_saved_values_in_pass_order() -> None:
    """Single pass prints one value; rolled passes list in pass order, at most four."""

    single = _scalar_pass(2.3141)
    single.shape, single.is_terminal_bool = (), False
    assert scalar_value_line(single) == "= 2.31"
    assert scalar_value_line(_rolled(0.87, 1.02, 0.95)) == "= 0.87, 1.02, 0.95"
    assert scalar_value_line(_rolled(0.5, 1.0, 1.5, 2.0, 2.5)) == "= 0.5, ..., 2.5"
    assert scalar_value_line(_rolled(float("nan"), float("inf"))) == "= nan, inf"
    assert scalar_value_line(_rolled(7)) == "= 7"
    assert scalar_value_line(_rolled(True)) == "= TRUE"  # a non-terminal 0-d bool


def test_scalar_value_line_claims_nothing_it_cannot_read() -> None:
    """No row for a terminal test, an unsaved pass, or a tensor with dimensions."""

    test_node = _scalar_pass(True)
    test_node.shape, test_node.is_terminal_bool = (), True
    assert scalar_value_line(test_node) is None
    unsaved = _scalar_pass(1.0, saved=False)
    unsaved.shape, unsaved.is_terminal_bool = (), False
    assert scalar_value_line(unsaved) is None
    one_dim = SimpleNamespace(shape=(1,), is_terminal_bool=False, has_saved_activation=True)
    assert scalar_value_line(one_dim) is None


class _EndsInNorm(nn.Module):
    """A model whose output is a 0-d norm."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).norm()


def test_drawn_zero_dim_node_reads_scalar_with_its_value(tmp_path: Any) -> None:
    """The drawing labels a 0-d node ``scalar, 4 B`` with a value row, never ``(), ``."""

    torch.manual_seed(0)
    trace = tl.trace(_EndsInNorm(), torch.ones(1, 3))
    try:
        dot = trace.draw(path=str(tmp_path / "norm"), file_format="dot").dot
    finally:
        trace.cleanup()
    cells = [re.sub(r"<[^>]+>", "", cell) for cell in re.findall(r"<TD[^>]*>(.*?)</TD>", dot)]
    assert "scalar, 4 B" in cells, cells
    assert any(re.match(r"^= -?\d", cell) for cell in cells), cells
    assert not any(cell.startswith("(), ") for cell in cells), cells
