"""Opt-in backward-node rows on ``draw_backward`` (U08.5).

Module-path labels and gradient values appear on backward nodes only when the
caller asks for them (``show_module_paths=True`` / ``show_grad_values=True``);
a default ``draw_backward`` is byte-for-byte unaffected by the new rows.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError


class _Block(nn.Module):
    """Nested block so module addresses carry a dotted path."""

    def __init__(self) -> None:
        """Initialize the layers."""
        super().__init__()
        self.proj = nn.Linear(3, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        return torch.relu(self.proj(x))


class _Net(nn.Module):
    """Two-level model with a parameter-owning nested module."""

    def __init__(self) -> None:
        """Initialize the submodules."""
        super().__init__()
        self.block = _Block()
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        return self.head(self.block(x))


def _backward_trace() -> tl.Trace:
    """Return a trace with one captured backward pass and saved gradients."""
    torch.manual_seed(0)
    trace = tl.trace(
        _Net(),
        torch.randn(2, 3, requires_grad=True),
        capture=tl.options.CaptureOptions(save_grads="all"),
    )
    trace.log_backward(trace[trace.output_layers[0]].out.sum())
    return trace


def _draw(trace: tl.Trace, tmp_path: Path, name: str, **kwargs: object) -> str:
    """Render the backward graph to DOT and return the source."""
    return trace.draw_backward(
        vis_outpath=str(tmp_path / name), vis_save_only=True, vis_fileformat="dot", **kwargs
    )


def test_backward_rows_absent_by_default(tmp_path: Path) -> None:
    """A default render has neither row, and explicit False equals the default."""
    trace = _backward_trace()
    try:
        default = _draw(trace, tmp_path, "default")
        explicit_off = _draw(
            trace, tmp_path, "off", show_module_paths=False, show_grad_values=False
        )
    finally:
        trace.cleanup()
    assert "module block.proj" not in default
    assert "grad norm" not in default
    assert "param block.proj.weight" not in default
    assert default == explicit_off


def test_backward_module_path_row_opt_in(tmp_path: Path) -> None:
    """``show_module_paths=True`` names the owning module, and params on AccumulateGrad."""
    trace = _backward_trace()
    try:
        dot = _draw(trace, tmp_path, "modules", show_module_paths=True)
        with pytest.raises(InvalidArgumentError):
            _draw(trace, tmp_path, "bad", show_module_paths="yes")
    finally:
        trace.cleanup()
    assert "module block.proj" in dot
    assert "module head" in dot
    assert "param block.proj.weight" in dot
    assert "grad norm" not in dot


def test_backward_grad_value_row_opt_in(tmp_path: Path) -> None:
    """``show_grad_values=True`` prints a finite norm and absmax per captured gradient."""
    trace = _backward_trace()
    try:
        dot = _draw(trace, tmp_path, "values", show_grad_values=True)
        unrolled = _draw(
            trace, tmp_path, "values_unrolled", show_grad_values=True, vis_mode="unrolled"
        )
    finally:
        trace.cleanup()
    rows = re.findall(r"grad norm ([0-9.eE+-]+), absmax ([0-9.eE+-]+)", dot)
    assert rows, "no gradient-value row was rendered"
    for norm_text, absmax_text in rows:
        norm, absmax = float(norm_text), float(absmax_text)
        assert norm >= absmax >= 0.0
    assert "module block.proj" not in dot
    assert re.search(r"grad norm [0-9.eE+-]+, absmax", unrolled)
