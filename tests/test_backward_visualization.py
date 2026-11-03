"""Smoke tests for backward grad_fn_handle visualization."""

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions


class _LinearReluModel(nn.Module):
    """Small model with module and functional ops."""

    def __init__(self) -> None:
        """Initialize the model."""
        super().__init__()
        self.fc = nn.Linear(3, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        return torch.relu(self.fc(x)).sum()


class _ViewModel(nn.Module):
    """Small model with a view op in the backward path."""

    def __init__(self) -> None:
        """Initialize the model."""
        super().__init__()
        self.fc = nn.Linear(6, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        y = self.fc(x)
        return y.view(2, 3, 2).sum()


class _DoubleFn(torch.autograd.Function):
    """Custom autograd function for visualization tests."""

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor) -> torch.Tensor:
        """Return doubled input."""
        return x * 2

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad: torch.Tensor) -> torch.Tensor:
        """Return doubled upstream grad."""
        return grad * 2


class _CustomModel(nn.Module):
    """Model using a custom autograd function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        return _DoubleFn.apply(x).sum()


def _log_backward_model(model: nn.Module, x: torch.Tensor) -> tl.Trace:
    """Return a Trace with backward metadata captured.

    Parameters
    ----------
    model:
        Model to log.
    x:
        Input tensor.

    Returns
    -------
    tl.Trace
        Model log after backward capture.
    """

    trace = tl.trace(model, x, capture=CaptureOptions(save_grads="all"))
    trace.log_backward(trace[trace.output_layers[0]].out.sum())
    return trace


def _log_two_backward_passes(model: nn.Module, x: torch.Tensor) -> tl.Trace:
    """Return a Trace with two captured backward passes.

    Parameters
    ----------
    model:
        Model to trace.
    x:
        Input tensor.

    Returns
    -------
    tl.Trace
        Trace with two backward passes.
    """

    trace = tl.trace(model, x, capture=CaptureOptions(save_grads="all"))
    loss = trace[trace.output_layers[0]].out.sum()
    trace.log_backward(loss, retain_graph=True)
    trace.log_backward(loss)
    return trace


def test_draw_backward_renders(tmp_path: Path) -> None:
    """draw_backward returns DOT source and writes a non-empty output file."""
    trace = _log_backward_model(_LinearReluModel(), torch.randn(2, 3, requires_grad=True))
    outpath = tmp_path / "backward_graph"

    dot = trace.draw(
        graph="backward",
        path=str(outpath),
        file_format="svg",
    ).dot

    assert "digraph" in dot
    assert "backward graph" in dot
    assert (tmp_path / "backward_graph.svg").stat().st_size > 0


def test_backward_graph_bwd_filter_limits_visible_passes(tmp_path: Path) -> None:
    """The bwd filter restricts unrolled backward rendering to selected passes."""
    trace = _log_two_backward_passes(_LinearReluModel(), torch.randn(2, 3, requires_grad=True))

    dot = trace.draw(
        graph="backward",
        path=str(tmp_path / "filtered"),
        file_format="svg",
        rolled=False,
        bwd=2,
    ).dot

    assert "cluster_backward_pass_2" in dot
    assert "cluster_backward_pass_1" not in dot
    assert "bwd 2" in dot


def test_draw_backward_errors_without_log_backward() -> None:
    """draw_backward errors clearly before explicit backward capture."""
    trace = tl.trace(
        _LinearReluModel(),
        torch.randn(2, 3, requires_grad=True),
        capture=CaptureOptions(save_grads="all"),
    )

    with pytest.raises(ValueError, match="call log_backward\\(loss\\) first"):
        trace.draw(graph="backward")
