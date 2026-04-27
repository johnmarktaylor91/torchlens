"""Smoke tests for backward grad_fn visualization."""

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.visualization.rendering import GRADIENT_ARROW_COLOR


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
        """Return doubled upstream gradient."""
        return grad * 2


class _CustomModel(nn.Module):
    """Model using a custom autograd function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        return _DoubleFn.apply(x).sum()


def _log_backward_model(model: nn.Module, x: torch.Tensor) -> tl.ModelLog:
    """Return a ModelLog with backward metadata captured.

    Parameters
    ----------
    model:
        Model to log.
    x:
        Input tensor.

    Returns
    -------
    tl.ModelLog
        Model log after backward capture.
    """

    model_log = tl.log_forward_pass(model, x, gradients_to_save="all")
    model_log.log_backward(model_log[model_log.output_layers[0]].activation.sum())
    return model_log


@pytest.mark.smoke
def test_show_backward_graph_renders(tmp_path: Path) -> None:
    """show_backward_graph returns DOT source and writes a non-empty output file."""
    model_log = _log_backward_model(_LinearReluModel(), torch.randn(2, 3, requires_grad=True))
    outpath = tmp_path / "backward_graph"

    dot = model_log.show_backward_graph(
        vis_outpath=str(outpath),
        vis_save_only=True,
        vis_fileformat="svg",
    )

    assert "digraph" in dot
    assert "backward graph" in dot
    assert (tmp_path / "backward_graph.svg").stat().st_size > 0


@pytest.mark.smoke
def test_backward_graph_includes_grad_fn_nodes(tmp_path: Path) -> None:
    """Backward DOT contains expected grad_fn labels."""
    model_log = _log_backward_model(_LinearReluModel(), torch.randn(2, 3, requires_grad=True))

    dot = model_log.show_backward_graph(
        vis_outpath=str(tmp_path / "grad_fns"),
        vis_save_only=True,
        vis_fileformat="svg",
    )

    assert "addmm_back" in dot
    assert "relu_back" in dot
    assert GRADIENT_ARROW_COLOR in dot


@pytest.mark.smoke
def test_backward_graph_intervening_visual_distinction(tmp_path: Path) -> None:
    """Intervening grad_fns use the documented ``[i]`` label prefix."""
    model_log = _log_backward_model(_ViewModel(), torch.randn(2, 6, requires_grad=True))

    dot = model_log.show_backward_graph(
        vis_outpath=str(tmp_path / "intervening"),
        vis_save_only=True,
        vis_fileformat="svg",
    )

    assert "[i] " in dot


@pytest.mark.smoke
def test_backward_graph_custom_grad_fn_distinction(tmp_path: Path) -> None:
    """Custom autograd grad_fns use the documented ``[custom]`` suffix."""
    model_log = _log_backward_model(_CustomModel(), torch.randn(2, 3, requires_grad=True))

    dot = model_log.show_backward_graph(
        vis_outpath=str(tmp_path / "custom"),
        vis_save_only=True,
        vis_fileformat="svg",
    )

    assert "doublefn_back" in dot
    assert "[custom]" in dot


@pytest.mark.smoke
def test_backward_graph_cross_references_forward_layers(tmp_path: Path) -> None:
    """Backward node labels include corresponding forward layer labels."""
    model_log = _log_backward_model(_LinearReluModel(), torch.randn(2, 3, requires_grad=True))

    dot = model_log.show_backward_graph(
        vis_outpath=str(tmp_path / "cross_ref"),
        vis_save_only=True,
        vis_fileformat="svg",
    )

    assert "@linear_1_1" in dot
    assert "@relu_1_2" in dot


@pytest.mark.smoke
def test_show_backward_graph_top_level_function(tmp_path: Path) -> None:
    """The top-level ``tl.show_backward_graph`` helper renders a ModelLog."""
    model_log = _log_backward_model(_LinearReluModel(), torch.randn(2, 3, requires_grad=True))

    dot = tl.show_backward_graph(
        model_log,
        vis_outpath=str(tmp_path / "top_level"),
        vis_save_only=True,
        vis_fileformat="svg",
    )

    assert "addmm_back" in dot


@pytest.mark.smoke
def test_show_backward_graph_errors_without_log_backward() -> None:
    """show_backward_graph errors clearly before explicit backward capture."""
    model_log = tl.log_forward_pass(
        _LinearReluModel(),
        torch.randn(2, 3, requires_grad=True),
        gradients_to_save="all",
    )

    with pytest.raises(ValueError, match="call log_backward\\(loss\\) first"):
        model_log.show_backward_graph(vis_save_only=True)


@pytest.mark.smoke
def test_forward_graph_unchanged(tmp_path: Path) -> None:
    """Top-level forward graph output remains stable around backward rendering."""
    model = _LinearReluModel()
    x = torch.randn(2, 3, requires_grad=True)
    before = tmp_path / "forward_before"
    after = tmp_path / "forward_after"

    tl.show_model_graph(
        model,
        x,
        vis_outpath=str(before),
        vis_save_only=True,
        vis_fileformat="dot",
    )
    model_log = _log_backward_model(model, x)
    model_log.show_backward_graph(
        vis_outpath=str(tmp_path / "backward"),
        vis_save_only=True,
        vis_fileformat="svg",
    )
    tl.show_model_graph(
        model,
        x,
        vis_outpath=str(after),
        vis_save_only=True,
        vis_fileformat="dot",
    )

    assert (tmp_path / "forward_before.dot").read_text() == (
        tmp_path / "forward_after.dot"
    ).read_text()
