"""Tests for combined forward/backward visualization."""

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._visualization._render_leaf import (
    _module_key_for_grad_fn,
)
from torchlens._visualization.node_spec import NodeSpec


class _LinearReluModel(nn.Module):
    """Small module model for combined visualization tests."""

    def __init__(self) -> None:
        """Initialize layers."""
        super().__init__()
        self.fc = nn.Linear(3, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        return torch.relu(self.fc(x)).sum()


class _ViewModel(nn.Module):
    """Model that introduces intervening view-related grad_fns."""

    def __init__(self) -> None:
        """Initialize layers."""
        super().__init__()
        self.fc = nn.Linear(6, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        y = self.fc(x)
        return y.view(2, 3, 2).sum()


def _log_backward_model(model: nn.Module, x: torch.Tensor) -> tl.Trace:
    """Return a Trace with backward metadata captured.

    Parameters
    ----------
    model:
        Model to trace.
    x:
        Input tensor.

    Returns
    -------
    tl.Trace
        Trace with backward metadata.
    """

    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads="all"))
    trace.log_backward(trace[trace.output_layers[0]].out.sum())
    return trace


def _log_two_backward_passes(model: nn.Module, x: torch.Tensor) -> tl.Trace:
    """Return a Trace with two backward passes captured.

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

    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads="all"))
    loss = trace[trace.output_layers[0]].out.sum()
    trace.log_backward(loss, retain_graph=True)
    trace.log_backward(loss)
    return trace


def test_draw_combined_intervening_cluster_outside(tmp_path: Path) -> None:
    """The outside mode leaves intervening grad_fns out of module clusters."""
    trace = _log_backward_model(_ViewModel(), torch.randn(2, 6, requires_grad=True))
    intervening = next(
        grad_fn_handle for grad_fn_handle in trace.autograd.grad_fns if not grad_fn_handle.has_op
    )

    dot = trace.draw(
        graph="combined",
        path=str(tmp_path / "outside"),
        file_format="svg",
        intervening_cluster="outside",
    ).dot

    assert _module_key_for_grad_fn(trace, intervening, "outside") is None
    assert "cluster___intervening__" not in dot


def test_draw_combined_intervening_cluster_own_cluster(tmp_path: Path) -> None:
    """The own mode creates the dedicated intervening cluster."""
    trace = _log_backward_model(_ViewModel(), torch.randn(2, 6, requires_grad=True))
    intervening = next(
        grad_fn_handle for grad_fn_handle in trace.autograd.grad_fns if not grad_fn_handle.has_op
    )

    dot = trace.draw(
        graph="combined",
        path=str(tmp_path / "own"),
        file_format="svg",
        intervening_cluster="own",
    ).dot

    assert _module_key_for_grad_fn(trace, intervening, "own") == "__intervening__"
    assert "cluster___intervening__" in dot


def test_draw_combined_intervening_cluster_inherits() -> None:
    """Upstream and downstream inheritance modes are deterministic."""
    trace = _log_backward_model(_ViewModel(), torch.randn(2, 6, requires_grad=True))
    intervening = next(
        grad_fn_handle for grad_fn_handle in trace.autograd.grad_fns if not grad_fn_handle.has_op
    )

    upstream = _module_key_for_grad_fn(trace, intervening, "upstream")
    downstream = _module_key_for_grad_fn(trace, intervening, "downstream")

    assert upstream is None or isinstance(upstream, str)
    assert downstream is None or isinstance(downstream, str)


def test_draw_combined_rolled_raises_notimplementederror() -> None:
    """Combined rolled rendering is explicitly deferred."""
    trace = _log_backward_model(_LinearReluModel(), torch.randn(2, 3, requires_grad=True))

    with pytest.raises(NotImplementedError):
        trace.draw(graph="combined", rolled=True)


def test_draw_combined_backward_node_spec_fn_callback(tmp_path: Path) -> None:
    """Combined rendering accepts a separate backward node callback."""
    trace = _log_backward_model(_LinearReluModel(), torch.randn(2, 3, requires_grad=True))

    def backward_node_spec_fn(grad_fn_handle: object, default_spec: NodeSpec) -> NodeSpec:
        """Tint one backward node."""
        if getattr(grad_fn_handle, "type", "") == "relu":
            default_spec.fillcolor = "#ABCDEF"
        return default_spec

    dot = trace.draw(
        graph="combined",
        path=str(tmp_path / "callback"),
        file_format="svg",
        backward_node_spec_fn=backward_node_spec_fn,
    ).dot

    assert "#ABCDEF" in dot


@pytest.mark.smoke
def test_draw_combined_bwd_filter_keeps_selected_grad_fns(tmp_path: Path) -> None:
    """Combined rendering accepts a one-based backward pass filter."""
    trace = _log_two_backward_passes(_LinearReluModel(), torch.randn(2, 3, requires_grad=True))

    dot = trace.draw(
        graph="combined",
        path=str(tmp_path / "combined_filtered"),
        file_format="svg",
        bwd=2,
    ).dot

    assert "shown: 2" in dot
    assert "bwd 2" in dot
    assert "addmm_back" in dot


def test_forward_rolled_grad_overlay_labels_partial_passes(tmp_path: Path) -> None:
    """Rolled forward gradient arrows label pass ranges when not present on all passes."""
    trace = tl.trace(
        _LinearReluModel(),
        torch.randn(2, 3, requires_grad=True),
        capture=tl.options.CaptureOptions(save_grads=tl.in_backward_pass(2)),
    )
    loss = trace[trace.output_layers[0]].out.sum()
    trace.log_backward(loss, retain_graph=True)
    trace.log_backward(loss)

    dot = trace.draw(
        path=str(tmp_path / "forward_partial_bwd"),
        file_format="svg",
        rolled=True,
    ).dot

    assert "bwd 2" in dot


def test_draw_combined_module_clusters() -> None:
    """Paired backward nodes are placed in the forward op's module cluster."""
    trace = _log_backward_model(_LinearReluModel(), torch.randn(2, 3, requires_grad=True))
    addmm = next(
        grad_fn_handle
        for grad_fn_handle in trace.autograd.grad_fns
        if grad_fn_handle.type == "addmm"
    )

    assert _module_key_for_grad_fn(trace, addmm, "upstream") == "fc:1"
