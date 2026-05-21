"""Tests for combined forward/backward visualization."""

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.visualization.node_spec import NodeSpec
from torchlens.visualization.rendering import (
    GRADIENT_ARROW_COLOR,
    _module_key_for_grad_fn,
    _param_module_for_accumulate_grad,
)


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


class _ScaleModule(nn.Module):
    """Submodule with one parameter."""

    def __init__(self) -> None:
        """Initialize the parameter."""
        super().__init__()
        self.scale = nn.Parameter(torch.ones(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        return x * self.scale


class _SingleParamModel(nn.Module):
    """Model with one module-owned parameter for AccumulateGrad attribution."""

    def __init__(self) -> None:
        """Initialize submodules."""
        super().__init__()
        self.scale = _ScaleModule()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        return self.scale(x).sum()


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

    trace = tl.trace(model, x, gradients_to_save="all")
    trace.log_backward(trace[trace.output_layers[0]].out.sum())
    return trace


@pytest.mark.smoke
def test_draw_combined_tinymlp_smoke(tmp_path: Path) -> None:
    """draw_combined returns DOT with forward and backward nodes."""
    trace = _log_backward_model(_LinearReluModel(), torch.randn(2, 3, requires_grad=True))

    dot = trace.draw_combined(
        vis_outpath=str(tmp_path / "combined"),
        vis_save_only=True,
        vis_fileformat="svg",
    )

    assert "combined forward/backward graph" in dot
    assert "linear_1_1" in dot
    assert "addmm_back" in dot
    assert (tmp_path / "combined.svg").stat().st_size > 0


@pytest.mark.smoke
def test_draw_combined_top_level_function(tmp_path: Path) -> None:
    """Top-level draw_combined renders a Trace."""
    trace = _log_backward_model(_LinearReluModel(), torch.randn(2, 3, requires_grad=True))

    with pytest.warns(DeprecationWarning):
        dot = tl.draw_combined(
            trace,
            vis_outpath=str(tmp_path / "top_level_combined"),
            vis_save_only=True,
            vis_fileformat="svg",
        )

    assert "relu_back" in dot


def test_draw_combined_module_clusters() -> None:
    """Paired backward nodes are placed in the forward op's module cluster."""
    trace = _log_backward_model(_LinearReluModel(), torch.randn(2, 3, requires_grad=True))
    addmm = next(
        grad_fn_handle
        for grad_fn_handle in trace.grad_fns
        if grad_fn_handle.grad_fn_type == "addmm"
    )

    assert _module_key_for_grad_fn(trace, addmm, "upstream") == "fc:1"


def test_draw_combined_intervening_cluster_outside(tmp_path: Path) -> None:
    """The outside mode leaves intervening grad_fns out of module clusters."""
    trace = _log_backward_model(_ViewModel(), torch.randn(2, 6, requires_grad=True))
    intervening = next(
        grad_fn_handle for grad_fn_handle in trace.grad_fns if not grad_fn_handle.has_op
    )

    dot = trace.draw_combined(
        vis_outpath=str(tmp_path / "outside"),
        vis_save_only=True,
        vis_fileformat="svg",
        intervening_cluster="outside",
    )

    assert _module_key_for_grad_fn(trace, intervening, "outside") is None
    assert "cluster___intervening__" not in dot


def test_draw_combined_intervening_cluster_own_cluster(tmp_path: Path) -> None:
    """The own mode creates the dedicated intervening cluster."""
    trace = _log_backward_model(_ViewModel(), torch.randn(2, 6, requires_grad=True))
    intervening = next(
        grad_fn_handle for grad_fn_handle in trace.grad_fns if not grad_fn_handle.has_op
    )

    dot = trace.draw_combined(
        vis_outpath=str(tmp_path / "own"),
        vis_save_only=True,
        vis_fileformat="svg",
        intervening_cluster="own",
    )

    assert _module_key_for_grad_fn(trace, intervening, "own") == "__intervening__"
    assert "cluster___intervening__" in dot


def test_draw_combined_intervening_cluster_inherits() -> None:
    """Upstream and downstream inheritance modes are deterministic."""
    trace = _log_backward_model(_ViewModel(), torch.randn(2, 6, requires_grad=True))
    intervening = next(
        grad_fn_handle for grad_fn_handle in trace.grad_fns if not grad_fn_handle.has_op
    )

    upstream = _module_key_for_grad_fn(trace, intervening, "upstream")
    downstream = _module_key_for_grad_fn(trace, intervening, "downstream")

    assert upstream is None or isinstance(upstream, str)
    assert downstream is None or isinstance(downstream, str)


def test_draw_combined_accumulategrad_attribution() -> None:
    """AccumulateGrad attribution uses the serialized label-keyed parameter map."""
    trace = _log_backward_model(_SingleParamModel(), torch.randn(2, 3, requires_grad=True))
    accumulate_grad = next(
        grad_fn_handle
        for grad_fn_handle in trace.grad_fns
        if grad_fn_handle.grad_fn_type == "accumulategrad"
    )

    assert _param_module_for_accumulate_grad(trace, accumulate_grad) == "scale:1"
    assert trace._grad_fn_param_refs[accumulate_grad.label] == "scale.scale"


def test_draw_combined_rolled_raises_notimplementederror() -> None:
    """Combined rolled rendering is explicitly deferred."""
    trace = _log_backward_model(_LinearReluModel(), torch.randn(2, 3, requires_grad=True))

    with pytest.raises(NotImplementedError):
        trace.draw_combined(vis_mode="rolled", vis_save_only=True)


def test_draw_combined_backward_node_spec_fn_callback(tmp_path: Path) -> None:
    """Combined rendering accepts a separate backward node callback."""
    trace = _log_backward_model(_LinearReluModel(), torch.randn(2, 3, requires_grad=True))

    def backward_node_spec_fn(grad_fn_handle: object, default_spec: NodeSpec) -> NodeSpec:
        """Tint one backward node."""
        if getattr(grad_fn_handle, "grad_fn_type", "") == "relu":
            default_spec.fillcolor = "#ABCDEF"
        return default_spec

    dot = trace.draw_combined(
        vis_outpath=str(tmp_path / "callback"),
        vis_save_only=True,
        vis_fileformat="svg",
        backward_node_spec_fn=backward_node_spec_fn,
    )

    assert "#ABCDEF" in dot


def test_draw_combined_correspondence_edge_constraint_false(tmp_path: Path) -> None:
    """Paired forward/backward edges are dashed and unconstrained."""
    trace = _log_backward_model(_LinearReluModel(), torch.randn(2, 3, requires_grad=True))

    dot = trace.draw_combined(
        vis_outpath=str(tmp_path / "correspondence"),
        vis_save_only=True,
        vis_fileformat="svg",
    )

    assert f'color="{GRADIENT_ARROW_COLOR}"' in dot
    assert "constraint=false" in dot
    assert "style=dashed" in dot
