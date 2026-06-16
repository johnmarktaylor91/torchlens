"""Runtime handle accessors for live Torch objects."""

from __future__ import annotations

import gc
import weakref
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl


class _HandleModel(nn.Module):
    """Small module with params, buffers, submodules, and autograd nodes."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.register_buffer("scale", torch.ones(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a differentiable forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Scaled linear output.
        """

        return self.linear(x) * self.scale


def _make_trace() -> tuple[_HandleModel, tl.Trace]:
    """Return a live model and trace with backward metadata enabled.

    Returns
    -------
    tuple[_HandleModel, tl.Trace]
        Source model and captured trace.
    """

    torch.manual_seed(0)
    model = _HandleModel()
    x = torch.randn(4, 3, requires_grad=True)
    trace = tl.trace(model, x, layers_to_save="all", save_grads="all")
    trace[trace.output_layers[0]].out.sum().backward()
    return model, trace


def _first_grad_fn_with_handle(trace: tl.Trace) -> Any:
    """Return a GradFn whose live handle can be resolved.

    Parameters
    ----------
    trace:
        Trace containing backward metadata.

    Returns
    -------
    Any
        GradFn record with a retained live autograd object.
    """

    for grad_fn in trace.grad_fns:
        handle = grad_fn.handle
        if handle is not None:
            return grad_fn
    raise AssertionError("Expected at least one live GradFn handle")


def test_runtime_handles_return_live_objects_by_identity() -> None:
    """Live handles resolve to the source model/autograd objects by identity."""

    model, trace = _make_trace()
    param = trace.params["linear.weight"]
    buffer = trace.buffers["scale"]
    module = trace.modules["linear"]
    root_module = trace.modules["self"]
    grad_fn = _first_grad_fn_with_handle(trace)

    assert param.handle is model.get_parameter("linear.weight")
    assert buffer.handle is model.get_buffer("scale")
    assert module.handle is model.get_submodule("linear")
    assert root_module.handle is model
    assert id(grad_fn.handle) == grad_fn.grad_fn_object_id


def test_runtime_handles_return_none_after_source_release_without_raising() -> None:
    """Handles return ``None`` once source-model reachability is gone."""

    model, trace = _make_trace()
    param = trace.params["linear.weight"]
    buffer = trace.buffers["scale"]
    module = trace.modules["linear"]
    model_ref = weakref.ref(model)

    assert param.handle is not None
    assert buffer.handle is not None
    assert module.handle is not None

    param.release_param_ref()
    del model
    gc.collect()

    assert model_ref() is None
    assert param.handle is None
    assert buffer.handle is None
    assert module.handle is None


def test_grad_fn_handle_returns_none_when_trace_refs_are_unavailable() -> None:
    """GradFn handles return ``None`` when retained autograd refs are removed."""

    _model, trace = _make_trace()
    grad_fn = _first_grad_fn_with_handle(trace)
    trace._backward_gradfn_refs = []
    if grad_fn.op is not None:
        grad_fn.op.grad_fn_handle = None

    assert grad_fn.handle is None


def test_portable_round_trip_runtime_handles_are_none(tmp_path: Path) -> None:
    """Portable ``.tlspec`` loads do not restore computed runtime handles."""

    pytest.importorskip("safetensors")
    _model, trace = _make_trace()
    path = tmp_path / "runtime_handles.tlspec"

    tl.save(trace, path, level="portable")
    loaded = tl.load(path)

    assert loaded.params["linear.weight"].handle is None
    assert loaded.buffers["scale"].handle is None
    assert loaded.modules["linear"].handle is None
    assert loaded.modules["self"].handle is None
    assert _first_grad_fn_with_handle(trace).handle is not None
    assert all(grad_fn.handle is None for grad_fn in loaded.grad_fns)


def test_param_handle_does_not_cache_after_release() -> None:
    """Param.handle read-through does not repopulate ``_param_ref``."""

    model, trace = _make_trace()
    param = trace.params["linear.weight"]

    param.release_param_ref()
    assert param._param_ref is None
    assert param.handle is model.get_parameter("linear.weight")
    assert param._param_ref is None
    assert param.value is model.get_parameter("linear.weight")
    assert param._param_ref is model.get_parameter("linear.weight")
