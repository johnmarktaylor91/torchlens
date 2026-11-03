"""Runtime handle accessors for live Torch objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads="all"), save="all")
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

    for grad_fn in trace.autograd.grad_fns:
        handle = grad_fn.handle
        if handle is not None:
            return grad_fn
    raise AssertionError("Expected at least one live GradFn handle")


def test_portable_round_trip_runtime_handles_are_none(tmp_path: Path) -> None:
    """Portable ``.tlspec`` loads do not restore computed runtime handles."""

    import safetensors  # noqa: F401

    _model, trace = _make_trace()
    path = tmp_path / "runtime_handles.tlspec"

    trace.save(path, level="portable")
    loaded = tl.load(path)

    assert loaded.params["linear.weight"].handle is None
    assert loaded.buffers["scale"].handle is None
    assert loaded.modules["linear"].handle is None
    assert loaded.modules["self"].handle is None
    assert _first_grad_fn_with_handle(trace).handle is not None
    assert all(grad_fn.handle is None for grad_fn in loaded.autograd.grad_fns)
