"""Strict typed accessor behavior tests."""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from torchlens._errors import AmbiguousOpLookupError
from torchlens.data_classes.buffer import Buffer
from torchlens.data_classes.grad_fn import GradFn
from torchlens.data_classes.layer import Layer
from torchlens.data_classes.module import Module
from torchlens.data_classes.op import Op
from torchlens.data_classes.param import Param


class StrictAccessorModel(torch.nn.Module):
    """Model with a repeated parameterized layer plus param and buffer surfaces."""

    def __init__(self) -> None:
        """Initialize the test module."""

        super().__init__()
        self.lin = torch.nn.Linear(3, 3)
        self.register_buffer("scale", torch.ones(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a repeated module call."""

        y = self.lin(x)
        z = self.lin(y)
        return z * self.scale


def _strict_trace() -> tl.Trace:
    """Return a trace with a multi-pass Linear layer."""

    torch.manual_seed(0)
    return tl.trace(
        StrictAccessorModel(),
        torch.randn(2, 3, requires_grad=True),
        layers_to_save="all",
        save_grads="all",
    )


def test_layers_accessor_always_returns_layer_for_pass_label() -> None:
    """Pass-qualified layer labels still resolve to the aggregate Layer."""

    trace = _strict_trace()

    assert isinstance(trace.layers["linear_1_1:1"], Layer)
    assert trace.layers["linear_1_1:1"] is trace.layers["linear_1_1"]


def test_ops_accessor_returns_op_for_pass_label() -> None:
    """Pass-qualified op labels resolve to Op records."""

    trace = _strict_trace()

    assert isinstance(trace.ops["linear_1_1:1"], Op)
    assert trace.ops["linear_1_1:1"].label == "linear_1_1:1"


def test_ops_accessor_bare_multi_pass_label_is_ambiguous() -> None:
    """A bare multi-pass Layer label is rejected by trace.ops."""

    trace = _strict_trace()

    with pytest.raises(AmbiguousOpLookupError):
        trace.ops["linear_1_1"]


def test_ambiguous_op_lookup_error_remains_value_error_compatible() -> None:
    """Existing callers that catch ValueError still catch the new error."""

    trace = _strict_trace()

    try:
        trace.ops["linear_1_1"]
    except ValueError as exc:
        assert isinstance(exc, AmbiguousOpLookupError)
    else:  # pragma: no cover - this branch would be a regression.
        raise AssertionError("Expected ValueError-compatible ambiguous lookup")


def test_modules_params_and_buffers_accessors_return_strict_types() -> None:
    """Aggregate accessors do not return pass-call objects for pass notation."""

    trace = _strict_trace()

    assert isinstance(trace.modules["lin:1"], Module)
    assert isinstance(trace.params["lin.weight:1"], Param)
    assert isinstance(trace.buffers["scale:1"], Buffer)


def test_grad_fns_accessor_returns_grad_fn_for_pass_label() -> None:
    """GradFn pass-qualified lookup returns a GradFn aggregate."""

    trace = _strict_trace()
    trace[trace.output_layers[0]].out.sum().backward()
    grad_fn = trace.grad_fns[0]

    assert isinstance(trace.grad_fns[f"{grad_fn.label}:1"], GradFn)
    assert trace.grad_fns[f"{grad_fn.label}:1"] is grad_fn
