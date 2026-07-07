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


class DuplicateShortNameParamModel(torch.nn.Module):
    """Model with two parameters sharing the short name ``weight``."""

    def __init__(self) -> None:
        """Initialize duplicate short-name modules."""

        super().__init__()
        self.left = torch.nn.Linear(3, 3, bias=False)
        self.right = torch.nn.Linear(3, 3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run both linear layers."""

        return self.left(x) + self.right(x)


class DuplicateShortNameBufferModel(torch.nn.Module):
    """Model with two buffers sharing the short name ``running_mean``."""

    def __init__(self) -> None:
        """Initialize duplicate short-name buffer modules."""

        super().__init__()
        self.left = torch.nn.BatchNorm1d(3)
        self.right = torch.nn.BatchNorm1d(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Read both duplicate buffers."""

        return x + self.left.running_mean + self.right.running_mean


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


def test_trace_getitem_layer_label_always_returns_layer() -> None:
    """Convenience layer-label lookup is deterministic across pass counts."""

    trace = _strict_trace()

    assert isinstance(trace["mul_1_2"], Layer)
    assert trace["mul_1_2"] is trace.layers["mul_1_2"]
    assert isinstance(trace["linear_1_1"], Layer)
    assert trace["linear_1_1"] is trace.layers["linear_1_1"]
    assert isinstance(trace["linear_1_1:1"], Op)


def test_ops_accessor_bare_multi_pass_label_is_ambiguous() -> None:
    """A bare multi-pass Layer label is rejected by trace.ops."""

    trace = _strict_trace()

    with pytest.raises(AmbiguousOpLookupError):
        trace.ops["linear_1_1"]


def test_layer_ops_accessor_bare_multi_pass_label_is_ambiguous() -> None:
    """A bare multi-pass Layer label is rejected by the scoped layer.ops accessor."""

    trace = _strict_trace()
    layer = trace.layers["linear_1_1"]

    with pytest.raises(AmbiguousOpLookupError):
        layer.ops["linear_1_1"]


def test_layer_ops_contains_is_false_for_ambiguous_label() -> None:
    """Scoped Op membership returns bool when lookup is ambiguous."""

    trace = _strict_trace()
    layer = trace.layers["linear_1_1"]

    assert "linear_1_1" not in layer.ops


def test_param_accessor_contains_is_false_for_ambiguous_short_name() -> None:
    """Param membership is false when indexing would raise ambiguity."""

    trace = tl.trace(DuplicateShortNameParamModel(), torch.randn(2, 3))

    assert "weight" not in trace.params
    with pytest.raises(AmbiguousOpLookupError):
        trace.params["weight"]


def test_buffer_accessor_contains_is_false_for_ambiguous_short_name() -> None:
    """Buffer membership returns bool when short-name lookup is ambiguous."""

    trace = tl.trace(DuplicateShortNameBufferModel(), torch.randn(2, 3))

    assert "running_mean" not in trace.buffers
    with pytest.raises(AmbiguousOpLookupError):
        trace.buffers["running_mean"]


def _legacy_param_state(address: str) -> dict[str, object]:
    """Return a minimal legacy Param state without ``co_parent_params``."""

    return {
        "name": "weight",
        "address": address,
        "all_addresses": [address],
        "shape": (),
        "num_params": 0,
        "num_params_trainable": 0,
        "num_params_frozen": 0,
        "is_trainable": False,
        "param_memory": 0,
        "_grad_memory": 0,
        "dtype": None,
        "device": None,
        "used_by_ops": [],
        "used_by_layers": [],
        "num_uses_by_ops": 0,
        "num_uses_by_layers": 0,
        "num_calls": 0,
    }


def test_legacy_param_restore_gets_independent_co_parent_list() -> None:
    """Legacy Param states do not share a class-level co_parent_params list."""

    first = Param.__new__(Param)
    second = Param.__new__(Param)

    with pytest.warns(DeprecationWarning):
        first.__setstate__(_legacy_param_state("left.weight"))
    with pytest.warns(DeprecationWarning):
        second.__setstate__(_legacy_param_state("right.weight"))
    first.co_parent_params.append("other.weight")

    assert first.co_parent_params == ["other.weight"]
    assert second.co_parent_params == []


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
