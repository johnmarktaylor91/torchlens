"""Strict typed accessor behavior tests."""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from torchlens._data_classes.buffer import Buffer
from torchlens._data_classes.grad_fn import GradFn
from torchlens._data_classes.layer import Layer
from torchlens._data_classes.module import Module
from torchlens._data_classes.op import Op
from torchlens._data_classes.param import Param
from torchlens._errors import AmbiguousOpLookupError
from torchlens._selection.errors import AmbiguousSiteError


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
        capture=tl.options.CaptureOptions(layers_to_save="all", save_grads="all"),
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


def test_trace_getitem_never_returns_a_layer() -> None:
    """``trace[...]`` returns Ops; Layers come only from ``trace.layers[...]`` (walk access RC-260)."""

    trace = _strict_trace()

    assert isinstance(trace["mul_1_2"], Op)
    assert trace["mul_1_2"] is trace.ops["mul_1_2"]
    assert isinstance(trace["linear_1_1:1"], Op)
    assert isinstance(trace["MUL_1_2"], Op)
    for label in trace.layer_labels:
        if trace.layers[label].num_passes == 1:
            assert not isinstance(trace[label], Layer)
    with pytest.raises(AmbiguousSiteError) as info:
        trace["linear_1_1"]
    assert info.value.fields["code"] == "multipass_bare_label_ambiguous"
    message = str(info.value)
    assert "trace.layers['linear_1_1']" in message
    assert "'linear_1_1:last'" in message and "'linear_1_1:all'" in message
    assert "'linear_1_1:1'" in message and "'linear_1_1:2'" in message
    assert isinstance(trace.layers["linear_1_1"], Layer)


def test_last_pass_token_names_the_final_pass() -> None:
    """``label:last`` resolves to the final pass Op on every door; passes stay 1-based."""

    trace = _strict_trace()
    last = trace.ops["linear_1_1:2"]

    assert trace["linear_1_1:last"] is last
    assert trace.ops["linear_1_1:last"] is last
    assert trace[tl.label("linear_1_1:last")] is last
    assert trace["mul_1_2:last"] is trace.ops["mul_1_2"]
    assert trace.layers["linear_1_1:last"] is trace.layers["linear_1_1"]
    assert "linear_1_1:last" in trace.ops
    assert trace[tl.label("linear_1_1:last")].pass_number == 2
    with pytest.raises(KeyError):
        trace["linear_1_1:-1"]
    with pytest.raises(KeyError):
        trace.layers["linear_1_1:0"]
    with pytest.raises(KeyError):
        trace["linear_1_1:3"]


def test_all_pass_token_selects_every_pass() -> None:
    """``label:all`` is the set spelling: a Selection of every pass of the layer."""

    trace = _strict_trace()

    every = trace["linear_1_1:all"]
    assert isinstance(every, tl.Selection)
    assert every.labels == ("linear_1_1:1", "linear_1_1:2")
    assert trace[tl.label("linear_1_1:all")].labels == every.labels
    assert every.labels == trace.layers["linear_1_1"].__selection__().labels
    single = trace["mul_1_2:all"]
    assert isinstance(single, tl.Selection)
    assert single.labels == ("mul_1_2",)
    assert trace.layers["linear_1_1:all"] is trace.layers["linear_1_1"]
    assert "linear_1_1:all" not in trace.ops
    with pytest.raises(KeyError, match="resolve_all"):
        trace.ops["linear_1_1:all"]
    with pytest.raises(KeyError):
        trace["no_such_layer:all"]


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
        "num_calls": 0,
    }


def test_legacy_param_restore_refuses_below_floor() -> None:
    """Pre-floor Param states refuse typed instead of resurrecting.

    Flipped from the pre-2.33-floor resurrection pin (which asserted the
    DeprecationWarning restore path and co_parent_params isolation): species-1
    deletion 3 makes unversioned legacy state refuse with the floor named.
    The co-parent isolation behavior it used to pin is covered for supported
    states by the strict-binder suites.
    """

    from torchlens._io import ArtifactTooOldError

    first = Param.__new__(Param)

    with pytest.raises(ArtifactTooOldError, match="saved by torchlens 2.x") as excinfo:
        first.__setstate__(_legacy_param_state("left.weight"))
    assert excinfo.value.fields["code"] == "artifact_version_below_floor"


def test_ambiguous_op_lookup_error_remains_value_error_compatible() -> None:
    """Existing callers that catch ValueError still catch the new error."""

    trace = _strict_trace()

    try:
        trace.ops["linear_1_1"]
    except ValueError as exc:
        assert isinstance(exc, AmbiguousOpLookupError)
        assert exc.fields["code"] == "ambiguous_op_lookup"
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
    grad_fn = trace.autograd.grad_fns[0]

    assert isinstance(trace.autograd.grad_fns[f"{grad_fn.label}:1"], GradFn)
    assert trace.autograd.grad_fns[f"{grad_fn.label}:1"] is grad_fn


class _IdleBranchModel(torch.nn.Module):
    """Model with a registered submodule the forward never calls and one inf-making op."""

    def __init__(self) -> None:
        """Register a used and an unused Linear."""

        super().__init__()
        self.used = torch.nn.Linear(3, 3)
        self.idle = torch.nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Divide by zero once so one op output is non-finite."""

        y = self.used(x)
        return y / torch.zeros_like(y)


def test_every_collection_serves_its_labels() -> None:
    """``.labels`` on layers, ops and modules lists the lookup labels in order (walk names option b)."""

    trace = _strict_trace()

    assert trace.layers.labels == tuple(trace.layer_labels)
    assert trace.ops.labels == tuple(op.label for op in trace.ops)
    assert "linear_1_1:2" in trace.ops.labels
    assert trace.modules.labels == tuple(trace.modules.keys())
    assert "lin" in trace.modules.labels
    assert trace.params.labels == tuple(param.address for param in trace.params)
    for label in trace.layers.compute.labels:
        assert trace.layers.compute[label] is trace.layers[label]


def test_ops_compute_filter_matches_the_trace_level_view() -> None:
    """``trace.ops.compute`` is the Op spelling of ``trace.layers.compute``."""

    trace = _strict_trace()

    compute = trace.ops.compute
    assert list(compute) == list(trace.compute_ops)
    assert compute.labels == trace.compute_ops.labels
    assert not any(op.is_input or op.is_output or op.is_buffer for op in compute)
    assert {op.layer_label for op in compute} == set(trace.layers.compute.labels)
    assert list(trace.input_ops.compute) == []


def test_ops_nonfinite_and_modules_uncalled_filters() -> None:
    """``trace.ops.nonfinite`` and ``trace.modules.uncalled`` mirror the Trace-level answers."""

    torch.manual_seed(0)
    trace = tl.trace(
        _IdleBranchModel(),
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(layers_to_save="all"),
    )

    nonfinite = trace.ops.nonfinite
    assert nonfinite.labels == tuple(trace.nonfinite_ops)
    assert len(nonfinite) >= 1
    assert all(isinstance(op, Op) for op in nonfinite)
    assert all(not torch.isfinite(op.out).all() for op in nonfinite)
    assert trace.ops.compute.nonfinite.labels == nonfinite.labels

    assert list(trace.modules.uncalled) == list(trace.uncalled_modules)
    assert "idle" in trace.modules.uncalled
    assert "idle" not in trace.modules.labels
