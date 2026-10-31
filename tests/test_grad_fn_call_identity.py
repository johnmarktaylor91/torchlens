"""grind-p3 T11: GradFnCall equality, hashing, and ordinal_index safety/cost.

The dataclass-generated ``__eq__`` compared raw field tuples, so two
like-labeled calls carrying saved multi-element gradient tensors raised an
untyped elementwise torch ``RuntimeError`` -- which also detonated inside
``ordinal_index``'s ``list(...).index(self)`` equality scan -- and the class
was unhashable. ``ordinal_index`` additionally rebuilt the accessor and ran
that scan per access (O(N^2) sweep, measured exponent 2.02); it now resolves
through a per-trace identity-position map.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.data_classes._trace_accessors import TraceGradFnCallAccessor
from torchlens.data_classes.grad_fn_call import GradFnCall


class _TraceStandIn:
    """Weakref-able Trace stand-in exposing a real GradFnCall accessor."""

    def __init__(self, calls: list[GradFnCall]) -> None:
        self._calls = calls
        self._backward_projection_revision = 0

    @property
    def grad_fn_calls(self) -> TraceGradFnCallAccessor:
        """Return the accessor over the stand-in's calls."""

        return TraceGradFnCallAccessor(
            OrderedDict((f"{call.label}:{index}", call) for index, call in enumerate(self._calls))
        )


def _twin_calls() -> tuple[GradFnCall, GradFnCall]:
    """Two like-labeled calls with distinct multi-element grad payloads."""

    left = GradFnCall(call_index=1, label="AddmmBackward0_1", grad_inputs=(torch.randn(4),))
    right = GradFnCall(call_index=1, label="AddmmBackward0_1", grad_inputs=(torch.randn(4),))
    return left, right


@pytest.mark.smoke
def test_eq_on_tensor_payloads_never_raises():
    """Value comparison of like-labeled calls is typed and safe."""

    left, right = _twin_calls()
    assert (left == right) is False  # distinct random payloads
    same = GradFnCall(call_index=1, label="AddmmBackward0_1", grad_inputs=left.grad_inputs)
    assert left == same
    assert left != object()
    assert left == left


def test_grad_fn_call_is_hashable_and_eq_consistent():
    """Equal calls hash equal; the class is usable in sets/dicts."""

    left, _right = _twin_calls()
    same = GradFnCall(call_index=1, label="AddmmBackward0_1", grad_inputs=left.grad_inputs)
    assert hash(left) == hash(same)
    assert len({left, same}) == 1


@pytest.mark.smoke
def test_ordinal_index_survives_like_labeled_tensor_payload_twins():
    """The position scan never runs elementwise tensor equality.

    Pre-fix, ``ordinal_index`` on the SECOND of two like-labeled calls with
    saved multi-element gradients crashed with the untyped torch
    ``RuntimeError`` from the equality scan; identity resolution returns the
    correct position.
    """

    left, right = _twin_calls()
    trace = _TraceStandIn([left, right])
    left.source_trace = trace
    right.source_trace = trace
    assert left.ordinal_index == 0
    assert right.ordinal_index == 1


def test_ordinal_index_on_live_backward_trace():
    """Full sweep over a real backward capture is correct and monotone."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    x = torch.randn(2, 4)
    trace = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(layers_to_save="all", save_grads="all"),
    )
    trace.log_backward(trace[trace.output_layers[0]].out.sum())
    calls = list(trace.grad_fn_calls.values())
    assert calls, "backward capture produced no GradFnCall records"
    assert [call.ordinal_index for call in calls] == list(range(len(calls)))
    trace.cleanup()


def test_ordinal_index_without_trace_is_minus_one():
    """A trace-less call keeps the historical -1 sentinel."""

    orphan = GradFnCall(call_index=1, label="orphan")
    assert orphan.ordinal_index == -1


def test_ordinal_index_sweep_is_linear_not_quadratic():
    """Per-access cost may not grow with N (identity map, not an eq scan).

    Counts accessor rebuilds instead of wall time (load-robust): a sweep over
    N calls must build the accessor O(1) times, not once per access.
    """

    calls = [
        GradFnCall(call_index=index, label=f"Fn_{index}", grad_inputs=(torch.randn(2),))
        for index in range(200)
    ]
    trace = _TraceStandIn(calls)
    for call in calls:
        call.source_trace = trace

    builds = 0
    original = _TraceStandIn.grad_fn_calls.fget

    def _counting(self: _TraceStandIn) -> TraceGradFnCallAccessor:
        nonlocal builds
        builds += 1
        return original(self)

    _TraceStandIn.grad_fn_calls = property(_counting)
    try:
        positions = [call.ordinal_index for call in calls]
    finally:
        _TraceStandIn.grad_fn_calls = property(original)
    assert positions == list(range(len(calls)))
    assert builds <= 2, f"ordinal_index rebuilt the accessor {builds} times for one sweep"
