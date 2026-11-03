"""Replay feeds each parent argument in the layout the function saw at capture.

A saved op output is a compact copy, so the output of a slice like ``x[:, 0:5]`` of a
``(4, 30)`` input is stored with stride ``(5, 1)`` although the next op read it with
stride ``(30, 1)``. Replaying that op from the compact copy can round differently (a BLAS
matmul on some CPUs picks a different kernel for a strided operand), so a fork edit of
``scale(0.5)`` did not return exactly half the captured value. Capture now records the
argument's layout on its ``ParentRef`` and replay re-lays the resolved value to it.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._utils._tensor_layout import layout_of, relay_to_layout
from torchlens.intervention._replay import _resolve_arg_component
from torchlens.intervention._types import ParentRef
from torchlens.options import CaptureOptions


class _SlicedLinear(nn.Module):
    """A bias-free linear over a column slice of the input, as in a per-block encoder."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(5, 8, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.lin(x[:, 0:5]))


class _PlainLinear(nn.Module):
    """The same linear over a contiguous input."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(5, 8, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.lin(x))


def _capture(model: nn.Module, x: torch.Tensor) -> tl.Trace:
    torch.manual_seed(0)
    return tl.trace(model, x, capture=CaptureOptions(replayable=True))


def _linear_op(trace: tl.Trace) -> object:
    return next(op for op in trace.ops if op.func_name == "linear")


def _sliced_trace() -> tl.Trace:
    torch.manual_seed(0)
    model = _SlicedLinear()
    torch.manual_seed(1)
    return _capture(model, torch.rand(4, 30))


def test_capture_records_the_strided_parent_layout() -> None:
    """The linear's parent ref carries the slice's stride, not the compact copy's."""

    trace = _sliced_trace()
    ref = _linear_op(trace).call.args_template.args[0]
    assert isinstance(ref, ParentRef)
    assert ref._capture_layout == ((4, 5), (30, 1), 0)
    assert trace[ref.parent_label].out.stride() == (5, 1)


def test_capture_records_no_layout_for_a_contiguous_parent() -> None:
    """A row-major argument needs no re-lay, so nothing is recorded."""

    torch.manual_seed(0)
    trace = _capture(_PlainLinear(), torch.rand(4, 5))
    ref = _linear_op(trace).call.args_template.args[0]
    assert isinstance(ref, ParentRef)
    assert ref._capture_layout is None


def test_replay_resolves_the_parent_in_its_capture_layout() -> None:
    """The replay engine hands the function the slice's stride with the saved values."""

    trace = _sliced_trace()
    op = _linear_op(trace)
    ref = op.call.args_template.args[0]
    value = _resolve_arg_component(ref, op, trace, {}, strict=True)
    assert value.stride() == (30, 1)
    assert value.storage_offset() == 0
    assert torch.equal(value, trace[ref.parent_label].out)


def test_fork_scale_halves_the_strided_site_exactly() -> None:
    """``fork.do(scale(0.5))`` on the linear returns exactly half the captured output."""

    trace = _sliced_trace()
    label = _linear_op(trace).label
    original = trace[label].out.detach().clone()
    fork = trace.fork()
    fork.do(tl.label(label), tl.scale(0.5))
    assert torch.equal(fork[label].out, original * 0.5)
    assert torch.equal(trace[label].out, original)


def test_layout_of_skips_contiguous_empty_and_overlapping_tensors() -> None:
    """Only a non-overlapping, non-contiguous plain tensor yields a layout."""

    base = torch.arange(24.0).reshape(4, 6)
    assert layout_of(base) is None
    assert layout_of(base[:, :0]) is None
    assert layout_of(torch.ones(3, 1).expand(3, 4)) is None
    assert layout_of(base[:, 2:4]) == ((4, 2), (6, 1), 2)
    assert layout_of(base.t()) == ((6, 4), (1, 6), 0)
    assert layout_of([1, 2]) is None


def test_relay_keeps_values_and_takes_the_recorded_layout() -> None:
    """The re-laid tensor has the recorded stride and offset and the same values."""

    base = torch.arange(24.0).reshape(4, 6)
    layout = layout_of(base[:, 2:4])
    compact = base[:, 2:4].contiguous()
    relaid = relay_to_layout(compact, layout)
    assert relaid.stride() == (6, 1)
    assert relaid.storage_offset() == 2
    assert torch.equal(relaid, compact)


@pytest.mark.parametrize(
    "layout",
    [None, ((4, 2), (2, 1), 0), ((2, 4), (6, 1), 0)],
    ids=["no-layout", "same-stride", "other-shape"],
)
def test_relay_returns_the_value_itself_when_nothing_to_do(layout: object) -> None:
    """No layout, a matching stride or a shape mismatch leave the value untouched."""

    compact = torch.arange(8.0).reshape(4, 2)
    assert relay_to_layout(compact, layout) is compact


def test_relay_keeps_the_gradient_path() -> None:
    """A differentiable replay still reaches the frontier leaf through the re-lay."""

    leaf = torch.arange(8.0).reshape(4, 2).requires_grad_(True)
    relaid = relay_to_layout(leaf, ((4, 2), (6, 1), 2))
    assert relaid.stride() == (6, 1)
    (relaid * torch.arange(8.0).reshape(4, 2)).sum().backward()
    assert torch.equal(leaf.grad, torch.arange(8.0).reshape(4, 2))
