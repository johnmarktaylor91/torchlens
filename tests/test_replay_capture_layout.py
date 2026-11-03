"""Replay feeds each parent argument in the layout the function saw at capture.

A saved op output is a compact copy, so the output of a slice like ``x[:, 0:5]`` of a
``(4, 30)`` input is stored with stride ``(5, 1)`` although the next op read it with
stride ``(30, 1)``. Replaying that op from the compact copy can round differently (a BLAS
matmul on some CPUs picks a different kernel for a strided operand), so a fork edit of
``scale(0.5)`` did not return exactly half the captured value. Capture now records the
argument's layout on its ``ParentRef`` and replay re-lays the resolved value to it.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from torchlens._utils._tensor_layout import layout_of
from torchlens.intervention._replay import _resolve_arg_component
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


def test_replay_resolves_the_parent_in_its_capture_layout() -> None:
    """The replay engine hands the function the slice's stride with the saved values."""

    trace = _sliced_trace()
    op = _linear_op(trace)
    ref = op.call.args_template.args[0]
    value = _resolve_arg_component(ref, op, trace, {}, strict=True)
    assert value.stride() == (30, 1)
    assert value.storage_offset() == 0
    assert torch.equal(value, trace[ref.parent_label].out)


def test_layout_of_skips_contiguous_empty_and_overlapping_tensors() -> None:
    """Only a non-overlapping, non-contiguous plain tensor yields a layout."""

    base = torch.arange(24.0).reshape(4, 6)
    assert layout_of(base) is None
    assert layout_of(base[:, :0]) is None
    assert layout_of(torch.ones(3, 1).expand(3, 4)) is None
    assert layout_of(base[:, 2:4]) == ((4, 2), (6, 1), 2)
    assert layout_of(base.t()) == ((6, 4), (1, 6), 0)
    assert layout_of([1, 2]) is None
