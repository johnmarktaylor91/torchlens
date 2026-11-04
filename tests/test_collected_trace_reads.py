"""Op and Layer reads on a view whose Trace may be gone: a value or a typed refusal, never None.

A Trace with no outside references is freed by refcount, so the Trace behind a
chained temporary (``tl.trace(model, x).output_ops[0].out``) can be gone before
the read runs. Whether a view keeps its Trace alive is the lifetime contract's
business; these tests pin what holds either way: a read answers a value, and a
read that needs a Trace that was collected refuses with ``RecordBindingError``
code ``trace_reference_collected`` and a remedy. A silent ``None`` (read as
"not saved" or "no such relation") or an untyped attribute miss is a wrong
answer.
"""

from __future__ import annotations

import gc
import weakref
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import RecordBindingError


class _ConvNet(nn.Module):
    """Conv, relu and linear: ops with parents, an atomic module and a receptive field."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3)
        self.fc = nn.Linear(2 * 4 * 4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.relu(self.conv(x)).flatten(1))


_OP_READS = (
    "out",
    "bytes_read",
    "bytes_written",
    "atomic_module",
    "input_ops",
    "layer",
    "containers",
    "input_shapes",
    "receptive_field",
)
_LAYER_READS = ("out", "parents", "children", "receptive_field")


def _model_and_input() -> tuple[nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    return _ConvNet().eval(), torch.randn(1, 1, 6, 6)


def _assert_value_or_typed_refusal(view: Any, name: str, trace_ref: weakref.ref) -> None:
    """Read ``view.<name>``: a value (never None), or the typed refusal once the Trace is gone.

    A read the view can still answer from its own record (``bytes_written`` from
    the Op's shape and dtype) may answer after the Trace is gone; one that needs
    the Trace refuses. Neither may answer ``None``.
    """

    try:
        value = getattr(view, name)
    except RecordBindingError as exc:
        assert trace_ref() is None, f"{name} refused while its Trace is alive"
        assert exc.fields["code"] == "trace_reference_collected", name
        assert "variable" in exc.fields["remedy"], name
    else:
        assert value is not None, f"{name} read None"


def _held_view(kind: str) -> tuple[Any, weakref.ref]:
    """A view held past ``del trace`` and a collection, with a weakref to its Trace."""

    model, x = _model_and_input()
    trace = tl.trace(model, x)
    view = trace["conv2d_1_1"] if kind == "op" else trace.layers["conv2d_1_1"]
    trace_ref = weakref.ref(trace)
    del trace
    gc.collect()
    return view, trace_ref


def test_chained_output_op_out_is_the_output_or_refuses_typed() -> None:
    """``tl.trace(m, x).output_ops[0].out`` is the model output or a typed refusal."""

    model, x = _model_and_input()
    try:
        out = tl.trace(model, x).output_ops[0].out
    except RecordBindingError as exc:
        assert exc.fields["code"] == "trace_reference_collected"
        assert "variable" in exc.fields["remedy"]  # names the fix: keep the Trace in a variable
    else:
        assert out is not None, "a chained temporary read None from Op.out"
        assert torch.equal(out, model(x))


def test_chained_receptive_field_is_a_view_or_refuses_typed() -> None:
    """``tl.trace(m, x)[label].receptive_field`` is a view or a typed refusal."""

    model, x = _model_and_input()
    try:
        view = tl.trace(model, x)["conv2d_1_1"].receptive_field
    except RecordBindingError as exc:
        assert exc.fields["code"] == "trace_reference_collected"
        assert "variable" in exc.fields["remedy"]  # names the fix: keep the Trace in a variable
    else:
        assert view.at((0, 0)) is not None


@pytest.mark.parametrize("name", _OP_READS)
def test_held_op_read_is_value_or_typed_refusal(name: str) -> None:
    """Each Op read answers its value or refuses typed once the Trace variable is gone."""

    op, trace_ref = _held_view("op")
    _assert_value_or_typed_refusal(op, name, trace_ref)


@pytest.mark.parametrize("name", _LAYER_READS)
def test_held_layer_read_is_value_or_typed_refusal(name: str) -> None:
    """Each Layer read answers its value or refuses typed once the Trace variable is gone."""

    layer, trace_ref = _held_view("layer")
    _assert_value_or_typed_refusal(layer, name, trace_ref)


def test_live_trace_reads_are_values() -> None:
    """The same reads on a live Trace answer values (the refusal is only for a dead Trace)."""

    model, x = _model_and_input()
    trace = tl.trace(model, x)
    op = trace["conv2d_1_1"]
    for name in _OP_READS:
        assert getattr(op, name) is not None, name
    layer = trace.layers["conv2d_1_1"]
    for name in _LAYER_READS:
        assert getattr(layer, name) is not None, name


def test_unknown_member_on_a_collected_op_stays_an_attribute_error() -> None:
    """A misspelled name is still an ``AttributeError``, never the collected-Trace refusal."""

    op, _trace_ref = _held_view("op")
    with pytest.raises(AttributeError):
        _ = op.no_such_member_anywhere
    assert not hasattr(op, "no_such_member_anywhere")
