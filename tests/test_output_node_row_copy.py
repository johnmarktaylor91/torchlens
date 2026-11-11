"""The output node reads its parent op's row (``_op_output_copy._copy_for_output``).

Every field of the row copy must equal the former eager whole-schema copy
(``_copy_with_shallow_fields`` with the output-node field sets), keep the same
sharing (a container is shared with the parent exactly when the eager copy
shared it), and stay equal to an eager copy after the model's parameters,
buffers and inputs are mutated in place after capture.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._constants import LAYER_PASS_LOG_FIELD_ORDER
from torchlens._data_classes import _op_copy, _op_output_copy, _op_save, op as op_module
from torchlens._ir.refs import DeviceRef, DtypeRef
from torchlens._utils._record_group_table import op_attr

#: Fields that differ between two separate traces of one model by design.
_PER_TRACE_FIELDS = frozenset({"source_trace"})
_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")


class _TwoOutputs(nn.Module):
    """Two outputs reading a parameter, a buffer and the input."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)
        self.norm = nn.BatchNorm1d(3)
        self.register_buffer("offset", torch.ones(3))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.norm(self.fc(x)) + self.offset
        if hidden.sum() > -1e9:
            hidden = hidden * 1.5
        return torch.relu(hidden), {"twice": hidden * 2}["twice"]


def _eager_copy(op: Any, *, _store: Any = None) -> Any:
    """The former eager output-node copy."""

    return _op_copy._copy_with_shallow_fields(
        op,
        _op_copy._OUTPUT_NODE_REPLACED_FIELDS,
        _store=_store,
        unread=_op_copy._OUTPUT_COPY_UNREAD_DERIVED_FIELDS,
    )


def _frozen(value: Any) -> Any:
    """Return a comparison key that later in-place mutation cannot change."""

    if isinstance(value, torch.Tensor):
        return ("tensor", value.detach().clone())
    if isinstance(value, dict):
        return ("dict", tuple((repr(key), _frozen(item)) for key, item in value.items()))
    if isinstance(value, list | tuple):
        return (type(value).__name__, tuple(_frozen(item) for item in value))
    if isinstance(value, set | frozenset):
        return ("set", tuple(sorted(repr(item) for item in value)))
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return _ADDRESS.sub("0x", repr(value))


def _same(left: Any, right: Any) -> bool:
    """Compare two frozen keys, tensors by exact value."""

    if isinstance(left, tuple) and left and left[0] == "tensor":
        return (
            isinstance(right, tuple)
            and right[0] == "tensor"
            and right[1].dtype == left[1].dtype
            and torch.equal(left[1], right[1])
        )
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(map(_same, left, right))
    return bool(left == right)


def _read(op: Any, field: str) -> Any:
    """Read one persisted field the way the copy reads it."""

    return getattr(op, op_attr(field), None)


def _output_fields(trace: Any) -> dict[tuple[str, str], Any]:
    """Freeze every field of every output op."""

    return {
        (str(op.label), field): _frozen(_read(op, field))
        for op in trace.ops
        if op.is_output
        for field in LAYER_PASS_LOG_FIELD_ORDER
        if field not in _PER_TRACE_FIELDS
    }


def _trace(model: nn.Module, x: torch.Tensor) -> Any:
    """Trace ``model`` with a fixed seed."""

    return tl.trace(model, x, capture=tl.options.CaptureOptions(seed=5))


def test_intercepted_read_fields_match_the_op_facade() -> None:
    """The row copy routes exactly the reads ``Op.__getattribute__`` intercepts through getattr."""

    assert _op_output_copy._INTERCEPTED_READ_FIELDS == op_module._INTERCEPTED_READ_FIELDS


def test_row_copy_equals_the_eager_copy_for_every_op() -> None:
    """Every field equals the eager copy's, and sharing with the parent is unchanged."""

    torch.manual_seed(0)
    model = _TwoOutputs().eval()
    trace = _trace(model, torch.randn(2, 4))
    checked = 0
    for op in trace.ops:
        row_copy = _op_output_copy._copy_for_output(op)
        eager_copy = _eager_copy(op)
        for field in LAYER_PASS_LOG_FIELD_ORDER:
            parent_value = _read(op, field)
            row_value = _read(row_copy, field)
            eager_value = _read(eager_copy, field)
            assert _same(_frozen(row_value), _frozen(eager_value)), (op.label, field)
            if isinstance(row_value, DtypeRef | DeviceRef):
                continue
            if isinstance(parent_value, list | dict | set):
                assert (row_value is parent_value) == (eager_value is parent_value), (
                    op.label,
                    field,
                )
            checked += 1
    assert checked > 0


def test_row_copied_output_nodes_equal_an_eager_copy_after_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Output nodes read after in-place mutation equal eager copies read at capture."""

    torch.manual_seed(0)
    model = _TwoOutputs().eval()
    x = torch.randn(2, 4)
    model_state = {key: value.clone() for key, value in model.state_dict().items()}
    x_start = x.clone()

    with monkeypatch.context() as patch:
        patch.setattr(_op_save, "_copy_for_output", _eager_copy)
        eager = _trace(model, x)
    eager_at_capture = _output_fields(eager)

    model.load_state_dict(model_state)
    x.copy_(x_start)
    row_copied = _trace(model, x)
    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.5)
        for buffer in model.buffers():
            if buffer.is_floating_point():
                buffer.mul_(-3.0)
            else:
                buffer.add_(2)
        x.add_(7.0)
    after_mutation = _output_fields(row_copied)

    assert len(eager.output_layers) == 2
    assert eager_at_capture.keys() == after_mutation.keys()
    differing = [
        key for key, value in eager_at_capture.items() if not _same(value, after_mutation[key])
    ]
    assert differing == []
