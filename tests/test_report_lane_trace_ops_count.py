"""sumfam F1: ``len(trace.ops)`` equals ``trace._num_ops``; iterators agree.

Branch A of the sumfam memo (accepted with the defaults.2 batch, JMT
2026-10-12): the accessor's length agrees with the same-named count, and the
alias-inclusive row sequence keeps its own spelling, ``trace._layer_list``.
Label lookup still resolves every row (the documented asymmetry, pinned
below).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes.aten_op import OpRef
from torchlens.validation.invariants import MetadataInvariantError, check_metadata_invariants


def _assert_counts_agree(trace: Any) -> None:
    counted = [op for op in trace._layer_list if not (op.is_input or op.is_output or op.is_buffer)]
    assert len(trace.ops) == trace._num_ops == len(counted)
    assert list(trace.ops) == counted
    assert trace.ops.values() == counted
    assert trace.ops.keys() == [op.label for op in counted]
    assert [label for label, _ in trace.ops.items()] == trace.ops.keys()
    assert [trace.ops[index] for index in range(len(trace.ops))] == counted


# Kept by shrink-r4 (per-id subset rule): sole coverage of _data_classes/_trace_accessors.py.
def test_counts_agree_on_a_recurrent_model_with_buffers() -> None:
    class _Recurrent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cell = nn.Linear(3, 3)
            self.register_buffer("scale", torch.ones(3))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for _ in range(3):
                x = torch.tanh(self.cell(x)) * self.scale
            return x

    trace = tl.trace(_Recurrent(), torch.randn(1, 3))
    assert any(op.is_buffer for op in trace._layer_list)
    _assert_counts_agree(trace)


def _armed_two_op_trace() -> Any:
    class _TwoOp(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(x + 1).sum()

    return tl.trace(_TwoOp(), torch.ones(2), capture=tl.options.CaptureOptions(aten_ops=True))


# Kept by shrink-r4 (per-id subset rule): sole coverage of validation/_invariants_primitive_ops.py, campaign mutant kills.
def test_a_wrong_op_row_index_still_raises() -> None:
    trace = _armed_two_op_trace()
    row = next(row for row in trace._primitive_op_profile.primitive_ops if row.parent_op_refs)
    ref = row.parent_op_refs[0]
    wrong = next(
        index
        for index, op in enumerate(trace._layer_list)
        if index != ref.op_row_index and op.layer_label != ref.op_label
    )
    row.parent_op_refs = (
        OpRef(op_row_index=wrong, op_label=ref.op_label, func_call_id=ref.func_call_id),
    )

    with pytest.raises(MetadataInvariantError) as excinfo:
        check_metadata_invariants(trace)
    assert excinfo.value.check_name == "primitive_op_invariants"
