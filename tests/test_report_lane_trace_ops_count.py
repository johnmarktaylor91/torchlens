"""sumfam F1: ``len(trace.ops)`` equals ``trace.num_ops``; iterators agree.

Branch A of the sumfam memo (accepted with the defaults.2 batch, JMT
2026-10-12): the accessor's length agrees with the same-named count, and the
alias-inclusive row sequence keeps its own spelling, ``trace.layer_list``.
Label lookup still resolves every row (the documented asymmetry, pinned
below).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch._aten_capture import _activate_aten_recording_for_tests
from torchlens.data_classes.aten_op import OpRef
from torchlens.validation.invariants import MetadataInvariantError, check_metadata_invariants


def _sequential_trace() -> Any:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    return tl.trace(model, torch.randn(2, 4))


def _assert_counts_agree(trace: Any) -> None:
    counted = [op for op in trace.layer_list if not (op.is_input or op.is_output or op.is_buffer)]
    assert len(trace.ops) == trace.num_ops == len(counted)
    assert list(trace.ops) == counted
    assert trace.ops.values() == counted
    assert trace.ops.keys() == [op.label for op in counted]
    assert [label for label, _ in trace.ops.items()] == trace.ops.keys()
    assert [trace.ops[index] for index in range(len(trace.ops))] == counted


def test_sequential_linear_relu_linear_counts_three() -> None:
    trace = _sequential_trace()

    assert trace.num_ops == 3
    assert len(trace.ops) == 3
    assert len(trace.layer_list) == 5
    _assert_counts_agree(trace)
    assert [op.func_name for op in trace.ops] == ["linear", "relu", "linear"]


def test_alias_rows_stay_addressable_but_uncounted() -> None:
    """The asymmetry: ``in`` and ``[label]`` reach every row, ``keys()`` does not."""

    trace = _sequential_trace()
    input_label = trace.input_layers[0]
    output_label = trace.output_layers[0]

    assert trace.ops[input_label].is_input
    assert trace.ops[output_label].is_output
    assert input_label in trace.ops and output_label in trace.ops
    counted_labels = trace.ops.keys()  # a list: membership here differs from ``in trace.ops``
    assert input_label not in counted_labels
    assert output_label not in counted_labels
    assert trace.layer_list[0].is_input and trace.layer_list[-1].is_output


def test_derived_accessors_and_trace_indexing_keep_every_row() -> None:
    """Only ``trace.ops`` is the counted view (why F1 was first reverted).

    The accessors built over chosen rows list exactly those rows, boundary rows
    included, and ``trace[int]`` indexes every row like ``trace[slice]`` and
    ``len(trace)``.
    """

    trace = _sequential_trace()
    rows = list(trace.layer_list)

    assert [op.layer_label for op in trace.input_ops] == list(trace.input_layers)
    assert [op.layer_label for op in trace.output_ops] == list(trace.output_layers)
    assert len(trace.input_ops) == trace.num_input_ops == 1
    assert len(trace.output_ops) == trace.num_output_ops == 1
    saved = [op for op in rows if op.has_saved_activation]
    assert list(trace.saved_ops) == saved and any(op.is_input for op in trace.saved_ops)
    assert len(trace) == len(rows) == 5
    assert [trace[index] for index in range(len(trace))] == rows
    assert trace[0] is rows[0] and trace[0].is_input


def test_counts_agree_on_a_real_model() -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.GPT2Config(n_layer=1, n_embd=16, n_head=2, n_positions=16, vocab_size=32)
    torch.manual_seed(0)
    model = transformers.GPT2Model(config).eval()
    trace = tl.trace(model, torch.randint(0, 32, (1, 6)))

    assert len(trace.layer_list) > trace.num_ops
    _assert_counts_agree(trace)


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
    assert any(op.is_buffer for op in trace.layer_list)
    _assert_counts_agree(trace)


def _armed_two_op_trace() -> Any:
    class _TwoOp(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(x + 1).sum()

    with _activate_aten_recording_for_tests():
        return tl.trace(_TwoOp(), torch.ones(2))


def test_primitive_profile_dense_rows_are_the_layer_list() -> None:
    """Validation re-form: the dense table the FK check reads is unchanged.

    ``op_row_index`` indexes the full row sequence (inputs and outputs
    included). Before F1 the check read ``list(trace.ops)``, which iterated the
    accessor's every-row list; it now reads ``list(trace.layer_list)``, the
    identical sequence.
    """

    trace = _armed_two_op_trace()
    profile = trace._primitive_op_profile
    assert profile is not None and profile.primitive_ops

    every_row = trace.ops._list  # the sequence list(trace.ops) produced before F1
    assert list(trace.layer_list) == every_row
    assert len(trace.layer_list) > len(trace.ops)
    for row in profile.primitive_ops:
        for ref in row.parent_op_refs:
            op = trace.layer_list[ref.op_row_index]
            assert op.layer_label == ref.op_label
            assert op.func_call_id == ref.func_call_id
    assert check_metadata_invariants(trace)


def test_a_wrong_op_row_index_still_raises() -> None:
    trace = _armed_two_op_trace()
    row = next(row for row in trace._primitive_op_profile.primitive_ops if row.parent_op_refs)
    ref = row.parent_op_refs[0]
    wrong = next(
        index
        for index, op in enumerate(trace.layer_list)
        if index != ref.op_row_index and op.layer_label != ref.op_label
    )
    row.parent_op_refs = (
        OpRef(op_row_index=wrong, op_label=ref.op_label, func_call_id=ref.func_call_id),
    )

    with pytest.raises(MetadataInvariantError) as excinfo:
        check_metadata_invariants(trace)
    assert excinfo.value.check_name == "primitive_op_invariants"
