"""A final Op label resolves through ``trace.ops`` to its own Op, or refuses.

Raw labels (``{type}_{type_idx}_{raw_idx}``) and final labels
(``{type}_{type_idx}_{step_idx}``) share one shape. Once grouping folds passes
into one layer the two numberings drift apart, and one Op's unsuffixed raw
spelling can equal another Op's final label (seen on mistral, where
``trace.ops["to_10_109"]`` returned the op whose raw label was
``to_10_109_raw`` instead of the op the graph edges name). These tests plant
that shadowing, a final label claimed by two Ops, and a mis-resolving accessor,
and require the accessor to answer correctly or refuse and the metadata
invariants to fail.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes._trace_accessors import _TRACE_OP_ACCESSOR_ATTR, TraceOpAccessor
from torchlens._data_classes.op import Op
from torchlens._errors import AmbiguousOpLookupError
from torchlens.errors import MetadataInvariantError
from torchlens.validation.invariants import _check_op_label_resolution


class _LoopThenSites(nn.Module):
    """A two-pass relu loop followed by single-pass ops, so numberings drift."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(2):
            x = torch.relu(x - 0.1)
        x = torch.sigmoid(x)
        x = torch.tanh(x + 1.0)
        return x * 2.0


@pytest.fixture
def trace() -> Iterator[tl.Trace]:
    """Capture the drift model once per test."""

    model = _LoopThenSites()
    captured = tl.trace(model, torch.randn(3, 4))
    try:
        yield captured
    finally:
        tl.release_model(model)
        captured.cleanup()


def _single_pass_pair(trace: tl.Trace) -> tuple[Op, Op]:
    """Return an earlier and a later single-pass computational Op."""

    singles = [op for op in trace.ops if op.num_passes == 1]
    assert len(singles) >= 2
    return singles[0], singles[-1]


def _fresh_ops(trace: tl.Trace) -> TraceOpAccessor:
    """Drop the memoized accessor so the next ``trace.ops`` rebuilds its indexes."""

    trace.__dict__.pop(_TRACE_OP_ACCESSOR_ATTR, None)
    return trace.ops


def test_every_label_resolves_to_its_own_op(trace: tl.Trace) -> None:
    """On a clean capture every final label and raw label names its own Op."""

    _check_op_label_resolution(trace)
    for op in trace.layer_list:
        assert trace.ops[op.label] is op
        assert trace.ops[op._label_raw] is op


def test_final_label_outranks_another_ops_raw_spelling(trace: tl.Trace) -> None:
    """Plant the mistral shape: an earlier Op's raw spelling equals a later final label."""

    earlier, later = _single_pass_pair(trace)
    final_label = later.layer_label
    earlier._label_raw = f"{final_label}_raw"
    ops = _fresh_ops(trace)

    assert ops[final_label] is later
    assert ops._resolve_substring_by_scan(final_label) is later
    # The shadowed raw spelling stays reachable through its unambiguous forms.
    assert ops[f"{final_label}_raw"] is earlier
    assert ops.by_raw_index(earlier.raw_index) is earlier
    _check_op_label_resolution(trace)


@dataclass
class _Row:
    """Duck-typed Op row: the label attributes ``TraceOpAccessor`` reads."""

    label: str
    label_short: str
    _label_raw: str
    layer_label: str
    layer_label_short: str
    raw_index: int

    @property
    def raw_label(self) -> str:
        """The unsuffixed raw spelling, as ``Op.raw_label`` derives it."""

        return self._label_raw.removesuffix("_raw")


def _rows_accessor(*rows: _Row) -> TraceOpAccessor:
    """Build an accessor over stand-in rows, each a single-pass layer."""

    calls = {row.layer_label: 1 for row in rows}
    return TraceOpAccessor(list(rows), calls)  # type: ignore[arg-type]


def test_raw_shadow_resolves_to_the_final_label_owner() -> None:
    """Row-level plant of the mistral shape, independent of any capture layout."""

    shadow = _Row("to_9_101:1", "to_9:1", "to_10_109_raw", "to_9_101", "to_9", 117)
    owner = _Row("to_10_109:1", "to_10:1", "to_11_117_raw", "to_10_109", "to_10", 125)
    accessor = _rows_accessor(shadow, owner)
    assert accessor["to_10_109"] is owner
    assert accessor._resolve_substring_by_scan("to_10_109") is owner
    assert accessor["to_10_109_raw"] is shadow
    # A raw spelling nothing final claims still resolves.
    assert accessor["to_11_117"] is owner


def test_unsuffixed_raw_label_is_a_raw_spelling_not_a_collision() -> None:
    """An unsuffixed ``_label_raw`` equal to another op's final label is outranked, not a clash.

    The collapse-optimizer fixtures plant this shape (an op whose raw label is
    the input's final label ``input_1``). It is a raw spelling, so the final
    label's owner wins and nothing refuses; only a suffixed raw label is a
    final-tier alias.
    """

    owner = _Row("input_1:1", "input_1:1", "input_1_raw", "input_1", "input_1", 0)
    planted = _Row("linear_1_1:1", "linear_1:1", "input_1", "linear_1_1", "linear_1", 1)
    accessor = _rows_accessor(owner, planted)
    assert accessor["input_1"] is owner
    assert accessor._resolve_substring_by_scan("input_1") is owner
    assert accessor["input_1_raw"] is owner
    assert accessor["linear_1_1"] is planted


def test_final_label_claimed_by_two_ops_refuses() -> None:
    """Two rows sharing one final alias make every lookup of it refuse."""

    first = _Row("relu_1_2:1", "relu_1:1", "relu_1_2_raw", "relu_1_2", "relu_1", 2)
    second = _Row("relu_2_3:1", "relu_1:1", "relu_2_3_raw", "relu_2_3", "relu_2", 3)
    accessor = _rows_accessor(first, second)
    with pytest.raises(AmbiguousOpLookupError, match="claimed by 2 different ops"):
        accessor["relu_1:1"]
    with pytest.raises(AmbiguousOpLookupError, match="claimed by 2 different ops"):
        accessor._resolve_substring_by_scan("relu_1:1")


def test_duplicate_pass_qualified_label_refuses() -> None:
    """A row list holding two rows under one ``label`` refuses that label."""

    first = _Row("relu_1_2:1", "relu_1:1", "relu_1_2_raw", "relu_1_2", "relu_1", 2)
    second = _Row("relu_1_2:1", "relu_2:1", "relu_2_3_raw", "relu_2_3", "relu_2", 3)
    accessor = _rows_accessor(first, second)
    with pytest.raises(AmbiguousOpLookupError, match="claimed by 2 different ops"):
        accessor["relu_1_2:1"]


def test_validation_fails_on_a_misresolving_label(trace: tl.Trace) -> None:
    """Validation fails when ``trace.ops`` answers a final label with another Op."""

    earlier, later = _single_pass_pair(trace)
    ops = _fresh_ops(trace)
    indexes = ops._ensure_alias_indexes()
    assert indexes is not None
    indexes.final[later.layer_label] = earlier
    with pytest.raises(
        MetadataInvariantError, match="resolves through trace.ops to a different op"
    ):
        _check_op_label_resolution(trace)


def test_validation_fails_on_a_colliding_label(trace: tl.Trace) -> None:
    """Validation fails when two Ops claim one final label."""

    earlier, later = _single_pass_pair(trace)
    ops = _fresh_ops(trace)
    indexes = ops._ensure_alias_indexes()
    assert indexes is not None
    indexes.collisions[later.layer_label] = [earlier, later]
    with pytest.raises(MetadataInvariantError, match="does not resolve to it"):
        _check_op_label_resolution(trace)


def test_validation_fails_on_a_dangling_label(trace: tl.Trace) -> None:
    """Validation fails when an Op's own label resolves to nothing."""

    _, later = _single_pass_pair(trace)
    ops = _fresh_ops(trace)
    indexes = ops._ensure_alias_indexes()
    assert indexes is not None
    del ops._dict[later.label]
    for alias in (later.label, later.label_short, later._label_raw, later.layer_label):
        indexes.final.pop(alias, None)
    indexes.layer.pop(later.layer_label, None)
    indexes.raw.pop(later.raw_label, None)
    with pytest.raises(MetadataInvariantError, match="does not resolve to it"):
        _check_op_label_resolution(trace)
