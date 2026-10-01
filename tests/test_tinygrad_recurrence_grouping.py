"""Tinygrad recurrence grouping: multi-pass layers, relabel safety, tamper oracles."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

tinygrad = pytest.importorskip("tinygrad")

from tinygrad import Tensor  # noqa: E402

import torchlens as tl  # noqa: E402
from torchlens.backends.tinygrad import TinygradBackend  # noqa: E402

pytestmark = [pytest.mark.backend_tinygrad, pytest.mark.smoke]


def _weight() -> Any:
    """Return a deterministic non-degenerate weight matrix.

    Returns
    -------
    Any
        Weight whose repeated application produces distinct per-pass outputs.
    """

    return Tensor.arange(16, dtype="float").reshape(4, 4) / 20.0


def _repeated_fn(x: Any, w: Any) -> Any:
    """Apply one matmul+relu block twice with a shared weight."""

    for _ in range(2):
        x = (x @ w).relu()
    return x


def _repeated_trace() -> Any:
    """Return a grouped trace of the shared-weight repeated block."""

    return tl.trace(_repeated_fn, (Tensor.ones(2, 4), _weight()), backend="tinygrad")


def _grouped_ops(trace: Any, func_name: str) -> list[Any]:
    """Return the ops of one multi-pass grouped layer, in pass order.

    Parameters
    ----------
    trace:
        Grouped tinygrad trace.
    func_name:
        Backend function name shared by the grouped ops.

    Returns
    -------
    list[Any]
        Multi-pass ops with that function name, in pass order.
    """

    return [op for op in trace.layer_list if op.func_name == func_name and op.num_passes > 1]


def test_tinygrad_repeated_block_groups_into_passes() -> None:
    """A repeated matmul+relu block becomes multi-pass layers that validate."""

    trace = _repeated_trace()
    # The tinygrad graph is reconstructed from the final UOp DAG: matmul
    # decomposes to mul + reduce (and relu to cmplt/where), so the repeated
    # loop body shows up as 2-pass layers on those decomposed ops.
    mul_ops = _grouped_ops(trace, "<lambda>")
    where_ops = _grouped_ops(trace, "where")

    assert trace.recurrence_detection is True
    assert mul_ops, "no grouped multi-pass ops found for the repeated body"
    assert where_ops
    assert {op.num_passes for op in where_ops} == {2}
    assert [op.pass_index for op in where_ops] == [1, 2]
    assert {op.layer_label for op in where_ops} == {where_ops[0].layer_label}
    assert list(where_ops[0].recurrent_ops) == [op.label for op in where_ops]
    assert trace.layer_num_calls[where_ops[0].layer_label] == 2
    assert TinygradBackend().validate_trace(trace) is True


def test_tinygrad_grouping_relabel_keeps_lookups_coherent() -> None:
    """Raw labels stay resolvable and per-op backend addresses stay unique."""

    trace = _repeated_trace()
    where_ops = _grouped_ops(trace, "where")
    pass2 = where_ops[1]

    assert trace[pass2._label_raw] is pass2
    assert trace[pass2.label] is pass2
    assert pass2.label == f"{pass2.layer_label}:2"
    addresses = [op.backend_address for op in trace.layer_list]
    assert len(addresses) == len(set(addresses)), "backend addresses collided"


def test_tinygrad_recurrence_detection_off_preserves_single_pass_layout() -> None:
    """The =False switch keeps the historical ungrouped layout."""

    trace = tl.trace(
        _repeated_fn,
        (Tensor.ones(2, 4), _weight()),
        backend="tinygrad",
        capture=tl.options.CaptureOptions(recurrence_detection=False),
    )

    assert trace.recurrence_detection is False
    assert all(op.num_passes == 1 for op in trace.layer_list)
    assert TinygradBackend().validate_trace(trace) is True


def test_tinygrad_tamper_dangling_sidecar_label_fails_validation() -> None:
    """A sidecar capture keyed to a label no op owns must fail, not skip."""

    trace = _repeated_trace()
    captures = list(trace.tinygrad_uop_captures)
    where_ops = _grouped_ops(trace, "where")
    target = next(
        index
        for index, capture in enumerate(captures)
        if capture.label_raw == where_ops[0]._label_raw
    )
    captures[target] = dataclasses.replace(captures[target], label_raw="where_9_99_raw")
    trace.tinygrad_uop_captures = tuple(captures)

    assert TinygradBackend().validate_trace(trace) is False


def test_tinygrad_tamper_swapped_sidecar_labels_fail_validation() -> None:
    """A stale-label oracle checking the WRONG pass's payload must FAIL.

    If grouping desynchronized the label-keyed sidecar from the ops it
    witnessed (each capture resolving to the other pass), replay would
    compare pass-1 UOp structure against pass-2 saved payloads. With a
    non-degenerate weight those differ, and validation must fail rather than
    silently bless the association.
    """

    trace = _repeated_trace()
    captures = list(trace.tinygrad_uop_captures)
    where_ops = _grouped_ops(trace, "where")
    raw_labels = {op._label_raw for op in where_ops}
    indices = [index for index, capture in enumerate(captures) if capture.label_raw in raw_labels]
    assert len(indices) == 2
    first, second = indices
    label_first = captures[first].label_raw
    label_second = captures[second].label_raw
    captures[first] = dataclasses.replace(captures[first], label_raw=label_second)
    captures[second] = dataclasses.replace(captures[second], label_raw=label_first)
    trace.tinygrad_uop_captures = tuple(captures)

    assert TinygradBackend().validate_trace(trace) is False


def test_tinygrad_tamper_stale_parent_positions_fail_validation() -> None:
    """A capture whose frozen parent positions disagree with the graph fails."""

    trace = _repeated_trace()
    captures = list(trace.tinygrad_uop_captures)
    where_ops = _grouped_ops(trace, "where")
    target = next(
        index
        for index, capture in enumerate(captures)
        if capture.label_raw == where_ops[1]._label_raw and capture.parent_arg_positions
    )
    original = captures[target].parent_arg_positions
    tampered = tuple((position, "cmplt_9_99_raw") for position, _label in original)
    captures[target] = dataclasses.replace(captures[target], parent_arg_positions=tampered)
    trace.tinygrad_uop_captures = tuple(captures)

    assert TinygradBackend().validate_trace(trace) is False


def test_tinygrad_derived_grads_survive_grouping() -> None:
    """Grouped traces keep the same derived-grad surface as ungrouped ones."""

    def loss_fn(x: Any, w: Any) -> Any:
        """Return a scalar loss through the repeated block."""

        return _repeated_fn(x, w).sum()

    from torchlens.backends.tinygrad import GradOptions

    x = Tensor.ones(2, 4)
    grouped = tl.trace(
        loss_fn,
        (x, _weight()),
        backend="tinygrad",
        grad_options=GradOptions(input_grad_argnums=(0,), intermediate_grads=True),
    )
    ungrouped = tl.trace(
        loss_fn,
        (x, _weight()),
        backend="tinygrad",
        grad_options=GradOptions(input_grad_argnums=(0,), intermediate_grads=True),
        capture=tl.options.CaptureOptions(recurrence_detection=False),
    )

    assert set(grouped.derived_grads.keys()) == set(ungrouped.derived_grads.keys())
    # Records key on final op labels, which differ across the two layouts;
    # compare in raw-label space (the per-op capture identity).
    grouped_raw = {grouped[label]._label_raw for label in grouped.intermediate_derived_grads}
    ungrouped_raw = {ungrouped[label]._label_raw for label in ungrouped.intermediate_derived_grads}
    assert ungrouped_raw, "reference ungrouped trace produced no records"
    assert grouped_raw == ungrouped_raw
