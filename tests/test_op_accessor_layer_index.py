"""Bare-layer-label Op reads are served from the layer-label index, not a row scan.

The captures keep live views (``save_mode="view"``) so the GradFn read test can
differentiate through the recorded tensors the way the Jacobian lens entry does.

``GradFn.op`` resolves its bare layer label through ``trace.ops.resolve_all`` once
per op on every backward pass. When that read scanned every row, a Jacobian fit
on Mamba-130m (about 24k ops, 96 passes per prompt) took six times longer than
the same fit with an indexed read. The gate is deterministic: after one warming
read, the row list refuses iteration and the reads must still answer.
"""

from __future__ import annotations

import gc
import pickle
import warnings
from collections.abc import Iterator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes._trace_accessors import TraceOpAccessor
from torchlens.options import CaptureOptions


class _NoScan(list):
    """A row key list that fails the test the moment anything iterates it.

    The ops door holds one row key per op and reads every record through that
    list, so a scan of the op rows is an iteration of it.
    """

    def __iter__(self) -> Iterator:  # pragma: no cover - the failure path
        raise AssertionError("resolve_all scanned the op rows instead of using the index")


class _Twice(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(torch.relu(self.lin(x)))


def _trace() -> tl.Trace:
    torch.manual_seed(0)
    x = torch.randn(2, 4, requires_grad=True)
    return tl.trace(_Twice().eval(), x, capture=CaptureOptions(save_mode="view"))


def test_resolve_all_bare_layer_label_reads_the_index() -> None:
    trace = _trace()
    ops = trace.ops
    multi_pass = next(
        op for op in trace._layer_list if trace._layer_num_calls.get(op.layer_label, 0) > 1
    )
    single_pass = next(
        op for op in trace._layer_list if trace._layer_num_calls.get(op.layer_label, 0) == 1
    )
    expected_multi = ops.resolve_all(multi_pass.layer_label)
    expected_single = ops.resolve_all(single_pass.layer_label)
    assert len(expected_multi) == 2 and expected_single == [single_pass]

    ops._keys = _NoScan(ops._keys)
    assert ops.resolve_all(multi_pass.layer_label) == expected_multi
    assert ops.resolve_all(multi_pass.short_layer_label) == expected_multi
    assert ops.resolve_all(single_pass.layer_label) == expected_single
    assert ops.resolve_all(f"{multi_pass.layer_label}:last") == [expected_multi[-1]]
    assert ops.resolve_all("no_such_layer_9") == []


def test_grad_fn_op_reads_do_not_scan_the_rows() -> None:
    trace = _trace()
    out = trace[trace.output_layers[0]].out
    source = next(op for op in trace._layer_list if not op.is_input).out
    torch.autograd.grad(out.sum(), source, retain_graph=True)
    grad_fns = [gf for gf in trace.autograd.grad_fns if gf.op_label is not None]
    if not grad_fns:
        pytest.skip("no GradFn carries an op label in this capture")
    resolved = [gf.op for gf in grad_fns]
    trace.ops._keys = _NoScan(trace.ops._keys)
    assert [gf.op for gf in grad_fns] == resolved


def _held_values(ops: TraceOpAccessor) -> list[Any]:
    """Every value the door's own containers and lazy indexes hold."""

    indexes = ops._alias_indexes
    assert indexes is not None and ops._layer_label_index is not None
    assert ops._counted_cache is not None and ops._resolved_op_cache
    held: list[Any] = [*ops._keys, *ops._counted_cache]
    for mapping in (ops._label_keys, ops._raw_index_lookup, ops._resolved_op_cache):
        held.extend(mapping.values())
    for mapping in (indexes.final, indexes.raw):
        held.extend(mapping.values())
    for groups in (indexes.layer, indexes.collisions, ops._layer_label_index):
        for members in groups.values():
            held.extend(members)
    return held


def test_ops_door_holds_row_keys_and_reads_the_same_records() -> None:
    trace = _trace()
    ops = trace.ops
    rows = list(trace._layer_list)
    single_pass = next(op for op in rows if trace._layer_num_calls.get(op.layer_label, 0) == 1)
    multi_pass = [op for op in rows if trace._layer_num_calls.get(op.layer_label, 0) > 1]
    assert ops[single_pass.short_label] is single_pass
    assert ops._ensure_alias_indexes() is not None
    # A bare multi-pass layer label is the lookup that builds the layer label index.
    layer_label = multi_pass[0].layer_label
    assert ops.resolve_all(layer_label) == [op for op in rows if op.layer_label == layer_label]
    assert ops._resolved_op(single_pass.label) is single_pass
    assert ops._resolved_op(single_pass.label) is single_pass  # the memoized read
    assert len(ops) == len(list(ops)) > 0

    held = _held_values(ops)
    assert held and all(type(value) is int for value in held)
    assert all(found is row for found, row in zip(ops._list, rows, strict=True))
    assert all(ops._dict[row.label] is row for row in rows)
    assert all(ops.by_raw_index(row.raw_index) is row for row in rows)
    assert [ops[index] for index in range(len(ops))] == list(ops) == ops.values()


def test_reloaded_trace_with_a_backward_capture_keys_its_ops_door_by_row() -> None:
    # Reloading a backward capture reads an ops door before the trace has its
    # core; the door the user gets afterwards must still key by store row.
    trace = _trace()
    trace.backward(trace.output.sum())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # tensor hooks are not serialized
        restored = pickle.loads(pickle.dumps(trace))
    door = restored.ops
    rows = list(restored._layer_list)
    tables = restored.__dict__["_trace_core"]._views
    assert door._table.rows is tables.ops and not tables.detached
    assert door._keys and all(type(key) is int and key >= 0 for key in door._keys)
    assert all(found is row for found, row in zip(door._list, rows, strict=True))


def test_held_ops_door_keeps_its_records_after_the_trace_is_gone() -> None:
    ops = _trace().ops
    gc.collect()
    labels = [op.label for op in ops]
    assert labels and [ops[label].label for label in labels] == labels
    assert [op.label for op in ops.compute] == labels


def test_ops_door_state_carries_its_members_only() -> None:
    trace = _trace()
    door = trace.ops.compute
    members = list(door)
    assert 0 < len(members) < len(trace._layer_list)
    state = door.__getstate__()
    carried = [row for row in state["_table"].rows if row is not None]
    carried.extend(state["_table"].detached.values())
    assert len(carried) == len(members)
    assert all(any(row is member for member in members) for row in carried)

    restored = TraceOpAccessor.__new__(TraceOpAccessor)
    restored.__setstate__(state)
    assert list(restored) == members and restored[members[0].label] is members[0]

    # A door pickled while it still held the records carries them as ``_list``.
    legacy = TraceOpAccessor.__new__(TraceOpAccessor)
    legacy.__setstate__(
        {"_list": members, "_layer_num_calls": dict(trace._layer_num_calls), "_counted_view": False}
    )
    assert list(legacy) == members and legacy[members[-1].label] is members[-1]
