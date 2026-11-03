"""Op and Layer concern groups (fields.1): group reads and typed refusals of the flat spellings.

The cost group is the first applied group: ``op.cost.flops_forward`` reads the
same persisted column ``op.flops_forward`` used to, derived members compute on
read from the group, and every former flat spelling refuses typed
(``facade_redirect``) with a remedy naming the new spelling, on a live trace
and after a pickle round trip.
"""

from __future__ import annotations

import pickle
from collections.abc import Iterator

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import FacadeTeachingError
from torchlens._utils._record_group_table import (
    ACTIVE_GROUPS,
    LAYER_MEMBER_MOVES,
    LAYER_MOVED_MEMBERS,
    OP_MEMBER_MOVES,
    OP_MOVED_MEMBERS,
    record_member,
)

_OP_COST_ROWS = [row for row in OP_MEMBER_MOVES if row[1] == "cost"]
_LAYER_COST_ROWS = [row for row in LAYER_MEMBER_MOVES if row[1] == "cost"]


@pytest.fixture(scope="module")
def trace() -> Iterator[tl.Trace]:
    """A small two-layer trace with one repeated module call (a multi-pass layer)."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    captured = tl.trace(model, torch.randn(2, 3))
    yield captured
    captured.cleanup()


def test_cost_group_is_applied() -> None:
    """The cost rows are live in the refusal tables of both records."""

    assert "cost" in ACTIVE_GROUPS
    assert len(_OP_COST_ROWS) == 16
    assert len(_LAYER_COST_ROWS) == 17
    assert {old for old, _, _ in _OP_COST_ROWS} <= set(OP_MOVED_MEMBERS)
    assert {old for old, _, _ in _LAYER_COST_ROWS} <= set(LAYER_MOVED_MEMBERS)


def test_group_classes_are_the_public_types(trace: tl.Trace) -> None:
    """``op.cost`` and ``layer.cost`` are ``tl.types.OpCost`` and ``tl.types.LayerCost``."""

    op = trace.ops["linear_1_1"]
    layer = trace.layers["linear_1"]
    assert type(op.cost) is tl.types.OpCost
    assert type(layer.cost) is tl.types.LayerCost
    assert "flops_forward" in dir(op.cost)
    assert "total_flops_forward" in dir(layer.cost)
    assert "OpCost" in repr(op.cost)


@pytest.mark.parametrize(("old", "leaf"), [(old, leaf) for old, _, leaf in _OP_COST_ROWS])
def test_op_flat_spelling_refuses_typed(trace: tl.Trace, old: str, leaf: str) -> None:
    """Each former flat Op cost spelling refuses typed, naming ``op.cost.<leaf>``."""

    op = trace.ops["linear_1_1"]
    with pytest.raises(FacadeTeachingError) as excinfo:
        getattr(op, old)
    assert excinfo.value.fields["code"] == "facade_redirect"
    assert f"op.cost.{leaf}" in str(excinfo.value)
    assert not hasattr(op, old)
    getattr(op.cost, leaf)


@pytest.mark.parametrize(("old", "leaf"), [(old, leaf) for old, _, leaf in _LAYER_COST_ROWS])
def test_layer_flat_spelling_refuses_typed(trace: tl.Trace, old: str, leaf: str) -> None:
    """Each former flat Layer cost spelling refuses typed, naming ``layer.cost.<leaf>``."""

    layer = trace.layers["linear_1"]
    with pytest.raises(FacadeTeachingError) as excinfo:
        getattr(layer, old)
    assert excinfo.value.fields["code"] == "facade_redirect"
    assert f"layer.cost.{leaf}" in str(excinfo.value)
    assert not hasattr(layer, old)


def test_layer_flat_spelling_refuses_writes(trace: tl.Trace) -> None:
    """Writing or deleting a former flat Layer spelling refuses instead of shadowing it."""

    layer = trace.layers["linear_1"]
    with pytest.raises(FacadeTeachingError):
        layer._cost_flops_forward = 1
    with pytest.raises(FacadeTeachingError):
        del layer._cost_flops_forward
    assert layer.cost.flops_forward == trace.ops["linear_1_1"].cost.flops_forward


def test_op_cost_reads_the_persisted_columns(trace: tl.Trace) -> None:
    """Stored cost members read the same column the flat spelling read."""

    op = trace.ops["linear_1_1"]
    assert int(op.cost.flops_forward) == 2 * 2 * 3 * 4 + 2 * 4
    assert op.cost.flops_forward == op._cost_flops_forward
    assert op.cost.duration == op._cost_duration
    assert int(op.cost.activation_memory) == 2 * 4 * 4
    assert int(op.cost.macs_forward) == 2 * 3 * 4
    assert op.cost.flops_total == (op.cost.flops_forward or 0) + (op.cost.flops_backward or 0)


def test_layer_cost_matches_its_ops(trace: tl.Trace) -> None:
    """Layer cost members agree with the first pass and the per-pass sums."""

    layer = trace.layers["linear_1"]
    ops = list(layer.ops.values())
    assert layer.cost.flops_forward == ops[0].cost.flops_forward
    assert layer.cost.macs_forward == ops[0].cost.macs_forward
    assert int(layer.cost.total_flops_forward) == sum(int(op.cost.flops_forward) for op in ops)
    assert (
        layer.cost.total_flops == layer.cost.total_flops_forward + layer.cost.total_flops_backward
    )
    assert layer.cost.total_macs is None
    assert layer.cost.duration == ops[0].cost.duration


def test_groups_survive_a_pickle_round_trip(trace: tl.Trace) -> None:
    """A reloaded trace serves the same group values and still refuses the flat spellings."""

    loaded = pickle.loads(pickle.dumps(trace))
    try:
        for label in ("linear_1_1", "relu_1_2"):
            assert loaded.ops[label].cost.flops_forward == trace.ops[label].cost.flops_forward
        layer = loaded.layers["linear_1"]
        assert layer.cost.flops_forward == trace.layers["linear_1"].cost.flops_forward
        assert layer.cost.activation_memory == trace.layers["linear_1"].cost.activation_memory
        with pytest.raises(FacadeTeachingError):
            layer.cost.flops_forward
        with pytest.raises(FacadeTeachingError):
            loaded.ops["linear_1_1"].cost.flops_forward
    finally:
        loaded.cleanup()


def test_record_member_follows_moves(trace: tl.Trace) -> None:
    """``record_member`` reads a former name through its group on an Op or Layer only."""

    op = trace.ops["linear_1_1"]
    layer = trace.layers["linear_1"]
    assert record_member(op, "func_duration") == op.cost.duration
    assert record_member(layer, "total_flops_total") == layer.cost.total_flops
    assert record_member(op, "label") == op.label

    class _Other:
        flops_forward = 7

    assert record_member(_Other(), "flops_forward") == 7
    assert record_member(object(), "flops_forward", None) is None
