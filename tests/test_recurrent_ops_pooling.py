"""Canonical-container pooling tests for ``recurrent_ops`` and Layer ``equivalent_ops``.

``recurrent_ops`` shares ONE canonical list per recurrence group (mirroring the
``equivalent_ops`` canonical-set pooling), with a copy-on-read barrier in
``Op.__getattribute__`` so no holder can alias-corrupt the group. ``Layer``
stores the canonical ``equivalent_ops`` set instead of retaining a private
barrier copy per Layer, with the same copy-on-read isolation on read.
"""

from __future__ import annotations

import pickle
from collections.abc import Iterator

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.postprocess.loop_detection import _apply_recurrence_assignments
from torchlens.postprocess.loop_grouping_adapter import RecurrenceAssignment

STEPS = 6
DIM = 8


class _LoopCell(nn.Module):
    """Hand-written recurrence producing multi-pass linear/tanh groups."""

    def __init__(self) -> None:
        """Initialize the shared linear cell."""

        super().__init__()
        self.lin = nn.Linear(2 * DIM, DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run ``STEPS`` recurrent steps over one shared cell."""

        h = torch.zeros(x.shape[0], DIM)
        for t in range(STEPS):
            h = torch.tanh(self.lin(torch.cat([h, x[:, t]], dim=-1)))
        return h


@pytest.fixture(scope="module")
def loop_trace() -> Iterator[tl.Trace]:
    """Trace of the recurrent cell, shared across tests (read-only)."""

    torch.manual_seed(0)
    model = _LoopCell()
    x = torch.randn(2, STEPS, DIM)
    trace = tl.trace(model, x)
    try:
        yield trace
    finally:
        trace.cleanup()


def _multi_pass_groups(trace: tl.Trace) -> dict[str, list]:
    """Return ``layer_label -> member ops`` for every multi-pass group."""

    groups: dict[str, list] = {}
    for op in trace:
        if op.num_passes > 1:
            groups.setdefault(op.layer_label, []).append(op)
    return groups


@pytest.mark.smoke
def test_recurrent_ops_share_one_canonical_list_per_group(loop_trace) -> None:
    """Every member of a recurrence group backs onto ONE raw list object."""

    groups = _multi_pass_groups(loop_trace)
    assert groups, "model must produce multi-pass groups"
    for members in groups.values():
        assert len(members) == STEPS
        raw_ids = {id(op._slot("recurrent_ops")) for op in members}
        assert len(raw_ids) == 1, "recurrence group must share one canonical list"


@pytest.mark.smoke
def test_recurrent_ops_values_and_order_are_correct(loop_trace) -> None:
    """Pooling never changes the observed value: ordered member labels."""

    for op in loop_trace:
        value = op.recurrent_ops
        assert type(value) is tuple
        assert op.label in value
        assert len(value) == op.num_passes
        members = [loop_trace[label] for label in value]
        raw_orders = [member.raw_index for member in members]
        assert raw_orders == sorted(raw_orders), "group order must follow raw capture order"
        assert value[op.pass_index - 1] == op.label


@pytest.mark.smoke
def test_recurrent_ops_reads_are_alias_safe(loop_trace) -> None:
    """Reads share ONE immutable group view; mutation is impossible (M7).

    The live group views (FORK-1, decided 2026-08-12) supersede the
    fresh-copy-per-read barrier: sharing an immutable tuple across members
    and readers is alias-safe by construction.
    """

    groups = _multi_pass_groups(loop_trace)
    members = next(iter(groups.values()))
    first, second = members[0], members[1]
    assert first.recurrent_ops is first.recurrent_ops, "group view reads are identity-stable"
    assert first.recurrent_ops is second.recurrent_ops, "members share ONE view"

    stolen = first.recurrent_ops
    expected = tuple(stolen)
    with pytest.raises(AttributeError):
        stolen.append("corrupted")
    assert first.recurrent_ops == expected
    assert second.recurrent_ops == expected


@pytest.mark.smoke
def test_assignment_apply_pools_lists_without_pooling_member_keys() -> None:
    """Shared assignment tuples yield one list while keys remain member-specific."""

    class _Node:
        """Minimal mutable assignment target."""

        pass

    members = ("first", "second")
    nodes = {label: _Node() for label in members}
    assignments = {
        label: RecurrenceAssignment(
            layer_label="shared_layer",
            recurrent_labels=members,
            pass_index=index,
            num_passes=2,
            equivalence_key=f"member_key_{index}",
        )
        for index, label in enumerate(members, start=1)
    }

    _apply_recurrence_assignments(nodes, assignments)  # type: ignore[arg-type]

    assert nodes["first"].recurrent_ops is nodes["second"].recurrent_ops
    assert nodes["first"].equivalence_class == "member_key_1"
    assert nodes["second"].equivalence_class == "member_key_2"


@pytest.mark.smoke
def test_layer_equivalent_ops_shares_canonical_set(loop_trace) -> None:
    """Layers back onto the ops' canonical equivalence container (M8 mirror).

    The dict era stored the shared canonical set per Layer; the M8 aggregate
    facade stores NOTHING — the mirror read resolves to the group's ONE
    cached immutable view, the same object the op read returns.
    """

    for layer in loop_trace.layers:
        first_pass = layer.ops[0]
        assert "equivalent_ops" not in layer.__dict__, (
            "Layer must not retain a private equivalent_ops copy (M8 mirror)"
        )
        assert layer.equivalent_ops is first_pass.equivalent_ops, (
            "Layer reads must resolve to the ops' one shared group view"
        )


@pytest.mark.smoke
def test_layer_equivalent_ops_reads_are_alias_safe(loop_trace) -> None:
    """Layer reads share the group's ONE immutable view (M7 live views)."""

    layer = loop_trace.layers[0]
    value = layer.equivalent_ops
    assert type(value) is frozenset
    assert layer.equivalent_ops is value, "group view reads are identity-stable"

    expected = frozenset(value)
    with pytest.raises(AttributeError):
        value.add("corrupted")
    assert layer.equivalent_ops == expected
    assert layer.ops[0].equivalent_ops == expected


@pytest.mark.smoke
def test_pooled_fields_survive_pickle_round_trip(loop_trace) -> None:
    """Pickle round-trips preserve values for both pooled fields."""

    restored = pickle.loads(pickle.dumps(loop_trace))
    for op in loop_trace:
        twin = restored[op.label]
        assert twin.recurrent_ops == op.recurrent_ops
        assert twin.equivalent_ops == op.equivalent_ops
    for layer, twin_layer in zip(loop_trace.layers, restored.layers):
        assert twin_layer.equivalent_ops == layer.equivalent_ops
