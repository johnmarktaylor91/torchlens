"""Deterministic counters and equivalence for op removal's whole-row release.

Husking a removed op used to read every stored field through the facade
(hydrating each pooled container cell it was about to drop) and then delete
each field through its descriptor. ``OpRowStore.release_row`` and its twins
drop the row's set cells directly. These tests pin that cleanup hydrates no
pooled cell and dispatches no field descriptor, and that the direct release
leaves every store shape in exactly the state the facade husk left.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from _hermetic_counters import quiet_gc, thread_counting

import torchlens as tl
from torchlens._data_classes import op as op_mod
from torchlens._data_classes._state_adapter import state_items
from torchlens._data_classes.cleanup import _clear_entry_attributes
from torchlens._trace_core import op_store
from torchlens._trace_core.op_store import (
    _MISSING,
    DetachedOpStore,
    OpRowStore,
    PooledCell,
    mark_op_row_released,
)


def _chain(depth: int) -> torch.nn.Module:
    torch.manual_seed(0)
    layers: list[torch.nn.Module] = []
    for _ in range(depth):
        layers.extend((torch.nn.Linear(4, 4), torch.nn.ReLU()))
    return torch.nn.Sequential(*layers).eval()


def _trace(depth: int = 4) -> Any:
    torch.manual_seed(0)
    return tl.trace(_chain(depth), torch.randn(2, 4))


def _facade_husk(op: Any) -> None:
    """The pre-release husk: read every field, delete each one read."""

    store = object.__getattribute__(op, "_core")
    row = object.__getattribute__(op, "_row")
    mark_op_row_released(store, row)
    for attr, _ in list(state_items(op)):
        delattr(op, attr)
    op_mod._detach_op_husk(op)


def _raw_state(store: Any) -> tuple[Any, ...]:
    """Presence of every cell plus the overlay's keys and tombstones."""

    rows = store._rows
    if rows is not None:
        present = tuple(tuple(value is not _MISSING for value in row) for row in rows)
    else:
        present = tuple(
            tuple(store.cell_get(row, fid) is not _MISSING for fid in range(store.layout.n_fields))
            for row in range(len(store))
        )
    overlay = store._overlay
    return (
        type(store).__name__,
        present,
        tuple(sorted(overlay)),
        tuple(sorted(key for key, value in overlay.items() if value is _MISSING)),
    )


def test_cleanup_hydrates_no_pooled_cell_and_dispatches_no_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trace cleanup releases rows without per-field reads or descriptor deletes."""

    trace = _trace()
    store = object.__getattribute__(trace.ops[0], "_core")
    pooled = sum(
        store.cell_get(row, fid).__class__ is PooledCell
        for row in range(len(store))
        for fid in range(store.layout.n_fields)
    )
    assert pooled > 0, "the fixture must carry pooled container cells"
    counts = {"hydrate": 0, "delete": 0}
    # Only this thread's calls count, with pending garbage collected first and
    # the cyclic collector off, so finalizers of objects earlier tests left
    # behind cannot hydrate or delete inside the window.
    monkeypatch.setattr(
        PooledCell, "hydrate", thread_counting(PooledCell.hydrate, counts, "hydrate")
    )
    monkeypatch.setattr(
        op_mod._OpField,
        "__delete__",
        thread_counting(op_mod._OpField.__delete__, counts, "delete"),
    )
    with quiet_gc():
        trace.cleanup()
    assert counts == {"hydrate": 0, "delete": 0}
    assert all(
        store.cell_get(row, fid) is _MISSING
        for row in range(len(store))
        for fid in range(store.layout.n_fields)
    )


@pytest.mark.parametrize("index", [0, 3, -1])
def test_release_matches_facade_husk_on_sealed_trace(index: int) -> None:
    """Twin traces: the direct release leaves the same store state as the facade husk."""

    facade_trace, direct_trace = _trace(), _trace()
    facade_op, direct_op = facade_trace.ops[index], direct_trace.ops[index]
    facade_store = object.__getattribute__(facade_op, "_core")
    direct_store = object.__getattribute__(direct_op, "_core")
    assert _raw_state(facade_store) == _raw_state(direct_store)
    _facade_husk(facade_op)
    _clear_entry_attributes(direct_op)
    assert _raw_state(facade_store) == _raw_state(direct_store)
    assert facade_store._sweep_plan is None and direct_store._sweep_plan is None
    for husk in (facade_op, direct_op):
        assert type(object.__getattribute__(husk, "_core")) is DetachedOpStore
        assert list(state_items(husk)) == []


def _synthetic_store(n_rows: int) -> OpRowStore:
    """A building store with pooled, packable, container and unset cells."""

    layout = op_store.OpStoreLayout(("a", "b", "c", "d", "e"))
    store = OpRowStore(layout)
    pooled = PooledCell.from_value([])
    for row in range(n_rows):
        cells = [row, pooled, [row], _MISSING, float(row) if row % 2 else _MISSING]
        store.adopt_row(cells)
    return store


def _per_cell_release(store: Any, row: int) -> None:
    """``cell_del`` on every field id, the facade husk's store-level effect."""

    for fid in range(store.layout.n_fields):
        store.cell_del(row, fid)


@pytest.mark.parametrize("n_rows", [8, op_store._TRANSPOSE_MIN_ROWS + 4])
@pytest.mark.parametrize("freeze", [False, True])
@pytest.mark.parametrize("cow_shared", [False, True])
def test_release_row_matches_per_cell_delete(n_rows: int, freeze: bool, cow_shared: bool) -> None:
    """Building, sealed row-major and columnar stores: same state as per-cell deletes."""

    twins = [_synthetic_store(n_rows), _synthetic_store(n_rows)]
    for store in twins:
        if freeze:
            store.freeze()
        store._cow_shared = cow_shared
    per_cell, direct = twins
    assert type(per_cell) is type(direct)
    for row in (0, 1, n_rows - 1):
        _per_cell_release(per_cell, row)
        direct.release_row(row)
        assert _raw_state(per_cell) == _raw_state(direct)
    # A second release of the same row is a no-op on both.
    _per_cell_release(per_cell, 1)
    direct.release_row(1)
    assert _raw_state(per_cell) == _raw_state(direct)


def test_detached_release_row_clears_every_cell() -> None:
    """The single-row store drops every set cell."""

    store = DetachedOpStore(op_store.OpStoreLayout(("a", "b", "c")))
    store.adopt_row([1, _MISSING, PooledCell.from_value({})])
    store.release_row(0)
    assert all(store.cell_get(0, fid) is _MISSING for fid in range(3))


def test_patched_field_attribute_keeps_the_facade_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """An Op attribute that is not a field descriptor disables the direct release."""

    trace = _trace()
    op = trace.ops[1]
    store = object.__getattribute__(op, "_core")
    row = object.__getattribute__(op, "_row")
    name = op_mod._OP_SLOT_NAMES[0]
    descriptor = vars(op_mod.Op)[name]
    monkeypatch.setattr(
        op_mod.Op,
        name,
        property(descriptor.__get__, descriptor.__set__, descriptor.__delete__),
    )
    assert op_mod.release_op_row(op, store, row) is False


def test_every_stored_field_attribute_qualifies_for_row_release() -> None:
    """Each stored field's ``Op`` attribute is a field descriptor or a registered overlay.

    A property installed over a stored field after its registration ran (or never
    registered) silently sends every op removal back to the per-field facade path;
    this names the field instead.
    """

    namespace = vars(op_mod.Op)
    unqualified = [
        (name, type(namespace.get(name)).__name__)
        for name in op_mod._OP_SLOT_NAMES
        if namespace.get(name).__class__ not in op_mod._ROW_RELEASE_FIELD_CLASSES
        and namespace.get(name) not in op_mod._ROW_RELEASE_PROPERTIES
    ]
    assert unqualified == []
