"""Unit suite for carried cells: derived cells that hold their own raw material.

``CarriedFamily`` (``torchlens/_trace_core/cell_states.py``) lets a pipeline
step store one ``LazyCell`` per row, holding that row's raw material, where
it would have stored a computed value; the first read runs the family
resolver on the material. These tests cover every store shape (building,
sealed, columnar, fork views, the detached row), the freeze seam and its
relation-view interning, compaction, the state stream and pickling.
"""

from __future__ import annotations

import copy
import pickle
from typing import Any

import pytest

from torchlens._data_classes._compaction import _compact_store_rows
from torchlens._trace_core.cell_states import (
    _MISSING,
    STATE_DERIVED,
    STATE_SET,
    CarriedFamily,
    LazyCell,
    LazyFields,
    cell_state,
)
from torchlens._trace_core.op_store import DetachedOpStore, OpRowStore, OpStoreLayout, OpStoreView
from torchlens._trace_core.relation_views import convert_row_cells

NAMES = ("plain", "shape", "joined")
PLAIN, SHAPE, JOINED = range(3)
LAYOUT = OpStoreLayout(NAMES)
#: Row count over the columnar-transpose threshold (512).
COLUMNAR_ROWS = 600
SMALL_ROWS = 4
SHAPES = ("building", "sealed", "columnar", "fork", "fork_columnar", "fork_of_fork", "detached")
CALLS: list[tuple[Any, ...]] = []


def _join(store: Any, row: int, name: str, material: Any) -> list[Any]:
    """Module-level resolver (a pickled cell names it): material plus the row's ``plain``."""

    CALLS.append((name, row, material))
    return [material, store.cell_get(row, PLAIN)]


def _lacking(store: Any, row: int, name: str, material: Any) -> Any:
    """Resolver that wrongly returns the unset sentinel."""

    return _MISSING


FAMILY = CarriedFamily("joins", ("joined",), _join)


@pytest.fixture(autouse=True)
def _clear_calls() -> Any:
    """Start every test with an empty resolver log."""

    CALLS.clear()
    yield
    CALLS.clear()


def _row(row: int) -> list[Any]:
    """Return a row whose ``joined`` cell is carried with material ``("m", row)``."""

    return [f"p{row}", ["s", row], FAMILY.carry("joined", ("m", row))]


def _store(shape: str) -> Any:
    """Return a store of the named shape whose rows all carry ``joined``."""

    if shape == "detached":
        detached = DetachedOpStore(LAYOUT)
        detached.adopt_row(_row(0))
        return detached
    columnar = "columnar" in shape
    base = OpRowStore(LAYOUT)
    for row in range(COLUMNAR_ROWS if columnar else SMALL_ROWS):
        base.adopt_row(_row(row))
    if shape == "building":
        return base
    base.freeze()
    assert (base._rows is None) == columnar
    if shape in ("sealed", "columnar"):
        return base
    view = OpStoreView(base)
    return OpStoreView(view) if shape == "fork_of_fork" else view


@pytest.mark.parametrize("shape", SHAPES)
def test_carried_cell_resolves_once_from_its_material(shape: str) -> None:
    """A carried cell starts derived, answers its resolver once and then holds the value."""

    store = _store(shape)
    assert cell_state(store, 0, JOINED) == STATE_DERIVED
    assert CALLS == []
    value = store.cell_get(0, JOINED)
    assert value == [("m", 0), "p0"]
    assert store.cell_get(0, JOINED) is value
    assert CALLS == [("joined", 0, ("m", 0))]
    assert cell_state(store, 0, JOINED) == STATE_SET
    if len(store) > 1:
        # Other rows stay unresolved: a carried family is per cell.
        assert cell_state(store, 1, JOINED) == STATE_DERIVED
        assert store.cell_get(1, JOINED) == [("m", 1), "p1"]


@pytest.mark.parametrize("shape", ("fork", "fork_columnar", "fork_of_fork"))
def test_fork_view_answer_stays_in_the_fork(shape: str) -> None:
    """A fork answers into its own overlay; the base cell stays carried."""

    view = _store(shape)
    base = view.base
    while isinstance(base, OpStoreView):
        base = base.base
    forked = view.cell_get(0, JOINED)
    assert forked == [("m", 0), "p0"]
    assert cell_state(base, 0, JOINED) == STATE_DERIVED
    original = base.cell_get(0, JOINED)
    assert original == forked and original is not forked


@pytest.mark.parametrize("shape", SHAPES)
def test_state_stream_answers_without_writing_back(shape: str) -> None:
    """``items`` (what a save and a record pickle read) answers and leaves the cell carried."""

    store = _store(shape)
    items = dict(store.items(0))
    assert items["joined"] == [("m", 0), "p0"]
    assert cell_state(store, 0, JOINED) == STATE_DERIVED


@pytest.mark.parametrize("shape", SHAPES)
def test_raw_read_returns_the_carried_cell(shape: str) -> None:
    """A raw read neither resolves nor hides the cell; ``material_of`` peeks at it."""

    store = _store(shape)
    raw = store.cell_get(0, JOINED, True)
    assert raw.__class__ is LazyCell
    assert FAMILY.material_of(raw) == ("m", 0)
    assert FAMILY.material_of("not carried") is _MISSING
    assert CALLS == []


@pytest.mark.parametrize("shape", SHAPES)
def test_delete_and_write_before_read_are_final(shape: str) -> None:
    """Deleting or overwriting a carried cell never runs its resolver."""

    store = _store(shape)
    assert store.cell_del(0, JOINED) is True
    assert store.cell_get(0, JOINED) is _MISSING
    if len(store) > 1:
        store.cell_set(1, JOINED, "written")
        assert store.cell_get(1, JOINED) == "written"
    assert CALLS == []


def test_freeze_keeps_carried_cells_and_interning_skips_them() -> None:
    """The freeze seam's relation-view interning leaves a carried cell carried."""

    base = OpRowStore(LAYOUT)
    for row in range(SMALL_ROWS):
        base.adopt_row(_row(row))
    rows = base.rows_building()
    convert_row_cells(
        rows,
        LAYOUT.fid_by_name,
        {},
        tuple_fields=("shape", "joined"),
        frozenset_fields=(),
    )
    assert rows[0][SHAPE] == ("s", 0)
    assert rows[0][JOINED].__class__ is LazyCell
    base.freeze()
    assert cell_state(base, 0, JOINED) == STATE_DERIVED
    assert base.cell_get(0, JOINED) == [("m", 0), "p0"]


def test_columnar_freeze_keeps_carried_cells() -> None:
    """The columnar transpose stores carried cells as objects, unresolved."""

    store = _store("columnar")
    for row in (0, COLUMNAR_ROWS - 1):
        assert cell_state(store, row, JOINED) == STATE_DERIVED
    assert CALLS == []
    assert store.cell_get(COLUMNAR_ROWS - 1, JOINED)[1] == f"p{COLUMNAR_ROWS - 1}"


def test_compaction_leaves_carried_cells_alone() -> None:
    """Compaction's pooling pass skips a carried cell without answering it."""

    base = OpRowStore(LAYOUT)
    for row in range(SMALL_ROWS):
        base.adopt_row(_row(row))
    _compact_store_rows(base, {})
    for row in range(SMALL_ROWS):
        assert cell_state(base, row, JOINED) == STATE_DERIVED
    assert CALLS == []


@pytest.mark.parametrize("shape", SHAPES)
def test_unresolved_store_cells_pickle_with_their_material(shape: str) -> None:
    """Pickling an unresolved cell carries its material; the copy resolves the same."""

    store = _store(shape)
    raw = store.cell_get(0, JOINED, True)
    restored = pickle.loads(pickle.dumps(raw))
    assert restored.__class__ is LazyCell
    assert restored.state == STATE_DERIVED
    assert restored.material == ("m", 0)
    assert restored.source is _join
    detached = DetachedOpStore(LAYOUT)
    detached.adopt_row(["p0", None, restored])
    assert detached.cell_get(0, JOINED) == [("m", 0), "p0"]


def test_detached_row_pickles_unresolved() -> None:
    """A whole detached row holding a carried cell round-trips through pickle."""

    cells = pickle.loads(pickle.dumps(_row(3)))
    detached = DetachedOpStore(LAYOUT)
    detached.adopt_row(cells)
    assert cell_state(detached, 0, JOINED) == STATE_DERIVED
    assert detached.cell_get(0, JOINED) == [("m", 3), "p3"]


def test_detached_twin_copied_through_reads_resolves_in_the_source() -> None:
    """Copying a row by ordinary reads (``detach_record``'s way) answers each carried cell."""

    base = _store("sealed")
    cells = [base.cell_get(1, fid) for fid in range(LAYOUT.n_fields)]
    twin = DetachedOpStore(LAYOUT)
    twin.adopt_row(cells)
    assert twin.cell_get(0, JOINED) is base.cell_get(1, JOINED)
    assert CALLS == [("joined", 1, ("m", 1))]


def test_copies_share_the_carried_cell() -> None:
    """``copy`` and ``deepcopy`` of a row keep the one carried cell, like a shared marker."""

    cell = FAMILY.carry("joined", ("m", 9))
    assert copy.copy(cell) is cell
    assert copy.deepcopy([cell])[0] is cell


def test_shared_markers_still_refuse_to_pickle() -> None:
    """Only carried cells pickle; a registry's shared marker never enters a stream."""

    fields = LazyFields(NAMES)
    fields.register_default("shape", factory=list)
    with pytest.raises(TypeError, match="storage marker"):
        pickle.dumps(fields.cell("shape"))


def test_family_refuses_bad_registrations_and_material() -> None:
    """A family needs fields, may not be called 'default', and a cell needs material."""

    with pytest.raises(ValueError, match="no fields"):
        CarriedFamily("empty", (), _join)
    with pytest.raises(ValueError, match="default"):
        CarriedFamily("default", ("joined",), _join)
    with pytest.raises(KeyError):
        FAMILY.carry("plain", 1)
    with pytest.raises(ValueError, match="raw material"):
        FAMILY.carry("joined", _MISSING)


def test_resolver_that_returns_no_value_raises() -> None:
    """A resolver lacking its material must raise; returning the sentinel is refused."""

    family = CarriedFamily("lacking", ("joined",), _lacking)
    detached = DetachedOpStore(LAYOUT)
    detached.adopt_row(["p0", None, family.carry("joined", "m")])
    with pytest.raises(RuntimeError, match="returned no value"):
        detached.cell_get(0, JOINED)
