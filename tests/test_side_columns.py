"""Unit suite for side-column families: derived fields whose raw values live beside the store.

``SideColumnFamily`` (``torchlens/_trace_core/side_columns.py``) gives each
field one shared ``LazyCell`` marker and keeps every row's raw value in one
side column per field, indexed by row, with one context per store. These
tests cover every store shape (building, sealed, columnar, fork views, the
detached row), the freeze seam and its relation-view interning, compaction,
the state stream's pending cells and pickling.
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
    LazyCell,
    LazyFields,
    cell_state,
)
from torchlens._trace_core.op_store import DetachedOpStore, OpRowStore, OpStoreLayout, OpStoreView
from torchlens._trace_core.relation_views import convert_row_cells
from torchlens._trace_core.side_columns import (
    PendingCell,
    SideColumnFamily,
    answer_pending,
    pending_cell,
)

NAMES = ("plain", "shape", "joined_probe")
PLAIN, SHAPE, JOINED = range(3)
FIELD = "joined_probe"
LAYOUT = OpStoreLayout(NAMES)
#: Row count over the columnar-transpose threshold (512).
COLUMNAR_ROWS = 600
SMALL_ROWS = 4
SHAPES = ("building", "sealed", "columnar", "fork", "fork_columnar", "fork_of_fork")
CALLS: list[tuple[Any, ...]] = []
CONTEXT = ("ctx", "shared")


def _join(raw: Any, context: Any, name: str) -> list[Any]:
    """Module-level resolver (a pickled pending cell names it): the raw value plus the context."""

    CALLS.append((name, raw))
    return [raw, context[1]]


def _lacking(raw: Any, context: Any, name: str) -> Any:
    """Resolver that wrongly returns the unset sentinel."""

    return _MISSING


FAMILY = SideColumnFamily("side_probe", (FIELD,), _join)
LACKING = SideColumnFamily("side_lacking", (FIELD,), _lacking)
MARKER = FAMILY.markers[FIELD]


@pytest.fixture(autouse=True)
def _clear_calls() -> Any:
    """Start every test with an empty resolver log."""

    CALLS.clear()
    yield
    CALLS.clear()


def _fill(base: OpRowStore, n_rows: int, family: SideColumnFamily = FAMILY) -> None:
    """Adopt ``n_rows`` rows whose ``joined_probe`` cell holds the shared marker."""

    side = family.bind(base, CONTEXT)
    for row in range(n_rows):
        marker = family.hold(side, FIELD, row, ("m", row))
        assert base.adopt_row([f"p{row}", ["s", row], marker]) == row


def _store(shape: str) -> Any:
    """Return a store of the named shape whose rows all hold the marker."""

    columnar = "columnar" in shape
    base = OpRowStore(LAYOUT)
    _fill(base, COLUMNAR_ROWS if columnar else SMALL_ROWS)
    if shape == "building":
        return base
    base.freeze()
    assert (base._rows is None) == columnar
    if shape in ("sealed", "columnar"):
        return base
    view = OpStoreView(base)
    return OpStoreView(view) if shape == "fork_of_fork" else view


@pytest.mark.parametrize("shape", SHAPES)
def test_side_cell_resolves_once_from_its_raw_value(shape: str) -> None:
    """A side cell starts derived, answers its resolver once and then holds the value."""

    store = _store(shape)
    assert cell_state(store, 0, JOINED) == STATE_DERIVED
    assert CALLS == []
    value = store.cell_get(0, JOINED)
    assert value == [("m", 0), "shared"]
    assert store.cell_get(0, JOINED) is value
    assert CALLS == [(FIELD, ("m", 0))]
    assert cell_state(store, 0, JOINED) == STATE_SET
    # Other rows stay unresolved: the family answers per cell.
    assert cell_state(store, 1, JOINED) == STATE_DERIVED
    assert store.cell_get(1, JOINED) == [("m", 1), "shared"]


@pytest.mark.parametrize("shape", ("columnar", "fork_columnar"))
def test_every_row_holds_the_one_shared_marker(shape: str) -> None:
    """An unread cell costs no object: every row holds the field's single marker."""

    store = _store(shape)
    for row in range(COLUMNAR_ROWS):
        assert store.cell_get(row, JOINED, True) is MARKER
    assert CALLS == []


@pytest.mark.parametrize("shape", ("fork", "fork_columnar", "fork_of_fork"))
def test_fork_view_answer_stays_in_the_fork(shape: str) -> None:
    """A fork reads its root store's side column and answers into its own overlay."""

    view = _store(shape)
    base = view.base
    assert not isinstance(base, OpStoreView)
    forked = view.cell_get(0, JOINED)
    assert forked == [("m", 0), "shared"]
    assert cell_state(base, 0, JOINED) == STATE_DERIVED
    original = base.cell_get(0, JOINED)
    assert original == forked and original is not forked


@pytest.mark.parametrize("shape", SHAPES)
def test_pending_cell_answers_without_writing_back(shape: str) -> None:
    """A pending cell (what the op state stream yields) answers and leaves the cell derived."""

    store = _store(shape)
    pending = pending_cell(store, 0, store.cell_get(0, JOINED, True))
    assert pending.__class__ is PendingCell
    assert repr(pending) == f"PendingCell(side_probe.{FIELD}, row=0)"
    assert CALLS == []
    assert answer_pending(pending) == [("m", 0), "shared"]
    assert pending == [("m", 0), "shared"]
    assert pending == pending_cell(store, 0, MARKER)
    assert cell_state(store, 0, JOINED) == STATE_DERIVED
    assert answer_pending("plain value") == "plain value"
    assert pending_cell(store, 0, "not a marker") is _MISSING


@pytest.mark.parametrize("shape", SHAPES)
def test_pending_cell_pickles_raw_value_and_context(shape: str) -> None:
    """Pickling a pending cell carries the raw value plus the context; the load answers it."""

    store = _store(shape)
    payload = pickle.dumps(pending_cell(store, 2, MARKER))
    assert CALLS == []
    assert pickle.loads(payload) == [("m", 2), "shared"]
    assert CALLS == [(FIELD, ("m", 2))]
    assert cell_state(store, 2, JOINED) == STATE_DERIVED


@pytest.mark.parametrize("shape", SHAPES)
def test_raw_peek_returns_the_held_value(shape: str) -> None:
    """``raw`` peeks at the value an unread cell would resolve from, without resolving."""

    store = _store(shape)
    assert FAMILY.raw(store, 1, FIELD) == ("m", 1)
    assert CALLS == []
    store.cell_get(1, JOINED)
    assert FAMILY.raw(store, 1, FIELD) is _MISSING


@pytest.mark.parametrize("shape", SHAPES)
def test_delete_and_write_before_read_are_final(shape: str) -> None:
    """Deleting or overwriting a side cell never runs its resolver."""

    store = _store(shape)
    assert store.cell_del(0, JOINED) is True
    assert store.cell_get(0, JOINED) is _MISSING
    store.cell_set(1, JOINED, "written")
    assert store.cell_get(1, JOINED) == "written"
    assert CALLS == []


def test_freeze_keeps_markers_and_interning_skips_them() -> None:
    """The freeze seam's relation-view interning leaves a marker in place."""

    base = OpRowStore(LAYOUT)
    _fill(base, SMALL_ROWS)
    rows = base.rows_building()
    convert_row_cells(
        rows,
        LAYOUT.fid_by_name,
        {},
        tuple_fields=("shape", FIELD),
        frozenset_fields=(),
    )
    assert rows[0][SHAPE] == ("s", 0)
    assert rows[0][JOINED] is MARKER
    base.freeze()
    assert cell_state(base, 0, JOINED) == STATE_DERIVED
    assert base.cell_get(0, JOINED) == [("m", 0), "shared"]


def test_columnar_freeze_keeps_side_cells() -> None:
    """The columnar transpose stores the marker, unresolved; row ids index the side column."""

    store = _store("columnar")
    for row in (0, COLUMNAR_ROWS - 1):
        assert cell_state(store, row, JOINED) == STATE_DERIVED
    assert CALLS == []
    assert store.cell_get(COLUMNAR_ROWS - 1, JOINED)[0] == ("m", COLUMNAR_ROWS - 1)


def test_compaction_leaves_side_cells_alone() -> None:
    """Compaction's pooling pass skips a marker without answering it."""

    base = OpRowStore(LAYOUT)
    _fill(base, SMALL_ROWS)
    _compact_store_rows(base, {})
    for row in range(SMALL_ROWS):
        assert cell_state(base, row, JOINED) == STATE_DERIVED
    assert CALLS == []
    assert base.cell_get(3, JOINED) == [("m", 3), "shared"]


def test_hold_column_adopts_the_row_indexed_list() -> None:
    """A whole column is adopted as one list; rows never written raise on read."""

    base = OpRowStore(LAYOUT)
    side = FAMILY.bind(base, CONTEXT)
    by_row = [("c", 0), _MISSING, ("c", 2)]
    marker = FAMILY.hold_column(side, FIELD, by_row)
    for row in range(3):
        base.adopt_row([f"p{row}", None, marker])
    assert side.columns[FIELD] is by_row
    assert base.cell_get(2, JOINED) == [("c", 2), "shared"]
    with pytest.raises(LookupError, match="no raw"):
        base.cell_get(1, JOINED)


def test_rebinding_a_new_context_starts_fresh_columns() -> None:
    """Binding the same store to a new context drops the old columns."""

    base = OpRowStore(LAYOUT)
    _fill(base, SMALL_ROWS)
    side = FAMILY.bind(base, ("ctx", "other"))
    assert side.columns == {}
    assert FAMILY.bind(base, side.context) is side
    with pytest.raises(LookupError, match="no raw"):
        base.cell_get(0, JOINED)


def test_detached_store_holds_no_side_columns() -> None:
    """A detached row cannot bind; a marker copied into one raises instead of guessing."""

    detached = DetachedOpStore(LAYOUT)
    with pytest.raises(TypeError):
        FAMILY.bind(detached, CONTEXT)
    detached.adopt_row(["p0", None, MARKER])
    with pytest.raises(LookupError, match="holds no side columns"):
        detached.cell_get(0, JOINED)
    with pytest.raises(LookupError):
        pickle.dumps(pending_cell(detached, 0, MARKER))


def test_detached_twin_copied_through_reads_resolves_in_the_source() -> None:
    """Copying a row by ordinary reads (``detach_record``'s way) answers each side cell."""

    base = _store("sealed")
    cells = [base.cell_get(1, fid) for fid in range(LAYOUT.n_fields)]
    twin = DetachedOpStore(LAYOUT)
    twin.adopt_row(cells)
    assert twin.cell_get(0, JOINED) is base.cell_get(1, JOINED)
    assert CALLS == [(FIELD, ("m", 1))]


def test_copies_share_the_marker_and_markers_refuse_to_pickle() -> None:
    """``copy`` and ``deepcopy`` keep the one marker; no marker enters a pickle stream."""

    assert copy.copy(MARKER) is MARKER
    assert copy.deepcopy([MARKER])[0] is MARKER
    with pytest.raises(TypeError, match="storage marker"):
        pickle.dumps(MARKER)
    fields = LazyFields(NAMES)
    fields.register_default("shape", factory=list)
    with pytest.raises(TypeError, match="storage marker"):
        pickle.dumps(fields.cell("shape"))


def test_family_refuses_bad_registrations() -> None:
    """A family needs fields, a fresh name, and may not be called 'default'."""

    with pytest.raises(ValueError, match="no fields"):
        SideColumnFamily("side_empty", (), _join)
    with pytest.raises(ValueError, match="already registered"):
        SideColumnFamily("default", (FIELD,), _join)
    with pytest.raises(ValueError, match="already registered"):
        SideColumnFamily("side_probe", (FIELD,), _join)
    assert MARKER.__class__ is LazyCell and MARKER.state == STATE_DERIVED


def test_resolver_that_returns_no_value_raises() -> None:
    """A resolver lacking its raw value must raise; returning the sentinel is refused."""

    base = OpRowStore(LAYOUT)
    _fill(base, 1, LACKING)
    with pytest.raises(RuntimeError, match="returned no value"):
        base.cell_get(0, JOINED)
