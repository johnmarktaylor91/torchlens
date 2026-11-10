"""Unit suite for the four cell states of the row stores.

``torchlens/_trace_core/cell_states.py`` gives a cell four states that are
never conflated: set, default-omitted, derived and deleted. No product code
registers a field yet, so these tests are the mechanism's only consumer.
They cover the three behaviours (a derived cell answered by its family
resolver, a whole column derived on first touch, a default answered for an
omitted cell) and, for every state, read, delete, repeated delete and read
after delete on the building store, the sealed store, the columnar store
with its overlay, the detached row, fork views and the state stream a save
writes.
"""

from __future__ import annotations

import copy
import pickle
from typing import Any

import pytest

from torchlens._trace_core.cell_states import (
    _MISSING,
    STATE_DEFAULT,
    STATE_DELETED,
    STATE_DERIVED,
    STATE_SET,
    LazyCell,
    LazyFields,
    cell_state,
)
from torchlens._trace_core.op_store import (
    DetachedOpStore,
    OpRowStore,
    OpStoreLayout,
    OpStoreView,
    begin_cell_write_audit,
    end_cell_write_audit,
)

NAMES = ("plain", "flag", "bag", "depth", "note", "tick")
PLAIN, FLAG, BAG, DEPTH, NOTE, TICK = range(6)
LAYOUT = OpStoreLayout(NAMES)
INITIAL_STATE = {
    "plain": STATE_SET,
    "flag": STATE_DEFAULT,
    "bag": STATE_DEFAULT,
    "depth": STATE_DERIVED,
    "note": STATE_DERIVED,
    "tick": STATE_DERIVED,
}
#: Row count over the columnar-transpose threshold (512).
COLUMNAR_ROWS = 600
SMALL_ROWS = 4
SHAPES = (
    "building",
    "sealed",
    "columnar",
    "columnar_shared",
    "detached",
    "fork",
    "fork_columnar",
    "fork_of_fork",
)
MULTI_ROW_SHAPES = tuple(shape for shape in SHAPES if shape != "detached")


class _Fixture:
    """One registry over ``NAMES`` whose resolvers record their calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.fields = LazyFields(NAMES)
        self.fields.register_default("flag", False)
        self.fields.register_default("bag", factory=list)
        self.fields.register_family("depths", ("depth",), self._depths, whole_column=True)
        self.fields.register_family("notes", ("note",), self._note)
        self.fields.register_family("ticks", ("tick",), self._tick, cache=False)

    def _depths(self, store: Any, name: str) -> list[Any]:
        """Whole-column resolver: one value per row, computed from ``plain``."""

        self.calls.append(("depths", name))
        return [("depth", row, store.cell_get(row, PLAIN)) for row in range(len(store))]

    def _note(self, store: Any, row: int, name: str) -> list[Any]:
        """Per-cell resolver with an object-valued (cached) answer."""

        self.calls.append(("note", row))
        return ["note", row]

    def _tick(self, store: Any, row: int, name: str) -> tuple[Any, ...]:
        """Per-cell resolver whose answer is not cached."""

        self.calls.append(("tick", row))
        return ("tick", row)

    def seed(self, row: int) -> list[Any]:
        """Return a seed row whose one eager cell is ``plain``."""

        cells = self.fields.seed_row()
        cells[PLAIN] = f"p{row}"
        return cells

    def base(self, n_rows: int) -> OpRowStore:
        """Return a building store of ``n_rows`` seeded rows."""

        store = OpRowStore(LAYOUT)
        for row in range(n_rows):
            store.adopt_row(self.seed(row))
        return store

    def store(self, shape: str) -> Any:
        """Return a store of the named shape."""

        if shape == "detached":
            detached = DetachedOpStore(LAYOUT)
            detached.adopt_row(self.seed(0))
            return detached
        columnar = "columnar" in shape
        base = self.base(COLUMNAR_ROWS if columnar else SMALL_ROWS)
        if shape == "building":
            return base
        base.freeze()
        assert (base._rows is None) == columnar
        if shape in ("sealed", "columnar"):
            return base
        view = OpStoreView(base)
        if shape == "columnar_shared":
            # A base that has been forked tombstones deletes in its overlay.
            assert base._cow_shared
            return base
        if shape == "fork_of_fork":
            return OpStoreView(view)
        return view


@pytest.mark.parametrize("shape", SHAPES)
def test_seeded_cells_start_in_their_registered_state(shape: str) -> None:
    """A seed row starts each field in the state its registration names."""

    fx = _Fixture()
    store = fx.store(shape)
    for fid, name in enumerate(NAMES):
        assert cell_state(store, 0, fid) == INITIAL_STATE[name]
    assert fx.calls == []


@pytest.mark.parametrize("shape", SHAPES)
def test_reads_answer_each_state(shape: str) -> None:
    """Set, default-omitted and derived cells each answer their own way."""

    fx = _Fixture()
    store = fx.store(shape)

    assert store.cell_get(0, PLAIN) == "p0"

    # An immutable default is answered without a write.
    assert store.cell_get(0, FLAG) is False
    assert cell_state(store, 0, FLAG) == STATE_DEFAULT

    # A mutable default is created for the row and written back.
    bag = store.cell_get(0, BAG)
    assert bag == []
    assert store.cell_get(0, BAG) is bag
    assert cell_state(store, 0, BAG) == STATE_SET
    bag.append("mine")
    assert store.cell_get(0, BAG) == ["mine"]
    if len(store) > 1:
        other = store.cell_get(1, BAG)
        assert other == [] and other is not bag

    # A per-cell derived answer is cached, so identity is stable.
    note = store.cell_get(0, NOTE)
    assert note == ["note", 0]
    assert store.cell_get(0, NOTE) is note
    assert fx.calls.count(("note", 0)) == 1
    assert cell_state(store, 0, NOTE) == STATE_SET

    # An uncached family answers every read and stays derived.
    assert store.cell_get(0, TICK) == ("tick", 0)
    assert store.cell_get(0, TICK) == ("tick", 0)
    assert fx.calls.count(("tick", 0)) == 2
    assert cell_state(store, 0, TICK) == STATE_DERIVED

    # A whole-column family runs once, on the first touch of any row.
    last = len(store) - 1
    depth = store.cell_get(last, DEPTH)
    assert depth == ("depth", last, f"p{last}")
    assert fx.calls.count(("depths", "depth")) == 1
    for row in range(len(store)):
        assert cell_state(store, row, DEPTH) == STATE_SET
    assert store.cell_get(0, DEPTH) == ("depth", 0, "p0")
    assert store.cell_get(last, DEPTH) is depth
    assert fx.calls.count(("depths", "depth")) == 1


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("name", NAMES)
def test_delete_before_read_is_final(shape: str, name: str) -> None:
    """Deleting a cell in any state succeeds once and never answers it."""

    fx = _Fixture()
    store = fx.store(shape)
    fid = NAMES.index(name)

    assert store.cell_del(0, fid) is True
    assert fx.calls == []
    assert cell_state(store, 0, fid) == STATE_DELETED
    assert store.cell_get(0, fid) is _MISSING
    assert store.cell_del(0, fid) is False
    assert store.cell_get(0, fid) is _MISSING
    assert fx.calls == []

    # A whole-column pass triggered from another row leaves it deleted.
    if len(store) > 1:
        assert store.cell_get(1, fid) is not _MISSING
        assert store.cell_get(0, fid) is _MISSING
        assert store.cell_del(0, fid) is False

    # The state stream omits it, and reading the stream does not revive it.
    assert name not in dict(store.items(0))
    assert cell_state(store, 0, fid) == STATE_DELETED

    # Only an explicit write brings a deleted cell back.
    store.cell_set(0, fid, "again")
    assert store.cell_get(0, fid) == "again"
    assert cell_state(store, 0, fid) == STATE_SET
    assert store.cell_del(0, fid) is True
    assert store.cell_get(0, fid) is _MISSING


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("name", NAMES)
def test_delete_after_read_is_final(shape: str, name: str) -> None:
    """A cell that was answered deletes exactly like a stored one."""

    fx = _Fixture()
    store = fx.store(shape)
    fid = NAMES.index(name)

    value = store.cell_get(0, fid)
    assert value is not _MISSING and value.__class__ is not LazyCell
    assert store.cell_del(0, fid) is True
    calls = list(fx.calls)
    assert store.cell_get(0, fid) is _MISSING
    assert store.cell_del(0, fid) is False
    assert store.cell_get(0, fid) is _MISSING
    assert cell_state(store, 0, fid) == STATE_DELETED
    assert fx.calls == calls


@pytest.mark.parametrize("shape", SHAPES)
def test_release_row_deletes_every_state_without_answering(shape: str) -> None:
    """Op removal husking clears lazy cells as stored, never answering them."""

    fx = _Fixture()
    store = fx.store(shape)
    store.release_row(0)
    assert fx.calls == []
    for fid in range(len(NAMES)):
        assert cell_state(store, 0, fid) == STATE_DELETED
        assert store.cell_get(0, fid) is _MISSING
        assert store.cell_del(0, fid) is False
    assert dict(store.items(0)) == {}


@pytest.mark.parametrize("shape", MULTI_ROW_SHAPES)
def test_whole_column_fills_only_rows_still_derived(shape: str) -> None:
    """A row written or deleted before the first touch keeps its cell."""

    fx = _Fixture()
    store = fx.store(shape)
    store.cell_set(1, DEPTH, "mine")
    assert store.cell_del(2, DEPTH) is True

    assert store.cell_get(0, DEPTH) == ("depth", 0, "p0")
    assert fx.calls == [("depths", "depth")]
    assert store.cell_get(1, DEPTH) == "mine"
    assert store.cell_get(2, DEPTH) is _MISSING
    assert store.cell_del(2, DEPTH) is False
    assert store.cell_get(3, DEPTH) == ("depth", 3, "p3")
    assert fx.calls == [("depths", "depth")]


def test_whole_column_runs_again_for_rows_appended_later() -> None:
    """A building store derives a later row without touching filled ones."""

    fx = _Fixture()
    store = fx.base(2)
    first = store.cell_get(0, DEPTH)
    row = store.adopt_row(fx.seed(2))
    assert cell_state(store, row, DEPTH) == STATE_DERIVED
    assert store.cell_get(row, DEPTH) == ("depth", 2, "p2")
    assert fx.calls == [("depths", "depth"), ("depths", "depth")]
    assert store.cell_get(0, DEPTH) is first


def test_resolver_that_lacks_its_material_raises_and_stays_derived() -> None:
    """A failing resolver propagates; the cell stays derived, never defaulted."""

    fields = LazyFields(NAMES)

    def fail(store: Any, row: int, name: str) -> Any:
        raise LookupError("no raw material")

    fields.register_family("notes", ("note",), fail)
    fields.register_family("ticks", ("tick",), lambda store, row, name: _MISSING)
    fields.register_family("flags", ("flag",), lambda store, row, name: fields.cell("note"))
    fields.register_family("depths", ("depth",), lambda store, name: [1], whole_column=True)
    fields.register_family("bags", ("bag",), lambda store, name: [_MISSING, 2], whole_column=True)
    store = OpRowStore(LAYOUT)
    store.adopt_row(fields.seed_row())
    store.adopt_row(fields.seed_row())

    with pytest.raises(LookupError):
        store.cell_get(0, NOTE)
    with pytest.raises(RuntimeError, match="returned no value"):
        store.cell_get(0, TICK)
    with pytest.raises(RuntimeError, match="returned no value"):
        store.cell_get(0, FLAG)
    with pytest.raises(RuntimeError, match="returned 1 values"):
        store.cell_get(0, DEPTH)
    with pytest.raises(RuntimeError, match="returned no value"):
        store.cell_get(0, BAG)
    for fid in (NOTE, TICK, FLAG, DEPTH, BAG):
        assert cell_state(store, 0, fid) == STATE_DERIVED
    # Deleting a cell whose resolver would fail still succeeds once.
    assert store.cell_del(0, NOTE) is True
    assert store.cell_del(0, NOTE) is False


@pytest.mark.parametrize("columnar", [False, True])
def test_fork_answers_into_its_own_overlay(columnar: bool) -> None:
    """A fork's answers never reach the parent, and its resolver sees the fork."""

    fx = _Fixture()
    base = fx.base(COLUMNAR_ROWS if columnar else SMALL_ROWS)
    base.freeze()
    view = OpStoreView(base)

    fork_bag = view.cell_get(0, BAG)
    fork_bag.append("fork")
    assert cell_state(base, 0, BAG) == STATE_DEFAULT
    assert base.cell_get(0, BAG) == []
    assert view.cell_get(0, BAG) == ["fork"]

    fork_note = view.cell_get(0, NOTE)
    assert cell_state(base, 0, NOTE) == STATE_DERIVED
    assert base.cell_get(0, NOTE) is not fork_note

    view.cell_set(0, PLAIN, "forked")
    assert view.cell_get(0, DEPTH) == ("depth", 0, "forked")
    assert cell_state(base, 0, DEPTH) == STATE_DERIVED
    assert base.cell_get(0, DEPTH) == ("depth", 0, "p0")
    assert view.cell_get(0, DEPTH) == ("depth", 0, "forked")


@pytest.mark.parametrize("columnar", [False, True])
@pytest.mark.parametrize("name", NAMES)
def test_fork_carries_deletion_in_both_generations(columnar: bool, name: str) -> None:
    """A deletion made before a fork is deleted in the fork and in its fork."""

    fx = _Fixture()
    base = fx.base(COLUMNAR_ROWS if columnar else SMALL_ROWS)
    base.freeze()
    fid = NAMES.index(name)
    assert base.cell_del(0, fid) is True

    view = OpStoreView(base)
    grandchild = OpStoreView(view)
    for store in (view, grandchild):
        assert store.cell_get(0, fid) is _MISSING
        assert store.cell_del(0, fid) is False
        assert name not in dict(store.items(0))

    # A deletion in the fork stays in the fork and reaches its own forks.
    assert view.cell_del(1, fid) is True
    assert base.cell_get(1, fid) is not _MISSING
    late = OpStoreView(view)
    assert late.cell_get(1, fid) is _MISSING
    assert late.cell_del(1, fid) is False

    # A deletion in the parent after the fork does not reach the fork.
    assert base.cell_del(2, fid) is True
    assert view.cell_get(2, fid) is not _MISSING
    assert fx.calls == [] or name in ("depth", "note", "tick")


def test_default_created_after_a_fork_is_isolated_from_the_next_fork() -> None:
    """A mutable default written back registers for later fork isolation."""

    fx = _Fixture()
    base = fx.base(SMALL_ROWS)
    base.freeze()
    OpStoreView(base).isolate_mutable_cells()  # builds the fork-isolation index

    bag = base.cell_get(0, BAG)
    bag.append("before")
    view = OpStoreView(base)
    view.isolate_mutable_cells()
    bag.append("after")
    assert view.cell_get(0, BAG) == ["before"]
    assert base.cell_get(0, BAG) == ["before", "after"]


@pytest.mark.parametrize("shape", SHAPES)
def test_state_stream_carries_answers_and_never_markers(shape: str) -> None:
    """A save writes the answered values and does not materialize the row."""

    fx = _Fixture()
    store = fx.store(shape)
    state = dict(store.items(0))

    assert state == {
        "plain": "p0",
        "flag": False,
        "bag": [],
        "depth": ("depth", 0, "p0"),
        "note": ["note", 0],
        "tick": ("tick", 0),
    }
    assert cell_state(store, 0, FLAG) == STATE_DEFAULT
    assert cell_state(store, 0, BAG) == STATE_DEFAULT
    assert cell_state(store, 0, NOTE) == STATE_DERIVED
    assert cell_state(store, 0, TICK) == STATE_DERIVED
    assert pickle.loads(pickle.dumps(state)) == state


@pytest.mark.parametrize("shape", SHAPES)
def test_state_stream_carries_deletion(shape: str) -> None:
    """A deleted cell is absent from the stream, so a restored row has it unset."""

    fx = _Fixture()
    store = fx.store(shape)
    for fid in range(1, len(NAMES)):
        assert store.cell_del(0, fid) is True
    state = pickle.loads(pickle.dumps(dict(store.items(0))))
    assert state == {"plain": "p0"}
    assert fx.calls == []

    restored = DetachedOpStore(LAYOUT)
    restored.adopt_row([state.get(name, _MISSING) for name in NAMES])
    for fid in range(1, len(NAMES)):
        assert restored.cell_get(0, fid) is _MISSING
        assert restored.cell_del(0, fid) is False


def test_markers_are_shared_stable_and_never_serialized() -> None:
    """A marker survives row copies as itself and refuses to enter a stream."""

    fx = _Fixture()
    marker = fx.fields.cell("depth")
    assert marker is fx.fields.seed_row()[DEPTH]
    assert copy.copy(marker) is marker
    assert copy.deepcopy([marker])[0] is marker
    assert "0x" not in repr(marker)
    assert repr(marker) == "<lazy derived cell 'depth' of family 'depths'>"
    with pytest.raises(TypeError, match="never enters a state stream"):
        pickle.dumps(marker)


def test_registry_refuses_ambiguous_registrations() -> None:
    """One marker per field, and a shared default must be immutable."""

    fields = LazyFields(NAMES)
    fields.register_default("flag", False)
    assert fields.register_default("note", ("a", frozenset({1}))).source == ("a", frozenset({1}))
    with pytest.raises(ValueError, match="already registered"):
        fields.register_default("flag", True)
    with pytest.raises(ValueError, match="already registered"):
        fields.register_family("flags", ("flag",), lambda store, row, name: 1)
    with pytest.raises(KeyError):
        fields.register_default("absent", 0)
    with pytest.raises(TypeError, match="register a factory"):
        fields.register_default("bag", [])
    with pytest.raises(TypeError, match="register a factory"):
        fields.register_default("bag", ([],))
    with pytest.raises(ValueError, match="not both"):
        fields.register_default("bag", 1, factory=list)
    with pytest.raises(ValueError, match="registers no fields"):
        fields.register_family("empty", (), lambda store, row, name: 1)
    fields.register_family("depths", ("depth", "tick"), lambda store, name: [], whole_column=True)
    with pytest.raises(ValueError, match="already registered"):
        fields.register_family("depths", ("bag",), lambda store, name: [], whole_column=True)
    assert fields.family_fields("depths") == ("depth", "tick")
    assert fields.cell("depth").whole_column and fields.cell("depth").cache

    first = fields.seed_row()
    second = fields.seed_row()
    assert first == second and first is not second
    assert first[PLAIN] is _MISSING and first[BAG] is _MISSING
    assert first[FLAG] is fields.cell("flag")


@pytest.mark.parametrize("record_reads", [False, True])
def test_answering_a_read_is_not_a_step_write(record_reads: bool) -> None:
    """The step audit records the reads an answer makes, never a write."""

    fx = _Fixture()
    store = fx.base(SMALL_ROWS)
    begin_cell_write_audit(store, record_reads=record_reads)
    try:
        store.cell_get(0, NOTE)
        store.cell_get(0, DEPTH)
        store.cell_get(0, BAG)
        assert store.cell_get(0, FLAG) is False
    finally:
        result = end_cell_write_audit(store)
    assert result.written_columns == set()
    assert result.effective_write_columns == set()
    if record_reads:
        assert result.read_columns == {"note", "depth", "bag", "flag", "plain"}
    assert cell_state(store, 0, NOTE) == STATE_SET
    assert cell_state(store, 1, DEPTH) == STATE_SET


def test_mutating_a_created_default_is_still_a_step_write() -> None:
    """A default created and then changed in place reads as a write of its column."""

    fx = _Fixture()
    store = fx.base(SMALL_ROWS)
    begin_cell_write_audit(store, record_reads=True)
    try:
        store.cell_get(0, BAG).append("written")
        store.cell_set(1, NOTE, "explicit")
    finally:
        result = end_cell_write_audit(store)
    assert result.written_columns == {"bag", "note"}
    assert result.effective_write_columns == {"bag", "note"}


def test_op_facade_reads_and_deletes_lazy_cells_like_stored_ones() -> None:
    """Through the ``Op`` descriptors a lazy cell behaves as a stored field.

    ``del op.field`` succeeds once, a second delete raises, and a read after
    it raises ``AttributeError``: the deleted field never answers its default
    or its resolver again.
    """

    from torchlens._data_classes.op import _OP_STORE_LAYOUT, Op, _OpField

    plain = [
        name
        for name in _OP_STORE_LAYOUT.names
        if vars(Op).get(name).__class__ is _OpField and not name.startswith("_")
    ]
    default_name, factory_name, derived_name, column_name = plain[:4]
    calls: list[Any] = []
    fields = LazyFields(_OP_STORE_LAYOUT.names)
    fields.register_default(default_name, "the default")
    fields.register_default(factory_name, factory=dict)

    def resolve(store: Any, row: int, name: str) -> Any:
        calls.append(name)
        return [name, row]

    def resolve_column(store: Any, name: str) -> Any:
        calls.append(name)
        return [(name, row) for row in range(len(store))]

    fields.register_family("derived", (derived_name,), resolve)
    fields.register_family("columns", (column_name,), resolve_column, whole_column=True)

    store = DetachedOpStore(_OP_STORE_LAYOUT)
    store.adopt_row(fields.seed_row())
    op = Op.__new__(Op)
    object.__setattr__(op, "_core", store)
    object.__setattr__(op, "_row", 0)

    assert getattr(op, default_name) == "the default"
    created = getattr(op, factory_name)
    assert created == {} and getattr(op, factory_name) is created
    derived = getattr(op, derived_name)
    assert derived == [derived_name, 0] and getattr(op, derived_name) is derived
    assert getattr(op, column_name) == (column_name, 0)
    assert calls == [derived_name, column_name]

    for name in (default_name, factory_name, derived_name, column_name):
        delattr(op, name)
        with pytest.raises(AttributeError):
            delattr(op, name)
        with pytest.raises(AttributeError):
            getattr(op, name)
        assert not hasattr(op, name)
    assert calls == [derived_name, column_name]

    # Deleted before any read: the resolver and the default are never consulted.
    fresh_store = DetachedOpStore(_OP_STORE_LAYOUT)
    fresh_store.adopt_row(fields.seed_row())
    fresh = Op.__new__(Op)
    object.__setattr__(fresh, "_core", fresh_store)
    object.__setattr__(fresh, "_row", 0)
    for name in (default_name, factory_name, derived_name, column_name):
        delattr(fresh, name)
        with pytest.raises(AttributeError):
            delattr(fresh, name)
        with pytest.raises(AttributeError):
            getattr(fresh, name)
    assert calls == [derived_name, column_name]
