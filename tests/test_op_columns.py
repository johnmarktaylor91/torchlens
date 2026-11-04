"""Private row-indexed column view over the ``Op`` row store (postprocess fast path).

Two contracts. The counter contract: a field read or write through a bound
column accessor costs ONE Python-level call on a building store, against the
three to five the ``Op`` attribute facade pays (``__getattribute__``, the field
descriptor or overlay property, ``cell_get``). The semantic contract: every read
and write through the view equals the facade's, including ``PooledCell``
hydrate-on-read (a fresh container cached back into the reading row, stable
identity, per-row isolation), unset cells, relation-view normalization on
finished stores, the group/dataflow/fact/ancestor-bitset encodings of a finished
trace, and the step-audit read record.
"""

from __future__ import annotations

import functools
import sys
from collections import defaultdict
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes._op_columns import OpColumns, column_rows_by_label
from torchlens._data_classes.op import _OP_SLOT_NAMES, _OP_STORE_LAYOUT, Op, _OpField
from torchlens._trace_core.op_store import (
    _MISSING,
    DetachedOpStore,
    OpRowStore,
    PooledCell,
    begin_cell_write_audit,
    end_cell_write_audit,
)

_FID = _OP_STORE_LAYOUT.fid_by_name


def _plain_field_names() -> list[str]:
    """Return stored field names served by a plain generated ``_OpField``."""

    return [
        name
        for name in _OP_SLOT_NAMES
        if type(vars(Op)[name]) is _OpField and name not in {"out", "grad"}
    ]


_PLAIN_A, _PLAIN_B = _plain_field_names()[:2]


def _facade(store: Any, row: int) -> Op:
    """Return an ``Op`` facade bound to ``(store, row)`` without construction."""

    op = Op.__new__(Op)
    object.__setattr__(op, "_core", store)
    object.__setattr__(op, "_row", row)
    return op


def _store_with_rows(n_rows: int) -> OpRowStore:
    """Return a building op store with ``n_rows`` all-unset rows."""

    store = OpRowStore(_OP_STORE_LAYOUT)
    for _ in range(n_rows):
        store.new_row()
    return store


def _python_calls(fn: Callable[[], Any]) -> int:
    """Return the number of Python-level call events ``fn`` makes, itself excluded."""

    calls = 0

    def profiler(frame: Any, event: str, arg: Any) -> None:
        """Count Python function entries."""

        nonlocal calls
        if event == "call":
            calls += 1

    sys.setprofile(profiler)
    try:
        fn()
    finally:
        sys.setprofile(None)
    return calls - 1


# ---------------------------------------------------------------------------
# Counter contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["has_output_descendant", "output_descendants", _PLAIN_A])
def test_column_read_costs_one_python_call_against_facade(name: str) -> None:
    """A bound column read is one Python call; the facade pays at least three."""

    n_rows = 40
    store = _store_with_rows(n_rows)
    for row in range(n_rows):
        store.cell_set(row, _FID[name], {f"label_{row}"} if name == "output_descendants" else row)
    ops = [_facade(store, row) for row in range(n_rows)]
    get = OpColumns(store).getter(name)

    def facade_reads() -> None:
        """Read every row through the facade."""

        for op in ops:
            getattr(op, name)

    def column_reads() -> None:
        """Read every row through the bound column getter."""

        for row in range(n_rows):
            get(row)

    facade_calls = _python_calls(facade_reads)
    column_calls = _python_calls(column_reads)

    assert column_calls == n_rows, column_calls
    assert facade_calls >= 3 * n_rows, facade_calls
    assert [get(row) for row in range(n_rows)] == [getattr(op, name) for op in ops]


def test_column_write_costs_one_python_call_against_facade() -> None:
    """A bound column write is one Python call; the facade write pays three."""

    n_rows = 40
    store = _store_with_rows(n_rows)
    ops = [_facade(store, row) for row in range(n_rows)]
    put = OpColumns(store).setter("has_output_descendant")

    def facade_writes() -> None:
        """Write every row through the facade."""

        for op in ops:
            op.has_output_descendant = True

    def column_writes() -> None:
        """Write every row through the bound column setter."""

        for row in range(n_rows):
            put(row, False)

    facade_calls = _python_calls(facade_writes)
    column_calls = _python_calls(column_writes)

    assert column_calls == n_rows, column_calls
    assert facade_calls >= 3 * n_rows, facade_calls
    assert all(op.has_output_descendant is False for op in ops)


# ---------------------------------------------------------------------------
# PooledCell hydrate-on-read contract
# ---------------------------------------------------------------------------


def _nested_defaultdict() -> defaultdict:
    """Return a non-empty defaultdict whose member is a mutable list."""

    value: defaultdict = defaultdict(list)
    value["a"].append(1)
    return value


_PROTOTYPES: list[Callable[[], Any]] = [
    list,
    lambda: [1, 2],
    dict,
    lambda: {"a": 1},
    set,
    lambda: {"x", "y"},
    lambda: {"nested": [1, 2]},
    _nested_defaultdict,
]


@pytest.mark.parametrize("name", [_PLAIN_A, "output_descendants"])
@pytest.mark.parametrize("make", _PROTOTYPES)
def test_pooled_cell_read_matches_facade(name: str, make: Callable[[], Any]) -> None:
    """Pooled reads hydrate, cache back, stay row-isolated, exactly as the facade."""

    # Grade C (radon counts each assert): one straight-line scenario, one clause
    # of the PooledCell contract per assert; splitting it would scatter the contract.
    fid = _FID[name]
    sides = []
    for _ in range(2):
        store = _store_with_rows(3)
        pooled = PooledCell.from_value(make())
        for row in range(3):
            store.cell_set(row, fid, pooled)
        sides.append((store, pooled))
    (facade_store, facade_pool), (column_store, column_pool) = sides
    facade_op = _facade(facade_store, 0)
    get = OpColumns(column_store).getter(name)

    via_facade = getattr(facade_op, name)
    via_column = get(0)

    assert type(via_column) is type(via_facade)
    assert via_column == via_facade
    for store, value in ((facade_store, via_facade), (column_store, via_column)):
        assert store.cell_get(0, fid) is value, "the hydrated container is cached back"
        assert store.cell_get(1, fid).__class__ is PooledCell, "unread rows stay pooled"
    assert get(0) is via_column, "identity is stable across reads"
    assert getattr(facade_op, name) is via_facade

    _mutate(via_facade)
    _mutate(via_column)
    assert get(0) == getattr(facade_op, name)
    assert get(1) == getattr(_facade(facade_store, 1), name) == make()
    assert column_pool.prototype == facade_pool.prototype == make()
    if isinstance(via_column, dict) and via_column.get("nested") is not None:
        assert get(1)["nested"] is not via_column["nested"], "nested containers rebuild"


def _mutate(value: Any) -> None:
    """Mutate one hydrated container in place."""

    if isinstance(value, list):
        value.append("mutated")
    elif isinstance(value, set):
        value.add("mutated")
    else:
        value["mutated"] = True


# ---------------------------------------------------------------------------
# Unset cells, plain values, relation writes, refusals
# ---------------------------------------------------------------------------


def test_unset_cell_raises_like_facade_and_default_returns() -> None:
    """An unset cell raises the facade's AttributeError; a default getter returns it."""

    store = _store_with_rows(1)
    with pytest.raises(AttributeError) as facade_error:
        getattr(_facade(store, 0), _PLAIN_A)
    with pytest.raises(AttributeError) as column_error:
        OpColumns(store).getter(_PLAIN_A)(0)
    assert str(column_error.value) == str(facade_error.value)
    assert column_error.value.name == facade_error.value.name == _PLAIN_A

    sentinel = object()
    assert OpColumns(store).getter(_PLAIN_A, default=sentinel)(0) is sentinel
    assert getattr(_facade(store, 0), _PLAIN_A, sentinel) is sentinel
    assert store.cell_get(0, _FID[_PLAIN_A]) is _MISSING


def test_plain_value_is_returned_by_identity() -> None:
    """A plain stored value comes back as the same object the facade returns."""

    store = _store_with_rows(1)
    payload = ["kept", {"by": "identity"}]
    store.cell_set(0, _FID[_PLAIN_B], payload)
    assert OpColumns(store).getter(_PLAIN_B)(0) is getattr(_facade(store, 0), _PLAIN_B) is payload


@pytest.mark.parametrize("name", ["children", "output_descendants", "in_conditionals"])
def test_relation_writes_match_facade_on_building_and_detached_stores(name: str) -> None:
    """Building stores keep raw staging containers; finished stores normalize."""

    building_facade, building_column = _store_with_rows(1), _store_with_rows(1)
    staged = ["a", "b"]
    setattr(_facade(building_facade, 0), name, staged)
    OpColumns(building_column).setter(name)(0, staged)
    assert building_column.cell_get(0, _FID[name]) is staged
    assert building_facade.cell_get(0, _FID[name]) is staged

    detached_facade = DetachedOpStore(_OP_STORE_LAYOUT)
    detached_column = DetachedOpStore(_OP_STORE_LAYOUT)
    for store in (detached_facade, detached_column):
        store.adopt_row([_MISSING] * _OP_STORE_LAYOUT.n_fields)
    setattr(_facade(detached_facade, 0), name, ["a", "b"])
    OpColumns(detached_column).setter(name)(0, ["a", "b"])
    facade_cell = detached_facade.cell_get(0, _FID[name])
    column_cell = detached_column.cell_get(0, _FID[name])
    assert type(column_cell) is type(facade_cell)
    assert column_cell == facade_cell

    with pytest.raises(Exception) as facade_error:
        setattr(_facade(detached_facade, 0), name, 3)
    with pytest.raises(Exception) as column_error:
        OpColumns(detached_column).setter(name)(0, 3)
    assert type(column_error.value) is type(facade_error.value)
    assert str(column_error.value) == str(facade_error.value)


def test_view_refuses_fields_the_facade_intercepts() -> None:
    """Intercepted reads, guarded writes and unknown names stay facade-only."""

    columns = OpColumns(_store_with_rows(1))
    for name in ("out", "grad"):
        with pytest.raises(ValueError, match="facade"):
            columns.getter(name)
        with pytest.raises(ValueError, match="facade"):
            columns.setter(name)
    with pytest.raises(ValueError, match="stored Op field"):
        columns.getter("not_an_op_field")


def test_audited_store_records_column_reads_and_writes() -> None:
    """A step-audit window observes column reads and writes like facade ones."""

    store = _store_with_rows(2)
    store.cell_set(0, _FID[_PLAIN_A], 1)
    columns = OpColumns(store)
    begin_cell_write_audit(store, record_reads=True)
    try:
        assert columns.getter(_PLAIN_A)(0) == 1
        columns.setter(_PLAIN_B)(1, "written")
    finally:
        result = end_cell_write_audit(store)
    assert _PLAIN_A in result.read_columns
    assert _PLAIN_B in result.written_columns
    assert store.cell_get(1, _FID[_PLAIN_B]) == "written"


# ---------------------------------------------------------------------------
# Finished-trace parity: every field of every row, fork against fork
# ---------------------------------------------------------------------------


class _Branchy(nn.Module):
    """Small model with fan-out, a parameter reuse and an output tuple."""

    def __init__(self) -> None:
        """Build two linear layers."""

        super().__init__()
        self.first = nn.Linear(4, 4)
        self.second = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a reused-layer branch and a residual branch."""

        hidden = torch.relu(self.first(x))
        return self.second(hidden), self.first(hidden) + x


@pytest.fixture
def branchy_trace() -> Iterator[tl.Trace]:
    """Yield a finished trace of ``_Branchy`` and release it afterwards."""

    torch.manual_seed(0)
    trace = tl.trace(_Branchy(), torch.randn(2, 4))
    try:
        yield trace
    finally:
        trace.cleanup()


def _outcome(read: Callable[..., Any], *args: Any) -> tuple[str, Any]:
    """Return ``("value", read(*args))`` or ``("raise", (type, message))``."""

    try:
        return "value", read(*args)
    except Exception as exc:  # the facade's own refusal shapes are compared
        return "raise", (type(exc), str(exc))


def _same(left: Any, right: Any) -> bool:
    """Return whether two read results are equal (tensors by value)."""

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return type(left) is type(right) and torch.equal(left, right)
    if type(left) is not type(right):
        return False
    try:
        return bool(left == right)
    except Exception:  # objects without a usable equality compare by repr
        return repr(left) == repr(right)


def _field_mismatch(ops: tuple[Any, Any, Any], read_column: Callable[[], Any], name: str) -> str:
    """Return why one field read differs, ``"fork-owned"`` or ``""`` when it matches.

    ``ops`` is ``(fork A op, fork B op, fork C op)``: A and C read through the
    facade, ``read_column`` reads fork B's row through the view.
    """

    facade_op, column_op, control_op = ops
    kind_f, facade_value = _outcome(getattr, facade_op, name)
    kind_c, column_value = _outcome(read_column)
    kind_o, control_value = _outcome(getattr, control_op, name)
    if kind_f != kind_c:
        return f"{kind_f} vs {kind_c}"
    if kind_f == kind_o and _same(facade_value, control_value):
        return "" if _same(facade_value, column_value) else "differs from facade"
    _kind_b, own_value = _outcome(getattr, column_op, name)
    if type(column_value) is not type(control_value):
        return "fork-owned type differs"
    if own_value is column_value or _same(own_value, column_value):
        return "fork-owned"
    return "fork-owned differs from own facade"


def test_finished_trace_reads_match_facade_field_by_field(branchy_trace: tl.Trace) -> None:
    """Fork B read through the view equals fork A read through the facade.

    Fields whose facade reads already differ between two forks (fork-owned
    values such as the owning-trace reference) are detected with a third,
    facade-read fork C; for those the view must return the same type as the
    facade and exactly what fork B's own facade then reads.
    """

    forks = (branchy_trace.fork(), branchy_trace.fork(), branchy_trace.fork())
    facade_ops, column_ops, control_ops = (list(fork.ops) for fork in forks)
    assert len(facade_ops) == len(column_ops) == len(control_ops) > 3
    column_store = object.__getattribute__(column_ops[0], "_core")
    columns = OpColumns(column_store)
    names = [name for name in _OP_SLOT_NAMES if name not in {"out", "grad"}]
    getters = {name: columns.getter(name) for name in names}

    # Grade C (radon counts each assert and comprehension filter): the per-field
    # decision lives in ``_field_mismatch``; what remains is the fixture walk.
    mismatches = []
    fork_owned = set()
    for ops in zip(facade_ops, column_ops, control_ops, strict=True):
        row = object.__getattribute__(ops[1], "_row")
        assert object.__getattribute__(ops[1], "_core") is column_store
        for name in names:
            why = _field_mismatch(ops, functools.partial(getters[name], row), name)
            if why == "fork-owned":
                fork_owned.add(name)
            elif why:
                mismatches.append((row, name, why))
    assert not mismatches, mismatches[:10]
    assert len(fork_owned) < len(names) // 4, sorted(fork_owned)


def test_column_rows_by_label_maps_raw_labels_to_one_store() -> None:
    """The pass helper maps every raw label to its row on the shared store."""

    store = _store_with_rows(3)
    ops = {f"raw_{row}": _facade(store, row) for row in range(3)}
    mapped = column_rows_by_label(ops)
    assert mapped is not None
    columns, rows = mapped
    assert columns.store is store
    assert rows == {"raw_0": 0, "raw_1": 1, "raw_2": 2}

    ops["stray"] = _facade(_store_with_rows(1), 0)
    assert column_rows_by_label(ops) is None, "ops on two stores fall back to the facade"
