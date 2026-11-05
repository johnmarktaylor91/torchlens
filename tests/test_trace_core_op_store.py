"""Unit contract for the M5 Op row store (torchlens/_trace_core/op_store.py)."""

import collections
from typing import Any, NamedTuple

import pytest
import torch

from torchlens._trace_core import op_store
from torchlens._trace_core.eviction_walk import _EVICT_WALK_DEPTH_CEILING
from torchlens._trace_core.op_store import (
    _MISSING,
    DetachedOpStore,
    OpRowStore,
    OpStoreLayout,
)
from torchlens._utils._display_bounded import RecordTuple

LAYOUT = OpStoreLayout(("alpha", "beta", "gamma", "delta"))


@pytest.fixture(autouse=True)
def _always_transpose(monkeypatch):
    """Force the columnar transpose so tiny fixtures exercise the frozen path."""

    monkeypatch.setattr(op_store, "_TRANSPOSE_MIN_ROWS", 1)


def _store_with_rows(n_rows):
    store = OpRowStore(LAYOUT)
    for _ in range(n_rows):
        store.new_row()
    return store


class _Pair(NamedTuple):
    left: Any
    right: Any


class _ListSub(list):
    """A list subclass: evicted in place through its own item methods."""


class _LooseTuple(tuple):
    """A tuple subclass built from one iterable, so the rebuild takes the fallback."""


class _PinnedTuple(tuple):
    """A tuple subclass whose constructor takes three items: it cannot be rebuilt."""

    def __new__(cls, first: Any, second: Any, _tag: Any) -> "_PinnedTuple":
        return super().__new__(cls, (first, second))


class _PinnedSet(frozenset):
    """A frozenset subclass whose constructor takes two items: it cannot be rebuilt."""

    def __new__(cls, first: Any, second: Any) -> "_PinnedSet":
        return super().__new__(cls, (first, second))


class _Holder:
    """A custom object: a tensor inside it is the disclosed residual."""

    def __init__(self, payload: Any) -> None:
        self.payload = payload


class _Owner:
    """A weak-referenceable stand-in for an owning ``TraceCore``."""


def _assert_last_owner_evicts_nested_tensors() -> None:
    """The last owner's release evicts nested tensors in columns and the overlay."""

    tensors = [torch.full((1,), float(i)) for i in range(9)]
    store = _store_with_rows(3)
    base_cells = [
        tensors[0],
        _Pair(tensors[1], "meta"),
        {tensors[2], "x"},
        collections.OrderedDict(k=tensors[3]),
        frozenset({tensors[4]}),
        _ListSub([tensors[5], 1]),
        _LooseTuple((tensors[6], 2)),
        RecordTuple([tensors[7]], "Ops", "ops"),
    ]
    for index, value in enumerate(base_cells):
        store.cell_set(index // 4, index % 4, value)
    store.freeze()
    cyclic: list[Any] = [torch.ones(1)]
    cyclic.append(cyclic)
    deep: list[Any] = [tensors[8]]
    for _ in range(_EVICT_WALK_DEPTH_CEILING):
        deep = [deep]
    holder = _Holder(torch.ones(1))
    for fid, value in enumerate((cyclic, deep, holder, (1, "a"))):
        store.cell_set(2, fid, value)
    sub_cycle = _ListSub([torch.ones(1)])
    sub_cycle.append(sub_cycle)
    dict_cycle: collections.OrderedDict[str, Any] = collections.OrderedDict(t=torch.ones(1))
    dict_cycle["self"] = dict_cycle
    pinned = _PinnedTuple(torch.ones(1), 1, "tag")
    pinned_set = _PinnedSet(torch.ones(1), 1)
    ceiling_tuple: Any = (torch.ones(1),)
    ceiling_pair: Any = _Pair(torch.ones(1), 0)
    for _ in range(_EVICT_WALK_DEPTH_CEILING):
        ceiling_tuple, ceiling_pair = [ceiling_tuple], [ceiling_pair]
    later_cells = (
        sub_cycle,
        dict_cycle,
        pinned,
        pinned_set,
        ceiling_tuple,
        ceiling_pair,
        (tensors[0], "meta"),
        ((torch.ones(1),), 1),
    )
    for index, value in enumerate(later_cells):
        store.cell_set(index // 4, index % 4, value)

    owner = _Owner()
    store.adopt_payload_owner(owner)
    store.release_payload_owner(owner)

    assert store._payload_owner_count == 0
    base = [store._columns[index % 4].values[index // 4] for index in range(8)]
    assert base[0] is None
    assert type(base[1]) is _Pair and base[1] == (None, "meta")
    assert base[2] == {"x", None}
    assert type(base[3]) is collections.OrderedDict and base[3] == {"k": None}
    assert base[4] == frozenset({None})
    assert type(base[5]) is _ListSub and base[5] == [None, 1]
    assert type(base[6]) is _LooseTuple and base[6] == (None, 2)
    assert type(base[7]) is RecordTuple and base[7] == (None,) and base[7].title == "Ops"
    assert sub_cycle[0] is None and sub_cycle[1] is sub_cycle
    assert dict_cycle["t"] is None and dict_cycle["self"] is dict_cycle
    assert store.cell_get(0, 2) is pinned and isinstance(pinned[0], torch.Tensor), (
        "a tuple subclass that cannot be rebuilt keeps its payload"
    )
    assert store.cell_get(0, 3) is pinned_set
    ceiling_tuple_cell, ceiling_pair_cell = store.cell_get(1, 0), store.cell_get(1, 1)
    for _ in range(_EVICT_WALK_DEPTH_CEILING):
        ceiling_tuple_cell, ceiling_pair_cell = ceiling_tuple_cell[0], ceiling_pair_cell[0]
    assert isinstance(ceiling_tuple_cell[0], torch.Tensor)
    assert isinstance(ceiling_pair_cell.left, torch.Tensor)
    assert store.cell_get(1, 2) == (None, "meta")
    assert store.cell_get(1, 3) == ((None,), 1)
    evicted_cycle = store.cell_get(2, 0)
    assert evicted_cycle[0] is None and evicted_cycle[1] is evicted_cycle
    innermost = store.cell_get(2, 1)
    for _ in range(_EVICT_WALK_DEPTH_CEILING):
        innermost = innermost[0]
    assert innermost[0] is tensors[8], "cells past the depth ceiling keep their payload"
    assert store.cell_get(2, 2).payload is holder.payload
    assert store.cell_get(2, 3) == (1, "a")


class TestFrozenPhase:
    @pytest.mark.smoke
    def test_subclass_and_mixed_values_stay_exact_type(self):
        class Marked(int):
            pass

        store = _store_with_rows(2)
        store.cell_set(0, 1, Marked(3))
        store.cell_set(1, 1, Marked(4))
        store.cell_set(0, 0, 1)
        store.cell_set(1, 0, 1.0)
        store.freeze()
        assert type(store.cell_get(0, 1)) is Marked
        assert type(store.cell_get(0, 0)) is int
        assert type(store.cell_get(1, 0)) is float

    @pytest.mark.smoke
    def test_post_freeze_object_delete_releases_the_cell(self):
        store = _store_with_rows(2)
        big = object()
        store.cell_set(0, 2, big)
        store.cell_set(1, 2, "keep")
        store.freeze()
        assert store.cell_del(0, 2) is True
        assert store.cell_get(0, 2) is _MISSING
        assert store.cell_get(1, 2) == "keep"
        column = store._columns[2]
        assert all(value is not big for value in column.values)
        _assert_last_owner_evicts_nested_tensors()

    @pytest.mark.smoke
    def test_post_freeze_packed_delete_tombstones(self):
        store = _store_with_rows(1)
        store.cell_set(0, 1, 9)
        store.freeze()
        assert store.cell_del(0, 1) is True
        assert store.cell_get(0, 1) is _MISSING
        assert store.cell_del(0, 1) is False

    @pytest.mark.smoke
    def test_no_new_rows_after_freeze(self):
        store = _store_with_rows(1)
        store.freeze()
        with pytest.raises(RuntimeError):
            store.new_row()


class TestSmallStoreSeal:
    @pytest.mark.smoke
    def test_adopt_row_bulk_ingress(self):
        store = OpRowStore(LAYOUT)
        row = store.adopt_row(["a", _MISSING, None, 4])
        assert row == 0
        assert store.cell_get(0, 0) == "a"
        assert store.cell_get(0, 1) is _MISSING
        assert store.cell_get(0, 2) is None
        with pytest.raises(ValueError):
            store.adopt_row(["too", "short"])


class TestDetachedStore:
    @pytest.mark.smoke
    def test_single_row_roundtrip(self):
        store = DetachedOpStore(LAYOUT)
        store.cell_set(0, 0, "x")
        assert store.cell_get(0, 0) == "x"
        assert list(store.items(0)) == [("alpha", "x")]
        assert store.cell_del(0, 0) is True
        assert store.cell_del(0, 0) is False
        assert not store.frozen
