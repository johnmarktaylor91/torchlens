"""Deterministic call counters for the freeze-seam container pooling sweep.

Most container cells are empty (``[]``, ``{}``, ``set()``); the sweep used to
run the alias census helper and the pool-key builder on each one, and to build
a full content key for cells whose own container is aliased in the store
(which the alias guard then refused). Empty cells now take an inline path and
aliased cells are refused before keying; the pooled result is unchanged.
"""

from __future__ import annotations

import sys
from typing import Any

from torchlens._data_classes import op as op_mod
from torchlens._trace_core.op_store import _MISSING, OpRowStore, OpStoreLayout, PooledCell

_HELPERS = {
    op_mod._count_container_ids.__code__: "census",
    op_mod._container_pool_key.__code__: "key",
}


def _store(rows: list[list[Any]]) -> OpRowStore:
    store = OpRowStore(OpStoreLayout(("a", "b", "c")))
    for cells in rows:
        store.adopt_row(cells)
    return store


def _pool_counting(stores: list[tuple[Any, Any]]) -> dict[str, int]:
    counts = {"census": 0, "key": 0}

    def profile(frame: Any, event: str, arg: Any) -> None:
        if event == "call":
            name = _HELPERS.get(frame.f_code)
            if name is not None:
                counts[name] += 1

    sys.setprofile(profile)
    try:
        op_mod._pool_container_cells(stores, {})
    finally:
        sys.setprofile(None)
    return counts


def test_empty_container_cells_call_no_helper() -> None:
    """N rows of empty list/dict/set cells: zero census and key helper calls, one pool per class."""

    n_rows = 50
    store = _store([[[], {}, set()] for _ in range(n_rows)])
    counts = _pool_counting([(store, (0, 1, 2))])
    assert counts == {"census": 0, "key": 0}
    rows = store.rows_building()
    assert rows is not None
    for fid, empty in enumerate(([], {}, set())):
        cells = {id(row[fid]) for row in rows}
        assert len(cells) == 1, "every empty of one class shares one pooled cell"
        cell = rows[0][fid]
        assert cell.__class__ is PooledCell
        assert cell.hydrate() == empty and type(cell.hydrate()) is type(empty)


def test_aliased_container_cells_are_refused_before_keying() -> None:
    """One non-empty list held by every row: no key is built and nothing pools."""

    shared = ["x", "y"]
    store = _store([[shared, _MISSING, _MISSING] for _ in range(20)])
    counts = _pool_counting([(store, (0,))])
    assert counts["key"] == 0
    rows = store.rows_building()
    assert rows is not None
    assert all(row[0] is shared for row in rows)


def test_aliased_empty_container_is_not_pooled() -> None:
    """An empty container reachable from two cells keeps its alias (never pooled)."""

    shared: list[Any] = []
    holder = {"k": shared}
    store = _store([[shared, holder, _MISSING], [[], {}, _MISSING], [[], {}, _MISSING]])
    op_mod._pool_container_cells([(store, (0, 1))], {})
    rows = store.rows_building()
    assert rows is not None
    assert rows[0][0] is shared and rows[0][1] is holder
    assert rows[1][0].__class__ is PooledCell and rows[1][0] is rows[2][0]


def test_kind_table_singleton_compaction_unchanged() -> None:
    """Kind tables (``fids=None``) still compact one-string lists and pool empties."""

    store = _store([[["only"], [], {}], [["other"], [], {}]])
    op_mod._pool_container_cells([(store, None)], {})
    rows = store.rows_building()
    assert rows is not None
    assert rows[0][0] == "only" and rows[1][0] == "other"
    assert store.compacted_singleton(0, 0, rows[0][0])
    assert rows[0][1] is rows[1][1] and rows[0][1].__class__ is PooledCell
