"""Unit contract for the M5 Op row store (torchlens/_trace_core/op_store.py)."""

import pytest

from torchlens._trace_core import op_store
from torchlens._trace_core.op_store import (
    _MISSING,
    DetachedOpStore,
    OpRowStore,
    OpStoreLayout,
)

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
