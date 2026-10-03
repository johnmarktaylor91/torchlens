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


class TestBuildingPhase:
    def test_new_row_starts_all_missing(self):
        store = _store_with_rows(2)
        assert store.cell_get(1, 0) is _MISSING
        assert list(store.items(1)) == []

    def test_set_get_roundtrip_preserves_identity(self):
        store = _store_with_rows(1)
        value = ["mutable", "container"]
        store.cell_set(0, 2, value)
        assert store.cell_get(0, 2) is value

    def test_none_is_a_real_value_distinct_from_missing(self):
        store = _store_with_rows(1)
        store.cell_set(0, 0, None)
        assert store.cell_get(0, 0) is None
        assert store.cell_del(0, 0) is True
        assert store.cell_get(0, 0) is _MISSING

    def test_delete_unset_cell_reports_false(self):
        store = _store_with_rows(1)
        assert store.cell_del(0, 3) is False

    def test_items_follow_layout_order_and_skip_missing(self):
        store = _store_with_rows(1)
        store.cell_set(0, 3, "d")
        store.cell_set(0, 0, "a")
        assert list(store.items(0)) == [("alpha", "a"), ("delta", "d")]


class TestFrozenPhase:
    def test_freeze_packs_uniform_exact_numerics(self):
        store = _store_with_rows(3)
        for row, (flag, count, ratio) in enumerate(
            [(True, 7, 0.5), (False, -2, 1.25), (True, 10**12, -0.0)]
        ):
            store.cell_set(row, 0, flag)
            store.cell_set(row, 1, count)
            store.cell_set(row, 2, ratio)
        store.freeze()
        assert store.frozen
        assert store.cell_get(0, 0) is True
        assert type(store.cell_get(1, 1)) is int
        assert store.cell_get(2, 1) == 10**12
        value = store.cell_get(2, 2)
        assert type(value) is float and value.hex() == (-0.0).hex()

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

    def test_none_bearing_column_stays_object_backed(self):
        store = _store_with_rows(2)
        store.cell_set(0, 1, 5)
        store.cell_set(1, 1, None)
        store.freeze()
        assert store.cell_get(0, 1) == 5
        assert store.cell_get(1, 1) is None

    def test_missing_cells_survive_freeze(self):
        store = _store_with_rows(2)
        store.cell_set(0, 1, 5)
        store.freeze()
        assert store.cell_get(1, 1) is _MISSING
        assert store.cell_get(0, 1) == 5

    def test_post_freeze_writes_land_in_overlay(self):
        store = _store_with_rows(1)
        store.cell_set(0, 0, "base")
        store.freeze()
        store.cell_set(0, 0, "over")
        assert store.cell_get(0, 0) == "over"

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

    def test_post_freeze_packed_delete_tombstones(self):
        store = _store_with_rows(1)
        store.cell_set(0, 1, 9)
        store.freeze()
        assert store.cell_del(0, 1) is True
        assert store.cell_get(0, 1) is _MISSING
        assert store.cell_del(0, 1) is False

    def test_no_new_rows_after_freeze(self):
        store = _store_with_rows(1)
        store.freeze()
        with pytest.raises(RuntimeError):
            store.new_row()

    def test_shared_object_identity_survives_freeze(self):
        store = _store_with_rows(2)
        shared = {"canonical"}
        store.cell_set(0, 3, shared)
        store.cell_set(1, 3, shared)
        store.freeze()
        assert store.cell_get(0, 3) is shared
        assert store.cell_get(1, 3) is shared


class TestSmallStoreSeal:
    def test_small_store_seals_without_transpose(self, monkeypatch):
        monkeypatch.setattr(op_store, "_TRANSPOSE_MIN_ROWS", 512)
        store = _store_with_rows(2)
        store.cell_set(0, 0, "kept")
        store.freeze()
        assert store.frozen
        assert store.rows_building() is not None
        assert store.cell_get(0, 0) == "kept"
        with pytest.raises(RuntimeError):
            store.new_row()
        store.cell_set(0, 1, "post-seal")
        assert store.cell_get(0, 1) == "post-seal"
        assert store.cell_del(0, 1) is True

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
    def test_single_row_roundtrip(self):
        store = DetachedOpStore(LAYOUT)
        store.cell_set(0, 0, "x")
        assert store.cell_get(0, 0) == "x"
        assert list(store.items(0)) == [("alpha", "x")]
        assert store.cell_del(0, 0) is True
        assert store.cell_del(0, 0) is False
        assert not store.frozen
