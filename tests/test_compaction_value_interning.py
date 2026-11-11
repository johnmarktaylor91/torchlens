"""Part C1 of compaction interning: value objects interned at row ingest.

``seed_op_values`` with an intern table gives equal values to the uncached
path and one object per distinct source; non-zero durations stay one object
per row. The freeze's sweep then visits each distinct object of the repeated
value columns once; its result must equal the per-cell sweep's, cell for cell
and object for object.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes import _compaction
from torchlens._ir._op_row_seed import NAME_KEYS, seed_op_values
from torchlens._quantities import Duration


def _fields(dtype: Any, duration: Any, memory: int) -> dict[str, Any]:
    """Return a minimal fields dict that ``seed_op_values`` converts."""

    fields: dict[str, Any] = {
        "dtype_ref": None,
        "dtype": dtype,
        "device_ref": None,
        "out": None,
        "output_device": "cpu",
        "backend_address": None,
        "address": "fc1",
        "resolver_status": None,
        "param_memory": None,
        "func_duration": duration,
        "flops_forward": 10,
        "flops_backward": None,
    }
    for name in NAME_KEYS.bytes_fields:
        fields[name] = memory
    return fields


def test_interned_seeding_equals_uncached_and_shares_objects() -> None:
    """Equal sources share one object; values equal the uncached conversion."""

    cache: dict[Any, Any] = {}
    first = _fields(torch.float32, 0, 64)
    second = _fields(torch.float32, 0, 64)
    plain = _fields(torch.float32, 0, 64)
    seed_op_values(first, NAME_KEYS, cache)
    seed_op_values(second, NAME_KEYS, cache)
    seed_op_values(plain, NAME_KEYS)
    for name, value in plain.items():
        assert first[name] == value and type(first[name]) is type(value), name
    for name in ("dtype_ref", "device_ref", "activation_memory", "param_memory", "flops_forward"):
        assert first[name] is second[name], name
    assert first["func_duration"] is second["func_duration"]


def test_measured_durations_stay_one_object_per_row() -> None:
    """A non-zero duration is never shared (``test_duration_identity`` pins the freeze)."""

    cache: dict[Any, Any] = {}
    first = _fields(torch.float32, 0.5, 8)
    second = _fields(torch.float32, 0.5, 8)
    seed_op_values(first, NAME_KEYS, cache)
    seed_op_values(second, NAME_KEYS, cache)
    assert first["func_duration"] == second["func_duration"] == Duration(0.5)
    assert first["func_duration"] is not second["func_duration"]


class _Store:
    """Stand-in building store over plain rows."""

    def __init__(self, rows: list[list[Any]], names: list[str]) -> None:
        """Hold the rows and a layout with ``names``."""

        self._rows = rows
        self.layout = type("Layout", (), {"names": tuple(names)})()

    def rows_building(self) -> list[list[Any]]:
        """Return the rows."""

        return self._rows


def _sweep(rows: list[list[Any]], names: list[str], by_object: bool) -> dict[Any, Any]:
    """Run the immutable sweep, with or without the per-object path."""

    pool: dict[Any, Any] = {}
    saved = _compaction._BY_OBJECT_FIELDS
    _compaction._BY_OBJECT_FIELDS = saved if by_object else frozenset()
    try:
        _compaction._compact_store_rows(_Store(rows, names), pool)
    finally:
        _compaction._BY_OBJECT_FIELDS = saved
    return pool


def test_sweep_by_object_equals_per_cell_sweep() -> None:
    """Same cells, same objects, same pool as the per-cell sweep."""

    def build() -> tuple[list[list[Any]], list[str]]:
        """Rows whose ``type`` column mixes shared and fresh equal strings."""

        shared = "".join(["re", "lu"])
        rows = []
        for index in range(12):
            fresh = "".join(["re", "lu"]) if index % 4 == 0 else shared
            rows.append(["".join(["re", "lu"]), fresh, (index % 2, "a"), None])
        return rows, ["func_name", "type", "shape", "dtype"]

    rows_a, names = build()
    rows_b, _ = build()
    pool_a = _sweep(rows_a, names, by_object=True)
    pool_b = _sweep(rows_b, names, by_object=False)
    assert rows_a == rows_b and pool_a.keys() == pool_b.keys()
    for row_a, row_b in zip(rows_a, rows_b, strict=True):
        assert [row_a.index(c) for c in row_a] == [row_b.index(c) for c in row_b]
    shape_ids_a = {id(row[2]) for row in rows_a}
    shape_ids_b = {id(row[2]) for row in rows_b}
    assert len(shape_ids_a) == len(shape_ids_b)
    type_ids_a = {id(row[1]) for row in rows_a}
    type_ids_b = {id(row[1]) for row in rows_b}
    assert len(type_ids_a) == len(type_ids_b) == 1


def test_two_traces_share_no_interned_value() -> None:
    """Each trace's ingest owns its intern table: no reference object is shared."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    x = torch.randn(2, 4)
    first = tl.trace(model, x)
    second = tl.trace(model, x)
    for name in ("dtype_ref", "device_ref"):
        refs_first = {id(getattr(op, name)) for op in first.ops if getattr(op, name) is not None}
        refs_second = {id(getattr(op, name)) for op in second.ops if getattr(op, name) is not None}
        assert refs_first and refs_second, name
        assert refs_first.isdisjoint(refs_second), name
