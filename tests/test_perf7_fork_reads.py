"""Fork copy reads take atomic fast paths and return exactly what they did before.

A fork reads the parent's sealed cells through an ``OpStoreView``: every read
isolates mutable containers (``cow_copy_value``) so fork-side mutation never reaches
the parent. Reads of atomic values (``None``, labels, counts) and packed numeric
columns now return without the isolation call, and the copier skips the per-member
call for atomic members. These tests compare both against the former code, kept here
verbatim as the reference, and pin the fork's call count on a fixed capture.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._trace_core import op_store
from torchlens.options import CaptureOptions


def _reference_cow_copy(value: Any, translate: Callable[[Any], Any] | None) -> Any:
    """``cow_copy_value`` as it was before this change (verbatim body)."""

    cls = value.__class__
    if cls in op_store._COW_ATOMIC:
        return value
    if cls is dict:
        return {
            _reference_cow_copy(key, translate): _reference_cow_copy(item, translate)
            for key, item in value.items()
        }
    if cls is list:
        return [_reference_cow_copy(item, translate) for item in value]
    if cls is set:
        return {_reference_cow_copy(item, translate) for item in value}
    if cls is tuple or cls is frozenset:
        copied = [_reference_cow_copy(item, translate) for item in value]
        for original, item_copy in zip(value, copied, strict=True):
            if original is not item_copy:
                return cls(copied)
        return value
    if translate is not None:
        mapped = translate(value)
        if mapped is not None:
            return mapped
    return value


def _reference_cell_get(view: Any, row: int, fid: int) -> Any:
    """``OpStoreView.cell_get`` as it was before this change (verbatim body)."""

    key = row * view.base.layout.n_fields + fid
    value = view._overlay.get(key, op_store._NO_OVERLAY)
    if value is not op_store._NO_OVERLAY:
        return value
    value = view._base_overlay.get(key, op_store._NO_OVERLAY)
    if value is op_store._NO_OVERLAY:
        base_rows = view._base_rows
        if base_rows is not None:
            value = base_rows[row][fid]
        else:
            columns = view.base._columns
            assert columns is not None
            value = columns[fid].get(row)
    return view._isolate(key, value)


class _Record:
    """Stands in for a parent record facade the translator maps."""


_PARENT, _CHILD = _Record(), _Record()


def _translate(value: Any) -> Any:
    return _CHILD if value is _PARENT else None


def _shape(result: Any, original: Any, depth: int = 0) -> str:
    """Types, atomic values, and whether each object is the original one."""

    same = "same" if result is original else "new"
    if result is _CHILD:
        return "child"
    if isinstance(result, (str, int, float, bool, bytes, complex)) or result is None:
        return f"{type(result).__name__}:{result!r}:{same}"
    if depth > 6:
        return f"{type(result).__name__}:{same}"
    if isinstance(result, dict) and isinstance(original, dict):
        pairs = zip(result.items(), original.items(), strict=True)
        inner = ",".join(
            f"{_shape(rk, ok, depth + 1)}={_shape(rv, ov, depth + 1)}"
            for (rk, rv), (ok, ov) in pairs
        )
        return f"dict:{same}{{{inner}}}"
    if isinstance(result, (list, tuple)) and isinstance(original, (list, tuple)):
        inner = ",".join(_shape(r, o, depth + 1) for r, o in zip(result, original, strict=True))
        return f"{type(result).__name__}:{same}[{inner}]"
    if isinstance(result, (set, frozenset)):
        return f"{type(result).__name__}:{same}:{sorted(map(repr, result))}"
    return f"{type(result).__name__}:{same}"


def _cow_cases() -> list[Any]:
    tensor = torch.ones(2)
    return [
        None,
        "label",
        3,
        2.0,
        True,
        b"x",
        tensor,
        _PARENT,
        object(),
        ["a", 1, None, tensor, _PARENT, ["nested", tensor]],
        {"a": 1, "b": [1, 2], "c": _PARENT, 2: "two"},
        {"a", 1, None, ("t", 1)},
        ("a", 1, None),
        ("a", [1, 2]),
        ("a", _PARENT),
        frozenset({"a", 1}),
        frozenset({("a", 1), "b"}),
        ((("deep", [1]),),),
        [],
        {},
        (),
    ]


def test_cow_copy_matches_the_former_copier() -> None:
    for index, (value, twin) in enumerate(zip(_cow_cases(), _cow_cases(), strict=True)):
        got = op_store.cow_copy_value(value, _translate)
        expected = _reference_cow_copy(twin, _translate)
        assert _shape(got, value) == _shape(expected, twin), f"case {index}"
        if isinstance(got, (set, frozenset)):
            assert got == expected, f"case {index}"


class _Small(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class _Long(nn.Module):
    """About 545 ops: past the store's columnar-transpose threshold (512 rows)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        for _ in range(180):
            h = torch.tanh(h * 1.01 + 0.1)
        return h.sum(dim=-1)


def _read_shape(value: Any, base_value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return f"T:{'shared' if value is base_value else 'new'}"
    return _shape(value, base_value)


def _base_value(store: Any, row: int, fid: int) -> Any:
    return store.cell_get(row, fid)


@pytest.mark.parametrize(
    "model_cls",
    [_Small, pytest.param(_Long, marks=pytest.mark.heavy)],
    ids=["row_major", "columnar"],
)
def test_every_fork_cell_read_matches_the_former_read(model_cls: type[nn.Module]) -> None:
    torch.manual_seed(0)
    trace = tl.trace(
        model_cls(), torch.randn(2, 4), capture=CaptureOptions(intervention_ready=True)
    )
    store = trace._trace_core.ops
    assert (store._columns is not None) is (model_cls is _Long), "fixture store shape"
    current = trace.fork()._trace_core.ops
    former = trace.fork()._trace_core.ops
    n_fields = store.layout.n_fields
    mismatches = []
    for row in range(len(store)):
        for fid in range(n_fields):
            base_value = _base_value(store, row, fid)
            got = current.cell_get(row, fid)
            expected = _reference_cell_get(former, row, fid)
            if _read_shape(got, base_value) != _read_shape(expected, base_value):
                mismatches.append((row, store.layout.names[fid], got, expected))
            # A second read of an isolated container serves the same copy.
            is_container = isinstance(got, (dict, list, set, tuple, frozenset))
            if is_container and current.cell_get(row, fid) is not got:
                mismatches.append((row, store.layout.names[fid], "second read differs"))
    assert not mismatches, mismatches[:3]


def _events(action: Callable[[], Any]) -> int:
    calls = 0

    def count(_frame: Any, event: str, _arg: Any) -> None:
        nonlocal calls
        if event in ("call", "c_call"):
            calls += 1

    sys.setprofile(count)
    try:
        action()
    finally:
        sys.setprofile(None)
    return calls


@pytest.mark.parametrize(
    ("model_cls", "bar"),
    [
        # Measured on the base reads: 4,578 events; after this change: BARSMALL.
        (_Small, 4_300),
        # Measured on the base reads: 628,300 events; after this change: BARLONG.
        pytest.param(_Long, 520_000, marks=pytest.mark.heavy),
    ],
    ids=["row_major", "columnar"],
)
def test_reading_every_fork_cell_stays_cheap(model_cls: type[nn.Module], bar: int) -> None:
    """Profiler events for one full read of a fresh fork's op cells."""

    torch.manual_seed(0)
    trace = tl.trace(
        model_cls(), torch.randn(2, 4), capture=CaptureOptions(intervention_ready=True)
    )
    view = trace.fork()._trace_core.ops
    n_fields = view.base.layout.n_fields
    rows = range(len(view))

    def read_all() -> None:
        for row in rows:
            for fid in range(n_fields):
                view.cell_get(row, fid)

    events = _events(read_all)
    assert events <= bar, events
