"""The op store seals ahead of the finished mark.

Three contracts. The finished mark's sealed-store body leaves what the facade
body leaves (every op flag, the trace flag) on a columnar store (512 rows or
more) and on a sealed row-major store, and it writes no overlay entry. A
streamed out evicted after the seal does not stay in the columnar store's
column under an overlay entry. The seal runs inside the
pipeline step, before the mark, through the package-level names it resolves
at call time.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens._postprocess as pp
from torchlens._postprocess import _finished_marks
from torchlens._utils._record_group_table import OP_STORAGE_ATTRS, record_member

#: Rows at which ``OpRowStore.freeze`` transposes into columns.
_COLUMNAR_OPS = 600


class _Chain(nn.Module):
    """A plain chain of additions, ``steps`` ops long."""

    def __init__(self, steps: int) -> None:
        super().__init__()
        self.steps = steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add one ``steps`` times."""

        for _ in range(self.steps):
            x = x + 1
        return x


@pytest.fixture
def isolated_patch() -> Iterator[pytest.MonkeyPatch]:
    """Yield a private monkeypatch context that is always undone."""

    patch = pytest.MonkeyPatch()
    try:
        yield patch
    finally:
        patch.undo()


def _store(trace: Any) -> Any:
    """Return the trace's Op row store."""

    return trace._trace_core.ops


def _row(op: Any) -> int:
    """Return the op's store row."""

    return int(object.__getattribute__(op, "_row"))


def _flags(trace: Any) -> dict[str, Any]:
    """Return every op's finished flag (value and class) and the trace flag."""

    ops = {
        label: (op._tracing_finished, type(op._tracing_finished))
        for label, op in trace._layer_dict_all_keys.items()
    }
    return {"ops": ops, "trace": trace._tracing_finished}


def _trace(steps: int, patch: pytest.MonkeyPatch, *, facade: bool) -> tuple[Any, list[bool]]:
    """Trace a chain; return the trace and whether the sealed body wrote the flags."""

    real = _finished_marks._mark_sealed_rows
    used: list[bool] = []

    def spy(*args: Any) -> bool:
        """Record the sealed body's answer (or force the facade body)."""

        wrote = False if facade else real(*args)
        used.append(wrote)
        return wrote

    patch.setattr(_finished_marks, "_mark_sealed_rows", spy)
    trace = tl.trace(_Chain(steps), torch.ones(2, 3))
    patch.undo()
    return trace, used


@pytest.mark.parametrize("steps", [40, _COLUMNAR_OPS])
def test_sealed_body_matches_facade_body(steps: int, isolated_patch: pytest.MonkeyPatch) -> None:
    """The sealed-store body and the facade body leave the same flags."""

    sealed, sealed_used = _trace(steps, isolated_patch, facade=False)
    facade, facade_used = _trace(steps, isolated_patch, facade=True)
    try:
        assert sealed_used == [True]
        assert facade_used == [False]
        assert _store(sealed).frozen and _store(facade).frozen
        assert (_store(sealed)._rows is None) is (steps >= 512)
        assert _flags(sealed) == _flags(facade)
        assert len(_flags(sealed)["ops"]) >= steps
        assert all(value == (True, bool) for value in _flags(sealed)["ops"].values())
        assert _flags(sealed)["trace"] is True
    finally:
        sealed.cleanup()
        facade.cleanup()


def test_sealed_body_writes_the_packed_column_not_the_overlay(
    isolated_patch: pytest.MonkeyPatch,
) -> None:
    """On a columnar store the flags live in the boolean column itself."""

    trace, used = _trace(_COLUMNAR_OPS, isolated_patch, facade=False)
    try:
        store = _store(trace)
        fid = store.layout.fid_by_name["_tracing_finished"]
        column = store._columns[fid]
        assert used == [True]
        assert column.packed and column.values.dtype == bool
        assert not any(key % store.layout.n_fields == fid for key in store._overlay)
        rows = [_row(op) for op in trace._layer_list]
        assert all(bool(column.values[row]) for row in rows)
    finally:
        trace.cleanup()


def test_streamed_outs_evicted_after_the_seal_leave_the_column(tmp_path: Path) -> None:
    """An evicted streamed out is ``None`` in the column, with no overlay entry over it."""

    trace = tl.trace(
        _Chain(_COLUMNAR_OPS),
        torch.ones(2, 3),
        storage=tl.to_disk(tmp_path / "sealed_evict.tlspec"),
    )
    try:
        store = _store(trace)
        assert store._rows is None
        fid = store.layout.fid_by_name[OP_STORAGE_ATTRS.get("out", "out")]
        n_fields = store.layout.n_fields
        streamed = [
            op for op in trace._layer_list if record_member(op, "out_ref", None) is not None
        ]
        assert len(streamed) >= _COLUMNAR_OPS
        for op in streamed:
            assert op.out is None
            key = _row(op) * n_fields + fid
            assert key not in store._overlay
            assert store._columns[fid].values[_row(op)] is None
    finally:
        trace.cleanup()


def test_seal_runs_before_the_mark_through_the_package_names(
    isolated_patch: pytest.MonkeyPatch,
) -> None:
    """The step resolves the relation freeze through the package, before the mark."""

    real = pp._freeze_relation_views
    seen: list[tuple[bool, bool]] = []

    def observing(trace: Any) -> None:
        """Record the trace flag and the store's seal state, then freeze."""

        seen.append((trace._tracing_finished, _store(trace).frozen))
        real(trace)

    isolated_patch.setattr(pp, "_freeze_relation_views", observing)
    trace = tl.trace(_Chain(4), torch.ones(2, 3))
    try:
        assert seen == [(False, False)]
        assert trace._tracing_finished is True and _store(trace).frozen
    finally:
        trace.cleanup()
