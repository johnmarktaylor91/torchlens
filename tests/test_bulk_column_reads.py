"""Bulk readers read stored ``Op`` fields a whole column at a time, with facade values.

``OpColumns.column`` answers what the per-row ``getter`` answers for every row,
on the sealed row-major store a small trace keeps and on the frozen columnar
store of a trace over the transpose threshold, overlay writes included.
``Trace.to_pandas`` built through the column path equals the same table read
op by op through the facade, for a live trace, a fan-out trace with siblings
and co-parents, and a saved and loaded trace (whose store is read per op).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes._op_columns import OpColumns
from torchlens._data_classes.op import _OP_SLOT_NAMES
from torchlens._trace_core.op_store import OpRowStore, _SealedRowMajorOpRowStore
from torchlens.stats import _trace_table_rows


class _Chain(nn.Module):
    """A long elementwise chain: enough ops for the frozen columnar store."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 600 alternating elementwise ops."""

        for index in range(600):
            x = torch.relu(x) if index % 2 else x + 1
        return x


class _FanOut(nn.Module):
    """Fan-out and fan-in, so ops have siblings and co-parents."""

    def __init__(self) -> None:
        """Build three linear layers."""

        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)
        self.c = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a merged branch pair and one raw branch."""

        hidden = torch.relu(self.a(x))
        left, right = self.b(hidden), self.c(hidden)
        return left * right + hidden, torch.sigmoid(right)


def _trace(model: nn.Module) -> tl.Trace:
    """Return a finished trace of ``model`` on a fixed input."""

    torch.manual_seed(0)
    return tl.trace(model, torch.randn(2, 4))


@pytest.fixture(params=["row_major", "columnar"])
def store_trace(request: pytest.FixtureRequest) -> Iterator[tl.Trace]:
    """Yield a trace whose op store has the requested shape, released afterwards."""

    trace = _trace(_FanOut() if request.param == "row_major" else _Chain())
    store = object.__getattribute__(trace.ops[0], "_core")
    expected = _SealedRowMajorOpRowStore if request.param == "row_major" else OpRowStore
    assert store.__class__ is expected
    try:
        yield trace
    finally:
        trace.cleanup()


def _outcome(read: Callable[[], Any]) -> tuple[str, Any]:
    """Return ``("value", read())`` or ``("raise", exception type)``."""

    try:
        return "value", read()
    except Exception as exc:  # the facade's own refusal type is compared
        return "raise", type(exc)


def test_column_matches_getter_row_by_row(store_trace: tl.Trace) -> None:
    """Every stored field read whole equals the per-row getter, by identity."""

    ops = list(store_trace.ops)
    store = object.__getattribute__(ops[0], "_core")
    rows = [object.__getattribute__(op, "_row") for op in ops]
    view = OpColumns(store)
    store_trace.to_pandas()  # answers lazy cells, so the overlay holds cached cells
    compared = 0
    for name in _OP_SLOT_NAMES:
        try:
            get = view.getter(name)
        except ValueError:
            continue
        kind, column = _outcome(lambda name=name: view.column(name, rows))
        expected = [_outcome(lambda row=row, get=get: get(row)) for row in rows]
        if kind == "raise":
            assert ("raise", column) in expected, name
            continue
        compared += 1
        for value, (want_kind, want) in zip(column, expected, strict=True):
            assert want_kind == "value", name
            assert value is want or (type(value) is type(want) and repr(value) == repr(want)), name
    assert compared > 50


def _per_op_table(trace: tl.Trace, monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    """Return ``trace.to_pandas()`` with every cell read op by op through the facade."""

    with monkeypatch.context() as patch:
        patch.setattr(_trace_table_rows, "_column_plan", lambda ops: None)
        return trace.to_pandas()


def _assert_same_table(trace: tl.Trace, monkeypatch: pytest.MonkeyPatch) -> None:
    """Assert the column-path table equals the per-op table, cell types included."""

    table = trace.to_pandas()  # first read of the trace: the column path meets lazy cells
    reference = _per_op_table(trace, monkeypatch)
    pd.testing.assert_frame_equal(table, reference, check_exact=True)
    assert list(table.columns) == list(reference.columns)
    for column in table.columns:
        got = [type(value) for value in table[column]]
        assert got == [type(value) for value in reference[column]], column
        assert repr(list(table[column])) == repr(list(reference[column])), column


@pytest.mark.parametrize("model", [_FanOut, _Chain], ids=["fan_out", "chain"])
def test_to_pandas_equals_per_op_reads(model: type[nn.Module], monkeypatch: Any) -> None:
    """The column path's table equals the facade table on a live trace."""

    trace = _trace(model())
    try:
        assert any(trace.to_pandas()["has_siblings"]) or model is _Chain
        _assert_same_table(trace, monkeypatch)
    finally:
        trace.cleanup()


def test_to_pandas_equals_per_op_reads_after_load(tmp_path: Path, monkeypatch: Any) -> None:
    """A loaded trace's table (its store is read cell by cell) equals the facade table."""

    trace = _trace(_FanOut())
    path = tmp_path / "fan_out.tlspec"
    trace.save(str(path))
    trace.cleanup()
    loaded = tl.load(str(path))
    _assert_same_table(loaded, monkeypatch)
