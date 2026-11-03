"""F10 lovely item 8/9: edge semantics (D28) + stats view polish laws.

The two axes are independent: ``expand``/``split`` are VIEWS that change
the multiset; ``clone``/``contiguous`` copy/re-layout the SAME one. Every
uncertain case is fail-closed -- an absent mark is honest, a wrong mark
is not; ``data_ptr`` never participates.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention._edge_semantics import (
    classify_view_or_copy,
    distribution_relation,
)
from torchlens.report._stats_table import build_stats_table


class _EdgeZoo(nn.Module):
    """One forward exercising view/copy/multiset-changing/binary edges."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)  # multi-tensor op: no verdicts
        v = h.view(2, 2, 4)  # view + same_multiset
        c = v.clone()  # copy + same_multiset
        f = c.flatten()  # conditional view: storage unknown; same_multiset
        e = f.unsqueeze(0).expand(2, -1)  # expand: view, NOT same_multiset
        s = e.sum(dim=0)  # reduction: nothing
        return s + h.flatten()  # binary op: multi-parent, nothing


@pytest.fixture(scope="module")
def zoo_trace():
    """Intervention-ready capture with edge provenance."""

    trace = tl.trace(
        _EdgeZoo().eval(),
        torch.randn(2, 8),
        capture=tl.options.CaptureOptions(replayable=True),
    )
    yield trace
    trace.cleanup()


def _edges_by_child(trace):
    """Map child label -> its slot-0 positional edge record."""

    result = {}
    for op in trace._layer_list:
        for record in getattr(op, "edge_uses", ()) or ():
            if record.arg_kind == "positional" and tuple(record.arg_path) == (0,):
                result.setdefault(record.child_label, record)
    return result


# Kept by the shrink-r4 coverage proof: a sole campaign mutation kill.
def test_distribution_relation_verdicts(zoo_trace) -> None:
    """same_multiset holds exactly where semantics + geometry corroborate."""

    edges = _edges_by_child(zoo_trace)
    verdicts = {
        label.split("_")[0]: distribution_relation(zoo_trace, record)
        for label, record in edges.items()
    }
    assert verdicts["view"] is not None and verdicts["view"].relation == "same_multiset"
    assert verdicts["clone"] is not None
    assert verdicts["flatten"] is not None  # copy-or-view, SAME multiset either way
    assert verdicts.get("expand") is None  # a view of a DIFFERENT multiset
    assert verdicts.get("linear") is None
    assert verdicts.get("sum") is None
    assert verdicts.get("add") is None  # multi-parent: never marked
    proof = verdicts["view"]
    assert proof.basis == "func_semantics:view"
    assert proof.corroboration.startswith("numel_match:")


# Kept by the shrink-r4 coverage proof: a sole campaign mutation kill.
def test_classifier_is_slot0_only() -> None:
    """Only the positional slot-0 edge can carry a storage verdict."""

    assert classify_view_or_copy("view", (0,)) == "view"
    assert classify_view_or_copy("view", (1,)) == "unknown"
    assert classify_view_or_copy("clone", (0,)) == "copy"
    assert classify_view_or_copy(None, (0,)) == "unknown"
    assert classify_view_or_copy("reshape", (0,)) == "unknown"  # may copy


# Kept by the shrink-r4 coverage proof: sole branch coverage of report/_stats_table.py, report/_stats_view.py and more.
def test_stats_table_marks_budget_sort_fold(zoo_trace) -> None:
    """Item 9: marks from D28 only; budget typed; sorts disclosed."""

    table = build_stats_table(zoo_trace)
    assert "graph (execution) order" in repr(table)
    marked_rows = [row for row in table.rows if row.parent_mark]
    marked = {row.label.split("_")[0] for row in marked_rows}
    assert {"view", "clone", "flatten"} <= marked
    assert "expand" not in marked and "add" not in marked
    assert all("func_semantics:" in row.relation_basis for row in marked_rows)

    budgeted = zoo_trace.summary(view="stats", max_scan_elements=20)
    states = {row.state for row in budgeted.rows}
    assert "unscanned" in states
    assert "budget: max_scan_elements=20" in str(budgeted)
    unscanned = next(row for row in budgeted.rows if row.state == "unscanned")
    assert unscanned.value_stats is None and "budget" in unscanned.reason

    ranked = table.sort_rows("mean")
    assert "sorted by mean" in repr(ranked)
    with pytest.raises(ValueError, match="closed vocabulary"):
        table.sort_rows("vibes")

    folded = table.fold_same_as_parent()
    assert len(folded.rows) == len(table.rows) - len(marked_rows)
    assert "rows folded" in repr(folded)


# Kept by the shrink-r4 coverage proof: sole branch coverage of report/_stats_table.py, report/_summary_report.py.
def test_to_pandas_carries_relation_columns(zoo_trace) -> None:
    """The tabular exit preserves the mark and its proof provenance."""

    pytest.importorskip("pandas")
    frame = zoo_trace.summary(view="stats").to_pandas()
    assert "same_as_parent" in frame.columns
    assert "relation_basis" in frame.columns
    assert frame.attrs["ordering"].startswith("rows in graph")
