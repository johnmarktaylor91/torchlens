"""Orphan removal's bulk drop (postprocess Step 3).

The drop scrubs only the surviving ops that name a removed label. These tests
check that the finished trace equals the one the general per-op removal
produces, on a model whose orphans share equivalence classes with live ops,
read a parameter and drive a conditional, and that a trace without orphans
scrubs nothing.
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pytest

import torchlens as tl
from torchlens._data_classes import _trace_removal
from torchlens._postprocess import graph_traversal

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
from field_diff_lib.cases import SEED, build_model  # noqa: E402
from field_diff_lib.surface import snapshot_trace  # noqa: E402


def _general_drop(trace: Any, orphan_nodes: set[str]) -> None:
    """The general removal: scrub every survivor, then rebuild the layer dict."""

    ws = trace._raw_graph_ws
    trace._batch_remove_log_entries(
        [ws.raw_layer_dict[label] for label in orphan_nodes], remove_references=True
    )
    kept = [label for label in ws.raw_layer_labels_list if label not in orphan_nodes]
    ws.raw_layer_dict = OrderedDict((label, ws.raw_layer_dict[label]) for label in kept)
    ws.raw_layer_labels_list = kept


def _dump(save_args: bool = False) -> tuple[Any, dict[str, Any]]:
    """Trace the dead-branches model and return the trace and its canonical dump."""

    model, x = build_model("dead_branches")
    capture = tl.options.CaptureOptions(seed=SEED, save_args=save_args)
    trace = tl.trace(model, x, capture=capture)
    dump, _canon = snapshot_trace(trace, [(str(REPO_ROOT), "<repo>")], read_geometry=True)
    return trace, dump


@pytest.mark.parametrize("save_args", [False, True])
def test_bulk_drop_equals_the_general_removal(
    monkeypatch: pytest.MonkeyPatch, save_args: bool
) -> None:
    """Every public field, repr and table equal the general removal's, with orphans present."""

    bulk_trace, bulk = _dump(save_args)
    assert len(bulk_trace._orphan_labels) >= 6, bulk_trace._orphan_labels
    with monkeypatch.context() as patch:
        patch.setattr(graph_traversal, "drop_orphan_entries", _general_drop)
        _general_trace, general = _dump(save_args)
    assert bulk == general


def test_bulk_drop_scrubs_only_the_survivors_that_name_an_orphan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scrub visits a strict subset of the survivors, and every op it skips names no orphan."""

    seen: dict[str, Any] = {}
    real = _trace_removal.batch_remove_log_entries

    def spy(trace: Any, entries: Any, **kwargs: Any) -> None:
        seen["scrubbed"] = list(kwargs["scrub_entries"])
        seen["removed"] = {entry._label_raw for entry in entries}
        seen["survivors"] = [
            op
            for label, op in trace._raw_graph_ws.raw_layer_dict.items()
            if label not in seen["removed"]
        ]
        real(trace, entries, **kwargs)

    monkeypatch.setattr(_trace_removal, "batch_remove_log_entries", spy)
    model, x = build_model("dead_branches")
    tl.trace(model, x, capture=tl.options.CaptureOptions(seed=SEED))
    assert seen["removed"]
    assert 0 < len(seen["scrubbed"]) < len(seen["survivors"])


def test_no_orphans_scrubs_nothing_and_keeps_the_layer_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without orphans the general removal is never entered and the layer dict is kept."""

    calls: list[Any] = []
    monkeypatch.setattr(
        _trace_removal, "batch_remove_log_entries", lambda *args, **kwargs: calls.append(args)
    )
    dicts: list[Any] = []
    real_drop = graph_traversal.drop_orphan_entries

    def watching(trace: Any, orphan_nodes: set[str]) -> None:
        before = trace._raw_graph_ws.raw_layer_dict
        real_drop(trace, orphan_nodes)
        dicts.append((orphan_nodes, before is trace._raw_graph_ws.raw_layer_dict))

    monkeypatch.setattr(graph_traversal, "drop_orphan_entries", watching)
    model, x = build_model("chain40")
    tl.trace(model, x)
    assert dicts == [(set(), True)]
    assert calls == []
