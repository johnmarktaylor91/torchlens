"""Orphan removal's bulk drop (postprocess Step 3).

The drop scrubs only the surviving ops that name a removed label. These tests
check that the finished trace equals the one the general per-op removal
produces, on a model whose orphans share equivalence classes with live ops,
read a parameter and drive a conditional, and that a trace without orphans
scrubs nothing.
"""

from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
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


def _dump(save_args: bool = False) -> tuple[Any, Any]:
    """Trace the dead-branches model and return the trace and its canonical dump.

    The per-process trace counter (the name suffix and the run nonce) differs
    between two traces in one process, so both are replaced by fixed tokens.
    """

    model, x = build_model("dead_branches")
    capture = tl.options.CaptureOptions(seed=SEED, save_args=save_args)
    trace = tl.trace(model, x, capture=capture)
    dump, _canon = snapshot_trace(trace, [(str(REPO_ROOT), "<repo>")], read_geometry=True)
    dump["trace"]["event_stream"]["fields"].pop("run_nonce")
    text = json.dumps(dump, default=repr, sort_keys=True).replace(trace.name, "<trace>")
    return trace, json.loads(text)


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
    """The scrub visits exactly the survivors that name an orphan, never all of them."""

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
    removed = seen["removed"]
    naming = [op for op in seen["survivors"] if _trace_removal._names_removed_label(op, removed)]
    assert seen["scrubbed"] == naming
    assert len(seen["scrubbed"]) < len(seen["survivors"])


@pytest.mark.parametrize(
    "fields",
    [
        *({attr: ["live_1", "dead_1"]} for attr in _trace_removal._LABEL_ATTRS),
        {"parent_arg_positions": {"args": {0: "dead_1"}, "kwargs": {}}},
        {"out_versions_by_child": {"dead_1": 0}},
        {"conditional_arm_children": {0: {"then": ["dead_1"]}}},
        {"conditional_elif_children": {0: {1: ["dead_1"]}}},
        {"_edge_uses": [SimpleNamespace(parent_label="live_1", child_label="dead_1")]},
        {"edits": ["dead_1"]},
    ],
)
def test_names_removed_label_sees_every_scrubbed_field(fields: dict[str, Any]) -> None:
    """Each field the general scrub rewrites marks its op for the bulk drop's scrub."""

    assert _trace_removal._names_removed_label(SimpleNamespace(**fields), {"dead_1"})
    clean = SimpleNamespace(
        **{attr: ["live_1"] for attr in _trace_removal._LABEL_ATTRS}, edits=["live_1"]
    )
    assert not _trace_removal._names_removed_label(clean, {"dead_1"})


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
