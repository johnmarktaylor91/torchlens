"""Regression tests for r20i: IR live-record lane cleanup + state hygiene.

Covers hardening findings on ``torchlens/ir``:

* N8  -- ``LiveIndex.append`` left a stale parent edge (and double-listed the
  label) when a raw label was re-seen at the append boundary.
* N9  -- ``CaptureEvents.release_runtime_sidecars`` retained runtime handles
  (``backend_session``, ``grad_fn_handles_by_label_raw``, ``recent_events``)
  its docstring said it detaches.
* N10 -- ``CaptureEvents._copy_for_replay`` shared the mutable per-event dicts
  (``transform_config``, ``parent_arg_positions``) of "frozen" ``OpEvent``
  objects, so a replay projection could mutate the sealed source stream.
"""

from __future__ import annotations

import dataclasses
import gc
import weakref
from collections import deque

from test_ir_basic import _build_ir_instances

from torchlens._ir.capture_events import CaptureEvents
from torchlens._ir.events import OpEvent, ParentEdge
from torchlens._ir.live_index import LiveIndex


class _Handle:
    """Weak-referenceable runtime-handle sentinel."""


def _base_op_event() -> OpEvent:
    """Return a fully-populated ``OpEvent`` built from the shared IR fixture."""

    return _build_ir_instances()["op_event"]


def _op_event(label: str, parents: tuple[str, ...] = ()) -> OpEvent:
    """Build an ``OpEvent`` with a given raw label and parent labels."""

    edges = tuple(
        ParentEdge(parent_label_raw=parent, arg_position=index, _edge_use="arg")
        for index, parent in enumerate(parents)
    )
    return dataclasses.replace(_base_op_event(), label_raw=label, parents=edges)


# --------------------------------------------------------------------------- N8


def test_live_index_reseen_label_drops_stale_parent_edge() -> None:
    """Re-seeing a raw label must not leave a stale parent->child edge (N8)."""

    index = LiveIndex()
    index.append(_op_event("old"))
    index.append(_op_event("new"))
    index.append(_op_event("dup", ("old",)))
    index.append(_op_event("dup", ("new",)))

    # Label indexed exactly once, last event wins for parents.
    assert index.labels.count("dup") == 1
    assert index.parents("dup") == ("new",)
    # The prior parent must no longer list the re-seen label as its child.
    assert index.children("old") == ()
    assert index.children("new") == ("dup",)
    assert "dup" not in index.children_by_parent.get("old", [])


def test_live_index_reseen_label_matches_replace_semantics() -> None:
    """A re-seen ``append`` is equivalent to an explicit ``replace`` (N8)."""

    reappended = LiveIndex()
    reappended.append(_op_event("a"))
    reappended.append(_op_event("b", ("a",)))
    reappended.append(_op_event("b", ("a",)))  # re-seen with same parent

    replaced = LiveIndex()
    replaced.append(_op_event("a"))
    replaced.append(_op_event("b", ("a",)))
    replaced.replace(_op_event("b", ("a",)))

    assert reappended.labels == replaced.labels == ["a", "b"]
    assert reappended.children("a") == replaced.children("a") == ("b",)


def test_live_index_unique_labels_unchanged() -> None:
    """Unique-label appends (the normal capture path) are unaffected (N8)."""

    index = LiveIndex()
    index.append(_op_event("a"))
    index.append(_op_event("b", ("a",)))
    index.append(_op_event("c", ("a", "b")))

    assert index.labels == ["a", "b", "c"]
    assert index.parents("c") == ("a", "b")
    assert index.children("a") == ("b", "c")
    assert index.children("b") == ("c",)


def test_capture_events_append_keeps_live_index_consistent() -> None:
    """Duplicate label through ``CaptureEvents.append`` leaves no stale edge (N8)."""

    events = CaptureEvents()
    events.append(_op_event("old"))
    events.append(_op_event("new"))
    events.append(_op_event("dup", ("old",)))
    events.append(_op_event("dup", ("new",)))

    assert events.live_index.labels.count("dup") == 1
    assert events.live_index.children("old") == ()
    assert events.live_index.parents("dup") == ("new",)


# --------------------------------------------------------------------------- N9


def test_release_runtime_sidecars_detaches_runtime_handles() -> None:
    """release_runtime_sidecars must drop the handles its docstring names (N9)."""

    events = CaptureEvents()
    op = _op_event("linear_1_1_raw")
    events.append(op)
    events.backend_session = _Handle()
    events.grad_fn_handles_by_label_raw[op.label_raw] = _Handle()
    events.recent_events = deque([object()])

    events.release_runtime_sidecars()

    assert events.backend_session is None
    assert events.grad_fn_handles_by_label_raw == {}
    assert len(events.recent_events) == 0
    # Structural facts are retained (payload-free op events).
    assert len(events.op_events) == 1


def test_release_runtime_sidecars_lets_handles_be_collected() -> None:
    """The detached runtime handles must become garbage-collectible (N9)."""

    events = CaptureEvents()
    op = _op_event("linear_1_1_raw")
    events.append(op)
    session = _Handle()
    grad_handle = _Handle()
    events.backend_session = session
    events.grad_fn_handles_by_label_raw[op.label_raw] = grad_handle
    session_ref = weakref.ref(session)
    grad_ref = weakref.ref(grad_handle)

    events.release_runtime_sidecars()
    del session, grad_handle
    gc.collect()

    assert session_ref() is None
    assert grad_ref() is None


# -------------------------------------------------------------------------- N10


def _seeded_capture_events() -> tuple[CaptureEvents, OpEvent]:
    """Build a CaptureEvents with one event carrying mutable dict fields."""

    op = dataclasses.replace(
        _op_event("linear_1_1_raw"),
        transform_config={"orig": 1},
        parent_arg_positions={"args": {0: "input_1_0_raw"}, "kwargs": {}},
    )
    events = CaptureEvents()
    events.append(op)
    return events, op


def test_copy_for_replay_does_not_share_mutable_event_dicts() -> None:
    """A replay projection must not mutate the sealed source event dicts (N10)."""

    source, op = _seeded_capture_events()
    projection = source._copy_for_replay()

    # Mutate every mutable projection event field across all three lanes.
    projection.op_events[0].transform_config["INJECTED"] = 999
    projection.op_events[0].parent_arg_positions["args"][7] = "PWNED"
    projection.op_event_by_label_raw[op.label_raw].transform_config["VIA_BYLABEL"] = 1
    projection.live_index.by_raw_label[op.label_raw].parent_arg_positions["kwargs"]["k"] = "v"

    assert source.op_events[0].transform_config == {"orig": 1}
    assert source.op_events[0].parent_arg_positions == {"args": {0: "input_1_0_raw"}, "kwargs": {}}
    assert source.op_event_by_label_raw[op.label_raw].transform_config == {"orig": 1}
    assert source.live_index.by_raw_label[op.label_raw].parent_arg_positions == {
        "args": {0: "input_1_0_raw"},
        "kwargs": {},
    }


def test_copy_for_replay_projection_lanes_reference_same_event() -> None:
    """All projection lanes must reference one independent cloned event (N10)."""

    source, op = _seeded_capture_events()
    projection = source._copy_for_replay()

    cloned = projection.op_events[0]
    assert projection.op_event_by_label_raw[op.label_raw] is cloned
    assert projection.live_index.by_raw_label[op.label_raw] is cloned
    # The clone is a distinct object from the sealed source event.
    assert cloned is not source.op_events[0]


def test_copy_for_replay_projections_are_mutually_independent() -> None:
    """Two projections of one source must not affect each other (N10)."""

    source, _op = _seeded_capture_events()
    first = source._copy_for_replay()
    second = source._copy_for_replay()

    first.op_events[0].transform_config["X"] = 1

    assert "X" not in second.op_events[0].transform_config
    assert "X" not in source.op_events[0].transform_config
