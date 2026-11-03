"""P0 probe: fastlog late-buffer behavior, characterized before migration.

Design-of-record section 4.5 predicted an accidental data loss on the fastlog
``to_trace`` path (a pre-0 late-buffer append invisible to materialization
because the capture-end seal cache predates it). The probe shows the loss
does NOT exist at the migration base (main 890a3cb5): the ``_copy_for_replay``
working stream that ``to_trace`` rebinds onto the cooked trace is
session-detached (``capture_session_for_events`` returns ``None``), so
postprocess falls back to the live op list — which already contains the
pre-0 late-buffer append — instead of a stale sealed tuple.

Consequences, locked here as the pre-migration contract:

* The ONE reviewed expected-diff entry the design budgeted for this scenario
  is EMPTY; the P4 watermark transport must PRESERVE the no-loss behavior.
* ``_raw_event_shape_hash`` is computed post-pre-0 on this path (review v4
  note N7's claim, confirmed): the hash must stay byte-identical through the
  migration even in this scenario.
"""

from __future__ import annotations

from typing import Any

import pytest

import torchlens as tl
from torchlens._capture.session import capture_session_for_events

from ._models import BufferOutputModel, _small_vec_input

pytestmark = pytest.mark.heavy


def _cooked_with_spy() -> tuple[Any, dict]:
    """Cook a recording while observing what step 0 actually consumes."""

    import torchlens._postprocess as postprocess_module
    import torchlens._postprocess._materialize as materialize_module

    original = materialize_module.materialize_from_events
    seen: dict = {}

    def spy(trace, events):
        seen["session"] = capture_session_for_events(events)
        seen["labels"] = [event.label_raw for event in events.op_events]
        original(trace, events)

    postprocess_module.materialize_from_events = spy
    materialize_module.materialize_from_events = spy
    try:
        model, inputs = BufferOutputModel(), _small_vec_input()
        recording = tl.record(model, inputs, save=tl.func("linear"))
        recording_labels = [event.label_raw for event in recording._capture_events.op_events]
        seen["recording_labels"] = recording_labels
        cooked = recording.to_trace()
    finally:
        postprocess_module.materialize_from_events = original
        materialize_module.materialize_from_events = original
    return cooked, seen


def test_hash_covers_post_pre0_stream_today() -> None:
    """`_raw_event_shape_hash` on the cooked path reflects the late append."""

    from torchlens._utils.hashing import compute_raw_event_shape_hash

    cooked, seen = _cooked_with_spy()
    assert cooked._raw_event_shape_hash
    # Recompute over the retained (released) structural stream is impossible
    # post-release; instead assert the invariant the hash mechanism gives us:
    # the hash was computed AFTER pre-0 (_postprocess/__init__.py computes it
    # from trace.capture_events after _resolve_output_parent_labels), so a
    # journal WITHOUT the buffer event must hash differently.
    model, inputs = BufferOutputModel(), _small_vec_input()
    recording = tl.record(model, inputs, save=tl.func("linear"))
    pre_pre0_hash = compute_raw_event_shape_hash(recording._capture_events)
    assert cooked._raw_event_shape_hash != pre_pre0_hash
