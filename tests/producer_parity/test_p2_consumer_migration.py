"""P2 gates: reducer passthrough, handle-index reads, dead-field verification."""

from __future__ import annotations

import pytest

import torchlens as tl
from torchlens._ir.capture_events import CaptureEvents
from torchlens._utils.hashing import compute_raw_event_shape_hash

from ._models import SmallCNN, _cnn_input

# Markers are additive: a file-level smoke pytestmark would keep the heavy test
# in the `-m smoke` tier, so tier marks are applied per test instead.


def test_reducer_passthrough_is_the_raw_list_today() -> None:
    events = CaptureEvents()
    assert events.amended_op_records() is events.op_events
    assert events.amended_op_record("missing") is None


def test_hash_reads_through_the_reducer(monkeypatch: pytest.MonkeyPatch) -> None:
    """The persisted hash consumes the folded view, not the raw list."""

    import torchlens._postprocess as postprocess_module
    import torchlens._postprocess._materialize as materialize_module

    seen: dict = {}
    original = materialize_module.materialize_from_events

    def spy(trace, events):
        reads: list = []
        real = events.amended_op_records

        def recording():
            reads.append(True)
            return real()

        events.amended_op_records = recording  # type: ignore[method-assign]
        seen["hash_via_reducer"] = compute_raw_event_shape_hash(events)
        seen["reducer_reads"] = len(reads)
        events.amended_op_records = real  # type: ignore[method-assign]
        seen["hash_raw"] = compute_raw_event_shape_hash(events)
        original(trace, events)

    monkeypatch.setattr(postprocess_module, "materialize_from_events", spy)
    monkeypatch.setattr(materialize_module, "materialize_from_events", spy)
    tl.trace(SmallCNN(), _cnn_input())
    assert seen["reducer_reads"] >= 1, "hash did not read through the reducer"
    assert seen["hash_via_reducer"] == seen["hash_raw"], "passthrough must be byte-identical"
