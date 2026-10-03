"""Inverse-adapter identity proof: inserting the adapter changes nothing."""

from __future__ import annotations

import pytest

import torchlens as tl

from ._models import SmallCNN, _cnn_input
from ._oracle_adapter import op_event_from_record


def test_adapter_is_identity_on_op_events(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compat ``OpEvent`` inputs pass through the adapter unchanged.

    Post-P7 torch journals hold ``OpRecord`` rows; the identity leg guards
    the adapter's behavior on the compat ``OpEvent`` shape preview backends
    still emit until S15. Records project to genuine events (non-identity),
    and re-projecting those events is the identity.
    """

    import torchlens.postprocess as postprocess_module
    import torchlens.postprocess._materialize as materialize_module
    from torchlens.ir.events import OpEvent
    from torchlens.ir.op_record import OpRecord

    captured: list = []
    original = materialize_module.materialize_from_events

    def spy(trace, events):
        captured.extend(events.op_events)
        original(trace, events)

    monkeypatch.setattr(postprocess_module, "materialize_from_events", spy)
    monkeypatch.setattr(materialize_module, "materialize_from_events", spy)
    tl.trace(SmallCNN(), _cnn_input())

    assert captured
    assert all(isinstance(record, OpRecord) for record in captured)
    events = [op_event_from_record(record) for record in captured]
    assert all(isinstance(event, OpEvent) for event in events)
    for event in events:
        assert op_event_from_record(event) is event
