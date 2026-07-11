"""Stage 4b dual-projection parity tests."""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from torchlens.capture.projectors import RecordingProjector, TraceProjector


class DualProjectionToy(nn.Module):
    """Small mixed-operation model for sealed-core projection parity."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run two operations with distinct values."""

        return torch.relu(x + 1)


def test_one_execution_has_trace_and_recording_projection_parity() -> None:
    """Trace and Recording projections agree on facts from one sealed core."""

    recording = tl.record(
        DualProjectionToy(),
        torch.tensor([[-2.0, 3.0]]),
        save=lambda _ctx: True,
        random_seed=47,
    )

    assert len(recording._captured_run_cores) == 1
    core = recording._captured_run_cores[0]
    trace_events = TraceProjector(core).events()
    sparse = RecordingProjector().project((core,))

    assert tuple(
        (fact.event_id.raw_index, fact.event_id.label_raw) for fact in core.event_facts
    ) == tuple((event.raw_index, event.label_raw) for event in trace_events)
    assert [record.ctx.label_raw for record in sparse.records] == [
        event.label_raw for event in trace_events if event.predicate_matched
    ]
    assert [record.spec for record in sparse.records] == [
        event.capture_spec for event in trace_events if event.predicate_matched
    ]
    assert [record.ctx.label_raw for record in recording.records] == [
        record.ctx.label_raw for record in sparse.records
    ]
    for record, event in zip(
        sparse.records,
        (event for event in trace_events if event.predicate_matched),
    ):
        assert record.ram_payload is event.output.tensor.payload
