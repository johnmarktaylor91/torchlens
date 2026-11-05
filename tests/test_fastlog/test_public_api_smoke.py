"""Smoke tests for the public fastlog API."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import RecorderStateError
from torchlens.recording import RecordContext


class SimpleMlp(nn.Module):
    """Small MLP used by public API smoke tests."""

    def __init__(self) -> None:
        """Initialize the layers."""

        super().__init__()
        self.layers = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP."""

        return self.layers(x)


class MultiArgModel(nn.Module):
    """Model with multiple positional inputs and a keyword argument."""

    def forward(self, x: torch.Tensor, y: torch.Tensor, *, scale: int = 1) -> torch.Tensor:
        """Combine inputs with a scalar multiplier."""

        return (x + y) * scale


def _keep_all_ops(ctx: RecordContext) -> bool:
    """Keep every operation event."""

    return ctx.kind == "op"


def _keep_no_ops(ctx: RecordContext) -> bool:
    """Reject every operation event."""

    _ = ctx
    return False


def _assert_unique_label_lookups(recording: tl.Recording) -> None:
    """Assert each retained record is indexed exactly once per distinct label."""

    for record in recording.records:
        label_matches = recording[record.ctx.label]
        assert label_matches == [record]
        raw_label = record.ctx.raw_label
        if raw_label is not None:
            raw_matches = recording[raw_label]
            assert raw_matches == [record]


def _force_legacy_capture_event_projection(recording: tl.Recording) -> None:
    """Reset a lazy Recording to rebuild records from legacy ``_capture_events`` only."""

    assert object.__getattribute__(recording, "_capture_events") is not None
    object.__setattr__(recording, "_captured_run_cores", ())
    object.__getattribute__(recording, "records").clear()
    recording.by_pass.clear()
    recording.by_label.clear()
    recording.by_address.clear()
    object.__setattr__(recording, "_records_built", False)


def test_activation_payloads_by_raw_label_preserves_repeated_passes() -> None:
    """Repeated raw labels retain every pass payload instead of overwriting."""

    with tl.recording.Recorder(SimpleMlp(), save=_keep_all_ops) as recorder:
        recorder(torch.ones(1, 3))
        recorder(torch.ones(1, 3) * 2)

    recording = recorder.recording
    raw_label = recording.records[0].ctx.raw_label
    assert raw_label is not None
    payload = recording.activation_payloads_by_raw_label[raw_label]
    assert isinstance(payload, list)
    assert len(payload) == 2


def test_recorder_recording_before_exit_raises() -> None:
    """Recorder.recording is guarded until __exit__ finalizes."""

    with tl.recording.Recorder(SimpleMlp(), save=_keep_all_ops) as recorder:
        with pytest.raises(RecorderStateError):
            _ = recorder.recording
