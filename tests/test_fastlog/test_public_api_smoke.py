"""Smoke tests for the public fastlog API."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.fastlog import RecordContext, RecorderStateError


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


def test_record_keep_op_true_and_false() -> None:
    """One-shot record honors constant true and false operation predicates."""

    x = torch.ones(1, 3)
    kept = tl.fastlog.record(SimpleMlp(), x, keep_op=_keep_all_ops)
    skipped = tl.fastlog.record(SimpleMlp(), x, keep_op=_keep_no_ops)

    assert len(kept.records) > 0
    assert skipped.records == []


def test_record_multi_arg_and_kwargs_forward() -> None:
    """One-shot record supports tuple args plus explicit kwargs."""

    x = torch.ones(1, 3)
    y = torch.ones(1, 3)
    recording = tl.fastlog.record(
        MultiArgModel(),
        (x, y),
        {"scale": 2},
        keep_op=_keep_all_ops,
    )

    assert len(recording.records) > 0


def test_record_single_tensor_input_shorthand() -> None:
    """One-shot record treats a single tensor as one positional argument."""

    recording = tl.fastlog.record(SimpleMlp(), torch.ones(1, 3), keep_op=_keep_all_ops)

    assert len(recording.records) > 0


def test_recorder_context_records_multiple_forwards() -> None:
    """Recorder accumulates explicitly logged forwards across a loop."""

    with tl.fastlog.Recorder(SimpleMlp(), keep_op=_keep_all_ops) as recorder:
        for _ in range(5):
            recorder.log(torch.ones(1, 3))

    assert recorder.recording.n_passes == 5


def test_direct_model_call_inside_recorder_block_does_not_capture() -> None:
    """Only Recorder.log opens the capture scope."""

    model = SimpleMlp()
    with tl.fastlog.Recorder(model, keep_op=_keep_all_ops) as recorder:
        recorder.log(torch.ones(1, 3))
        before = len(recorder._state.recording.records)  # noqa: SLF001
        model(torch.ones(1, 3))
        after = len(recorder._state.recording.records)  # noqa: SLF001

    assert after == before


def test_dry_run_returns_events_without_tensor_payloads() -> None:
    """Dry-run returns event contexts and no retained tensor payloads."""

    trace = tl.fastlog.dry_run(SimpleMlp(), torch.ones(1, 3), keep_op=_keep_all_ops)

    assert trace.events
    assert all(not hasattr(event, "ram_payload") for event in trace.events)
    assert all(not hasattr(event, "disk_payload") for event in trace.events)


def test_recorder_recording_before_exit_raises() -> None:
    """Recorder.recording is guarded until __exit__ finalizes."""

    with tl.fastlog.Recorder(SimpleMlp(), keep_op=_keep_all_ops) as recorder:
        with pytest.raises(RecorderStateError):
            _ = recorder.recording
