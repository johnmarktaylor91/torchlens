"""Smoke coverage for predicate-mode capture dispatchers."""

from __future__ import annotations

import time

import torch
from torch import nn

from torchlens import _state
from torchlens.backends.torch.sources import log_source_tensor
from torchlens.data_classes.trace import Trace
from torchlens.backends.torch.model_prep import (
    _cleanup_model_session,
    _ensure_model_prepared,
    _prepare_model_session,
)
from torchlens.capture.projections import RecordingState, active_recording_state
from torchlens.fastlog.options import RecordingOptions
from torchlens.fastlog.types import CaptureSpec, Recording


class TinyMlp(nn.Module):
    """Two-layer MLP for predicate dispatcher smoke tests."""

    def __init__(self) -> None:
        """Initialize the test network."""

        super().__init__()
        self.layers = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the tiny network."""

        return self.layers(x)


def _empty_recording(history_size: int) -> Recording:
    """Build an empty in-memory Recording for dispatcher tests."""

    return Recording(
        records=[],
        by_pass={},
        by_label={},
        by_address={},
        bundle_path=None,
        n_ops=1,
        n_records=0,
        start_times=[],
        end_times=[],
        predicate_failures=[],
        predicate_failure_overflow_count=0,
        keep_op_repr=None,
        keep_module_repr=None,
        history_size=history_size,
    )


def _run_dispatcher_smoke(
    model: nn.Module,
    x: torch.Tensor,
    options: RecordingOptions,
) -> Recording:
    """Run a minimal predicate pass through existing dispatchers."""

    trace = Trace("TinyMlp")
    trace.capture_mode = "predicate"
    trace.capture_start_time = time.time()
    state = RecordingState(options=options, recording=_empty_recording(options.history_size))
    _ensure_model_prepared(model)
    _prepare_model_session(trace, model)
    try:
        with active_recording_state(state), _state.active_logging(trace):
            log_source_tensor(trace, x, "input", "input.x")
            model(x)
    finally:
        _cleanup_model_session(model, [x])
    trace._fastlog_recording = state.recording
    return Recording.from_capture_events(trace)


def test_predicate_dispatchers_fire_and_store_ram_payloads() -> None:
    """Predicate mode calls predicates and stores RAM payloads through dispatchers."""

    seen_kinds: list[str] = []

    def keep_op(ctx) -> bool | CaptureSpec | None:
        """Keep operation events and record predicate invocation."""

        seen_kinds.append(ctx.kind)
        return ctx.kind == "op"

    def keep_module(ctx) -> bool:
        """Keep module events and record predicate invocation."""

        seen_kinds.append(ctx.kind)
        return True

    recording = _run_dispatcher_smoke(
        TinyMlp(),
        torch.ones(1, 3),
        RecordingOptions(
            keep_op=keep_op,
            keep_module=keep_module,
            default_op=False,
            default_module=False,
            include_source_events=True,
        ),
    )

    assert "input" in seen_kinds
    assert "op" in seen_kinds
    assert "module_enter" in seen_kinds
    assert "module_exit" in seen_kinds
    assert len(recording.records) > 0
    assert any(record.ctx.kind == "op" and record.ram_payload is not None for record in recording)
