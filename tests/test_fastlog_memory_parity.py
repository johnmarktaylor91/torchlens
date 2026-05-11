"""Regression tests for fastlog's lightweight event projection."""

from __future__ import annotations

import sys

import torch
from torch import nn

import torchlens as tl


class ManyOps(nn.Module):
    """Model that emits many cheap tensor operations."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a fixed chain of elementwise operations."""

        for _ in range(500):
            x = x + 1
        return x


def test_fastlog_nonmatching_ops_do_not_materialize_oplogs() -> None:
    """Non-matching fastlog events stay lightweight and have no OpLog bridge."""

    recording = tl.fastlog.record(ManyOps(), torch.ones(1), keep_op=lambda ctx: False)
    events = recording.recording_trace.events
    op_events = recording._capture_events.op_events  # noqa: SLF001

    assert len(recording) == 0
    assert len(events) >= 500
    assert all(not hasattr(event, "materialized_log") for event in op_events)
    assert sys.getsizeof(events) / len(events) < 128
