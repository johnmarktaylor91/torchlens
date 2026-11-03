"""Public spellings of the Trace concern views and their text blocks."""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from torchlens.types import TraceHealth, TraceProvenance


def _tiny_trace() -> tl.Trace:
    """Capture a one-layer model."""

    torch.manual_seed(0)
    return tl.trace(nn.Sequential(nn.Linear(3, 2), nn.ReLU()), torch.randn(2, 3))


def test_trace_provenance_summary_is_public() -> None:
    """``trace.provenance.summary()`` returns the capture-provenance text block."""

    trace = _tiny_trace()
    try:
        assert isinstance(trace.provenance, TraceProvenance)
        assert isinstance(trace.health, TraceHealth)
        text = trace.provenance.summary()
        assert isinstance(text, str)
        assert "TorchLens Discoverability Summary" in text
        assert not hasattr(trace.provenance, "_summary")
    finally:
        trace.cleanup()
