"""Shared journal-grabbing probe for the P4 test files.

Intercepts step-0 materialization to hand tests the REAL journal entries a
tiny capture produced, without depending on postprocess internals beyond the
one compat seam every path routes through.

Since P7 there is ONE torch producer (decomposed ``OpRecord``). The
``"legacy"`` leg models the compat ``OpEvent`` journal shape that preview
backends still emit until S15: its templates are genuine ``OpEvent``s
synthesized from a decomposed capture through the retained inverse adapter
(``op_event_from_record``), which dies with ``OpEvent`` in S15.
"""

from __future__ import annotations

import importlib
from typing import Any

import torch
from torch import nn

import torchlens as tl

from ._oracle_adapter import op_event_from_record

_SEED = 20260812


class ProbeCNN(nn.Module):
    """Conv + relu + fc: enough op/facet variety for template events."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(_SEED + 60)
        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.fc = nn.Linear(2 * 4 * 4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv(x))
        return self.fc(h.flatten(1))


def probe_input() -> torch.Tensor:
    torch.manual_seed(_SEED + 61)
    return torch.randn(2, 1, 4, 4)


def grab_journal_events(monkeypatch: Any, producer: str) -> list[Any]:
    """Return raw journal entries from a tiny capture at the step-0 seam.

    ``producer`` selects the journal SHAPE: ``"decomposed"`` returns the
    captured ``OpRecord`` rows; ``"legacy"`` returns genuine compat
    ``OpEvent``s projected through the inverse adapter (the preview-journal
    stand-in until S15).

    Parameters
    ----------
    monkeypatch:
        Patch controller used to intercept materialization.
    producer:
        Journal shape to return.

    Returns
    -------
    list[Any]
        Detached journal templates from the tiny capture.
    """

    assert producer in ("legacy", "decomposed"), producer
    postprocess_module = importlib.import_module("torchlens._postprocess")
    materialize_module = importlib.import_module("torchlens._postprocess._materialize")
    original = materialize_module.materialize_from_events
    grabbed: list[Any] = []

    def observing(trace: Any, events: Any) -> None:
        if not grabbed:
            grabbed.extend(events.op_events)
        original(trace, events)

    monkeypatch.setattr(postprocess_module, "materialize_from_events", observing)
    monkeypatch.setattr(materialize_module, "materialize_from_events", observing)
    trace = tl.trace(ProbeCNN(), probe_input())
    try:
        assert grabbed, "journal interception grabbed no events"
        if producer == "legacy":
            return [op_event_from_record(entry) for entry in grabbed]
        return grabbed
    finally:
        trace.cleanup()
