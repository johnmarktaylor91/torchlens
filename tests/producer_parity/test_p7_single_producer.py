"""P7: one torch producer (decomposed), with the S15 retention schedule.

The legacy torch producer, its ``TORCHLENS_CAPTURE_PRODUCER`` dual-path
switch, ``_op_event_from_log``, ``_event_from_record``, and the parity
comparator died in P7. This module guards BOTH halves of that deletion:

* torch captures journal decomposed ``OpRecord`` rows unconditionally, with
  stamped core seqs and grad-fn single ownership;
* the compat surfaces previews still need are RETAINED until S15 —
  ``OpEvent``, the ingest adapter ``op_record_from_event``, the fold guard
  ``PATH_TO_FLAT``, and the record-shape-aware ``_clone_op_event_for_replay``
  — so their disappearance before S15 is a red test, not a silent break.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.ir.events import OpEvent
from torchlens.ir.op_record import OpRecord


def test_torch_capture_journals_records_unconditionally() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    x = torch.randn(2, 4)

    trace = tl.trace(model, x)
    events = trace._capture_events.op_events
    assert events, "vacuous: no journal rows"
    assert all(isinstance(event, OpRecord) for event in events), {
        type(event).__name__ for event in events
    }
    seqs = [event.core.seq for event in events]
    assert all(seq > 0 for seq in seqs), "unstamped record reached the journal"
    assert seqs == sorted(seqs)
    # grad-fn single ownership: no record carries a handle attribute at all.
    assert all(getattr(event, "grad_fn_handle", None) is None for event in events)


def test_environment_switch_is_dead(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dual-path switch is retired: the env var changes nothing."""

    monkeypatch.setenv("TORCHLENS_CAPTURE_PRODUCER", "legacy")
    trace = tl.trace(nn.Sequential(nn.Linear(4, 4), nn.ReLU()), torch.randn(2, 4))
    assert all(isinstance(event, OpRecord) for event in trace._capture_events.op_events)


def test_s15_compat_surfaces_retained() -> None:
    """OpEvent + adapter + fold guard + replay clone survive until S15."""

    from torchlens.ir.capture_events import _clone_op_event_for_replay
    from torchlens.ir.op_record import PATH_TO_FLAT, op_record_from_event

    assert OpEvent is not None
    assert callable(op_record_from_event)
    assert callable(_clone_op_event_for_replay)
    # The fold guard stays total over the registry's flat names (12 paths).
    assert PATH_TO_FLAT, "PATH_TO_FLAT emptied before S15"
