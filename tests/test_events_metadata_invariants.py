"""The engine hook slot for the metadata-invariant tripwire (torchlens._events).

Validation is a tripwire: the engine's backends must run ``check_metadata_invariants``
through the slot, the slot must be filled by importing ``torchlens.validation``, and an
empty slot must load the tripwire instead of skipping it.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _events
from torchlens.errors import MetadataInvariantError
from torchlens.validation import check_metadata_invariants


@pytest.fixture
def restore_hook() -> Iterator[None]:
    before = _events.registered_metadata_invariants()
    try:
        yield
    finally:
        _events._METADATA_INVARIANTS = before


def _plain_trace() -> tl.Trace:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    return tl.trace(model, torch.randn(2, 4))


def test_importing_validation_registers_the_tripwire() -> None:
    assert _events.registered_metadata_invariants() is check_metadata_invariants


def test_plain_capture_runs_the_registered_tripwire_once(restore_hook: None) -> None:
    calls: list[object] = []

    def counting(trace: tl.Trace) -> bool:
        calls.append(trace)
        return check_metadata_invariants(trace)

    _events.register_metadata_invariants(counting)
    trace = _plain_trace()
    assert _events.run_metadata_invariants(trace) is True
    assert len(calls) == 1 and calls[0] is trace


def test_empty_slot_loads_the_tripwire_instead_of_skipping(restore_hook: None) -> None:
    _events._METADATA_INVARIANTS = None
    trace = _plain_trace()
    assert _events.run_metadata_invariants(trace) is True
    assert _events.registered_metadata_invariants() is check_metadata_invariants


def test_tripwire_failures_propagate_through_the_slot(restore_hook: None) -> None:
    def failing(trace: tl.Trace) -> bool:
        raise MetadataInvariantError("hook_probe", "deliberate failure")

    _events.register_metadata_invariants(failing)
    with pytest.raises(MetadataInvariantError, match=r"\[hook_probe\] deliberate failure"):
        _events.run_metadata_invariants(_plain_trace())
