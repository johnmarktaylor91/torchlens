"""Fork capture-outcome derivation (the anti-masquerade contract).

``Trace.fork()`` is the sanctioned mutation surface, so a fork must never
carry its parent's blessed settle stamp by identity: pre-fix, a hand-edited
fork saved (and re-loaded) as a bit-identical ATTESTED COMPLETE product.
Forks now settle a DERIVED outcome through the same structural lattice an
attestation-less artifact uses (R06 doctrine: derivation emits HALTED /
UNATTESTED / UNKNOWN, never a blessed COMPLETE), with fork provenance in the
settlement note. These tests pin the derivation, the persisted round-trip,
and the capability parity that makes the change safe.
"""

import os
import pickle
from collections.abc import Iterator

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.capture.outcome import CaptureStatus


class _Net(nn.Module):
    """Small two-layer net."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run both layers with a relu between."""

        return self.fc2(torch.relu(self.fc1(x)))


@pytest.fixture(scope="module")
def parent_trace() -> Iterator[tl.Trace]:
    """Return a captured, attested-COMPLETE trace."""

    torch.manual_seed(0)
    trace = tl.trace(_Net(), torch.randn(4, 8), save=tl.func("relu"))
    try:
        yield trace
    finally:
        trace.cleanup()


def test_fork_outcome_is_derived_never_aliased(parent_trace: tl.Trace) -> None:
    """A fork's outcome is a fresh DERIVED record, not the parent's stamp."""

    fork = parent_trace.fork()
    assert parent_trace.outcome.status is CaptureStatus.COMPLETE
    assert parent_trace.outcome.derived is False
    assert fork.outcome is not parent_trace.outcome
    assert fork.outcome.status is CaptureStatus.UNATTESTED
    assert fork.outcome.derived is True
    assert fork.outcome.settlement_note == "forked_from=complete"


def test_mutated_fork_cannot_save_as_attested_complete(parent_trace: tl.Trace, tmp_path) -> None:
    """The RED-capable masquerade case.

    Pre-fix a hand-edited fork persisted the parent's attested COMPLETE
    payload verbatim, and the load coherence matrix adopted it (finished,
    not halted -> coherent), so the artifact presented as a blessed capture
    it never was.
    """

    fork = parent_trace.fork()
    # A hand edit on the mutation surface: overwrite one saved activation.
    op = fork["relu_1_2"].ops[0]
    op.out = torch.zeros_like(op.out)
    path = os.path.join(tmp_path, "mutated_fork.tlspec")
    tl.save(fork, path)
    restored = tl.load(path)
    assert restored.outcome.status is not CaptureStatus.COMPLETE
    assert restored.outcome.status is CaptureStatus.UNATTESTED
    assert restored.outcome.derived is True


def test_fork_outcome_survives_pickle_coherently(parent_trace: tl.Trace) -> None:
    """The derived UNATTESTED record round-trips pickle without degrading."""

    fork = parent_trace.fork()
    restored = pickle.loads(pickle.dumps(fork))
    assert restored.outcome.status is CaptureStatus.UNATTESTED
    assert restored.outcome.derived is True
    assert restored.outcome.settlement_note == "forked_from=complete"


def test_fork_of_fork_records_unattested_provenance(parent_trace: tl.Trace) -> None:
    """A fork chain derives at every level and names its immediate parent."""

    fork = parent_trace.fork()
    grandfork = fork.fork()
    assert grandfork.outcome.status is CaptureStatus.UNATTESTED
    assert grandfork.outcome.settlement_note == "forked_from=unattested"


def test_fork_keeps_analysis_capabilities(parent_trace: tl.Trace, tmp_path) -> None:
    """Capability parity: UNATTESTED loses nothing a fork could do before."""

    fork = parent_trace.fork()
    # save_analysis allows UNATTESTED.
    path = os.path.join(tmp_path, "fork.tlspec")
    tl.save(fork, path)
    assert tl.load(path)["relu_1_2"] is not None
    # validation entry allows UNATTESTED (the tripwire stays armed).
    from torchlens.capture.outcome import require_capture_capability

    require_capture_capability(fork, "validation_entry")
    require_capture_capability(fork, "backward")
