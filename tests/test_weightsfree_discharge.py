"""Discharge preflight + W1-RPT + identity-v0 envelope (memo items 10/12).

The comparable-twins preflight REFUSES typed and never refutes: the registry
stays untouched and the structure trace keeps its HYPOTHESIS status. The
report names structural deltas before any bare "digests differ" fallback.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._capture.structure_only import (
    StructureClaimStatus,
    StructureOnlyCapabilityError,
    claim_status_for,
    registered_discharge,
)
from torchlens.options import CaptureOptions


class Toy(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(4, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


def _twins() -> tuple[object, object]:
    structure = tl.trace(Toy(), torch.randn(2, 4), capture=CaptureOptions(structure_only=True))
    real = tl.trace(Toy(), torch.randn(2, 4))
    return structure, real


def _expect_incomparable(structure, real, reason: str) -> None:
    with pytest.raises(StructureOnlyCapabilityError) as excinfo:
        structure.discharge_against(real)
    assert excinfo.value.fields["code"] == "structure_only_discharge_incomparable"
    assert excinfo.value.fields["reason"] == reason
    # Refuse is NOT refute: nothing registered, status still HYPOTHESIS.
    assert registered_discharge(structure) is None
    assert claim_status_for(structure) is StructureClaimStatus.HYPOTHESIS


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/structure_only.py.
def test_rescue_asymmetry_refuses() -> None:
    """A rescue re-run on exactly one side breaks comparability (D10)."""

    structure, real = _twins()
    real._rescue_rerun = {"trigger": "test"}
    _expect_incomparable(structure, real, "rescue_asymmetry")


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/structure_only.py.
def test_opaque_witness_on_oracle_refuses() -> None:
    """A real oracle carrying an opaque host-write witness fails preflight
    (memo D7 composition rule): an UNVERIFIABLE oracle corroborates nothing."""

    from torchlens.backends.torch.completeness_witness import (
        _HOST_ESCAPE_MUTABLE_WRITEBACK,
    )

    structure, real = _twins()
    _HOST_ESCAPE_MUTABLE_WRITEBACK.add(real)
    try:
        _expect_incomparable(structure, real, "real_oracle_opaque_witness")
    finally:
        _HOST_ESCAPE_MUTABLE_WRITEBACK.discard(real)


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/structure_only.py.
def test_report_is_one_screen_and_names_verdict() -> None:
    structure, real = _twins()
    discharge = structure.discharge_against(real)
    text = discharge.report()
    assert "CORROBORATED" in text
    assert "identity-v0" in text
    assert "claims by kind" in text
    assert len(text.splitlines()) <= 12  # one screen


@pytest.mark.smoke
def test_rpt_names_count_delta_structurally() -> None:
    """W1-RPT: a record-count divergence is named structurally, never as a
    bare digest difference (3 phantom records once produced 300 apparent
    diffs and zero per-claim rows, costing two panel labs a round)."""

    class MaybeExtra(nn.Module):
        def __init__(self, extra: bool) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 4)
            self.extra = extra

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.fc(x)
            if self.extra:
                h = torch.relu(torch.relu(h))
            return h

    structure = tl.trace(
        MaybeExtra(False), torch.randn(2, 4), capture=CaptureOptions(structure_only=True)
    )
    real = tl.trace(MaybeExtra(True), torch.randn(2, 4))
    discharge = structure.discharge_against(real)
    assert discharge.verdict is StructureClaimStatus.REFUTED
    assert "structural alignment" in discharge.first_contradiction
    assert "digests differ" not in discharge.first_contradiction
    assert "structural alignment" in discharge.report()
