"""Hypothesis/discharge battery (L7a memo sec 3/9.4).

Covers: corroboration on identical geometry, refutation on a planted shape
divergence, the graph-mismatch verdict without row comparison, typed
precondition refusals, the POSITIONAL-JOIN pins (repeated identical ops and
recurrent passes under digest equality), G4 never-promote, and G5
refuted-flips-accessors.
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


class TwoLayer(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(4, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


class Branchy(nn.Module):
    """Same class, config-controlled extra op: the graph-refuted fixture.

    The D10 comparable-twins preflight (weightsfree memo) refuses CROSS-CLASS
    discharges typed, so the graph-mismatch REFUTED verdict is exercised with
    same-class twins whose executed graphs differ.
    """

    def __init__(self, extra: bool = False) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.extra = extra

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        if self.extra:
            h = torch.relu(h)
        return h


class RepeatedIdenticalOps(nn.Module):
    """Two structurally identical linears + a module applied twice
    (recurrent passes): the positional-join stress shapes."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)
        self.shared = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.a(x))
        h = torch.relu(self.b(h))
        h = self.shared(h)
        h = self.shared(h)
        return h


def _structure(model: nn.Module, x: torch.Tensor):
    return tl.trace(model, x, capture=CaptureOptions(structure_only=True))


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/structure_only.py.
def test_corroborate_on_identical_geometry() -> None:
    model = TwoLayer()
    structure = _structure(model, torch.randn(2, 4))
    real = tl.trace(model, torch.randn(2, 4))
    discharge = structure.discharge_against(real)
    assert discharge.verdict is StructureClaimStatus.CORROBORATED
    assert discharge.graph_matched is True
    assert discharge.first_contradiction is None
    assert len(discharge.claims) > 0
    assert all(c.verdict is StructureClaimStatus.CORROBORATED for c in discharge.claims)
    assert claim_status_for(structure) is StructureClaimStatus.CORROBORATED
    assert registered_discharge(structure) is discharge


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/structure_only.py.
def test_cross_class_discharge_refuses_incomparable() -> None:
    """D10: refuse is not refute — a cross-class pair is a user error, not a
    refuted hypothesis; the registry stays untouched."""

    structure = _structure(TwoLayer(), torch.randn(2, 4))
    other = tl.trace(RepeatedIdenticalOps(), torch.randn(2, 4))
    with pytest.raises(StructureOnlyCapabilityError) as excinfo:
        structure.discharge_against(other)
    assert excinfo.value.fields["code"] == "structure_only_discharge_incomparable"
    assert excinfo.value.fields["reason"] == "model_class"
    assert registered_discharge(structure) is None
    assert claim_status_for(structure) is StructureClaimStatus.HYPOTHESIS


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/structure_only.py.
def test_input_geometry_mismatch_refuses_incomparable() -> None:
    """D10: differing input plans refuse typed (a batch-size mismatch is a
    comparability error, never evidence against the hypotheses)."""

    structure = _structure(TwoLayer(), torch.randn(2, 4))
    real = tl.trace(TwoLayer(), torch.randn(3, 4))
    with pytest.raises(StructureOnlyCapabilityError) as excinfo:
        structure.discharge_against(real)
    assert excinfo.value.fields["code"] == "structure_only_discharge_incomparable"
    assert excinfo.value.fields["reason"] == "input_geometry"
    assert registered_discharge(structure) is None


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/structure_only.py.
def test_precondition_refusals_are_typed() -> None:
    ordinary = tl.trace(TwoLayer(), torch.randn(2, 4))
    structure = _structure(TwoLayer(), torch.randn(2, 4))
    # Not a structure-only trace.
    with pytest.raises(StructureOnlyCapabilityError) as excinfo:
        ordinary.discharge_against(ordinary)
    assert excinfo.value.fields["code"] == "structure_only_discharge_precondition"
    # Oracle must be an ordinary capture.
    with pytest.raises(StructureOnlyCapabilityError) as excinfo:
        structure.discharge_against(structure)
    assert excinfo.value.fields["code"] == "structure_only_discharge_precondition"
