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
from torchlens.capture.structure_only import (
    StructureClaimStatus,
    StructureOnlyCapabilityError,
    claim_status_for,
    registered_discharge,
    require_structure_only_capability,
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


def test_positional_join_covers_repeated_ops_and_recurrent_passes() -> None:
    model = RepeatedIdenticalOps()
    structure = _structure(model, torch.randn(2, 4))
    real = tl.trace(model, torch.randn(2, 4))
    discharge = structure.discharge_against(real)
    assert discharge.verdict is StructureClaimStatus.CORROBORATED
    # The join compared every layer positionally, including both passes of
    # the shared module and both identical linears.
    assert len({c.site for c in discharge.claims}) == len(structure.layer_list)


def test_refute_on_planted_shape_divergence() -> None:
    """The doctored-config row (weightsfree memo sec 6): same class, same
    input, drifted width — the graph digests agree but the shape hypotheses
    contradict; refuted with the first contradicting site named."""

    structure = _structure(TwoLayer(width=4), torch.randn(2, 4))
    real = tl.trace(TwoLayer(width=5), torch.randn(2, 4))
    discharge = structure.discharge_against(real)
    assert discharge.verdict is StructureClaimStatus.REFUTED
    assert discharge.first_contradiction is not None
    assert "shape" in discharge.first_contradiction
    assert claim_status_for(structure) is StructureClaimStatus.REFUTED


def test_graph_mismatch_refutes_at_structure_without_row_comparison() -> None:
    structure = _structure(Branchy(extra=False), torch.randn(2, 4))
    other = tl.trace(Branchy(extra=True), torch.randn(2, 4))
    discharge = structure.discharge_against(other)
    assert discharge.verdict is StructureClaimStatus.REFUTED
    assert discharge.graph_matched is False
    assert discharge.claims == ()
    # W1-RPT: the count/position delta is named STRUCTURALLY before any
    # bare "digests differ" fallback.
    assert "structural alignment" in discharge.first_contradiction


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


def test_never_promote_g4_and_refuted_flips_accessors_g5() -> None:
    model = TwoLayer()
    structure = _structure(model, torch.randn(2, 4))
    # G4: born hypothesis; ordinary reads never promote.
    assert claim_status_for(structure) is StructureClaimStatus.HYPOTHESIS
    _ = structure.layer_list
    _ = structure["relu_1_2"].shape
    assert claim_status_for(structure) is StructureClaimStatus.HYPOTHESIS
    # Hypothesis rows pass the chokepoint pre-discharge.
    row = require_structure_only_capability(structure, "shapes_dtypes")
    assert row is not None and row.status_v1 == "supported_hypothesis"
    # G5: a REFUTED discharge flips hypothesis consumers to typed refusals
    # (doctored-width twin: same class and input, drifted interior shapes).
    real = tl.trace(TwoLayer(width=6), torch.randn(2, 4))
    discharge = structure.discharge_against(real)
    assert discharge.verdict is StructureClaimStatus.REFUTED
    with pytest.raises(StructureOnlyCapabilityError) as excinfo:
        require_structure_only_capability(structure, "shapes_dtypes")
    assert excinfo.value.fields["code"] == "structure_only_refuted_hypothesis"
    # Structural rows stay readable (the graph IS a structural fact).
    assert require_structure_only_capability(structure, "graph_structure") is not None


def test_discharge_never_mutates_either_trace() -> None:
    model = TwoLayer()
    structure = _structure(model, torch.randn(2, 4))
    real = tl.trace(model, torch.randn(2, 4))
    before_structure = set(vars(structure))
    before_real = set(vars(real))
    structure.discharge_against(real)
    assert set(vars(structure)) == before_structure
    assert set(vars(real)) == before_real
    assert real.structure_only is False
