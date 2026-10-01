"""Conformance packs C0-C2, profiles, mutation plants (MEMO 4, build B4).

Anti-vacuity, executed: the well-behaved fake provider passes every cell
BEFORE any plant is trusted to fail one; every plant in the closed set turns
its intended cell red; toys earn NOTHING (the runner refuses every claim on
config-built evidence); the report is canonical and its attestation digest
recomputes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from torchlens.conformance import (
    CLAIM_MIN_PRETRAINED_FAMILIES,
    FAKE_PLANTS,
    ConformanceReport,
    RosterModel,
    claim_string,
    run_conformance,
)
from torchlens.conformance._adapters import resolve_adapter
from torchlens.errors import ConfigurationError

pytestmark = pytest.mark.smoke


def _build_tiny() -> tuple[Any, Any]:
    """Zero-arg factory for the cheap hermetic roster model."""

    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)
    ).eval()
    return model, torch.randn(2, 8)


def _tiny_roster(realism: str = "config_built", families: int = 1) -> tuple[RosterModel, ...]:
    """Cheap roster rows; realism is a TEST INPUT to the adjudicator here."""

    evidence = "pinned-rev-deadbeef" if realism == "pretrained" else ""
    return tuple(
        RosterModel(
            family=f"tiny{index}", realism=realism, build=_build_tiny, checkpoint_evidence=evidence
        )
        for index in range(families)
    )


def _rows(report: ConformanceReport, case_id: str) -> list[Any]:
    """Collect the result rows for one case id."""

    return [row for row in report.results if row.case_id == case_id]


# ---------------------------------------------------------------------------
# The reference ladder and the well-behaved fake (harness sanity first).
# ---------------------------------------------------------------------------


def test_torch_reference_passes_the_full_ladder(tmp_path: Path) -> None:
    """The reference backend passes C0-C2 on the hermetic roster."""

    report = run_conformance(
        "torch", tiers=("C0", "C1", "C2"), roster=_tiny_roster(), workdir=tmp_path
    )
    assert report.funnel["failed"] == 0
    assert report.funnel["skipped"] == 0
    assert report.earned_tier == "C2"
    assert report.claim_eligible is False  # toys earn nothing
    with pytest.raises(ConfigurationError) as excinfo:
        claim_string(report)
    assert excinfo.value.fields["code"] == "conformance_claim_unearned"


def test_wellbehaved_fake_passes_before_plants_are_trusted(tmp_path: Path) -> None:
    """Plant sanity: the unplanted fake passes every cell (never eligible)."""

    report = run_conformance(
        "fake", tiers=("C0", "C1", "C2"), roster=_tiny_roster(), workdir=tmp_path
    )
    assert report.funnel["failed"] == 0
    assert report.earned_tier == "C2"
    assert report.claim_eligible is False  # the fake provider never claims
    with pytest.raises(ConfigurationError):
        claim_string(report)


# ---------------------------------------------------------------------------
# Mutation plants: every plant turns its INTENDED cell red.
# ---------------------------------------------------------------------------

_PLANT_INTENDED_CELL = {
    "structural_failure": ("c1_capture_completes", ("C0", "C1")),
    "capability_undispatched": ("c1_capabilities_consumed", ("C0", "C1")),
    "output_corruption": ("c1_reference_parity", ("C0", "C1")),
    "roundtrip_field_loss": ("c2_structure_equal", ("C0", "C1", "C2")),
    "tamper_tripwire": ("c2_roundtrip_completes", ("C0", "C1", "C2")),
    "shared_adapter_assumption": ("c2_reload_is_distinct", ("C0", "C1", "C2")),
}


def test_plant_table_covers_the_closed_set() -> None:
    """The intended-cell table and FAKE_PLANTS are the same closed set."""

    assert set(_PLANT_INTENDED_CELL) == set(FAKE_PLANTS)


@pytest.mark.parametrize("plant", FAKE_PLANTS)
def test_every_plant_turns_its_intended_cell_red(plant: str, tmp_path: Path) -> None:
    """Each planted defect fails the exact cell it targets, nothing vacuous."""

    case_id, tiers = _PLANT_INTENDED_CELL[plant]
    report = run_conformance(
        "fake", tiers=tiers, roster=_tiny_roster(), workdir=tmp_path, _plant=plant
    )
    intended = _rows(report, case_id)
    assert intended, f"plant {plant} produced no {case_id} rows"
    assert any(row.outcome == "failed" for row in intended), plant
    assert report.earned_tier != tiers[-1]
    with pytest.raises(ConfigurationError) as excinfo:
        claim_string(report)
    assert excinfo.value.fields["code"] == "conformance_claim_unearned"


def test_unknown_plant_refuses_typed() -> None:
    """A plant outside the closed vocabulary refuses conformance_plant_unknown."""

    with pytest.raises(ConfigurationError) as excinfo:
        resolve_adapter("fake", plant="bogus_plant")
    assert excinfo.value.fields["code"] == "conformance_plant_unknown"


# ---------------------------------------------------------------------------
# Scope grammar, backend vocabulary, executed floor.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tiers", [("C1",), ("C0", "C2"), ("bogus",), ("C2", "C1", "C0")])
def test_non_cumulative_tier_requests_refuse(tiers: tuple[str, ...]) -> None:
    """Tiers must be a cumulative prefix from C0, in rung order."""

    with pytest.raises(ConfigurationError) as excinfo:
        run_conformance("fake", tiers=tiers)
    assert excinfo.value.fields["code"] == "conformance_scope_invalid"


def test_unknown_backend_refuses_typed() -> None:
    """v1 resolves exactly 'torch' and 'fake'; anything else refuses."""

    with pytest.raises(ConfigurationError) as excinfo:
        run_conformance("mystery-provider")
    assert excinfo.value.fields["code"] == "conformance_backend_unknown"
    assert "torchlens.ecosystem.plugins" in str(excinfo.value)


def test_executed_floor_zero_green_never_reads_green() -> None:
    """Fewer executed checks than the floor earns NO tier however green."""

    report = run_conformance("fake", tiers=("C0",), executed_floor=10)
    assert report.funnel["passed"] == report.funnel["executed"] == 3
    assert report.earned_tier == ""
    with pytest.raises(ConfigurationError) as excinfo:
        claim_string(report)
    assert excinfo.value.fields["code"] == "conformance_claim_unearned"


# ---------------------------------------------------------------------------
# The claim grammar (MEMO 4.2, closed).
# ---------------------------------------------------------------------------


def test_c0_claim_never_says_conformant() -> None:
    """C0 earns 'provider API compatible' and NEVER 'conformant'."""

    report = run_conformance("torch", tiers=("C0",))
    assert report.earned_tier == "C0"
    assert report.claim_eligible is True
    claim = claim_string(report)
    assert claim == "TorchLens provider API compatible, C0"
    assert "conformant" not in claim


def test_pretrained_family_floor_gates_c1_claims(tmp_path: Path) -> None:
    """C1 claims need >= 2 pretrained families; the scope rides the string.

    Realism labels are TEST INPUTS here: the tiny rows carry
    realism='pretrained' to drive the adjudicator, exactly like a provider
    would present pinned-checkpoint rows.
    """

    assert CLAIM_MIN_PRETRAINED_FAMILIES == 2
    one = run_conformance("torch", tiers=("C0", "C1"), roster=_tiny_roster("pretrained", 1))
    assert one.earned_tier == "C1" and one.claim_eligible is False
    with pytest.raises(ConfigurationError):
        claim_string(one)
    two = run_conformance("torch", tiers=("C0", "C1"), roster=_tiny_roster("pretrained", 2))
    assert two.claim_eligible is True
    claim = claim_string(two)
    assert claim == "TorchLens-conformant capture, C1 (scope: tiny0,tiny1)"


def test_c2_claim_spelling(tmp_path: Path) -> None:
    """The C2 claim string is the durable-capture spelling with scope."""

    report = run_conformance(
        "torch",
        tiers=("C0", "C1", "C2"),
        roster=_tiny_roster("pretrained", 2),
        workdir=tmp_path,
    )
    assert claim_string(report) == "TorchLens-conformant durable capture, C2 (scope: tiny0,tiny1)"


# ---------------------------------------------------------------------------
# Canonical report bytes and the attestation digest.
# ---------------------------------------------------------------------------


def test_attestation_digest_recomputes_over_canonical_bytes() -> None:
    """The SHA-256 attests the digest-free canonical form, third-party checkable."""

    from torchlens.conformance import _canonical_report_json

    report = run_conformance("fake", tiers=("C0",))
    recomputed = hashlib.sha256(
        _canonical_report_json(report, include_attestation=False).encode("utf-8")
    ).hexdigest()
    assert report.attestation_sha256 == recomputed
    payload = json.loads(report.to_canonical_json())
    assert payload["attestation_sha256"] == recomputed
    assert payload["environment"]["torchlens_version"]
    # Canonical bytes are cwd-independent.
    assert str(Path.cwd()) not in report.to_canonical_json()


def test_report_funnel_discloses_every_requested_case() -> None:
    """requested == executed + skipped; the funnel is the complete result set."""

    report = run_conformance("fake", tiers=("C0",))
    assert report.funnel["requested"] == len(report.results)
    assert report.funnel["requested"] == report.funnel["executed"] + report.funnel["skipped"]
