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

from tests.conformance import (
    ConformanceReport,
    RosterModel,
    claim_string,
    run_conformance,
)
from tests.conformance._adapters import FAKE_PLANTS
from torchlens.errors import ConfigurationError


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
            family=f"tiny{index}",
            realism=realism,
            builder=_build_tiny,
            checkpoint_evidence=evidence,
        )
        for index in range(families)
    )


def _rows(report: ConformanceReport, case_id: str) -> list[Any]:
    """Collect the result rows for one case id."""

    return [row for row in report.results if row.case_id == case_id]


# ---------------------------------------------------------------------------
# The reference ladder and the well-behaved fake (harness sanity first).
# ---------------------------------------------------------------------------


def _failed_ids(report: ConformanceReport) -> set[str]:
    """Case ids of the failed rows."""

    return {row.case_id for row in report.results if row.outcome == "failed"}


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/prepare/dispatch.py, _user_funcs_guards.py, campaign mutant kills.
def test_torch_reference_passes_the_full_ladder(tmp_path: Path) -> None:
    """The reference backend passes C0-C2 on the hermetic roster."""

    report = run_conformance(
        "torch", tiers=("C0", "C1", "C2"), roster=_tiny_roster(), workdir=tmp_path
    )
    assert _failed_ids(report) == set()
    assert report.funnel["skipped"] == 0
    assert report.earned_tier == "C2"
    assert report.is_claim_eligible is False  # toys earn nothing
    with pytest.raises(ConfigurationError) as excinfo:
        claim_string(report)
    assert excinfo.value.fields["code"] == "conformance_claim_unearned"


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
    """This table plus the protocol-era table (``_protocol_cases``) is the closed set."""

    from tests.conformance._protocol_cases import PLANT_INTENDED_CELL

    assert set(_PLANT_INTENDED_CELL).isdisjoint(PLANT_INTENDED_CELL)
    assert set(_PLANT_INTENDED_CELL) | set(PLANT_INTENDED_CELL) == set(FAKE_PLANTS)


# Kept by shrink-r4 (per-id subset rule): sole coverage of _io/_root_manifest.py, _io/bundle.py, _io/format_contract.py ....
@pytest.mark.parametrize("plant", sorted(_PLANT_INTENDED_CELL))
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


# ---------------------------------------------------------------------------
# Scope grammar, backend vocabulary, executed floor.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The claim grammar (MEMO 4.2, closed).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Canonical report bytes and the attestation digest.
# ---------------------------------------------------------------------------


def test_attestation_digest_recomputes_over_canonical_bytes() -> None:
    """The SHA-256 attests the digest-free canonical form, third-party checkable."""

    from tests.conformance import _canonical_report_json

    report = run_conformance("fake", tiers=("C0",))
    recomputed = hashlib.sha256(
        _canonical_report_json(report, include_attestation=False).encode("utf-8")
    ).hexdigest()
    assert report.attestation_sha256 == recomputed
    payload = json.loads(report.to_json())
    assert payload["attestation_sha256"] == recomputed
    assert payload["environment"]["torchlens_version"]
    # Canonical bytes are cwd-independent.
    assert str(Path.cwd()) not in report.to_json()
