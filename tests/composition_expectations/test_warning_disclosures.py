"""S-18 warning contract: chokepoint, contract table lockstep, ratchet (row 0.4).

ZERO of the package's warn sites carried a machine-readable code before this
wave (memo D7). The mechanism: ``TorchLensWarning`` gains the error base's
remedy chokepoint (``fields["remedy"]`` derived from the authored
``Remedy: ...`` tail), sites pass ``code=``, and the contract table lives at
``docs/reference/warning_contract.md``. First contracted row: the zero-fire
rerun warning -- the one warning standing between a user and a silently
un-applied ablation (memo 5.4).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.composition_expectations import _censuses
from tests.composition_expectations.test_diagnostic_site_contract import _ratchets

pytestmark = pytest.mark.compo

REPO_ROOT = Path(__file__).resolve().parents[2]
WARNING_CONTRACT_DOC = REPO_ROOT / "docs" / "reference" / "warning_contract.md"

_DOC_ROW_PATTERN = re.compile(r"^\| `([a-z0-9_]+)` \| (.*?) \| (.*?) \|$")


def _contract_table_codes(text: str) -> dict[str, tuple[str, str]]:
    """Parse a warning-contract table: code -> (warning, remedy class)."""

    rows: dict[str, tuple[str, str]] = {}
    for line in text.splitlines():
        match = _DOC_ROW_PATTERN.match(line.strip())
        if match is None:
            continue
        code, warning, remedy = match.groups()
        assert code not in rows, f"duplicate warning-contract row for {code!r}"
        rows[code] = (warning, remedy)
    return rows


def test_torchlens_warning_remedy_chokepoint() -> None:
    """The warning base derives fields['remedy'] from the Remedy: tail.

    The five-line chokepoint copied from the error base (memo 0.4 XS
    mechanism): authored-tail derivation, explicit ``remedy=`` kwarg wins,
    ``code=`` payload lands in fields.
    """

    from torchlens.errors import TorchLensWarning

    warning = TorchLensWarning("Something drifted. Remedy: re-run the capture", code="planted")
    assert warning.fields["code"] == "planted"
    assert warning.fields["remedy"] == "re-run the capture"

    explicit = TorchLensWarning("Msg. Remedy: derived tail", remedy="explicit wins")
    assert explicit.fields["remedy"] == "explicit wins"

    bare = TorchLensWarning("No remedy tail here")
    assert "remedy" not in bare.fields


def test_first_contracted_row_is_live() -> None:
    """The zero-fire rerun warning is coded at the site and behaviorally sound."""

    from torchlens.errors import TorchLensWarning

    coded_sites = {site.code: site for site in _censuses.warn_sites() if site.code}
    assert "rerun_zero_fire" in coded_sites, (
        "the first contracted S-18 site (intervention/rerun.py zero-fire) lost its code"
    )
    site = coded_sites["rerun_zero_fire"]
    assert site.module == "torchlens.intervention.rerun"
    assert site.category == "TorchLensWarning"

    # The warning the site constructs carries the machine-readable contract.
    constructed = TorchLensWarning(
        "Rerun hook plan entries fired at zero sites on the new inputs: ['p']. "
        "The rerun completed, but those interventions were no-ops. "
        "Remedy: resolve the target sites against the rerun trace",
        code="rerun_zero_fire",
        unfired_plan_ids=["p"],
    )
    assert constructed.fields["code"] == "rerun_zero_fire"
    assert constructed.fields["remedy"].startswith("resolve the target sites")
    assert constructed.fields["unfired_plan_ids"] == ["p"]
    assert isinstance(constructed, UserWarning), "category compatibility with pytest.warns"


def test_warning_contract_table_lockstep_both_directions() -> None:
    """Census codes and the contract table agree exactly, both ways."""

    table = _contract_table_codes(WARNING_CONTRACT_DOC.read_text(encoding="utf-8"))
    assert table, "the warning contract table parsed empty"
    census_codes = {site.code for site in _censuses.warn_sites() if site.code}
    missing_rows = census_codes - set(table)
    stale_rows = set(table) - census_codes
    assert not missing_rows, (
        "warn sites pass codes with no warning-contract row (add the row to "
        f"docs/reference/warning_contract.md in the same change): {sorted(missing_rows)}"
    )
    assert not stale_rows, (
        "warning-contract rows with no live coded site (drop or re-point the "
        f"row in the same change): {sorted(stale_rows)}"
    )
    for code, (warning, remedy) in table.items():
        assert warning.strip() and remedy.strip(), f"empty contract columns for {code!r}"


def test_warning_contract_lockstep_is_red_capable() -> None:
    """Planted drift in either direction is detected by the pure comparators."""

    table = _contract_table_codes(
        "| Code | Warning | Remedy class |\n|---|---|---|\n| `planted_code` | w | r |\n"
    )
    assert set(table) == {"planted_code"}
    # Direction 1: a coded site missing from the table.
    assert {"other_code"} - set(table) == {"other_code"}
    # Direction 2: a table row with no coded site.
    assert set(table) - {"other_code"} == {"planted_code"}


def test_s18_ratchets_hold_exactly() -> None:
    """Warn-site totals, uncoded count, and contract-row count match baselines.

    Teaching: a NEW user-facing warn site lands coded (with its contract row)
    in its own change; coding an existing site lowers ``s18_uncoded_sites``
    in the same change. The comfort number is one toy session (memo 3.5
    caveat); the ratchet is what makes the size risk irrelevant.
    """

    sites = _censuses.warn_sites()
    table = _contract_table_codes(WARNING_CONTRACT_DOC.read_text(encoding="utf-8"))
    ratchets = _ratchets()
    live = {
        "s18_total_sites": len(sites),
        "s18_uncoded_sites": sum(1 for site in sites if not site.has_code),
        "s18_contract_rows": len(table),
    }
    print("\nS-18 warning census:")
    for metric, count in live.items():
        print(f"  {metric}\t{count}\t(baseline {ratchets[metric]})")
    drifted = {
        metric: (ratchets[metric], count)
        for metric, count in live.items()
        if count != ratchets[metric]
    }
    assert not drifted, (
        "S-18 ratchet drift (baseline, live). New warn sites: code them + add "
        "their contract row in this change. Burn-down: lower "
        f"data/diagnostic_ratchets.tsv in this change. Drifted: {drifted}"
    )
