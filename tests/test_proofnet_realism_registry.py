"""The per-symbol realism registry + PR closure gate (memo D5/E1; F36).

Every live public symbol needs realism-graded test evidence and a docs
reference, or a dated owner-carrying gap row -- checklists rot, so the
registry is closure-checked against ``torchlens.__all__`` in BOTH
directions on every run. Realism ratchet: the R1-referenced count may not
shrink and the undocumented count may not grow (the launch number is the
fill rate, not the test count).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.smoke]

REGISTRY_PATH = Path(__file__).resolve().parent / "support" / "proofnet" / "realism_registry.tsv"

REALISM_TIERS = {"R1", "R0", "toy", "NONE"}
DOCS_TIERS = {"referenced", "executed-R0", "executed-R1", "NONE"}

#: FROZEN FLOORS/CEILINGS (F36 freeze, 2026-08-30; presence-grade census).
#: R1_REFERENCED_FLOOR only rises (realism fill rate is the launch number);
#: UNDOCUMENTED_CEILING only falls (three class names lacked a
#: ``torchlens.``-prefixed docs reference at the freeze -- D01 burns them).
R1_REFERENCED_FLOOR = 24
UNDOCUMENTED_CEILING = 3

#: Named realism rows OUTSIDE the symbol grid: whole-workflow recipes whose
#: rot would not surface through any one symbol's row. Each names a test
#: file that must exist (the SKELETON-RECIPE handoff lands here: the
#: user-wrapper -> trace -> site-key-align recipe, executed on cached real
#: distilgpt2/gpt2 -- foldA item 13's "one F36 realism row").
RECIPE_REALISM_ROWS: dict[str, str] = {
    "skeleton-change-recipe": "tests/test_skeleton_recipe_docs.py",
    "rg-workflow-gallery": "tests/workflow_gallery/rg_passed_ids.txt",
    "r1-core-plumbing": "tests/real_model/r1/test_realism_pretrained_core.py",
}


def _rows() -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    header: list[str] | None = None
    for line in REGISTRY_PATH.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if header is None:
            header = parts
            continue
        rows[parts[0]] = dict(zip(header, parts, strict=True))
    assert header == ["symbol", "realism_tier", "docs_tier", "owner"]
    return rows


def test_registry_closes_over_the_live_public_surface() -> None:
    """Both directions: every __all__ name has a row; no ghost rows."""

    import torchlens

    live = set(torchlens.__all__)
    registered = set(_rows())
    missing = live - registered
    ghosts = registered - live
    assert not missing, (
        f"public symbols with NO realism row (new surface gets no grace): {sorted(missing)}"
    )
    assert not ghosts, (
        f"realism rows for symbols no longer public (stale, delete): {sorted(ghosts)}"
    )


def test_rows_carry_closed_tiers_and_owners() -> None:
    """Schema teeth; NONE tiers require a non-freeze owner (a dated debt)."""

    for symbol, row in _rows().items():
        assert row["realism_tier"] in REALISM_TIERS, row
        assert row["docs_tier"] in DOCS_TIERS, row
        assert row["owner"], symbol
        if row["realism_tier"] == "NONE" or row["docs_tier"] == "NONE":
            assert row["owner"] != "F36-freeze", (
                f"{symbol}: a NONE tier must carry the owner burning it down,"
                " never the freeze stamp"
            )


def test_realism_fill_rate_only_improves() -> None:
    """R1-referenced count is a floor; undocumented count is a ceiling."""

    rows = _rows()
    r1_count = sum(1 for row in rows.values() if row["realism_tier"] == "R1")
    undocumented = sum(1 for row in rows.values() if row["docs_tier"] == "NONE")
    assert r1_count >= R1_REFERENCED_FLOOR, (
        f"R1-referenced symbols fell to {r1_count} (floor"
        f" {R1_REFERENCED_FLOOR}): real-model evidence was REMOVED from the"
        " suite without re-truing the registry"
    )
    assert undocumented <= UNDOCUMENTED_CEILING, (
        f"{undocumented} undocumented public symbols exceed the ceiling"
        f" {UNDOCUMENTED_CEILING}: new surface shipped without a docs"
        " reference (no grace for new surface, memo D5)"
    )
    assert undocumented == UNDOCUMENTED_CEILING, (
        "undocumented count fell below the ceiling -- lower"
        " UNDOCUMENTED_CEILING in the same commit (stale slack)"
    )


def test_recipe_realism_rows_exist_on_disk() -> None:
    """Whole-workflow realism rows (incl. the skeleton-change recipe) point
    at artifacts that exist; a vanished recipe test is a broken tripwire."""

    repo_root = Path(__file__).resolve().parents[1]
    missing = {
        row_id: path
        for row_id, path in RECIPE_REALISM_ROWS.items()
        if not (repo_root / path).exists()
    }
    assert not missing, f"recipe realism rows without artifacts: {missing}"
