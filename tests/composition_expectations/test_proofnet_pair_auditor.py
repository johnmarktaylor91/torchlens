"""The pair-coverage auditor (compo memo stratum 4; F36 waves A-D).

Enforcement authority over the declared witnessed rows: it computes covered
(product, verb) axis pairs from the M1 and edit matrices, and every
uncovered VALID pair fails until a human writes a witnessed cell or
registers an exclusion routed to N/A-or-gap. Constraints are themselves
registered and red-capability-tested: a planted over-broad constraint (one
that swallows a COVERED pair) must go red. The residue is a NUMBER
(valid / covered / excluded / uncovered), never folklore.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import pytest

from tests.composition_expectations.test_proofnet_edit_product import (
    EDIT_PRODUCT_MATRIX,
)
from tests.composition_expectations.test_proofnet_m1_product_verb import (
    M1_EXPECTATIONS,
    PRODUCT_NAMES,
    VERB_NAMES,
)

pytestmark = pytest.mark.compo


#: Memo D12 fixture-economics license (test_galleries.py lint):
COMPO_CONSTRUCTION_LICENSE = (
    "pure-data auditor (no products); license documents intent if probes are added"
)

ALL_VERBS = tuple(VERB_NAMES) + ("do",)

#: Edit-matrix product spellings normalized onto the M1 lifecycle axis.
EDIT_PRODUCT_TO_M1 = {
    "trace_live_fork": "trace_live",
    "trace_live_no_fork": "trace_live",
    "trace_loaded": "trace_loaded",
    "trace_structure_only": "trace_structure_only",
    "trace_halted": "trace_halted",
    "trace_not_intervention_ready": "trace_live",
}


@dataclass(frozen=True)
class Exclusion:
    """A registered constraint: one uncovered pair, routed with a reason."""

    product: str
    verb: str
    routing: str  # "n/a" (no such door by design) or "known-gap"
    reason: str


#: The exclusion registry. Every row is a conscious decision with a route;
#: "not built yet" is only legal under the known-gap routing with an owner.
EXCLUSIONS: tuple[Exclusion, ...] = (
    Exclusion(
        "recording",
        "do",
        "n/a",
        "Recording is a sparse event stream with no replay overlay; the edit"
        " door is to_trace() then do() (M1 pins the missing-door refusal)",
    ),
    Exclusion(
        "partial_trace",
        "do",
        "n/a",
        "a failed capture is N1-refused from every mutating engine; the"
        " partial product serves evidence only",
    ),
    Exclusion(
        "trace_slice",
        "do",
        "n/a",
        "slices feed do() through __selection__ on their SOURCE trace (the"
        " deep-chain suite executes that route); the presenter itself has no"
        " edit door",
    ),
    Exclusion(
        "bundle",
        "do",
        "n/a",
        "Bundle edits ride vary()/site_sweep (F03 territory); per-member do()"
        " goes through the member trace",
    ),
    Exclusion(
        "trace_episode",
        "do",
        "known-gap",
        "OWNER F36 remainder, dated 2026-08-30: do() on an episode fork must"
        " quarantine inherited episode evidence"
        " (episode_evidence_dropped_perturbed_replay); the cell is unfunded"
        " this wave -- fund it or route it N/A with the F42 owner's sign-off",
    ),
    Exclusion(
        "trace_episode_coupled",
        "do",
        "known-gap",
        "OWNER F36 remainder, dated 2026-08-30: same quarantine cell on the"
        " coupled product (replay door refusal IS pinned by the episode"
        " table; the do() quarantine disclosure is not)",
    ),
)


def covered_pairs() -> set[tuple[str, str]]:
    """Every (product, verb) pair some witnessed matrix cell drives."""

    pairs: set[tuple[str, str]] = set(M1_EXPECTATIONS)
    for edit_product in EDIT_PRODUCT_MATRIX:
        pairs.add((EDIT_PRODUCT_TO_M1[edit_product], "do"))
    return pairs


def audit(
    exclusions: tuple[Exclusion, ...],
) -> tuple[set[tuple[str, str]], set[tuple[str, str]], list[str]]:
    """Return (uncovered, excluded, violations) for the declared universe."""

    universe = set(itertools.product(PRODUCT_NAMES, ALL_VERBS))
    covered = covered_pairs() & universe
    excluded = {(row.product, row.verb) for row in exclusions}
    violations = [
        f"over-broad constraint: ({row.product}, {row.verb}) is excluded but a"
        " witnessed cell ALREADY covers it -- delete the exclusion"
        for row in exclusions
        if (row.product, row.verb) in covered
    ]
    uncovered = universe - covered - excluded
    return uncovered, excluded, violations


def test_residue_is_zero_after_registered_exclusions() -> None:
    """Uncovered valid pairs fail until witnessed or consciously excluded."""

    uncovered, excluded, violations = audit(EXCLUSIONS)
    universe_size = len(PRODUCT_NAMES) * len(ALL_VERBS)
    residue_report = (
        f"pair-coverage residue: valid={universe_size}"
        f" covered={len(covered_pairs())} excluded={len(excluded)}"
        f" uncovered={len(uncovered)}"
    )
    assert not violations, violations
    assert not uncovered, (
        f"{residue_report}; UNCOVERED pairs (write a witnessed cell or"
        f" register a routed exclusion): {sorted(uncovered)}"
    )


def test_exclusions_carry_routes_and_reasons() -> None:
    """Constraint schema teeth: closed routings; known-gaps carry owners."""

    for row in EXCLUSIONS:
        assert row.routing in {"n/a", "known-gap"}, row
        assert len(row.reason) > 20, f"({row.product}, {row.verb}): reason too thin"
        if row.routing == "known-gap":
            assert "OWNER" in row.reason and "dated" in row.reason, (
                f"({row.product}, {row.verb}): known-gap exclusions need an"
                " owner and a date (D11 teeth)"
            )


def test_auditor_flags_a_planted_over_broad_constraint() -> None:
    """Red-capability: excluding a COVERED pair must be a violation."""

    planted = EXCLUSIONS + (
        Exclusion("trace_live", "summary", "n/a", "planted over-broad constraint x" * 3),
    )
    _, _, violations = audit(planted)
    assert any("over-broad" in violation for violation in violations), (
        "the auditor accepted a constraint that swallows a covered pair --"
        " constraints are unchecked, which is how residue lies"
    )
