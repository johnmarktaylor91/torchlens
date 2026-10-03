"""Ledger contract lockstep (row 0.8): validation, red-capability, projection diff.

The ledger's closed vocabularies and per-state teeth are enforced here, every
rule is red-capability-proved with planted rows (a gate nobody has proved can
fail is not a gate -- memo section 9), production capability ids are
REFERENCED and resolved (never restated), and the generated projection is
regenerate-and-diff so a ledger change is always a reviewed diff.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from tests.composition_expectations.ledger import (
    DATA_DIR,
    LEDGER,
    CompositionRow as Row,
    ledger_violations,
    render_projection,
    row_violations,
)

pytestmark = pytest.mark.compo

TREE_ROOT = Path(__file__).resolve().parent


def _valid_row(**overrides: object) -> Row:
    """A compliant baseline row the plants mutate one rule at a time."""

    base = Row(
        row_id="CELL-PLANT",
        family="plant",
        generator_kind="hand",
        operation_ids=("tl.trace",),
        axis_values=("axis=value",),
        expected_state="SUPPORTED",
        risk_tags=(),
        oracle_kind="EXACT",
        evidence_node="tests/composition_expectations/test_contract_lockstep.py",
    )
    return dataclasses.replace(base, **overrides)  # type: ignore[arg-type]


def test_ledger_validates_clean() -> None:
    """Every seed row satisfies the full row contract."""

    violations = ledger_violations()
    assert not violations, "ledger rows violate the contract:\n  " + "\n  ".join(violations)
    print(f"\ncomposition ledger: {len(LEDGER)} rows, 0 violations")


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        pytest.param({"expected_state": "VIBES"}, "unknown state", id="closed-states"),
        pytest.param({"oracle_kind": "SMOKE"}, "unknown oracle kind", id="smoke-is-deleted"),
        pytest.param({"generator_kind": "vibes"}, "unknown generator", id="closed-generators"),
        pytest.param({"risk_tags": ("vibes",)}, "unknown risk tag", id="closed-risk-tags"),
        pytest.param({"tier": "warp"}, "unknown tier", id="closed-tiers"),
        pytest.param({"operation_ids": ()}, "no operation ids", id="operation-required"),
        pytest.param({"axis_values": ()}, "no axis values", id="axes-required"),
        pytest.param(
            {"evidence_node": ""}, "without an evidence node", id="supported-needs-evidence"
        ),
        pytest.param(
            {"risk_tags": ("accepted_then_discarded",), "oracle_kind": "INVARIANT"},
            "mandates oracle",
            id="risk-mandates-differential",
        ),
        pytest.param(
            {"risk_tags": ("persistence_hop",), "oracle_kind": "EXACT"},
            "mandates oracle",
            id="risk-mandates-invariant",
        ),
        pytest.param(
            {"risk_tags": ("pass_resolution",), "oracle_kind": "DIFFERENTIAL"},
            "mandates oracle",
            id="risk-mandates-exact",
        ),
        pytest.param(
            {"oracle_kind": "ADMISSION"},
            "requires a semantic witness",
            id="admission-never-supports",
        ),
        pytest.param(
            {"expected_state": "REFUSES-TYPED-TEACHING", "oracle_kind": "REFUSAL"},
            "needs refusal_code",
            id="refuses-needs-code-or-probe",
        ),
        pytest.param(
            {"expected_state": "N-A", "oracle_kind": "ADMISSION"},
            "needs reason",
            id="na-needs-reason-reviewer-trigger",
        ),
        pytest.param(
            {
                "expected_state": "N-A",
                "oracle_kind": "ADMISSION",
                "n_a_reason": "not built yet",
                "n_a_reviewer": "compo",
                "n_a_review_trigger": "wave A",
            },
            "never N/A",
            id="not-built-yet-is-never-na",
        ),
        pytest.param(
            {"expected_state": "KNOWN-GAP", "oracle_kind": "REFUSAL"},
            "KNOWN-GAP without",
            id="known-gap-needs-teeth",
        ),
    ],
)
def test_row_contract_is_red_capable(overrides: dict[str, object], fragment: str) -> None:
    """Each contract rule flags its planted violation."""

    violations = row_violations(_valid_row(**overrides))
    assert any(fragment in violation for violation in violations), (fragment, violations)


def test_admission_is_legal_on_refuses_and_probes() -> None:
    """The two sanctioned ADMISSION shapes pass."""

    refuses = _valid_row(
        expected_state="REFUSES-TYPED-TEACHING",
        oracle_kind="ADMISSION",
        refusal_code="some_code",
    )
    assert row_violations(refuses) == []
    probe = _valid_row(
        expected_state="KNOWN-GAP",
        oracle_kind="ADMISSION",
        generator_kind="generated-probe",
        gap_owner="o",
        gap_issue="i",
        gap_deadline="d",
        gap_reproducer="r",
        gap_auto_probe="p",
    )
    assert row_violations(probe) == []


def test_capability_row_references_resolve() -> None:
    """Ledger rows REFERENCE production capability ids; references must resolve.

    Tests never restate backend booleans (memo 3.1): any
    ``trace_option:<option>:<backend>`` operation id must exist in the
    in-package operation-grain rows.
    """

    from torchlens.backends._options import operation_grain_capability_rows
    from torchlens.backends.registry import registered_backend_specs

    live_ids = {
        row.row_id
        for spec in registered_backend_specs()
        for row in operation_grain_capability_rows(spec)
    }
    for row in LEDGER:
        for operation_id in row.operation_ids:
            if operation_id.startswith("trace_option:"):
                assert operation_id in live_ids, (
                    f"{row.row_id} references capability id {operation_id!r} that "
                    "production does not serve"
                )


def test_projection_regenerates_identically() -> None:
    """Regenerate-and-diff: the committed projection matches the ledger."""

    committed = (DATA_DIR / "composition_ledger.md").read_text(encoding="utf-8")
    live = render_projection()
    assert committed == live, (
        "the generated projection drifted from the ledger. Review the diff, then "
        "copy render_projection() over "
        "tests/composition_expectations/data/composition_ledger.md in this change."
    )


def test_every_tree_module_carries_the_compo_marker() -> None:
    """Composition test modules are `-m compo` selectable, per the algebra."""

    unmarked = [
        path.name
        for path in sorted(TREE_ROOT.glob("test_*.py"))
        if "pytest.mark.compo" not in path.read_text(encoding="utf-8")
    ]
    assert not unmarked, (
        "composition test modules must carry the compo selection marker "
        f"(pytestmark = [..., pytest.mark.compo]): {unmarked}"
    )


def test_known_gap_rows_render_with_teeth_in_projection() -> None:
    """The projection shows owner + deadline for every KNOWN-GAP row."""

    projection = render_projection()
    for row in LEDGER:
        assert row.row_id in projection, f"{row.row_id} missing from the projection"
        if row.expected_state == "KNOWN-GAP":
            assert row.gap_owner in projection and row.gap_deadline in projection, (
                f"{row.row_id}: gap teeth not rendered (state and gaps render TOGETHER)"
            )
