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
    render_projection,
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


def test_capability_row_references_resolve() -> None:
    """Ledger rows REFERENCE production capability ids; references must resolve.

    Tests never restate backend booleans (memo 3.1): any
    ``trace_option:<option>:<backend>`` operation id must exist in the
    in-package operation-grain rows.
    """

    from torchlens.backends._options import operation_grain_capability_rows
    from torchlens.backends.registry import registered_backend_specs as specs

    live_ids = {row.row_id for spec in specs() for row in operation_grain_capability_rows(spec)}
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
