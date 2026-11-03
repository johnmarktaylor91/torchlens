"""The declared table of validation exemptions.

Every case the validation tripwire admits without a passing check is one row
here: the op spellings or reason code it applies to, the exact case admitted,
the reason it is correct by design and outside the check's contract, the
guard (``module:symbol``) that narrows it at run time, and two paired tests:
``removal_test`` fails when the row is removed, ``negative_test`` proves the
unintended case still fails. The run-time registries in :mod:`exemptions`
(``SKIP_VALIDATION_ENTIRELY``, ``SKIP_PERTURBATION_ENTIRELY``,
``STRUCTURAL_ARG_POSITIONS``, ``STRUCTURAL_ARG_KWARG_ALIASES`` and
``CUSTOM_EXEMPTION_CHECKS``) are derived from this table, so a membership
edit is a table edit. Rows of kind ``early_exit``, ``posthoc``, ``recorder``,
``tolerance`` and ``invariant`` declare exemptions whose admitted case is
decided by their guard in code; the table is their single declaration and the
test ledgers (``tests/test_validation_exemption_ledger.py``,
``tests/test_validation_registry_ledger.py``) audit it row by row. The row
literals live in ``_exemption_rows_ops`` (op-keyed) and
``_exemption_rows_decisions`` (code-keyed) to stay under the file cap; this
module assembles them into ``EXEMPTION_TABLE`` and derives the registries.

Validation integrity (AGENTS.md) is LOCKED: a row is added only with a proof
that its case is outside the check's contract, a removal test and a negative
test. The table never widens a row to make a test pass.
"""

from __future__ import annotations

from ._exemption_rows_decisions import _DECISION_ROWS, _INVARIANT_ROWS, _TOLERANCE_ROWS
from ._exemption_rows_ops import _REGISTRY_ROWS
from ._exemption_schema import (
    ExemptionKind,
    ExemptionRow,
    custom_check_symbols,
    justifications,
    row_for,
    structural_kwarg_aliases,
    structural_positions,
)

__all__ = (
    "EXEMPTION_TABLE",
    "ExemptionKind",
    "ExemptionRow",
    "declared_custom_check_symbols",
    "declared_structural_kwarg_aliases",
    "declared_structural_positions",
    "exemption_row",
    "skip_perturbation_justifications",
    "skip_validation_justifications",
)

EXEMPTION_TABLE: tuple[ExemptionRow, ...] = (
    _REGISTRY_ROWS + _DECISION_ROWS + _TOLERANCE_ROWS + _INVARIANT_ROWS
)
"""Every exemption the validation tripwire admits, in one place."""


def skip_validation_justifications() -> dict[str, str]:
    """Derive ``SKIP_VALIDATION_ENTIRELY`` from the table."""

    return justifications(EXEMPTION_TABLE, "skip_validation")


def skip_perturbation_justifications() -> dict[str, str]:
    """Derive ``SKIP_PERTURBATION_ENTIRELY`` from the table."""

    return justifications(EXEMPTION_TABLE, "skip_perturbation")


def declared_structural_positions() -> dict[str, set[int]]:
    """Derive ``STRUCTURAL_ARG_POSITIONS`` from the table."""

    return structural_positions(EXEMPTION_TABLE)


def declared_structural_kwarg_aliases() -> dict[str, dict[int, set[str]]]:
    """Derive ``STRUCTURAL_ARG_KWARG_ALIASES`` from the table."""

    return structural_kwarg_aliases(EXEMPTION_TABLE)


def declared_custom_check_symbols() -> dict[str, str]:
    """Derive the ``CUSTOM_EXEMPTION_CHECKS`` spelling-to-predicate map from the table."""

    return custom_check_symbols(EXEMPTION_TABLE)


def exemption_row(name: str, kind: str | None = None) -> ExemptionRow:
    """Return the row declaring an op spelling or reason code, optionally by kind."""

    return row_for(EXEMPTION_TABLE, name, kind)
