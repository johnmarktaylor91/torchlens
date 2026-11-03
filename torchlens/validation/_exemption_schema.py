"""Row schema and derived views for the declared validation exemption table.

:mod:`torchlens.validation._exemption_table` declares every exemption the
validation tripwire admits as an :class:`ExemptionRow`. This module holds the
row type, the compact builders the table is written with, and the views that
derive the run-time registries (``SKIP_VALIDATION_ENTIRELY`` and friends in
:mod:`torchlens.validation.exemptions`) from the table. It imports nothing
from the validation pipeline, so the table can be read without torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ExemptionKind = Literal[
    "skip_validation",
    "skip_perturbation",
    "structural_position",
    "custom_check",
    "early_exit",
    "posthoc",
    "recorder",
    "tolerance",
    "invariant",
]

REGISTRY_KINDS: frozenset[str] = frozenset(
    {"skip_validation", "skip_perturbation", "structural_position", "custom_check"}
)
"""Kinds whose rows are derived into a run-time registry (keyed by op spelling)."""

DECISION_KINDS: frozenset[str] = frozenset(
    {"early_exit", "posthoc", "recorder", "tolerance", "invariant"}
)
"""Kinds whose rows declare a guarded decision (keyed by reason code or constant)."""


@dataclass(frozen=True)
class ExemptionRow:
    """One declared exemption.

    Parameters
    ----------
    names:
        Op spellings (registry rows) or the reason code (decision rows) the
        row applies to. Registry keys are exact captured ``func_name`` values.
    kind:
        Which tier admits the case; registry kinds are derived into the
        run-time registries, the others are declarations of a guarded decision.
    admits:
        The exact case admitted, as the guard decides it.
    reason:
        Why the case is outside the check's contract. For ``skip_validation``
        and ``skip_perturbation`` rows this is the justification string the
        pipeline emits, verbatim.
    guard:
        ``module:symbol`` of the predicate or constant that narrows the row.
    removal_test:
        Pytest node id that fails when the row is removed.
    negative_test:
        Pytest node id proving the unintended case still fails; ``None`` is a
        declared gap, reported by the table tests.
    positions:
        Structural argument positions (``structural_position`` rows only).
    kwarg_aliases:
        Keyword spellings of those positions (``structural_position`` only).
    """

    names: tuple[str, ...]
    kind: ExemptionKind
    admits: str
    reason: str
    guard: str
    removal_test: str
    negative_test: str | None
    positions: frozenset[int] = frozenset()
    kwarg_aliases: tuple[tuple[int, frozenset[str]], ...] = field(default_factory=tuple)

    @property
    def name(self) -> str:
        """Return the canonical spelling or reason code of the row."""

        return self.names[0]

    @property
    def guard_module(self) -> str:
        """Return the module part of ``guard``."""

        return self.guard.partition(":")[0]

    @property
    def guard_symbol(self) -> str:
        """Return the symbol part of ``guard``."""

        return self.guard.partition(":")[2]


def rows_of_kind(table: tuple[ExemptionRow, ...], kind: str) -> tuple[ExemptionRow, ...]:
    """Return the rows of ``table`` whose kind is ``kind``, in table order."""

    return tuple(row for row in table if row.kind == kind)


def justifications(table: tuple[ExemptionRow, ...], kind: str) -> dict[str, str]:
    """Return ``{spelling: justification}`` for a skip registry kind.

    The justification is the row's ``reason``, emitted verbatim by the
    pipeline, so the dict equals the registry literal it replaces.
    """

    return {name: row.reason for row in rows_of_kind(table, kind) for name in row.names}


def structural_positions(table: tuple[ExemptionRow, ...]) -> dict[str, set[int]]:
    """Return ``{spelling: positions}`` for the ``structural_position`` rows."""

    return {
        name: set(row.positions)
        for row in rows_of_kind(table, "structural_position")
        for name in row.names
    }


def structural_kwarg_aliases(table: tuple[ExemptionRow, ...]) -> dict[str, dict[int, set[str]]]:
    """Return ``{spelling: {position: {kwarg, ...}}}`` for rows that carry aliases."""

    return {
        name: {position: set(aliases) for position, aliases in row.kwarg_aliases}
        for row in rows_of_kind(table, "structural_position")
        if row.kwarg_aliases
        for name in row.names
    }


def custom_check_symbols(table: tuple[ExemptionRow, ...]) -> dict[str, str]:
    """Return ``{spelling: guard symbol}`` for the ``custom_check`` rows."""

    return {
        name: row.guard_symbol for row in rows_of_kind(table, "custom_check") for name in row.names
    }


def row_for(table: tuple[ExemptionRow, ...], name: str, kind: str | None = None) -> ExemptionRow:
    """Return the row declaring ``name`` (an op spelling or a reason code).

    A spelling may be declared under two kinds (``new`` is both an
    uninitialized-by-design skip and a structural template), so ``kind``
    selects among them; without it the first declaring row in table order wins.

    Raises
    ------
    KeyError
        When no row declares ``name`` (under ``kind`` when given).
    """

    for row in table:
        if name in row.names and (kind is None or row.kind == kind):
            return row
    raise KeyError(name if kind is None else f"{kind}:{name}")
