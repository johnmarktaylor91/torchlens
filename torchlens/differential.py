"""The ONE guarded cross-capture delta projection (leverage B15, D-2/D-7).

``diff_report(subject, reference)`` enumerates BOTH captures' structural
site keys and settles every position into exactly one row of a closed
status vocabulary — the projection summary/cost/themes/netron/Model-Explorer
consumers read instead of re-deriving their own joins:

- ``added``              key present on the subject only (declared, never guessed)
- ``removed``            key present on the reference only
- ``unresolved``         the guarded join refused the pairing (verdict disclosed)
- ``excluded_machinery`` an engine-inserted op (D-7: excluded from model-effect
                         rows with an explicit count, disclosed as a typed
                         structural addition — never painted as a changed
                         model site, never hidden)
- ``unreachable``        joined, but a retained payload is missing on at least
                         one side — no value claim is possible (distinct from
                         zero BY CONSTRUCTION)
- ``zero``               joined, both payloads retained, delta exactly zero
- ``changed``            joined, both payloads retained, delta nonzero

Pairing authority is the shipped guarded site join
(:mod:`torchlens.postprocess._site_join`): ``corroborated`` and
``positional`` verdicts admit comparison, refused verdicts become
``unresolved`` rows, and labels never enter the join key (leverage D-2).
The report is SESSION-ONLY and never persisted. Every spelling here is
DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from ._errors import InvalidArgumentError

__all__ = ["DiffRow", "DifferentialReport", "diff_report"]

#: Engine-inserted machinery layer types (leverage D-7 exclusion rule).
_MACHINERY_LAYER_TYPES = frozenset({"interventionreplacement"})

#: Closed row-status vocabulary (leverage B15).
_ROW_STATUSES = (
    "added",
    "removed",
    "unresolved",
    "excluded_machinery",
    "unreachable",
    "zero",
    "changed",
)


@dataclass(frozen=True)
class DiffRow:
    """One settled position of the differential projection.

    ``site_key`` is the structural key; ``subject_label`` / ``reference_label``
    are DISCLOSURE (display only — labels never authorize a pairing, and a
    one-sided row carries only its own side's label). ``verdict`` is the
    guarded join's verdict where a join was attempted; ``max_abs_delta`` is
    populated only on value rows (``zero`` / ``changed``).
    """

    site_key: str
    status: str
    subject_label: str | None = None
    reference_label: str | None = None
    pass_index: int | None = None
    verdict: str | None = None
    max_abs_delta: float | None = None
    detail: str | None = None


@dataclass(frozen=True)
class DifferentialReport:
    """The settled two-capture differential: rows plus honest totals."""

    rows: tuple[DiffRow, ...]
    counts: dict[str, int] = field(default_factory=dict)

    def rows_with_status(self, status: str) -> tuple[DiffRow, ...]:
        """Return the rows carrying one closed-vocabulary status."""

        if status not in _ROW_STATUSES:
            raise InvalidArgumentError(
                f"unknown differential row status {status!r}; the closed "
                f"vocabulary is {sorted(_ROW_STATUSES)}",
                code="differential_status_invalid",
                remedy="pass one of the closed row statuses",
            )
        return tuple(row for row in self.rows if row.status == status)

    def __repr__(self) -> str:
        parts = ", ".join(
            f"{status}={self.counts.get(status, 0)}"
            for status in _ROW_STATUSES
            if self.counts.get(status, 0)
        )
        return f"DifferentialReport({len(self.rows)} rows: {parts or 'no rows'})"


def _ops_by_key(trace: Any, keys: dict[str, str]) -> dict[str, list[str]]:
    """Invert one profile's label->key map into key->[labels] (capture order)."""

    by_key: dict[str, list[str]] = {}
    for label, key in keys.items():
        by_key.setdefault(key, []).append(label)
    return by_key


def _pass_of(trace: Any, label: str) -> int:
    """Return one op's 1-based pass index (single-pass ops read 1)."""

    return int(getattr(trace.ops[label], "pass_index", 1) or 1)


@dataclass(frozen=True)
class _JoinContext:
    """The per-diff constants every row settler shares (one join, many rows)."""

    subject: Any
    reference: Any
    subject_coords: dict[str, tuple[str, str, int]]
    reference_by_coord: dict[tuple[str, str, int], str]


def _value_row(
    ctx: _JoinContext,
    key: str,
    subject_label: str,
    reference_label: str,
    verdict: str,
) -> DiffRow:
    """Settle one joined pair into unreachable / zero / changed."""

    subject = ctx.subject
    subject_op = subject.ops[subject_label]
    reference_op = ctx.reference.ops[reference_label]

    def _row(status: str, detail: str | None = None, max_abs_delta: float | None = None) -> DiffRow:
        """One settled row with the pair's shared identity fields."""

        return DiffRow(
            site_key=key,
            status=status,
            subject_label=subject_label,
            reference_label=reference_label,
            pass_index=_pass_of(subject, subject_label),
            verdict=verdict,
            max_abs_delta=max_abs_delta,
            detail=detail,
        )

    for side_name, op in (("subject", subject_op), ("reference", reference_op)):
        if not getattr(op, "has_saved_activation", False):
            return _row("unreachable", detail=f"payload not retained on {side_name}")
    subject_value = subject_op.out
    reference_value = reference_op.out
    if not isinstance(subject_value, torch.Tensor) or not isinstance(reference_value, torch.Tensor):
        return _row("unreachable", detail="non-tensor payload")
    if tuple(subject_value.shape) != tuple(reference_value.shape):
        return _row(
            "unreachable",
            detail=(
                f"shape drift {tuple(subject_value.shape)!r} vs {tuple(reference_value.shape)!r}"
            ),
        )
    delta = subject_value.detach().cpu().to(torch.float64) - reference_value.detach().cpu().to(
        torch.float64
    )
    max_abs = float(delta.abs().max().item()) if delta.numel() else 0.0
    return _row("zero" if max_abs == 0.0 else "changed", max_abs_delta=max_abs)


def _joined_rows(
    ctx: _JoinContext,
    key: str,
    row: Any,
    subject_labels: list[str],
) -> list[DiffRow]:
    """Settle one joined key's occurrence pairs into rows.

    Pairing coordinate = (key, pass-qualified call instance, within-instance
    position) — exactly what the join's cardinality guard verifies equal, so
    a ``joined`` verdict licenses it (never labels, never a global ordinal).
    """

    subject = ctx.subject
    rows: list[DiffRow] = []
    for subject_label in subject_labels:
        if getattr(subject.ops[subject_label], "layer_type", None) in _MACHINERY_LAYER_TYPES:
            rows.append(
                DiffRow(
                    site_key=key,
                    status="excluded_machinery",
                    subject_label=subject_label,
                    pass_index=_pass_of(subject, subject_label),
                    verdict=row.verdict.value,
                    detail="engine-inserted op: typed structural addition, "
                    "excluded from model-effect rows (D-7)",
                )
            )
            continue
        coord = ctx.subject_coords.get(subject_label)
        reference_label = None if coord is None else ctx.reference_by_coord.get(coord)
        if reference_label is None:
            rows.append(
                DiffRow(
                    site_key=key,
                    status="unresolved",
                    subject_label=subject_label,
                    pass_index=_pass_of(subject, subject_label),
                    verdict=row.verdict.value,
                    detail="occurrence coordinate absent on the reference",
                )
            )
            continue
        rows.append(_value_row(ctx, key, subject_label, reference_label, row.verdict.value))
    return rows


def diff_report(subject: Any, reference: Any) -> DifferentialReport:
    """Settle two captures into the guarded differential projection.

    Parameters
    ----------
    subject:
        The capture whose perspective the delta takes (``subject - reference``).
    reference:
        The one explicit reference capture.

    Returns
    -------
    DifferentialReport
        One row per structural position of the UNION of both captures, each
        in the closed added / removed / unresolved / excluded_machinery /
        unreachable / zero / changed vocabulary.

    Raises
    ------
    InvalidArgumentError
        ``differential_subject_is_reference`` on self-comparison (vacuous by
        construction); ``site_key_unavailable`` (from the site profiler) when
        either capture predates structural site keys — the projection's
        pairing authority is the guarded join, and a keyless capture has
        nothing sound to join on.
    """

    if subject is reference:
        raise InvalidArgumentError(
            "diff_report(subject, reference) received the SAME capture on "
            "both sides: the differential of a capture with itself is "
            "identically zero, so every row would be vacuous. Pass the OTHER "
            "run as the reference (a fork after do() is a different trace).",
            code="differential_subject_is_reference",
            remedy="pass two different captures",
        )
    from .postprocess._site_join import (
        join_site_profiles,
        occurrence_coordinates,
        site_profile,
    )

    subject_profile = site_profile(subject)
    reference_profile = site_profile(reference)
    join_rows = join_site_profiles(subject_profile, reference_profile)
    subject_by_key = _ops_by_key(subject, subject_profile.keys)
    reference_by_key = _ops_by_key(reference, reference_profile.keys)
    ctx = _JoinContext(
        subject=subject,
        reference=reference,
        subject_coords=occurrence_coordinates(subject, subject_profile.keys),
        reference_by_coord={
            coord: label
            for label, coord in occurrence_coordinates(reference, reference_profile.keys).items()
        },
    )

    rows: list[DiffRow] = []
    for key in sorted(set(subject_by_key) | set(reference_by_key)):
        subject_labels = subject_by_key.get(key, [])
        reference_labels = reference_by_key.get(key, [])
        if not reference_labels:
            for label in subject_labels:
                machinery = (
                    getattr(subject.ops[label], "layer_type", None) in _MACHINERY_LAYER_TYPES
                )
                rows.append(
                    DiffRow(
                        site_key=key,
                        status="excluded_machinery" if machinery else "added",
                        subject_label=label,
                        pass_index=_pass_of(subject, label),
                        detail=(
                            "engine-inserted op: typed structural addition, "
                            "excluded from model-effect rows (D-7)"
                            if machinery
                            else "declared addition (subject only)"
                        ),
                    )
                )
            continue
        if not subject_labels:
            rows.extend(
                DiffRow(
                    site_key=key,
                    status="removed",
                    reference_label=label,
                    pass_index=_pass_of(reference, label),
                    detail="declared removal (reference only)",
                )
                for label in reference_labels
            )
            continue
        row = join_rows[key]
        if not row.joined:
            rows.extend(
                DiffRow(
                    site_key=key,
                    status="unresolved",
                    subject_label=label,
                    pass_index=_pass_of(subject, label),
                    verdict=row.verdict.value,
                    detail="the guarded join refused this cohort",
                )
                for label in subject_labels
            )
            continue
        rows.extend(_joined_rows(ctx, key, row, subject_labels))

    counts: dict[str, int] = {}
    for row_record in rows:
        counts[row_record.status] = counts.get(row_record.status, 0) + 1
    return DifferentialReport(rows=tuple(rows), counts=counts)
