"""Derived worst-of-members capture-outcome fold for ``Bundle`` (foldB D9).

A ``Bundle`` has no settlement of its own -- members remain the settlement
authority -- so the container speaks through the DERIVED fold in this module:
``Bundle.outcome`` reports the most severe member status (COMPLETE never
blessed above the weakest member), ``derived=True``, disclosure-only. The
fold is reported beside results and never gates a read on a member it does
not describe (foldB D8: capability gating stays PER MEMBER at the member
whose facts a read cites). Split out of ``torchlens/bundle/__init__.py`` to
respect its size ceiling (R43 split-never-raise).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..capture.outcome import CaptureOutcome

# Worst-of-members fold severity, least -> most severe, keyed on the frozen
# CaptureStatus values (string keys keep torchlens.capture lazy at import).
# The order is capability-table restrictiveness under the landed
# CAPTURE_OUTCOME_CAPABILITIES matrix (refusal-cell count), with the two ties
# broken by epistemic weakness: UNATTESTED sits above COMPLETE because an
# unattested member is never blessed COMPLETE, FAILED above ABORTED_NONFINITE
# because an abort is a deliberate, better-understood stop, and UNKNOWN is the
# top by doctrine (unprovable, most restrictive).
_OUTCOME_FOLD_SEVERITY: dict[str, int] = {
    "complete": 0,
    "unattested": 1,
    "halted": 2,
    "aborted_nonfinite": 3,
    "failed": 4,
    "unknown": 5,
}


def _fold_member_outcomes(
    member_outcomes: Sequence[tuple[str, CaptureOutcome | None]],
) -> CaptureOutcome:
    """Fold member outcomes into one derived worst-of-members disclosure.

    This is the Bundle outcome-authority fold (foldB D9): a DERIVED
    container-level status reported beside results, never a settlement and
    never a gate on a member it does not describe (D8 -- capability gating
    stays PER MEMBER at the member whose facts a read cites). COMPLETE is
    never blessed above the weakest member; a member with no settled outcome
    contributes UNKNOWN fail-closed; an empty domain (unreachable through the
    public constructor, which requires at least one Trace) lands on UNKNOWN.

    Parameters
    ----------
    member_outcomes:
        Ordered ``(member_name, settled_outcome_or_None)`` pairs, as read
        through :func:`torchlens.capture.outcome.outcome_for`.

    Returns
    -------
    CaptureOutcome
        Frozen derived record: the most severe member status under
        ``_OUTCOME_FOLD_SEVERITY``, ``derived=True``, and a
        ``settlement_note`` naming the driving member plus status counts.
    """

    from ..capture.outcome import CaptureOutcome, CaptureStatus

    if not member_outcomes:
        return CaptureOutcome(
            status=CaptureStatus.UNKNOWN,
            derived=True,
            settlement_note=(
                "bundle worst-of-members fold over 0 members: no member "
                "outcomes to fold (fail-closed)"
            ),
        )
    counts: Counter[str] = Counter()
    unsettled = 0
    worst_name, worst_status = member_outcomes[0][0], CaptureStatus.COMPLETE
    worst_rank = -1
    for name, outcome in member_outcomes:
        if outcome is None:
            unsettled += 1
            status = CaptureStatus.UNKNOWN
        else:
            status = outcome.status
        counts[status.value] += 1
        rank = _OUTCOME_FOLD_SEVERITY[status.value]
        if rank > worst_rank:
            worst_name, worst_status, worst_rank = name, status, rank
    count_text = ", ".join(
        f"{value}={counts[value]}" for value in _OUTCOME_FOLD_SEVERITY if counts[value]
    )
    note = (
        f"bundle worst-of-members fold over {len(member_outcomes)} members; "
        f"driven by member {worst_name!r} (status {worst_status.value!r}); "
        f"status counts: {count_text}"
    )
    if unsettled:
        note += f"; {unsettled} member(s) carry no settled outcome (treated unknown, fail-closed)"
    return CaptureOutcome(status=worst_status, derived=True, settlement_note=note)
