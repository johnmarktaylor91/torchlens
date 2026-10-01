"""Bundle-level retained-bytes aggregation + retention preflight (F03 item 5c).

The shipped byte accountant is PER-CAPTURE (every fork gets its own
``SaveBudget``), so before this module no object knew a bundle's total
retained footprint — a user with ``save_budget="auto"`` running a 144-capture
sweep got no protection (ledger memo 3.5). Two repairs:

- :func:`bundle_retained_bytes` — the cross-member aggregation, explicitly
  labelled a LOWER BOUND: per-capture accountants meter retained activation
  storage, not full member RSS (the panel measured ~255 MB claimed vs
  ~497 MB actual per live-hook member), and members without an accountant
  contribute zero and are NAMED rather than silently averaged in.
- :func:`preflight_retention_projection` — the typed BEFORE-the-first-
  candidate refusal: when the projected retention cannot fit the ceiling,
  the call names the projection, the ceiling, and the remedies instead of
  discovering the overrun at candidate 97.

Every spelling DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._save_budget import format_bytes
from ..errors.episode import BundleExperimentError

if TYPE_CHECKING:
    from . import Bundle

__all__ = ["bundle_retained_bytes", "preflight_retention_projection"]


def bundle_retained_bytes(bundle: Bundle) -> dict[str, Any]:
    """Aggregate retained activation bytes across member accountants.

    Returns
    -------
    dict
        ``total_bytes`` (int), ``per_member`` ({name: bytes}),
        ``unmetered_members`` (names with no accountant — counted at zero,
        never guessed), and ``basis="lower_bound"``: the figure meters
        retained activation storage only, never full member RSS, and
        cross-fork shared storage is charged per accountant.
    """

    per_member: dict[str, int] = {}
    unmetered: list[str] = []
    for name, member in bundle.members.items():
        accountant = getattr(member, "_save_budget_accountant", None)
        ledgers = getattr(accountant, "ledgers", None)
        if not ledgers:
            per_member[name] = 0
            unmetered.append(name)
            continue
        per_member[name] = int(
            sum(int(getattr(ledger, "committed_bytes", 0)) for ledger in ledgers.values())
        )
    return {
        "total_bytes": sum(per_member.values()),
        "per_member": per_member,
        "unmetered_members": unmetered,
        "basis": "lower_bound",
    }


def preflight_retention_projection(
    *,
    per_candidate_bytes: int,
    retained_candidates: int,
    ceiling_bytes: int,
    current_bytes: int = 0,
    operation: str = "site_sweep",
) -> dict[str, Any]:
    """Project a retention request against a byte ceiling BEFORE any work.

    The projection is ``current + per_candidate * retained_candidates`` with
    every input a LOWER BOUND (the aggregation basis above), so a refusal is
    conservative in the honest direction: a request the lower bound already
    breaks cannot fit.

    Returns
    -------
    dict
        The projection record (also suitable for the operation chronology):
        ``projected_bytes``, ``ceiling_bytes``, ``current_bytes``,
        ``per_candidate_bytes``, ``retained_candidates``,
        ``basis="lower_bound"``.

    Raises
    ------
    BundleExperimentError
        ``retention_projection_over_budget`` when the projection crosses the
        ceiling — BEFORE the first candidate runs, naming the projection,
        the ceiling, and the remedies.
    """

    projected = int(current_bytes) + int(per_candidate_bytes) * int(retained_candidates)
    record: dict[str, Any] = {
        "projected_bytes": projected,
        "ceiling_bytes": int(ceiling_bytes),
        "current_bytes": int(current_bytes),
        "per_candidate_bytes": int(per_candidate_bytes),
        "retained_candidates": int(retained_candidates),
        "basis": "lower_bound",
    }
    if projected > ceiling_bytes:
        raise BundleExperimentError(
            f"{operation} retention projection {format_bytes(projected)} "
            f"({retained_candidates} retained candidate(s) x "
            f"{format_bytes(per_candidate_bytes)} + {format_bytes(current_bytes)} "
            f"already retained; lower bound) exceeds the "
            f"{format_bytes(ceiling_bytes)} ceiling. Refusing BEFORE the first "
            "candidate. Remedies: retain fewer members (retain=top_k(k) or "
            "'none'), raise the byte ceiling, or run on the replay lane where "
            "retained members are cheaper.",
            code="retention_projection_over_budget",
            **record,
        )
    return record
