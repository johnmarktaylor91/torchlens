"""Bundle card and outcome fold: ``repr``, ``str`` and the derived outcome of a ``Bundle``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._capture.outcome import outcome_for
from .._utils.fail_open import fail_open
from ._outcome_fold import _fold_member_outcomes

if TYPE_CHECKING:
    from .._capture.outcome import CaptureOutcome
    from . import Bundle


def _member_outcome_token(member: Any) -> str:
    """One member's settled outcome-status token, degrading to ``unknown``."""

    outcome = fail_open(lambda: getattr(member, "outcome", None), lambda _error: None)
    return getattr(getattr(outcome, "status", None), "value", None) or "unknown"


def outcome_distribution(self: Bundle) -> dict[str, int]:
    """Count member capture outcomes by settled status (``Bundle.outcome_distribution``)."""

    counts: dict[str, int] = {}
    for member in self._members.values():
        token = _member_outcome_token(member)
        counts[token] = counts.get(token, 0) + 1
    return counts


def bundle_repr(self: Bundle) -> str:
    """One-line bundle card: members, baseline, outcome distribution."""

    distribution = self._outcome_distribution()
    outcome_note = ""
    if any(token != "complete" for token in distribution):
        pairs = ", ".join(f"{token}={count}" for token, count in sorted(distribution.items()))
        outcome_note = f", outcomes=({pairs})"
    base = (
        f"Bundle(n_members={len(self)}, names={self.names!r}, "
        f"baseline={self._baseline_name!r}, "
        f"structurally_consistent={self.is_structurally_consistent}"
        f"{outcome_note})"
    )
    if not self._effect_tables:
        return base
    # D3c: silent truncation reading as "covered everything" is the
    # failure mode this sprint keeps finding — an engine-built bundle
    # states its candidate accounting and policy in its own repr.
    table = next(reversed(self._effect_tables.values()))
    counts: dict[str, int] = {}
    for row in table.rows:
        if row.candidate_id == "__baseline__":
            continue
        counts[row.status] = counts.get(row.status, 0) + 1
    attempted = sum(counts.values())
    summary = ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
    return (
        f"{base[:-1]}, effects=[attempted={attempted}, {summary}, "
        f"retain={table.retain_policy}, lane={table.lane}])"
    )


def bundle_str(self: Bundle) -> str:
    """Bounded bundle view: repr line + per-member one-liners (``Bundle.__str__``)."""

    from ..stats._envelope import COLLECTION_MAX_CHILDREN

    lines = [self.__repr__()]
    names = list(self.names)
    for name in names[:COLLECTION_MAX_CHILDREN]:
        member = self._members[name]
        marker = " (baseline)" if name == self._baseline_name else ""
        lines.append(f"  {name}{marker}: {member!r}")
    if len(names) > COLLECTION_MAX_CHILDREN:
        lines.append(f"  ... {len(names) - COLLECTION_MAX_CHILDREN} more members")
    return "\n".join(lines)


def bundle_outcome(self: Bundle) -> CaptureOutcome:
    """Derived worst-of-members capture-outcome fold (``Bundle.outcome``)."""

    return _fold_member_outcomes(
        [(name, outcome_for(member)) for name, member in self._members.items()]
    )
