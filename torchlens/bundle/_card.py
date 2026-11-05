"""Bundle card and folds: ``repr``, ``str``, ``summary``, the member table, the agent dump
and the derived outcome of a ``Bundle``."""

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


def bundle_summary(self: Bundle) -> str:
    """Multi-line bundle card (``Bundle.summary``): members, outcomes, baseline, relations."""

    distribution = self._outcome_distribution()
    members = ", ".join(
        f"{name} (baseline)" if name == self._baseline_name else name for name in self.names
    )
    outcomes = ", ".join(f"{token}={count}" for token, count in sorted(distribution.items()))
    relation_count = fail_open(lambda: len(self.relations()), lambda _error: 0)
    return "\n".join(
        [
            self.__repr__(),
            f"  members ({len(self)}): {members or 'none'}",
            f"  outcomes: {outcomes or 'none'}",
            f"  baseline: {self._baseline_name!r}",
            f"  relations: {relation_count}",
            f"  structurally consistent: {self.is_structurally_consistent}",
        ]
    )


def bundle_table(self: Bundle) -> Any:
    """One stacked member-keyed table (``Bundle.to_pandas``).

    The leading ``member`` column names the member; the remaining columns are
    each member's own ``Trace.to_pandas()`` rows, concatenated in ``names``
    order with a fresh index. An empty bundle gives an empty frame that still
    carries the ``member`` column.
    """

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
        ) from exc

    frames = []
    for name in self.names:
        frame = self._members[name].to_pandas().copy()
        frame.insert(0, "member", name)
        frames.append(frame)
    if not frames:
        return pd.DataFrame({"member": pd.Series([], dtype=object)})
    return pd.concat(frames, ignore_index=True)


def bundle_agent_json(self: Bundle, *, max_ops: int | None = None) -> dict[str, Any]:
    """The ``torchlens.agent_bundle.v1`` dump (``Bundle.to_agent_json``).

    Card facts first, then every member's own ``Trace.to_agent_json`` dump
    keyed by member name; ``max_ops`` is passed through to each member.
    """

    return {
        "schema": "torchlens.agent_bundle.v1",
        "member_count": len(self),
        "names": list(self.names),
        "baseline_name": self._baseline_name,
        "outcome": dict(self._outcome_distribution()),
        "structurally_consistent": bool(self.is_structurally_consistent),
        "members": {
            name: self._members[name].to_agent_json(max_ops=max_ops) for name in self.names
        },
    }
