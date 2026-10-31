"""The diagnostic frontier: first-dirty sites plus one hop (B6, F16).

"Flagged ops" is unbounded -- one injected NaN flags 65% of distilgpt2's
ops because nonfinite values propagate, so the flagged set is the whole
downstream cone (treescope memo F-E). The FRONTIER is the payload: each
minimal flagged site (a flagged op with no flagged parents), its direct
parents (the last clean values), and its direct children. The frontier
contains the clean-to-dirty transition -- the picture that answers "where
did it come from".

This module is the ONE reusable graph query (memo B6): the offline
report's default array policy, the card's "where did this come from"
link, and any future viewer all consume it.

Distribution-sketch slot disposition (memo B6's second half): the typed
optional slot ships as ``TensorStats.histogram_counts`` /
``histogram_edges`` / ``histogram_evidence`` (C02) -- method, bounds, and
exact/sampled evidence ride the record, absent (``None``) by default on
unsupported payloads. This lane CONSUMES that slot rather than minting a
second one.

Every spelling is DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["FrontierSite", "NonfiniteFrontier", "nonfinite_frontier"]


@dataclass(frozen=True)
class FrontierSite:
    """One frontier member.

    Attributes
    ----------
    label:
        Pass-qualified op label.
    role:
        ``"first_dirty"`` (minimal flagged site), ``"parent"`` (last clean
        value feeding it), or ``"child"`` (direct downstream op).
    """

    label: str
    role: str


@dataclass(frozen=True)
class NonfiniteFrontier:
    """The resolved frontier plus its honesty facts.

    Attributes
    ----------
    flagged_total:
        Total flagged (nonfinite-carrying) ops -- the number the report's
        ``K flagged, M shown`` disclosure prints.
    sites:
        Deterministic-priority members: first-dirty sites in execution
        order, then their parents, then their children (deduplicated).
    coverage:
        The trace's nonfinite-coverage disclosure (basis + unchecked
        counts) -- read it before trusting an empty frontier.
    """

    flagged_total: int
    sites: tuple[FrontierSite, ...]
    coverage: str


def _qualify(label: str) -> str:
    """Normalize one label into pass-qualified space.

    ``trace.nonfinite_ops`` spells every op pass-qualified (``:1`` on
    single-pass ops) while relation edges keep BARE labels for single-pass
    layers -- a bare edge label always means pass 1, so qualifying it makes
    the two spaces comparable.
    """

    return label if ":" in label else f"{label}:1"


def _parents_of(trace: Any, label: str) -> tuple[str, ...]:
    """Pass-qualified parent labels of one op, degrade-safe."""

    try:
        return tuple(_qualify(str(p)) for p in (trace[label].parents or ()))
    except Exception:  # noqa: BLE001 - a missing record contributes no edges
        return ()


def _children_of(trace: Any, label: str) -> tuple[str, ...]:
    """Pass-qualified child labels of one op, degrade-safe."""

    try:
        return tuple(_qualify(str(c)) for c in (trace[label].children or ()))
    except Exception:  # noqa: BLE001 - a missing record contributes no edges
        return ()


def nonfinite_frontier(trace: Any) -> NonfiniteFrontier:
    """Compute the nonfinite diagnostic frontier for one trace.

    A pure query over the recorded dataflow: no payload reads, no device
    transfers, no mutation. Deterministic order: first-dirty sites in the
    trace's flagged-op order, then parents, then children, first
    occurrence wins.

    Parameters
    ----------
    trace:
        Finished ``Trace`` (loaded traces work; the coverage line
        discloses the evidence basis either way).

    Returns
    -------
    NonfiniteFrontier
        Frontier sites plus the ``flagged_total`` and coverage facts.
    """

    flagged = tuple(str(label) for label in (getattr(trace, "nonfinite_ops", ()) or ()))
    flagged_set = frozenset(flagged)
    coverage = str(getattr(trace, "nonfinite_coverage", "unknown"))

    first_dirty = [
        label
        for label in flagged
        if not any(parent in flagged_set for parent in _parents_of(trace, label))
    ]
    ordered: list[FrontierSite] = []
    seen: set[str] = set()
    for label in first_dirty:
        if label not in seen:
            seen.add(label)
            ordered.append(FrontierSite(label, "first_dirty"))
    for label in first_dirty:
        for parent in _parents_of(trace, label):
            if parent not in seen:
                seen.add(parent)
                ordered.append(FrontierSite(parent, "parent"))
    for label in first_dirty:
        for child in _children_of(trace, label):
            if child not in seen:
                seen.add(child)
                ordered.append(FrontierSite(child, "child"))
    return NonfiniteFrontier(
        flagged_total=len(flagged),
        sites=tuple(ordered),
        coverage=coverage,
    )
