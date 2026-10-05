"""Frontier records and merge algebra for the collapse optimizer.

The bounded beam in :mod:`torchlens.visualization.collapse_optimizer` keeps at
most one point per rendered node count, at most ``FRONTIER_CAP`` node-count
buckets, and drops subtrees above ``K_CAP`` rendered nodes. This module holds
the address-independent decision records and the pure merge helpers that apply
those bounds; the optimizer owns the search, memoization and watchdog metering.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

K_CAP = 64


FRONTIER_CAP = 32


@dataclass(frozen=True)
class _DecisionPoint:
    """Address-independent DP frontier point plus reconstruction witness."""

    k: int
    cost: float
    decision: Any
    box_costs: tuple[float, ...]
    priority: tuple[int, ...]


@dataclass(frozen=True)
class _ModuleDecision:
    """Memoized module-level choice."""

    kind: Literal["box", "expand"]
    segments: tuple[_SegmentDecision, ...] = ()


@dataclass(frozen=True)
class _SegmentDecision:
    """Memoized choices for one segmented child sequence."""

    components: tuple[_ComponentDecision, ...]


@dataclass(frozen=True)
class _ComponentDecision:
    """Memoized role-component treatment."""

    kind: Literal["boxes", "folded", "expanded", "segmented"]
    member_ks: tuple[int, ...] = ()
    run_indices: tuple[tuple[int, ...], ...] = ()


def _retain_best_frontier_point(
    best_by_count: dict[int, tuple[Any, tuple[float, int, tuple[Any, ...], tuple[Any, ...]]]],
    point: Any,
) -> None:
    """Retain ``point`` if it is the best candidate seen for its node count."""

    sort_key = _point_sort_key(point)
    incumbent = best_by_count.get(point.k)
    if incumbent is None or sort_key < incumbent[1]:
        best_by_count[point.k] = (point, sort_key)


def _frontier_from_best_by_count(
    best_by_count: Mapping[int, tuple[Any, tuple[float, int, tuple[Any, ...], tuple[Any, ...]]]],
) -> tuple[Any, ...]:
    """Return the beam-capped deterministic frontier from per-count bests.

    Dropping node-count buckets beyond ``FRONTIER_CAP`` is the beam bound
    described in the module docstring.
    """

    ordered = sorted(best_by_count.values(), key=lambda item: item[1])
    return tuple(sorted((item[0] for item in ordered[:FRONTIER_CAP]), key=lambda point: point.k))


def _decision_sort_key(
    cost: float,
    k: int,
    priority: tuple[int, ...],
) -> tuple[float, int, tuple[int, ...], tuple[Any, ...]]:
    """Return the deterministic ordering key for a decision candidate."""

    return (cost, k, priority, ())


def _retain_best_decision_candidate(
    best_by_count: dict[
        int, tuple[_DecisionPoint, tuple[float, int, tuple[Any, ...], tuple[Any, ...]]]
    ],
    point: _DecisionPoint,
    sort_key: tuple[float, int, tuple[int, ...], tuple[Any, ...]],
) -> None:
    """Retain ``point`` when its precomputed key wins its node-count bucket."""

    incumbent = best_by_count.get(point.k)
    if incumbent is None or sort_key < incumbent[1]:
        best_by_count[point.k] = (point, sort_key)


def _merge_module_segment_frontiers(
    left_points: Sequence[_DecisionPoint],
    right_points: Sequence[_DecisionPoint],
) -> tuple[_DecisionPoint, ...]:
    """Merge module expand points with child-segment decisions."""

    best_by_count: dict[
        int, tuple[_DecisionPoint, tuple[float, int, tuple[Any, ...], tuple[Any, ...]]]
    ] = {}
    for left in left_points:
        left_decision = cast(_ModuleDecision, left.decision)
        for right in right_points:
            k = left.k + right.k
            if k > K_CAP:
                continue
            cost = round(left.cost + right.cost, 6)
            priority = (*left.priority, *right.priority)
            sort_key = _decision_sort_key(cost, k, priority)
            incumbent = best_by_count.get(k)
            if incumbent is not None and incumbent[1] <= sort_key:
                continue
            right_decision = cast(_SegmentDecision, right.decision)
            point = _DecisionPoint(
                k=k,
                cost=cost,
                decision=_ModuleDecision(
                    "expand",
                    (*left_decision.segments, right_decision),
                ),
                box_costs=(*left.box_costs, *right.box_costs),
                priority=priority,
            )
            _retain_best_decision_candidate(best_by_count, point, sort_key)
    return cast(tuple[_DecisionPoint, ...], _frontier_from_best_by_count(best_by_count))


def _merge_segment_component_frontiers(
    left_points: Sequence[_DecisionPoint],
    right_points: Sequence[_DecisionPoint],
) -> tuple[_DecisionPoint, ...]:
    """Merge segment points with one role-component treatment frontier."""

    best_by_count: dict[
        int, tuple[_DecisionPoint, tuple[float, int, tuple[Any, ...], tuple[Any, ...]]]
    ] = {}
    for left in left_points:
        left_decision = cast(_SegmentDecision, left.decision)
        for right in right_points:
            k = left.k + right.k
            if k > K_CAP:
                continue
            cost = round(left.cost + right.cost, 6)
            priority = (*left.priority, *right.priority)
            sort_key = _decision_sort_key(cost, k, priority)
            incumbent = best_by_count.get(k)
            if incumbent is not None and incumbent[1] <= sort_key:
                continue
            right_decision = cast(_ComponentDecision, right.decision)
            point = _DecisionPoint(
                k=k,
                cost=cost,
                decision=_SegmentDecision((*left_decision.components, right_decision)),
                box_costs=(*left.box_costs, *right.box_costs),
                priority=priority,
            )
            _retain_best_decision_candidate(best_by_count, point, sort_key)
    return cast(tuple[_DecisionPoint, ...], _frontier_from_best_by_count(best_by_count))


def _merge_component_member_frontiers(
    left_points: Sequence[_DecisionPoint],
    right_points: Sequence[_DecisionPoint],
) -> tuple[_DecisionPoint, ...]:
    """Merge expanded role-component points with one member frontier."""

    best_by_count: dict[
        int, tuple[_DecisionPoint, tuple[float, int, tuple[Any, ...], tuple[Any, ...]]]
    ] = {}
    for left in left_points:
        left_decision = cast(_ComponentDecision, left.decision)
        for right in right_points:
            k = left.k + right.k
            if k > K_CAP:
                continue
            cost = round(left.cost + right.cost, 6)
            priority = (*left.priority, *right.priority)
            sort_key = _decision_sort_key(cost, k, priority)
            incumbent = best_by_count.get(k)
            if incumbent is not None and incumbent[1] <= sort_key:
                continue
            point = _DecisionPoint(
                k=k,
                cost=cost,
                decision=_ComponentDecision(
                    "expanded",
                    (*left_decision.member_ks, right.k),
                ),
                box_costs=(*left.box_costs, *right.box_costs),
                priority=priority,
            )
            _retain_best_decision_candidate(best_by_count, point, sort_key)
    return cast(tuple[_DecisionPoint, ...], _frontier_from_best_by_count(best_by_count))


def _point_sort_key(point: Any) -> tuple[float, int, tuple[Any, ...], tuple[Any, ...]]:
    """Return deterministic ordering key for frontier points."""

    if isinstance(point, _DecisionPoint):
        return (point.cost, point.k, point.priority, ())
    fold_addresses = tuple(fold.representative for fold in point.folds)
    return (point.cost, point.k, tuple(sorted(point.selected)), fold_addresses)
