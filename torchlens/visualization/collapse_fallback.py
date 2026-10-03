"""Deterministic coarse-but-legal fallback collapse planner (memo D5(iv)).

Every spelling here is DOCUMENTED-UNSTABLE pending the naming sprint.

When the quality (frontier) planner is refused by the admission gate,
abandoned by the watchdog, capped by the allocation meter, or produces an
empty frontier (the K_CAP cliff), the render must NEVER degrade to an
uncollapsed wall — a 1,688-of-1,712-node "plan" was the silent contract
violation the panel started from. This module builds the deterministic
replacement by REUSING the shipped v1 significance-greedy machinery
(memo D5(iv), review r1 3b): the revision-cached ``analyze_collapse`` scores
order candidate module boxes, the existing readable-band policy decides how
many to take, and the standard fold discovery adds "+N more" runs where the
raw-op count admits it. Cost is one analysis plus a bounded number of
O(universe) plan builds; no frontier state is ever consulted, so the result
is deterministic in (trace revision, context) alone.

The resulting plan flows through the SAME honesty gates as every other plan:
it is a normal typed :class:`~.collapse_plan.CollapsePlan`, its
``scored_k`` equals its realized count by construction (three-way parity),
and callers disclose the planner tier (``planner="linear_fallback"`` /
``"floor_fallback"``) plus the D4 no-silent-floor warnings.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from .auto_collapse import (
    ModuleRepeatFold,
    _readable_band_high,
    analyze_collapse,
    resolve_repeat_folds,
)
from .collapse_plan import CollapsePlan, count

if TYPE_CHECKING:
    from ..data_classes.module import Module
    from ..data_classes.trace import Trace
    from .collapse_plan import RenderContext
    from .source_graph import SourceGraph


def _top_level_fallback_addresses(trace: Trace) -> tuple[str, ...]:
    """Return direct child module addresses suitable for floor fallback boxes.

    Parameters
    ----------
    trace:
        Trace being optimized.

    Returns
    -------
    tuple[str, ...]
        Direct children of ``self`` that hide at least one rendered operation.
    """

    addresses: list[str] = []
    for module in trace.modules:
        address = str(getattr(module, "address", ""))
        if address in {"", "self"}:
            continue
        parent = getattr(module, "address_parent", None)
        if parent not in {"", "self", None}:
            continue
        if int(getattr(module, "num_layers", 0) or 0) <= 1:
            continue
        addresses.append(address)
    return tuple(sorted(dict.fromkeys(addresses)))


def _fallback_collapse_fn(selected: frozenset[str]) -> Callable[[Module], bool]:
    """Return a collapse predicate selecting exactly ``selected`` addresses."""

    def collapse_fn(module: Module) -> bool:
        """Return whether ``module`` is selected by the fallback planner."""

        return module.address in selected

    return collapse_fn


def _greedy_box_selection(trace: Trace, band_high: int, universe_count: int) -> frozenset[str]:
    """Return the significance-greedy module-box selection.

    Walks eligible modules in the v1 signal-size score order (the canonical
    ``analyze_collapse`` ranking; deterministic tiebreak on address),
    skipping any module whose ancestor is already selected (a nested box
    inside a selected box hides nothing extra), and stops once the
    optimistic projected count reaches the readable band. The projection
    uses per-module hidden-op counts, so it is optimistic about overlaps;
    the caller builds the real plan once from the returned set and uses the
    REALIZED count for every disclosure (memo D7: disclosures come from the
    realized plan, never from a scored estimate).
    """

    analysis = analyze_collapse(trace)
    ranked = sorted(
        (
            (address, score)
            for address, score in analysis.scores.items()
            if score > 0.0 and analysis.signals[address].eligible
        ),
        key=lambda item: (-item[1], item[0]),
    )
    selected: set[str] = set()
    projected = universe_count
    for address, _score in ranked:
        if projected <= band_high:
            break
        if any(address != ancestor and address.startswith(f"{ancestor}.") for ancestor in selected):
            continue
        selected.add(address)
        hidden = int(analysis.signals[address].hidden_ops or 0)
        projected -= max(hidden - 1, 0)
    return frozenset(selected)


def linear_fallback_plan(
    trace: Trace,
    context: RenderContext,
    source_graph: SourceGraph | None,
    plan_builder: Callable[..., CollapsePlan],
    raw_op_ceiling: int,
) -> tuple[frozenset[str], dict[str, ModuleRepeatFold], CollapsePlan]:
    """Build the deterministic fallback selection, folds, and plan.

    Parameters
    ----------
    trace:
        Trace being rendered.
    context:
        Render context.
    source_graph:
        Optional already-built normalized source graph for this walk.
    plan_builder:
        ``collapse_optimizer._collapse_plan_for_source_or_trace`` — injected
        to avoid a circular import; called as
        ``plan_builder(trace, collapse_fn, folds, context, source_graph)``.
    raw_op_ceiling:
        The defensive raw-op constant; fold discovery (which walks module
        tables, not the rendered universe) is skipped above it and the skip
        is the caller's disclosure to make.

    Returns
    -------
    tuple
        ``(selected_addresses, repeat_folds, plan)`` where ``plan`` is never
        empty and never silently equal to the full graph without the caller
        disclosing it (the near-uncollapsed warning reads the realized
        count).
    """

    full_plan = plan_builder(trace, None, None, context, source_graph)
    universe_count = count(full_plan)
    band_high = _readable_band_high(trace)
    selected = _greedy_box_selection(trace, band_high, universe_count)
    if not selected:
        return frozenset(), {}, full_plan
    collapse_fn = _fallback_collapse_fn(selected)
    folds: dict[str, ModuleRepeatFold] = {}
    if len(trace.ops) <= raw_op_ceiling:
        # The shipped band-gated fold policy: folds are discovered against
        # the greedy selection exactly as the v1 render path would discover
        # them, so a run of selected same-structure boxes still reads as
        # "box +N more" in the fallback overview.
        folds = resolve_repeat_folds(trace, collapse_fn, context, fold_repeats=None)
    boxed_plan = plan_builder(trace, collapse_fn, folds, context, source_graph)
    if not 0 < count(boxed_plan) < universe_count:
        return frozenset(), {}, full_plan
    return selected, folds, boxed_plan
