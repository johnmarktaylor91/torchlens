"""Typed collapse event ladder: one truth for auto, floats, and the schedule.

Every spelling here is DOCUMENTED-UNSTABLE pending the naming sprint.

Collapse memo item 9 (D8). The measured public float slider was not
consumable: densenet201 spent ``t`` in ``[0, 0.984]`` between 713 and 366
visible nodes (all unreadable) with its whole readable range in the last
1.6% of travel, resnet152's first in-band step sat at ``t=1.0``, and
duplicate ``t`` values and no-op steps existed. The documented story
("``auto`` = the first in-band schedule point") was not what the code did.

This module makes them the same truth: ONE nested ladder of typed
condensation events (:class:`CollapseEvent`) is derived from the max-mode
endpoint, and ``auto``, float-level selection, and the public
``Drawing.collapse_schedule()`` all read it.

- Interior ``t`` maps GEOMETRICALLY in realized count:
  ``target(t) = full * (max/full)**t``, i.e.
  ``t(count) = log(full/count) / log(full/max)`` -- so equal slider travel
  means equal condensation RATIO, killing the dead zone.
- ``t`` is strictly increasing: steps that do not change the realized count
  coalesce into the next effective stop, which lists EVERY event it
  contains (nothing semantic is lost).
- Endpoints are byte-identical to ``collapse="none"`` / ``collapse="max"``
  (the ``t=1.0`` plan IS the max result's plan, interned nodes included).
- ``auto`` = the first ladder point meeting the readable-band count cap
  (the documented contract, now real), else the DISCLOSED strongest point:
  ``band_missed=True`` + ``strongest_plan_count`` computed from the
  REALIZED plan (memo D7), never a silent near-uncollapsed render.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from .auto_collapse import (
    ModuleRepeatFold,
    _readable_band_high,
    _revision_scoped,
    analyze_collapse,
)
from .collapse_estimator import PATHOLOGICAL_OP_MULTIPLIER
from .collapse_plan import (
    ChildSegment,
    CollapsePlan,
    CollapseSchedule,
    CollapseScheduleStep,
    ModuleBox,
    PlanNode,
    RawOp,
    collapse_plan_for_source_graph,
    count,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .._data_classes.trace import Trace
    from .collapse_plan import RenderContext
    from .source_graph import SourceGraph


@dataclass(frozen=True)
class CollapseEvent:
    """One typed condensation event on the ladder (memo D8).

    Parameters
    ----------
    kind:
        ``"box"`` (module interior hidden behind one box), ``"fold"``
        (a legal repeated run elided behind a representative), or
        ``"segment"`` (a run of different ops/children squeezed into one
        capsule; realized only at the max endpoint).
    addresses:
        Represented source addresses: the boxed module, every fold member
        (representative first), or the segment's members/ops.
    replacement:
        Human-readable description of the rendered replacement unit.
    legality:
        Legality witness token: which grammar admitted the event
        (``"box"``, ``"fold"``, ``"segment"``). Fold legality was proven by
        the v2 run grammar at max selection; the ladder never re-derives it.
    score:
        Deterministic ordering score (the DP box cost that sequenced this
        event; lower collapses earlier).
    fold:
        The fold descriptor for ``kind="fold"`` events, so consumers can
        re-apply the exact fold without re-deriving legality.
    """

    kind: str
    addresses: tuple[str, ...]
    replacement: str
    legality: str
    score: float
    fold: ModuleRepeatFold | None = None


def _ladder_t(full_count: int, visible_count: int, max_count: int) -> float:
    """Return the geometric interior ``t`` for one realized count (D8)."""

    if full_count <= max_count or visible_count >= full_count:
        return 0.0
    if visible_count <= max_count:
        return 1.0
    ratio = math.log(full_count / visible_count) / math.log(full_count / max_count)
    return round(min(max(ratio, 0.0), 1.0), 6)


def _collapsed_addresses_for_result(result: Any) -> frozenset[str]:
    """Return module addresses hidden by an optimizer result.

    Parameters
    ----------
    result:
        Optimizer result to inspect.

    Returns
    -------
    frozenset[str]
        Collapsed module addresses represented by selected boxes, run folds,
        and child segments.
    """

    addresses = set(result.selected)
    addresses.update(result.repeat_folds)
    for segment in (result.segments or {}).values():
        addresses.update(segment.members)
    return frozenset(addresses)


def _reported_collapsed_addresses(result: Any) -> frozenset[str]:
    """Return the honest public hidden set for an optimizer result.

    Extends :func:`_collapsed_addresses_for_result` with the concrete op
    labels hidden by operation segments, which hide rendered nodes without
    collapsing any module address.
    """

    addresses = set(_collapsed_addresses_for_result(result))
    for segment in (result.segments or {}).values():
        if segment.kind == "op":
            addresses.update(str(op) for op in segment.ops)
    return frozenset(addresses)


def _optimizer() -> Any:
    """Return the optimizer module (runtime import; breaks the cycle)."""

    from . import collapse_optimizer

    return collapse_optimizer


def _event_costs(trace: Trace, context: RenderContext, max_result: Any) -> dict[str, float]:
    """Return the DP box-cost ordering map for the max result's addresses."""

    optimizer = _optimizer()
    addresses = _collapsed_addresses_for_result(max_result)
    if not addresses:
        return {}
    analysis = analyze_collapse(trace)
    child_addresses = optimizer._child_address_map(trace)
    state = optimizer._OptimizerState(
        trace=trace,
        context=context,
        analysis=analysis,
        child_addresses=child_addresses,
        hidden_counts=optimizer._rendered_module_hidden_counts(trace, context),
        structural_digests=optimizer._structural_digest_map(trace, child_addresses, analysis),
        expanded_cache={},
        role_components_cache={},
        child_segments_cache={},
        single_member_expanded_cache={},
        box_cost_cache={},
        branch_salience_cache={},
        output_shape_cache={},
        weights=optimizer.OptimizerWeights(),
        g_star=max_result.g_star or 1.0,
        total_ops=optimizer._optimizer_total_units(trace, context),
        allow_folds=True,
        allow_segments=True,
        max_salience_floor=None,
        rendered_own_units=optimizer._rendered_own_unit_map(trace, context),
    )
    costs: dict[str, float] = {}
    for address in addresses:
        signal = analysis.signals.get(address)
        if signal is None:
            costs[address] = math.inf
        else:
            costs[address] = optimizer._cached_box_cost(trace, signal, state)
    return costs


def _source_graph(trace: Trace, context: RenderContext) -> SourceGraph:
    """Build the normalized source graph (the one deferred import site)."""

    from .source_graph import build_source_graph

    return build_source_graph(trace, context)


def build_event_ladder(
    trace: Trace,
    context: RenderContext,
    max_result: Any,
) -> tuple[CollapseEvent, ...]:
    """Return the ordered typed condensation events for one max endpoint.

    Box and fold events are ordered by their DP box cost (cheapest-to-hide
    first; deterministic address tiebreak); segment events come last because
    segments are a max-endpoint post-pass and realize only there.
    """

    costs = _event_costs(trace, context, max_result)
    fold_by_rep: dict[str, ModuleRepeatFold] = {}
    fold_members: set[str] = set()
    for fold in max_result.repeat_folds.values():
        if fold.representative not in fold_by_rep:
            fold_by_rep[fold.representative] = fold
            fold_members.update(fold.addresses)
    events: list[CollapseEvent] = []
    for address in sorted(set(max_result.selected) - fold_members):
        events.append(
            CollapseEvent(
                kind="box",
                addresses=(address,),
                replacement=f"module box @{address}",
                legality="box",
                score=costs.get(address, math.inf),
            )
        )
    for representative, fold in sorted(fold_by_rep.items()):
        member_costs = [costs.get(address, math.inf) for address in fold.addresses]
        events.append(
            CollapseEvent(
                kind="fold",
                addresses=tuple(fold.addresses),
                replacement=(f"repeat fold @{representative} (+{len(fold.addresses) - 1} more)"),
                legality="fold",
                score=min(member_costs) if member_costs else math.inf,
                fold=fold,
            )
        )
    events.sort(key=lambda event: (event.score, event.addresses))
    for _name, segment in sorted((max_result.segments or {}).items()):
        events.append(
            CollapseEvent(
                kind="segment",
                addresses=tuple(segment.members) or tuple(str(op) for op in segment.ops),
                replacement=segment.label,
                legality="segment",
                score=math.inf,
            )
        )
    return tuple(events)


@_revision_scoped
def collapse_schedule(
    trace: Trace,
    context: RenderContext,
    weights: Any | None = None,
) -> CollapseSchedule:
    """Return the monotone public float collapse schedule (ladder-backed).

    Parameters
    ----------
    trace:
        Trace being rendered.
    context:
        Rendering context.
    weights:
        Accepted for signature compatibility and ignored: the public float
        schedule is weight-independent and always derives from the
        default-weight max plan, so caching by context alone is sound.

    Returns
    -------
    CollapseSchedule
        Ordered nested schedule from the full graph to the current max
        plan. ``t`` is strictly increasing (geometric in realized count),
        no-op steps are coalesced into their next effective stop, and each
        step carries the typed events it applies.
    """

    _ = weights
    optimizer = _optimizer()
    if len(trace._layer_list) > optimizer.COLLAPSE_OPTIMIZER_MAX_OPS * PATHOLOGICAL_OP_MULTIPLIER:
        # Pathological pre-gate (memo D5(i); r8 R60-13 preserved): the
        # revision snapshot below is an uncached O(N) deep walk, pointless
        # when the optimizer will decline. Over-budget but non-pathological
        # traces proceed: select_collapse_plan serves the deterministic
        # fallback endpoint, so the schedule stays a real ladder.
        over_source_graph = _source_graph(trace, context)
        over_full_plan = collapse_plan_for_source_graph(over_source_graph, None, None)
        over_full_count = count(over_full_plan)
        optimizer.select_collapse_plan(trace, context, mode="max", source_graph=over_source_graph)
        return CollapseSchedule(
            (
                CollapseScheduleStep(
                    t=0.0,
                    target_count=over_full_count,
                    visible_count=over_full_count,
                    collapsed_addresses=frozenset(),
                    plan=over_full_plan,
                ),
            )
        )
    from .auto_collapse import _collapse_graph_revision

    revision = _collapse_graph_revision(trace)
    cache_entry = optimizer._SCHEDULE_CACHE.get(trace)
    if cache_entry is None or cache_entry[0] != revision:
        cached_by_context: dict[RenderContext, CollapseSchedule] = {}
        optimizer._SCHEDULE_CACHE[trace] = (revision, cached_by_context)
    else:
        cached_by_context = cache_entry[1]
    cached = cached_by_context.get(context)
    if cached is not None:
        return cached
    source_graph = _source_graph(trace, context)
    node_pool: dict[PlanNode, PlanNode] = {}
    full_plan = collapse_plan_for_source_graph(source_graph, None, None, node_pool=node_pool)
    full_count = count(full_plan)
    max_result = optimizer.select_collapse_plan(
        trace, context, mode="max", source_graph=source_graph
    )
    max_plan = CollapsePlan(
        nodes=tuple(node_pool.setdefault(node, node) for node in max_result.plan.nodes),
        context=max_result.plan.context,
    )
    first_step = CollapseScheduleStep(
        t=0.0,
        target_count=full_count,
        visible_count=full_count,
        collapsed_addresses=frozenset(),
        plan=full_plan,
    )
    if max_result.declined:
        schedule = CollapseSchedule((first_step,))
        cached_by_context[context] = schedule
        return schedule
    max_count = max_result.visible_count
    if full_count <= max_count:
        schedule = CollapseSchedule(
            (
                first_step,
                CollapseScheduleStep(
                    t=1.0,
                    target_count=max_count,
                    visible_count=max_count,
                    collapsed_addresses=_reported_collapsed_addresses(max_result),
                    plan=max_plan,
                    events=build_event_ladder(trace, context, max_result),
                ),
            )
        )
        cached_by_context[context] = schedule
        return schedule
    ladder_events = build_event_ladder(trace, context, max_result)
    interior_steps = _interior_ladder_steps(
        ladder_events, source_graph, node_pool, (full_count, max_count)
    )
    steps = [first_step, *interior_steps]
    # The endpoint step (t=1.0) absorbs every still-pending event (identity
    # against the ONE ladder construction); its plan IS the max plan
    # (interned nodes), keeping the byte-identity contract.
    applied_ids = {id(event) for step in interior_steps for event in step.events}
    remaining = tuple(event for event in ladder_events if id(event) not in applied_ids)
    steps.append(
        CollapseScheduleStep(
            t=1.0,
            target_count=max_count,
            visible_count=max_count,
            collapsed_addresses=_reported_collapsed_addresses(max_result),
            plan=max_plan,
            events=remaining,
        )
    )
    schedule = CollapseSchedule(tuple(steps))
    cached_by_context[context] = schedule
    return schedule


def _interior_ladder_steps(
    events: tuple[CollapseEvent, ...],
    source_graph: SourceGraph,
    node_pool: dict[PlanNode, PlanNode],
    counts: tuple[int, int],
) -> list[CollapseScheduleStep]:
    """Realize the interior (strictly count-reducing) ladder stops (D8).

    ``counts`` is ``(full_count, max_count)``. Only count-reducing stops are
    public; events that changed nothing coalesce into the next effective
    stop; counts at or below the endpoint's belong to the endpoint itself.
    Segment events never realize interior (max-endpoint post-pass).
    """

    full_count, max_count = counts
    optimizer = _optimizer()
    steps: list[CollapseScheduleStep] = []
    selected: set[str] = set()
    folds: dict[str, ModuleRepeatFold] = {}
    pending: list[CollapseEvent] = []
    previous_count = full_count
    for event in events:
        if event.kind == "segment":
            continue
        selected.update(event.addresses)
        if event.fold is not None:
            for address in event.fold.addresses:
                folds[address] = event.fold
        pending.append(event)
        plan = collapse_plan_for_source_graph(
            source_graph,
            optimizer._collapse_fn_from_selected(frozenset(selected)),
            dict(folds),
            node_pool=node_pool,
        )
        visible_count = count(plan)
        if previous_count > visible_count > max_count:
            steps.append(
                CollapseScheduleStep(
                    t=_ladder_t(full_count, visible_count, max_count),
                    target_count=visible_count,
                    visible_count=visible_count,
                    collapsed_addresses=frozenset(selected),
                    plan=plan,
                    events=tuple(pending),
                )
            )
            pending = []
            previous_count = visible_count
    return steps


def _ladder_state_through(
    steps: tuple[CollapseScheduleStep, ...],
    chosen: CollapseScheduleStep,
) -> tuple[frozenset[str], dict[str, ModuleRepeatFold]]:
    """Accumulate the (selected, folds) state up to and including one step."""

    selected: set[str] = set()
    folds: dict[str, ModuleRepeatFold] = {}
    for step in steps:
        for event in step.events:
            if event.kind == "segment":
                continue
            selected.update(event.addresses)
            if event.fold is not None:
                for address in event.fold.addresses:
                    folds[address] = event.fold
        if step is chosen:
            break
    return frozenset(selected), folds


def auto_from_ladder(
    trace: Trace,
    context: RenderContext,
    source_graph: SourceGraph | None,
) -> Any:
    """Return the auto-mode result: the first in-band ladder point, evened.

    ``auto`` is the first schedule step whose realized count meets the
    readable-band cap, evened by :func:`_even_auto_plan`. When no step
    reaches the band, the DISCLOSED strongest point is served: the max
    endpoint with ``band_missed=True`` and ``strongest_plan_count`` from the
    REALIZED plan (the reason never says "floor").
    """

    optimizer = _optimizer()
    schedule = collapse_schedule(trace, context)
    max_result = optimizer.select_collapse_plan(
        trace, context, mode="max", source_graph=source_graph
    )
    if max_result.declined:
        return max_result
    band_high = _readable_band_high(trace)
    chosen = next((step for step in schedule.steps if step.visible_count <= band_high), None)
    if chosen is None:
        strongest = schedule.steps[-1]
        band_note = (
            f"band_missed: strongest_plan_count={strongest.visible_count} "
            f"exceeds the readable band ({band_high}); planner="
            f"{max_result.planner}"
        )
        # A fallback-tier endpoint already carries its cause and remedy
        # (K_CAP diagnosis, "module= focus" guidance); the band note joins
        # it rather than clobbering it (memo D7: protected reasons).
        base_reason = max_result.reason
        return replace(
            max_result,
            band_missed=True,
            strongest_plan_count=strongest.visible_count,
            reason=f"{base_reason}; {band_note}" if base_reason else band_note,
        )
    if chosen is schedule.steps[-1]:
        return max_result
    ladder_state = _ladder_state_through(schedule.steps, chosen)
    folds = ladder_state[1]
    plan, segments, selected = _even_auto_plan(trace, context, source_graph, chosen, ladder_state)
    return optimizer.OptimizerResult(
        selected=selected,
        repeat_folds=folds,
        plan=plan,
        visible_count=count(plan),
        analyze_ms=analyze_collapse(trace).elapsed_ms,
        select_ms=0.0,
        g_star=max_result.g_star,
        segments=segments,
        planner="ladder",
        scored_k=count(plan),
    )


#: ``auto`` folds a run of at least this many structurally identical,
#: consecutive sibling boxes into one segment (two read fine side by side).
EVEN_FOLD_MIN_RUN = 3


def _within(address: str, ancestor: str) -> bool:
    """Return whether ``address`` is ``ancestor`` or inside it."""

    return address == ancestor or address.startswith(ancestor + ".")


def _parent_address(address: str) -> str:
    """Return the module parent address (``"self"`` for top-level modules)."""

    return address.rsplit(".", 1)[0] if "." in address else "self"


def _even_auto_plan(
    trace: Trace,
    context: RenderContext,
    source_graph: SourceGraph | None,
    chosen: CollapseScheduleStep,
    ladder_state: tuple[frozenset[str], dict[str, ModuleRepeatFold]],
) -> tuple[CollapsePlan, dict[str, Any], frozenset[str]]:
    """Even out the chosen ladder point: box stray small modules, fold runs.

    The ladder orders boxes by cost, so its first in-band point can leave a
    small module expanded between boxed siblings (convnext's stem and
    stage-2 downsample) and a run of identical blocks as one box each
    (convnext's nine stage-2 blocks). Both passes only condense: the evened
    plan never has more visible nodes than the ladder point. ``ladder_state``
    is the ``(selected, folds)`` pair accumulated through ``chosen``.
    """

    selected, folds = ladder_state
    plan = chosen.plan
    boxes = _even_auto_boxes(trace, analyze_collapse(trace).signals, selected)
    if boxes != selected:
        evened = collapse_plan_for_source_graph(
            source_graph or _source_graph(trace, context),
            _optimizer()._collapse_fn_from_selected(boxes),
            dict(folds),
        )
        if count(evened) <= count(plan):
            plan, selected = evened, boxes
    plan, segments = _fold_identical_runs(trace, context, plan)
    return plan, segments, selected


def _even_auto_boxes(
    trace: Trace, signals: Mapping[str, Any], selected: frozenset[str]
) -> frozenset[str]:
    """Box fully expanded small modules that sit beside condensed siblings.

    A module qualifies when it may render as a box, runs once, contains no
    box, is inside no box, and hides no more ops than the smallest box that
    is a sibling or a sibling's child: left expanded it reads as stray raw
    ops between boxes. It must also be the only expanded sibling of its
    structure: when identical siblings are still expanded, the ladder is part
    way through that family and its in-band point stays as the ladder left it
    (a stack of identical blocks keeps "just enough" boxed). Outermost first,
    so the parent the user wrote wins over its child.
    """

    boxes = set(selected)
    children: dict[str, list[str]] = {}
    for address in signals:
        children.setdefault(_parent_address(address), []).append(address)
    for address in sorted(signals, key=lambda item: (item.count("."), item)):
        signal = signals[address]
        if not _may_box_expanded(trace, address, signal, boxes):
            continue
        siblings = [item for item in children[_parent_address(address)] if item != address]
        if _has_expanded_twin(trace, signal, siblings, boxes, signals):
            continue
        sibling_box_ops = _subtree_box_ops(siblings, boxes, signals)
        if sibling_box_ops and signal.hidden_ops <= min(sibling_box_ops):
            boxes.add(address)
    return frozenset(boxes)


def _may_box_expanded(trace: Trace, address: str, signal: Any, boxes: set[str]) -> bool:
    """Return whether a single-call, box-eligible module is fully expanded."""

    if signal.num_calls > 1 or not _optimizer()._eligible_box(trace, address, signal):
        return False
    return not any(_within(address, box) or _within(box, address) for box in boxes)


def _has_expanded_twin(
    trace: Trace, signal: Any, siblings: list[str], boxes: set[str], signals: Mapping[str, Any]
) -> bool:
    """Return whether a sibling with the same structure is also fully expanded."""

    return any(
        signals[sibling].structural_digest == signal.structural_digest
        and _may_box_expanded(trace, sibling, signals[sibling], boxes)
        for sibling in siblings
    )


def _subtree_box_ops(roots: list[str], boxes: set[str], signals: Mapping[str, Any]) -> list[int]:
    """Return the hidden-op counts of boxes that are a root or a root's child.

    Only boxes one level down count: a box nested deeper (BERT's
    ``encoder.layer.0`` beside ``embeddings``) sits in its own cluster and
    does not make the module beside its ancestor read as stray.
    """

    return [
        signals[box].hidden_ops
        for box in boxes
        if box in signals and any(root in (box, _parent_address(box)) for root in roots)
    ]


def _fold_identical_runs(
    trace: Trace, context: RenderContext, plan: CollapsePlan
) -> tuple[CollapsePlan, dict[str, Any]]:
    """Replace runs of identical consecutive sibling boxes with segments.

    A run is a legal child segment (the max planner's own legality, at the
    auto dominance limit) whose members share one structural digest and
    number at least :data:`EVEN_FOLD_MIN_RUN`.
    """

    optimizer = _optimizer()
    analysis = analyze_collapse(trace)
    legality = (
        analysis,
        optimizer._rendered_module_hidden_counts(trace, context),
        optimizer._optimizer_total_units(trace, context),
    )
    nodes: list[PlanNode] = []
    segments: dict[str, Any] = {}
    covered: set[str] = set()
    index = 0
    while index < len(plan.nodes):
        node = plan.nodes[index]
        run = _identical_run(trace, context, legality, plan.nodes, index)
        if len(run) >= EVEN_FOLD_MIN_RUN:
            ops = optimizer._child_segment_covered_ops(analysis, run)
            descriptor = optimizer._make_child_segment_descriptor(trace, context, run, ops)
            segments[descriptor.name] = descriptor
            covered.update(str(label).rsplit(":", 1)[0] for label in ops)
            nodes.append(ChildSegment(run))
            index += len(run)
            continue
        if not (isinstance(node, RawOp) and isinstance(node.op, str) and node.op in covered):
            nodes.append(node)
        index += 1
    if not segments:
        return plan, segments
    optimizer._assert_segment_descriptor_parity(nodes, segments)
    return CollapsePlan(nodes=tuple(nodes), context=plan.context), segments


def _identical_run(
    trace: Trace,
    context: RenderContext,
    legality: tuple[Any, Any, int],
    nodes: tuple[PlanNode, ...],
    start: int,
) -> tuple[str, ...]:
    """Return the identical-digest prefix of the legal segment run at ``start``.

    ``legality`` is ``(analysis, rendered hidden counts, total units)``, the
    inputs of the max planner's segment legality.
    """

    if not isinstance(nodes[start], ModuleBox):
        return ()
    analysis, hidden_counts, total_units = legality
    run = _optimizer()._legal_plan_child_segment_run(
        trace, context, analysis, nodes, start, hidden_counts, total_units, dominance_limit=0.75
    )
    if not run:
        return ()
    signals = analysis.signals
    digest = signals[run[0]].structural_digest
    length = 0
    while length < len(run) and signals[run[length]].structural_digest == digest:
        length += 1
    return tuple(run[:length])
