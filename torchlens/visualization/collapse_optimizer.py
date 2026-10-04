"""Bounded beam-search tree-cut optimizer for v2 auto collapse.

The selection is a memoized tree cut with a deterministic, bounded frontier,
not an exact optimizer:

- every frontier merge keeps at most one point per rendered node count and at
  most ``FRONTIER_CAP`` node-count buckets, so reachable counts can be pruned
  (a beam, with classic beam suboptimality);
- per-count pruning minimizes the additive sum of box costs, while the global
  objective (:func:`_global_q` / :func:`_global_max_q`) also adds a
  non-additive ``w_max * max(box_costs)`` term, so a same-count point that
  wins globally can be pruned locally;
- subtrees larger than ``K_CAP`` rendered nodes are dropped from frontiers
  entirely.

The result is a deterministic approximation of the stated objective. "DP" and
"frontier" in the helper names refer to the memoized structure, not to an
optimality guarantee.
"""

from __future__ import annotations

import hashlib
import math
import time
import warnings
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, cast

from .._errors import InvalidArgumentError
from ..errors._base import TorchLensWarning
from ..utils.display import user_stacklevel
from ._collapse_disclosures import _warn_budget_fallback
from ._collapse_runs import FlowAdjacency, _flow_adjacency, longest_uniform_legal_run
from ._collapse_signatures import fingerprints_for
from ._segment_descriptors import (
    _child_segment_covered_ops,
    _crosses_module_call_boundary,
    _effective_render_module_stack,
    _make_child_segment_descriptor,
    _make_op_segment_descriptor,
)
from .auto_collapse import (
    GENERIC_CONTAINER_CLASSES,
    RUN_FOLD_MIN_LENGTH,
    ChildCondensedFlowGraph,
    CollapseAnalysis,
    ModuleCollapseSignals,
    ModuleRepeatFold,
    _collapse_graph_revision,
    _flow_ordered_child_addresses,
    _is_trunk_collapse,
    _make_run_fold,
    _module_output_shape_tuple,
    _paired_external_connector,
    _readable_band_high,
    _rendered_module_hidden_counts,
    _revision_scoped,
    _run_fold_is_chain_interval,
    _shape_channel_dim,
    _shape_spatial_dims,
    analyze_collapse,
)
from .collapse_estimator import (
    ACTIVE_BUDGET,
    PATHOLOGICAL_OP_MULTIPLIER,
    QUALITY_PLANNER_BUDGET_MS,
    WATCHDOG_FLOOR_MS,
    WATCHDOG_SLACK,
    EstimatorDiagnostics,
    FallbackDegrade,
    SelectionBudget,
    charge_frontier_allocation,
    predicted_select_ms,
    watchdog_enabled,
)
from .collapse_fallback import _top_level_fallback_addresses, linear_fallback_plan
from .collapse_ladder import collapse_schedule  # noqa: F401  (re-export; moved in F11)
from .collapse_plan import (
    ChildSegment,
    CollapsePlan,
    CollapseSchedule,
    EllipsisNode,
    ModuleBox,
    OpSegment,
    PlanNode,
    RawOp,
    RenderContext,
    RepeatFold,
    SegmentDescriptor,
    collapse_plan_for_source_graph,
    collapse_plan_for_trace,
    count,
)

if TYPE_CHECKING:
    from ..data_classes.module import Module
    from ..data_classes.trace import Trace
    from .auto_collapse import ChildCondensedFlowGraph
    from .source_graph import SourceGraph


K_CAP = 64
FRONTIER_CAP = 32
MAX_SALIENCE_FLOOR = 0.75

#: Preflight compute ceiling for the v2 collapse selection (b8 R60). The
#: frontier DP is measured superlinear (~n^1.75) in rendered op count with no
#: internal time budget, so above this many ops the optimizer DECLINES with a
#: disclosed warning instead of silently burning CPU-hours: ``draw`` renders
#: uncollapsed, ``Trace.collapse_plan()`` refuses typed
#: (``collapse_plan_unavailable``), and the schedule degrades to its single
#: full-graph step. Calibration: the largest shipped-suite models
#: (densenet201 at 1,517 ops, maxvit_t at 1,307) must stay admitted -- their
#: schedules are the suite's slow cells at tens of seconds -- while the
#: extrapolated multi-thousand-op cost (an hour-class compute at ~5,000 ops)
#: is exactly what the ceiling exists to refuse.
COLLAPSE_OPTIMIZER_MAX_OPS = 2000


@dataclass(frozen=True)
class OptimizerWeights:
    """Cost weights for the v2 tree-cut optimizer.

    Parameters
    ----------
    w_grain:
        Weight for deviation from the target hidden-mass grain.
    w_landmark:
        Penalty for hiding landmark crossings.
    w_trunk:
        Penalty for collapsing nearly all input-output trunk structure.
    w_generic:
        Penalty for generic container labels.
    w_dom:
        Penalty for one module dominating the full trace.
    w_sal:
        Penalty for hiding salient child-level branching, divided by same-role
        sibling multiplicity.
    fold_intrinsic:
        Small intrinsic fold cost, so folds win only under band pressure.
    w_max:
        Weight for the maximum selected box cost in global selection.
    w_k:
        Linear in-band preference for smaller rendered cuts.
    """

    w_grain: float = 1.0
    w_landmark: float = 1.2
    w_trunk: float = 1.0
    w_generic: float = 0.15
    w_dom: float = 1.5
    w_sal: float = 0.8
    fold_intrinsic: float = 0.15
    w_max: float = 0.3
    w_k: float = 2.5
    segment_intrinsic: float = 0.03


@dataclass(frozen=True)
class RoleComponent:
    """Flow-ordered sibling role component.

    Parameters
    ----------
    members:
        Direct child module addresses in parent flow order.
    """

    members: tuple[str, ...]


@dataclass(frozen=True)
class OptimizerResult:
    """Selected v2 collapse plan and renderer adapter state.

    Parameters
    ----------
    selected:
        Module addresses rendered as boxes.
    repeat_folds:
        Fold descriptors keyed by every folded address.
    plan:
        Renderer-faithful selected collapse plan.
    visible_count:
        Count of rendered nodes implied by ``plan``.
    analyze_ms:
        Time spent in shared v1/R2 analysis.
    select_ms:
        Time spent in v2 selection.
    g_star:
        Winning grain target.
    declined:
        Whether the optimizer declined and callers should use v1.
    reason:
        Human-readable decline reason.
    segments:
        Segment descriptors keyed by segment node name.
    level:
        Max-mode ladder level that produced the plan.
    planner:
        Which planner produced the plan: ``"frontier"`` (the DP optimizer)
        or ``"floor_fallback"`` (the conservative plan used when the DP
        frontier is empty). A ``"floor_fallback"`` plan can be the ENTIRE
        uncollapsed graph -- callers must disclose it, never present it as
        an optimized result (collapse memo D4: no silent floor).
    k_cap_exhausted:
        True when the frontier emptied because the root's own rendered
        units exceed ``K_CAP`` -- every candidate point overshoots the cap
        (the measured cliff mechanism on cached decoders).
    root_own_units:
        Rendered units the root owns directly (outside every top-level
        child module); the variable that drives ``k_cap_exhausted``.
    """

    selected: frozenset[str]
    repeat_folds: Mapping[str, ModuleRepeatFold]
    plan: CollapsePlan
    visible_count: int
    analyze_ms: float
    select_ms: float
    g_star: float | None
    declined: bool = False
    reason: str | None = None
    segments: Mapping[str, SegmentDescriptor] | None = None
    level: str | None = None
    planner: str = "frontier"
    k_cap_exhausted: bool = False
    root_own_units: int = 0
    #: The DP's scored node count for the winning frontier point, or ``None``
    #: when no frontier point directly produced the plan (floor fallback,
    #: schedule steps, post-pass segment condensation). The three-way parity
    #: gate pins ``scored_k == count(plan) == emitted render units`` -- the
    #: phantom-k class is exactly a divergence here.
    scored_k: int | None = None
    #: Band-miss disclosure (memo D7): True when no ladder point reaches the
    #: readable band; the served plan is the DISCLOSED strongest point and
    #: ``strongest_plan_count`` carries its REALIZED count (never a scored
    #: estimate, never the word "floor" -- the beam proves no bound).
    band_missed: bool = False
    strongest_plan_count: int | None = None
    #: Estimator disclosure for field refits (memo item 14): the
    #: ``(U, W, predicted_ms, actual_ms, tier, formula version)`` tuple, or
    #: ``None`` on paths that never consulted the admission gate (declined
    #: pathological entries, schedule-step projections).
    estimator: EstimatorDiagnostics | None = None

    def __post_init__(self) -> None:
        """Refuse construction when segment descriptors were dropped.

        The renderer materializes segment boxes from ``segments``, not from
        the plan nodes, so a segmented ``plan`` published with a smaller
        descriptor mapping silently renders more nodes than ``visible_count``
        claims. Every construction site (including ``dataclasses.replace``)
        must therefore keep descriptor cardinality equal to the number of
        segment nodes in the plan.
        """

        segment_nodes = sum(isinstance(node, (ChildSegment, OpSegment)) for node in self.plan.nodes)
        descriptor_count = len(self.segments or {})
        # r-b7 R24-2: a raise, not an assert — this guards the label-honesty
        # contract on the DEFAULT draw(collapse=) path, and `python -O` strips
        # asserts, which would let segment boxes under-report hidden calls.
        if descriptor_count != segment_nodes:
            raise RuntimeError(
                f"OptimizerResult segment descriptor cardinality {descriptor_count} != "
                f"plan segment nodes {segment_nodes}; publishing this result would "
                "silently render hidden structure"
            )


def _assert_visible_plan(visible_count: int, origin: str) -> None:
    """Raise when a collapse plan about to be published has no visible nodes.

    T9 (grind-p3): a raise, not an assert — these guards run on the DEFAULT
    ``draw(collapse=)`` path and ``python -O`` strips asserts, which would
    let an empty plan render a blank graph with no diagnosis.

    Parameters
    ----------
    visible_count:
        Visible node count of the plan being published.
    origin:
        Human-readable name of the publishing path, used in the error.
    """

    if visible_count <= 0:
        raise RuntimeError(f"{origin} produced no visible nodes")


@dataclass(frozen=True)
class _FrontierPoint:
    """One DP frontier point for a subtree."""

    k: int
    cost: float
    nodes: tuple[PlanNode, ...]
    selected: frozenset[str]
    folds: tuple[ModuleRepeatFold, ...]
    box_costs: tuple[float, ...]


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


@dataclass(frozen=True)
class _OptimizerState:
    """Immutable shared state for one g-star DP run."""

    trace: Trace
    context: RenderContext
    analysis: CollapseAnalysis
    child_addresses: Mapping[str, tuple[str, ...]]
    hidden_counts: Mapping[str, int]
    structural_digests: Mapping[str, str]
    expanded_cache: dict[
        str,
        tuple[ChildCondensedFlowGraph | None, tuple[str, ...], tuple[str, ...]],
    ]
    role_components_cache: dict[tuple[str, tuple[str, ...]], tuple[RoleComponent, ...]]
    child_segments_cache: dict[tuple[str, ...], tuple[tuple[str, ...], ...]]
    single_member_expanded_cache: dict[_MemoKey, tuple[_DecisionPoint, ...]]
    box_cost_cache: dict[str, float]
    branch_salience_cache: dict[str, float]
    output_shape_cache: dict[tuple[str, str], tuple[int, ...] | None]
    weights: OptimizerWeights
    g_star: float
    total_ops: int
    allow_folds: bool
    allow_segments: bool
    max_salience_floor: float | None
    rendered_own_units: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class _MemoKey:
    """Memo key for structurally equivalent module subtrees."""

    digest: str
    landmark_bucket: int
    trunk: bool
    num_calls: int | None = None
    rolled_mass: int | None = None


_RESULT_CACHE: weakref.WeakKeyDictionary[
    object,
    tuple[
        tuple[object, ...],
        dict[tuple[RenderContext, str, OptimizerWeights], OptimizerResult],
    ],
] = weakref.WeakKeyDictionary()
#: Traces whose over-ceiling decline already warned (r8 R60-13: declined
#: results skip the result cache -- the revision snapshot that keyed it is
#: itself the O(N) cost being avoided -- so warn-once rides its own set).
_CEILING_WARNED_TRACES: weakref.WeakSet = weakref.WeakSet()
_SCHEDULE_CACHE: weakref.WeakKeyDictionary[
    object,
    tuple[tuple[object, ...], dict[RenderContext, CollapseSchedule]],
] = weakref.WeakKeyDictionary()
_BOX_UNITS_CACHE: weakref.WeakKeyDictionary[
    object,
    tuple[
        tuple[object, ...],
        dict[RenderContext, Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]]],
    ],
] = weakref.WeakKeyDictionary()


@_revision_scoped
def select_collapse_plan(
    trace: Trace,
    context: RenderContext,
    weights: OptimizerWeights | None = None,
    mode: Literal["auto", "max"] = "auto",
    source_graph: SourceGraph | None = None,
) -> OptimizerResult:
    """Return the v2 auto-collapse plan for ``trace``.

    Parameters
    ----------
    trace:
        Trace being rendered.
    context:
        Rendering context.
    weights:
        Optional optimizer weights.
    mode:
        Collapse policy to select.
    source_graph:
        Optional schedule-local normalized source graph.

    Returns
    -------
    OptimizerResult
        Selected collapse result, or a declined result when unsupported.
    """

    resolved_weights = OptimizerWeights() if weights is None else weights
    # Pathological pre-gate (collapse memo D5(i), r8 R60-13 preserved): the
    # revision snapshot and even the universe build are pointless on a
    # 100k-op artifact, so far-out-of-envelope traces decline outright --
    # the ONE remaining uncollapsed decline. Everything below it degrades to
    # the deterministic fallback planner, never to an uncollapsed wall.
    op_count = len(trace.ops)
    if op_count > COLLAPSE_OPTIMIZER_MAX_OPS * PATHOLOGICAL_OP_MULTIPLIER:
        if source_graph is None:
            from .source_graph import build_source_graph

            source_graph = build_source_graph(trace, context)
        full_plan = collapse_plan_for_source_graph(source_graph, None, None)
        if mode == "max" or trace not in _CEILING_WARNED_TRACES:
            _CEILING_WARNED_TRACES.add(trace)
            warnings.warn(
                TorchLensWarning(
                    f"TorchLens is skipping smart collapse: this trace has "
                    f"{op_count} ops, above the pathological-input pre-gate "
                    f"({PATHOLOGICAL_OP_MULTIPLIER} x the defensive constant "
                    f"COLLAPSE_OPTIMIZER_MAX_OPS={COLLAPSE_OPTIMIZER_MAX_OPS}). "
                    "The graph renders uncollapsed; reduce the rendered "
                    "graph first with module= focus, vis_call_depth, or "
                    "rolled mode.",
                    code="collapse_pathological_skip",
                ),
                # r7 R19 (opus b6 LOW): a fixed stacklevel resolved to TorchLens's
                # own _trace_stats caller; blame the user's draw()/collapse_plan()
                # line instead (the entry depth differs per public spelling).
                stacklevel=user_stacklevel(),
            )
        return OptimizerResult(
            selected=frozenset(),
            repeat_folds={},
            plan=full_plan,
            visible_count=count(full_plan),
            analyze_ms=0.0,
            select_ms=0.0,
            g_star=None,
            declined=True,
            reason=(
                f"collapse_ops_ceiling: {op_count} ops exceed "
                f"{PATHOLOGICAL_OP_MULTIPLIER} x "
                f"COLLAPSE_OPTIMIZER_MAX_OPS={COLLAPSE_OPTIMIZER_MAX_OPS}"
            ),
        )
    # Weights are part of the cache identity: a weighted result must never be
    # served for a differently weighted call (stale-cache defect class).
    cache_key = (context, mode, resolved_weights)
    revision = _collapse_graph_revision(trace)
    cache_entry = _RESULT_CACHE.get(trace)
    if cache_entry is None or cache_entry[0] != revision:
        cached_by_context: dict[tuple[RenderContext, str, OptimizerWeights], OptimizerResult] = {}
        _RESULT_CACHE[trace] = (revision, cached_by_context)
    else:
        cached_by_context = cache_entry[1]
    cached = cached_by_context.get(cache_key)
    if cached is not None:
        if cached.planner == "linear_fallback":
            # N15 (themes memo item 1): an explicit ``max`` compaction request
            # that cannot get the quality planner re-warns on EVERY call,
            # cache hit included; ``auto`` keeps the once-per-trace dedupe.
            _warn_budget_fallback(trace, mode, cached)
        return cached
    # Admission gate (collapse memo D5): the variable is U -- the rendered
    # universe of the FULL plan (focus/depth/rolled reductions applied), so
    # context reductions are real remedies -- tiered by the measured
    # (U, W) work estimator. Over-budget requests degrade to the
    # deterministic fallback planner, never to an uncollapsed wall.
    if source_graph is None:
        from .source_graph import build_source_graph

        source_graph = build_source_graph(trace, context)
    budget = _admission_budget(trace, context, source_graph)
    if (
        budget.universe_count > COLLAPSE_OPTIMIZER_MAX_OPS
        or budget.predicted_ms > QUALITY_PLANNER_BUDGET_MS
    ):
        budget.fired = "admission"
        result = _budget_fallback_result(trace, context, source_graph, mode, budget)
        cached_by_context[cache_key] = result
        return result
    result = _run_admitted_selection(trace, context, source_graph, (mode, weights), budget)
    cached_by_context[cache_key] = result
    return result


def _admission_budget(
    trace: Trace,
    context: RenderContext,
    source_graph: SourceGraph | None,
) -> SelectionBudget:
    """Measure U and W and price the request (memo D5 admission inputs)."""

    admission_plan = _collapse_plan_for_source_or_trace(trace, None, None, context, source_graph)
    universe_count = count(admission_plan)
    width = _max_sibling_group_width(trace, context)
    return SelectionBudget(
        predicted_ms=predicted_select_ms(universe_count, width),
        universe_count=universe_count,
        max_sibling_width=width,
        started_at=time.perf_counter(),
    )


def _run_admitted_selection(
    trace: Trace,
    context: RenderContext,
    source_graph: SourceGraph | None,
    request: tuple[str, OptimizerWeights | None],
    budget: SelectionBudget,
) -> OptimizerResult:
    """Run the admitted quality planner under the armed budget (memo D5).

    ``request`` is the ``(mode, weights)`` pair (weights steer only the max
    substrate). The watchdog deadline arms here (never in CI/deterministic
    mode); a FallbackDegrade from the frontier chokepoints lands on the
    deterministic fallback planner; every result is stamped with
    EstimatorDiagnostics.
    """

    mode, weights = request
    if watchdog_enabled():
        budget.deadline = (
            time.perf_counter()
            + max(WATCHDOG_FLOOR_MS, WATCHDOG_SLACK * budget.predicted_ms) / 1000.0
        )
    budget_token = ACTIVE_BUDGET.set(budget)
    try:
        if mode == "max":
            result = _select_max_plan(trace, context, weights, source_graph)
        else:
            from .collapse_ladder import auto_from_ladder

            result = auto_from_ladder(trace, context, source_graph)
    except FallbackDegrade:
        return _budget_fallback_result(trace, context, source_graph, mode, budget)
    finally:
        ACTIVE_BUDGET.reset(budget_token)
    return replace(
        result,
        estimator=EstimatorDiagnostics(
            universe_count=budget.universe_count,
            max_sibling_width=budget.max_sibling_width,
            predicted_ms=budget.predicted_ms,
            actual_ms=(time.perf_counter() - budget.started_at) * 1000.0,
            tier=result.planner,
            peak_frontier_records=budget.peak_allocated,
            budget_fired=budget.fired,
        ),
    )


# The PUBLIC auto surface reads the typed event ladder (memo D8 unfreeze);
# the historical frontier auto survives below as _frontier_auto_result, the
# max path's internal substrate.


@dataclass(frozen=True)
class _PlanMemos:
    """Plan-local pricing memos shared across one selection's passes."""

    expanded_cache: dict[
        str, tuple[ChildCondensedFlowGraph | None, tuple[str, ...], tuple[str, ...]]
    ]
    output_shape_cache: dict[tuple[str, str], tuple[int, ...] | None]


def _frontier_auto_result(
    trace: Trace,
    context: RenderContext,
    resolved_weights: OptimizerWeights,
    source_graph: SourceGraph | None,
    memos: _PlanMemos | None = None,
) -> OptimizerResult:
    """Run the historical band-targeted auto frontier selection (internal).

    The PUBLIC auto surface reads the typed event ladder (memo D8 unfreeze,
    collapse_ladder.auto_from_ladder). This internal selector survives as
    the max path's substrate: _select_max_plan seeds its interval
    condensation from a band-targeted frontier cut and falls back to it
    when max's [3,20] legal-plan band produces nothing (memo D4(d) --
    documented, disclosed). Results never enter the public result cache.
    ``memos`` lets the max caller share its plan-local pricing caches so one
    selection prices each module once (the once-per-address shape-lookup
    pin); ``None`` builds call-local dicts.
    """

    analysis = analyze_collapse(trace)
    start = time.perf_counter()
    hidden_counts = _rendered_module_hidden_counts(trace, context)
    child_addresses = _child_address_map(trace)
    structural_digests = _structural_digest_map(trace, child_addresses, analysis)
    if memos is None:
        memos = _PlanMemos(expanded_cache={}, output_shape_cache={})
    expanded_cache = memos.expanded_cache
    output_shape_cache = memos.output_shape_cache
    best = _select_best_decision(
        trace=trace,
        context=context,
        analysis=analysis,
        child_addresses=child_addresses,
        hidden_counts=hidden_counts,
        structural_digests=structural_digests,
        expanded_cache=expanded_cache,
        output_shape_cache=output_shape_cache,
        weights=resolved_weights,
        allow_folds=False,
    )
    first_pass_point: _FrontierPoint | None = None
    first_pass_plan: CollapsePlan | None = None
    if best is not None:
        first_pass_point, first_pass_plan = _instantiate_best_point(
            trace=trace,
            context=context,
            analysis=analysis,
            child_addresses=child_addresses,
            hidden_counts=hidden_counts,
            structural_digests=structural_digests,
            expanded_cache=expanded_cache,
            output_shape_cache=output_shape_cache,
            best=best,
            source_graph=source_graph,
        )
    if first_pass_plan is None or count(first_pass_plan) > _readable_band_high(trace):
        best = _select_best_decision(
            trace=trace,
            context=context,
            analysis=analysis,
            child_addresses=child_addresses,
            hidden_counts=hidden_counts,
            structural_digests=structural_digests,
            expanded_cache=expanded_cache,
            output_shape_cache=output_shape_cache,
            weights=replace(resolved_weights, fold_intrinsic=0.05),
            allow_folds=True,
        )
        first_pass_point = None
        first_pass_plan = None
    if best is None:
        # No silent floor (collapse memo D4): the frontier emptied. The plan
        # served is the deterministic significance-greedy fallback
        # (memo D5(iv)), never a bare uncollapsed wall.
        selected, repeat_folds, plan = linear_fallback_plan(
            trace,
            context,
            source_graph,
            _collapse_plan_for_source_or_trace,
            COLLAPSE_OPTIMIZER_MAX_OPS,
        )
        _assert_visible_plan(count(plan), "collapse floor fallback")
        root_own_units = _root_own_unit_count(trace, context, analysis, child_addresses)
        k_cap_exhausted = root_own_units > K_CAP
        cause = (
            f"the root owns {root_own_units} rendered units, above K_CAP={K_CAP}, "
            "so every frontier candidate overshoots the cap"
            if k_cap_exhausted
            else "no optimizer frontier was produced"
        )
        return OptimizerResult(
            selected=selected,
            repeat_folds=repeat_folds,
            plan=plan,
            visible_count=count(plan),
            analyze_ms=analysis.elapsed_ms,
            select_ms=(time.perf_counter() - start) * 1000.0,
            g_star=None,
            reason=(
                f"floor_fallback: {cause}; reduce the rendered graph with "
                "module= focus, vis_call_depth, or rolled mode"
            ),
            planner="floor_fallback",
            k_cap_exhausted=k_cap_exhausted,
            root_own_units=root_own_units,
        )
    if first_pass_point is None or first_pass_plan is None:
        instantiated_point, plan = _instantiate_best_point(
            trace=trace,
            context=context,
            analysis=analysis,
            child_addresses=child_addresses,
            hidden_counts=hidden_counts,
            structural_digests=structural_digests,
            expanded_cache=expanded_cache,
            output_shape_cache=output_shape_cache,
            best=best,
            source_graph=source_graph,
        )
    else:
        instantiated_point = first_pass_point
        plan = first_pass_plan
    _, winning_g, _, _, _, _, _ = best
    repeat_folds = _fold_mapping(instantiated_point.folds)
    band_high = _readable_band_high(trace)
    if (
        count(plan) > band_high
        and not instantiated_point.selected
        and not instantiated_point.folds
        and not _frontier_can_reach_band(
            trace=trace,
            context=context,
            analysis=analysis,
            child_addresses=child_addresses,
            hidden_counts=hidden_counts,
            structural_digests=structural_digests,
            expanded_cache=expanded_cache,
            output_shape_cache=output_shape_cache,
            weights=replace(resolved_weights, fold_intrinsic=0.05),
            band_high=band_high,
            source_graph=source_graph,
        )
    ):
        segmented_plan, segments = _condense_plan_with_child_segments(
            trace,
            context,
            analysis,
            plan,
            hidden_counts,
            _optimizer_total_units(trace, context),
            dominance_limit=0.75,
            k_hi=band_high,
        )
        segmented_count = count(segmented_plan)
        if (
            segments
            and 1 <= segmented_count <= band_high
            and segmented_count < count(plan)
            and _plan_respects_max_dominance(trace, analysis, segmented_plan, segments, 0.75)
        ):
            plan = segmented_plan
        else:
            segments = {}
    else:
        segments = {}
    _assert_visible_plan(count(plan), "v2 collapse plan")
    return OptimizerResult(
        selected=instantiated_point.selected,
        repeat_folds=repeat_folds,
        plan=plan,
        visible_count=count(plan),
        analyze_ms=analysis.elapsed_ms,
        select_ms=(time.perf_counter() - start) * 1000.0,
        g_star=winning_g,
        segments=segments,
        # The scored claim binds only when the winning point's realization IS
        # the returned plan; the segment post-pass discloses via None.
        scored_k=instantiated_point.k if not segments else None,
    )


def select_collapse_level(
    trace: Trace,
    context: RenderContext,
    t: float,
    weights: OptimizerWeights | None = None,
) -> OptimizerResult:
    """Return the v2 collapse plan for a public float collapse level.

    Parameters
    ----------
    trace:
        Trace being rendered.
    context:
        Rendering context.
    t:
        Collapse level in ``[0.0, 1.0]``. ``0.0`` preserves the full graph and
        ``1.0`` is byte-identical to ``collapse="max"``.
    weights:
        Optional optimizer weights.

    Returns
    -------
    OptimizerResult
        Selected collapse result for ``t``.

    Raises
    ------
    ValueError
        If ``t`` is outside ``[0.0, 1.0]``.
    """

    if not 0.0 <= t <= 1.0:
        raise InvalidArgumentError(
            f"collapse float level must be in [0.0, 1.0]; received {t!r}",
            code="collapse_level_invalid",
            remedy="pass a collapse level between 0.0 and 1.0",
            argument="t",
        )
    if t == 1.0:
        return select_collapse_plan(trace, context, weights, mode="max")
    schedule = collapse_schedule(trace, context, weights)
    step = schedule.at(t)
    selected = step.collapsed_addresses
    repeat_folds: dict[str, ModuleRepeatFold] = {}
    segments: dict[str, SegmentDescriptor] = {}
    if t == 0.0:
        selected = frozenset()
    result = OptimizerResult(
        selected=selected,
        repeat_folds=repeat_folds,
        plan=step.plan,
        visible_count=step.visible_count,
        analyze_ms=0.0,
        select_ms=0.0,
        g_star=None,
        segments=segments,
        reason=None,
    )
    return result


# The public float schedule family (collapse_schedule, _schedule_t,
# _collapsed_addresses_for_result, _reported_collapsed_addresses,
# _schedule_ordered_addresses) moved to .collapse_ladder with the F11
# typed event ladder (collapse memo item 9).


def _select_max_plan(
    trace: Trace,
    context: RenderContext,
    weights: OptimizerWeights | None,
    source_graph: SourceGraph | None = None,
) -> OptimizerResult:
    """Return the max-mode v2 plan by condensing legal auto-plan intervals.

    Parameters
    ----------
    trace:
        Trace being optimized.
    context:
        Rendering context.
    weights:
        Optional optimizer weights forwarded to the auto selector.
    source_graph:
        Optional schedule-local normalized source graph.

    Returns
    -------
    OptimizerResult
        Max-mode result with segment descriptors, or an L3 auto fallback.
    """

    # One plan-local memo pair for the WHOLE max selection: the frontier
    # substrate and the level loop below price the same modules, and separate
    # dicts made one cold selection compute every module shape twice (the
    # once-per-address shape-lookup pin in tests/test_auto_collapse_metrics).
    memos = _PlanMemos(expanded_cache={}, output_shape_cache={})
    expanded_cache = memos.expanded_cache
    output_shape_cache = memos.output_shape_cache
    auto = _frontier_auto_result(
        trace,
        context,
        OptimizerWeights() if weights is None else weights,
        source_graph,
        memos=memos,
    )
    analysis = analyze_collapse(trace)
    hidden_counts = _rendered_module_hidden_counts(trace, context)
    child_addresses = _child_address_map(trace)
    structural_digests = _structural_digest_map(trace, child_addresses, analysis)
    total_ops = _optimizer_total_units(trace, context)
    auto_count = count(auto.plan)
    levels = (
        ("L0", 12, 0.60),
        ("L1", 12, 0.75),
        ("L2", min(20, auto_count), 0.75),
    )
    resolved_weights = OptimizerWeights() if weights is None else weights
    start = time.perf_counter()
    deferred_repair: (
        tuple[_FrontierPoint, CollapsePlan, dict[str, SegmentDescriptor], float] | None
    ) = None
    for level, k_hi, dominance_limit in levels:
        best = _select_best_decision(
            trace=trace,
            context=context,
            analysis=analysis,
            child_addresses=child_addresses,
            hidden_counts=hidden_counts,
            structural_digests=structural_digests,
            expanded_cache=expanded_cache,
            output_shape_cache=output_shape_cache,
            weights=replace(resolved_weights, fold_intrinsic=0.05),
            allow_folds=True,
            allow_segments=True,
            k_min=3,
            k_max=k_hi,
            objective="max",
            require_segments=True,
        )
        if best is not None:
            point, plan = _instantiate_best_point(
                trace=trace,
                context=context,
                analysis=analysis,
                child_addresses=child_addresses,
                hidden_counts=hidden_counts,
                structural_digests=structural_digests,
                expanded_cache=expanded_cache,
                output_shape_cache=output_shape_cache,
                best=best,
                prefer_instantiated_nodes=True,
                source_graph=source_graph,
            )
            point, plan = _repair_max_salience_floor(
                trace=trace,
                context=context,
                analysis=analysis,
                child_addresses=child_addresses,
                hidden_counts=hidden_counts,
                structural_digests=structural_digests,
                output_shape_cache=output_shape_cache,
                weights=replace(resolved_weights, fold_intrinsic=0.05),
                g_star=best[1],
                point=point,
                source_graph=source_graph,
            )
            segments = _segments_from_plan_nodes(trace, context, analysis, plan)
            plan_count = count(plan)
            if (
                plan_count > k_hi
                and plan_count <= min(20, auto_count)
                and plan_count < auto_count
                and _plan_respects_max_dominance(trace, analysis, plan, segments, 0.75)
            ):
                deferred_repair = (point, plan, segments, best[1])
            if (
                segments
                and 3 <= plan_count <= k_hi
                and plan_count < auto_count
                and _plan_respects_max_dominance(trace, analysis, plan, segments, dominance_limit)
            ):
                return OptimizerResult(
                    selected=point.selected,
                    repeat_folds=_fold_mapping(point.folds),
                    plan=plan,
                    visible_count=plan_count,
                    analyze_ms=analysis.elapsed_ms,
                    select_ms=(time.perf_counter() - start) * 1000.0,
                    g_star=best[1],
                    segments=segments,
                    level=level,
                    reason=None,
                )
            if level == "L2" and deferred_repair is not None:
                repair_point, repair_plan, repair_segments, repair_g = deferred_repair
                repair_count = count(repair_plan)
                if 3 <= repair_count <= k_hi:
                    return OptimizerResult(
                        selected=repair_point.selected,
                        repeat_folds=_fold_mapping(repair_point.folds),
                        plan=repair_plan,
                        visible_count=repair_count,
                        analyze_ms=analysis.elapsed_ms,
                        select_ms=(time.perf_counter() - start) * 1000.0,
                        g_star=repair_g,
                        segments=repair_segments,
                        level=level,
                        reason=None,
                    )
        plan, segments = _condense_plan_with_child_segments(
            trace,
            context,
            analysis,
            auto.plan,
            hidden_counts,
            total_ops,
            dominance_limit=dominance_limit,
            k_hi=k_hi,
        )
        plan_count = count(plan)
        if (
            segments
            and 3 <= plan_count <= k_hi
            and plan_count < auto_count
            and _plan_respects_max_dominance(trace, analysis, plan, segments, dominance_limit)
        ):
            return replace(
                auto,
                plan=plan,
                visible_count=plan_count,
                segments=segments,
                level=level,
                # Protected reasons (memo D7): a fallback-tier substrate's
                # diagnosed cause + remedy is a D4-mandated disclosure and
                # survives the segment condensation; a healthy substrate's
                # scored claim no longer binds, so its reason clears.
                reason=auto.reason if auto.planner == "floor_fallback" else None,
                # The plan was swapped: auto's scored claim no longer binds.
                scored_k=None,
            )
    # The auto fallback must keep auto's own segment descriptors: auto's
    # band-pressure branch can legitimately return a segmented plan, and
    # stripping ``segments`` here would publish segment plan nodes the
    # renderer cannot materialize (descriptor-loss honesty gap).
    return replace(
        auto,
        level="L3",
        # Protected reasons (memo D7): the auto substrate's own cause and
        # remedy (K_CAP diagnosis, "module= focus" guidance) survive the
        # max-band fallback note instead of being clobbered by it.
        reason=(
            f"fallback_to_auto: no legal max plan in [3,{min(20, auto_count)}]"
            + (f"; {auto.reason}" if auto.reason else "")
        ),
    )


def _repair_max_salience_floor(
    trace: Trace,
    context: RenderContext,
    analysis: CollapseAnalysis,
    child_addresses: Mapping[str, tuple[str, ...]],
    hidden_counts: Mapping[str, int],
    structural_digests: Mapping[str, str],
    output_shape_cache: dict[tuple[str, str], tuple[int, ...] | None],
    weights: OptimizerWeights,
    g_star: float,
    point: _FrontierPoint,
    source_graph: SourceGraph | None = None,
) -> tuple[_FrontierPoint, CollapsePlan]:
    """Expand selected max boxes that hide unique wide parallel fans.

    Parameters
    ----------
    trace:
        Trace being optimized.
    context:
        Rendering context.
    analysis:
        Shared collapse analysis.
    child_addresses:
        Renderer-tree child addresses.
    hidden_counts:
        Renderer-faithful hidden counts.
    structural_digests:
        Structural memo digests.
    output_shape_cache:
        Plan-local cache shared by optimizer states.
    weights:
        Optimizer weights for the repair pass.
    g_star:
        Winning target hidden-mass grain.
    point:
        Instantiated max-plan frontier point.
    source_graph:
        Optional schedule-local normalized source graph.

    Returns
    -------
    tuple[_FrontierPoint, CollapsePlan]
        Repaired point and its renderer-faithful plan.
    """

    state = _OptimizerState(
        trace=trace,
        context=context,
        analysis=analysis,
        child_addresses=child_addresses,
        hidden_counts=hidden_counts,
        structural_digests=structural_digests,
        expanded_cache={},
        role_components_cache={},
        child_segments_cache={},
        single_member_expanded_cache={},
        box_cost_cache={},
        branch_salience_cache={},
        output_shape_cache=output_shape_cache,
        weights=weights,
        g_star=g_star,
        total_ops=_optimizer_total_units(trace, context),
        allow_folds=True,
        allow_segments=False,
        max_salience_floor=MAX_SALIENCE_FLOOR,
        rendered_own_units=_rendered_own_unit_map(trace, context),
    )
    memo: dict[_MemoKey, tuple[_DecisionPoint, ...]] = {}
    nodes: list[PlanNode] = []
    selected = set(point.selected)
    folds = list(point.folds)
    box_costs = list(point.box_costs)
    changed = False
    for node in point.nodes:
        replacement = _max_salience_floor_replacement(node, state, memo)
        if replacement is None:
            nodes.append(node)
            continue
        address = _plan_module_address(node)
        if address is None:
            nodes.append(node)
            continue
        changed = True
        nodes.extend(replacement.nodes)
        selected = {
            candidate
            for candidate in selected
            if candidate != address and not candidate.startswith(f"{address}.")
        }
        selected.update(replacement.selected)
        folds.extend(replacement.folds)
        box_costs.extend(replacement.box_costs)
    if not changed:
        return point, CollapsePlan(nodes=point.nodes, context=context)
    rendered_plan = _collapse_plan_for_source_or_trace(
        trace,
        _collapse_fn_from_selected(frozenset(selected)),
        _fold_mapping(folds),
        context,
        source_graph,
    )
    rendered_plan, _ = _condense_plan_with_child_segments(
        trace,
        context,
        analysis,
        rendered_plan,
        hidden_counts,
        _optimizer_total_units(trace, context),
        dominance_limit=0.75,
        k_hi=20,
    )
    repaired = _FrontierPoint(
        k=count(rendered_plan),
        cost=point.cost,
        nodes=rendered_plan.nodes,
        selected=frozenset(selected),
        folds=tuple(folds),
        box_costs=tuple(box_costs),
    )
    return repaired, rendered_plan


def _max_salience_floor_replacement(
    node: PlanNode,
    state: _OptimizerState,
    memo: dict[_MemoKey, tuple[_DecisionPoint, ...]],
) -> _FrontierPoint | None:
    """Return an expanded replacement for a salient max-mode module box."""

    if not isinstance(node, ModuleBox):
        return None
    address = node.call.rsplit(":", 1)[0]
    signal = state.analysis.signals.get(address)
    if signal is None or _max_box_salience_score(address, signal, state) < MAX_SALIENCE_FLOOR:
        return None
    candidates: list[_FrontierPoint] = []
    for decision in _frontier_for_module(address, state, memo):
        # M2b: a box is recognized by its decision KIND. The historical
        # ``k == 1`` conjunct was only true under the phantom-k pricing.
        if isinstance(decision.decision, _ModuleDecision) and decision.decision.kind == "box":
            continue
        candidates.append(_instantiate_module(address, decision.k, state, memo))
    if not candidates:
        return None
    folded = [candidate for candidate in candidates if candidate.folds]
    pool = folded or candidates
    return min(pool, key=lambda candidate: (candidate.k, candidate.cost, repr(candidate.nodes)))


def _condense_plan_with_child_segments(
    trace: Trace,
    context: RenderContext,
    analysis: CollapseAnalysis,
    plan: CollapsePlan,
    hidden_counts: Mapping[str, int],
    total_ops: int,
    *,
    dominance_limit: float,
    k_hi: int,
) -> tuple[CollapsePlan, dict[str, SegmentDescriptor]]:
    """Replace legal consecutive child boxes with max-mode segment nodes.

    Parameters
    ----------
    trace:
        Trace being optimized.
    context:
        Rendering context.
    analysis:
        Shared collapse analysis.
    plan:
        Auto-mode plan to condense.
    hidden_counts:
        Renderer-faithful hidden counts by module.
    total_ops:
        Total rendered operation units.
    dominance_limit:
        Maximum fraction of ops one segment may hide.
    k_hi:
        Current ladder node ceiling.

    Returns
    -------
    tuple[CollapsePlan, dict[str, SegmentDescriptor]]
        Condensed plan and renderer descriptors.
    """

    _ = k_hi
    nodes: list[PlanNode] = []
    segments: dict[str, SegmentDescriptor] = {}
    hidden_raw_ops: set[str] = set()
    concrete_raw_labels, concrete_segment_ops = _concrete_plan_op_labels(trace, plan.nodes)
    box_owned_labels = _plan_box_owned_surfaced_labels(trace, context, plan.nodes)
    index = 0
    while index < len(plan.nodes):
        node = plan.nodes[index]
        if isinstance(node, RawOp) and isinstance(node.op, str) and node.op in hidden_raw_ops:
            index += 1
            continue
        if isinstance(node, (OpSegment, ChildSegment)):
            # The input plan may already be segmented (auto's band-pressure
            # branch re-condensed by the max ladder). Pass such nodes through
            # verbatim and rebuild their descriptors so the published mapping
            # stays in lockstep with the plan; dropping them here is exactly
            # the descriptor-loss class the parity tripwire guards against.
            if isinstance(node, OpSegment):
                concrete = concrete_segment_ops.get(index, tuple(node.ops))
                descriptor = _make_op_segment_descriptor(trace, context, node.ops, concrete)
            else:
                covered_ops = _child_segment_covered_ops(analysis, node.members)
                descriptor = _make_child_segment_descriptor(
                    trace, context, node.members, covered_ops
                )
            segments[descriptor.name] = descriptor
            nodes.append(node)
            index += 1
            continue
        op_run = _legal_plan_op_segment_run(
            trace,
            context,
            plan.nodes,
            index,
            total_ops,
            dominance_limit=dominance_limit,
            concrete_labels=concrete_raw_labels,
            box_owned_labels=box_owned_labels,
        )
        if op_run:
            concrete_run = tuple(
                concrete_raw_labels.get(index + offset, label)
                for offset, label in enumerate(op_run)
            )
            descriptor = _make_op_segment_descriptor(trace, context, op_run, concrete_run)
            segments[descriptor.name] = descriptor
            nodes.append(OpSegment(op_run))
            index += len(op_run)
            continue
        run = _legal_plan_child_segment_run(
            trace,
            context,
            analysis,
            plan.nodes,
            index,
            hidden_counts,
            total_ops,
            dominance_limit=dominance_limit,
        )
        if run:
            covered_ops = _child_segment_covered_ops(analysis, run)
            descriptor = _make_child_segment_descriptor(trace, context, run, covered_ops)
            segments[descriptor.name] = descriptor
            hidden_raw_ops.update(str(label).rsplit(":", 1)[0] for label in covered_ops)
            nodes.append(ChildSegment(run))
            index += len(run)
            continue
        nodes.append(node)
        index += 1
    _assert_segment_descriptor_parity(nodes, segments)
    return CollapsePlan(nodes=tuple(nodes), context=context), segments


def _segments_from_plan_nodes(
    trace: Trace,
    context: RenderContext,
    analysis: CollapseAnalysis,
    plan: CollapsePlan,
) -> dict[str, SegmentDescriptor]:
    """Build renderer segment descriptors from first-class plan nodes.

    Parameters
    ----------
    trace:
        Trace being optimized.
    context:
        Rendering context.
    analysis:
        Shared collapse analysis.
    plan:
        Plan containing first-class segment nodes.

    Returns
    -------
    dict[str, SegmentDescriptor]
        Descriptor mapping keyed by Graphviz segment node name.
    """

    segments: dict[str, SegmentDescriptor] = {}
    _, concrete_segment_ops = _concrete_plan_op_labels(trace, plan.nodes)
    for index, node in enumerate(plan.nodes):
        if isinstance(node, ChildSegment):
            covered_ops = _child_segment_covered_ops(analysis, node.members)
            descriptor = _make_child_segment_descriptor(trace, context, node.members, covered_ops)
            segments[descriptor.name] = descriptor
        elif isinstance(node, OpSegment):
            concrete = concrete_segment_ops.get(index, tuple(node.ops))
            descriptor = _make_op_segment_descriptor(trace, context, node.ops, concrete)
            segments[descriptor.name] = descriptor
    _assert_segment_descriptor_parity(plan.nodes, segments)
    return segments


def _plan_respects_max_dominance(
    trace: Trace,
    analysis: CollapseAnalysis,
    plan: CollapsePlan,
    segments: Mapping[str, SegmentDescriptor],
    dominance_limit: float,
) -> bool:
    """Return whether visible max boxes and segments obey the dominance cap."""

    total_ops = _optimizer_total_units(trace, plan.context)
    if total_ops <= 0:
        return True
    # Descriptors are built in plan order, so pair each segment plan node with
    # its descriptor positionally; identity keys such as member tuples are not
    # unique under per-pass segments of reused modules.
    ordered_segments = list(segments.values())
    position = 0
    for node in plan.nodes:
        if isinstance(node, ModuleBox):
            address = node.call.rsplit(":", 1)[0]
            signal = analysis.signals.get(address)
            if signal is not None and signal.hidden_ops / total_ops > dominance_limit:
                return False
        elif isinstance(node, (ChildSegment, OpSegment)):
            segment = ordered_segments[position] if position < len(ordered_segments) else None
            position += 1
            if segment is not None and segment.num_ops / total_ops > dominance_limit:
                return False
    return True


def _concrete_plan_op_labels(
    trace: Trace,
    nodes: Sequence[PlanNode],
) -> tuple[dict[int, str], dict[int, tuple[str, ...]]]:
    """Attribute pass-qualified op identity to plan nodes by occurrence order.

    Plan nodes carry pass-free render labels, so an op executed on multiple
    module passes appears several times under one base label. Rendered plans
    enumerate ops in execution order, which makes the ``k``-th plan occurrence
    of a base label the ``k``-th pass of that op layer.

    Parameters
    ----------
    trace:
        Trace being optimized.
    nodes:
        Plan-node sequence.

    Returns
    -------
    tuple[dict[int, str], dict[int, tuple[str, ...]]]
        Concrete labels for string ``RawOp`` nodes keyed by plan index, and
        concrete member labels for ``OpSegment`` nodes keyed by plan index.
        Entries that cannot be attributed to a concrete op are omitted from
        the raw mapping and left pass-free in the segment mapping.
    """

    valid = {str(op.label) for op in trace.ops}
    counts: dict[str, int] = {}
    raw: dict[int, str] = {}
    segment_ops: dict[int, tuple[str, ...]] = {}

    def resolve(base: str) -> str | None:
        """Return the concrete label for the next occurrence of ``base``."""

        if base in valid:
            return base
        counts[base] = counts.get(base, 0) + 1
        qualified = f"{base}:{counts[base]}"
        return qualified if qualified in valid else None

    for index, node in enumerate(nodes):
        if isinstance(node, RawOp) and isinstance(node.op, str):
            resolved = resolve(node.op)
            if resolved is not None:
                raw[index] = resolved
        elif isinstance(node, OpSegment):
            segment_ops[index] = tuple(resolve(op) or op for op in node.ops)
    return raw, segment_ops


def _assert_segment_descriptor_parity(
    nodes: Sequence[PlanNode],
    segments: Mapping[str, SegmentDescriptor],
) -> None:
    """Fail loudly when segment descriptors collide before renderer exposure.

    r-b7 R24-2: a raise, not an assert — label honesty must survive
    ``python -O``.
    """

    segment_nodes = sum(isinstance(node, (ChildSegment, OpSegment)) for node in nodes)
    if len(segments) != segment_nodes:
        raise RuntimeError(
            f"segment descriptor cardinality {len(segments)} != plan segment "
            f"nodes {segment_nodes}; a non-injective segment identity would "
            "silently drop rendered structure"
        )


def _legal_plan_op_segment_run(
    trace: Trace,
    context: RenderContext,
    nodes: Sequence[PlanNode],
    start: int,
    total_ops: int,
    *,
    dominance_limit: float,
    concrete_labels: Mapping[int, str],
    box_owned_labels: frozenset[str] = frozenset(),
) -> tuple[str, ...] | None:
    """Return a legal consecutive raw-op segment run at ``start``.

    Parameters
    ----------
    trace:
        Trace being optimized.
    context:
        Rendering context; its ``vis_mode`` fixes the cluster keyspace used
        for the module-call containment boundary.
    nodes:
        Plan-node sequence.
    start:
        Candidate start index.
    total_ops:
        Total rendered operation units.
    dominance_limit:
        Maximum segment dominance.
    concrete_labels:
        Pass-qualified op labels for string ``RawOp`` nodes by plan index.
    box_owned_labels:
        Pass-free labels of surfaced own-output ops that plan module boxes
        already account for in their remainder labels; a run never absorbs
        them.

    Returns
    -------
    tuple[str, ...] | None
        Operation labels for a legal segment, or ``None``.
    """

    labels: list[str] = []
    op_order = {str(op.label): index for index, op in enumerate(trace.ops)}
    previous_index: int | None = None
    previous_stack: tuple[str, ...] | None = None
    seen_stacks: dict[tuple[str, ...], tuple[str, ...]] = {}
    for offset, node in enumerate(nodes[start:]):
        if not isinstance(node, RawOp) or not isinstance(node.op, str):
            break
        # Label honesty (round-25): a collapsed atomic module's surfaced
        # own-output op is the box's separate sibling node, and the box
        # remainder label accounts for it. Absorbing it into a segment
        # double-represents the op -- counted by the box content label AND
        # claimed by the segment range label -- and renders structurally
        # identical sibling blocks inconsistently. Keep it a standalone raw
        # node exactly like its siblings' exit ops.
        if node.op in box_owned_labels:
            break
        concrete = concrete_labels.get(start + offset)
        if concrete is None:
            break
        current_index = op_order.get(concrete)
        if current_index is None:
            break
        op = trace.ops[concrete]
        if getattr(op, "is_input", False) or getattr(op, "is_output", False):
            break
        # Module-call containment (round-24 C1): an unrolled run must never
        # absorb two different CALLS of one reused module -- the segment
        # renders as one node in one cluster, so crossing a reuse boundary
        # nests later calls' work under the first call's cluster and leaves
        # the other call clusters falsely empty. Runs across distinct
        # single-call sibling modules stay legal: their exact call-qualified
        # LCA owner renders the segment at the honest common ancestor.
        # Rolled clusters merge passes per address (no call boundary to
        # cross), so rolled runs keep their historical shape.
        if context.vis_mode != "rolled":
            stack = _effective_render_module_stack(op)
            if previous_stack is not None and _crosses_module_call_boundary(previous_stack, stack):
                break
            pass_free = tuple(entry.rsplit(":", 1)[0] for entry in stack)
            if seen_stacks.setdefault(pass_free, stack) != stack:
                break
            previous_stack = stack
        if previous_index is not None and current_index != previous_index + 1:
            break
        labels.append(node.op)
        previous_index = current_index
    if len(labels) < 3:
        return None
    if total_ops > 0:
        max_len = max(3, int(total_ops * dominance_limit))
        labels = labels[:max_len]
    return tuple(labels) if len(labels) >= 3 else None


def _plan_box_owned_surfaced_labels(
    trace: Trace,
    context: RenderContext,
    nodes: Sequence[PlanNode],
) -> frozenset[str]:
    """Return surfaced own-output labels owned by module boxes in a plan.

    A collapsed atomic module keeps its own output op visible while its
    innermost box is dropped, so the op renders as the box's separate sibling
    node and the box remainder label subtracts it. Operation-segment runs must
    never absorb such an op: the segment range label would claim it while the
    box content label still counts it, double-representing one op and
    rendering structurally identical sibling blocks inconsistently
    (round-25).

    Parameters
    ----------
    trace:
        Trace being optimized.
    context:
        Render context for the plan.
    nodes:
        Plan-node sequence being condensed.

    Returns
    -------
    frozenset[str]
        Pass-free render labels of surfaced own-output ops whose owning
        module renders as a collapsed box (or repeat-fold representative) in
        ``nodes``.
    """

    addresses = {node.call.rsplit(":", 1)[0] for node in nodes if isinstance(node, ModuleBox)}
    addresses.update(
        node.rep.call.rsplit(":", 1)[0] for node in nodes if isinstance(node, RepeatFold)
    )
    if not addresses:
        return frozenset()
    units = _module_render_box_units(trace, context)
    labels: set[str] = set()
    for address in addresses:
        unit = units.get(address)
        if unit is not None:
            labels.update(unit[1])
    return frozenset(labels)


def _legal_plan_child_segment_run(
    trace: Trace,
    context: RenderContext,
    analysis: CollapseAnalysis,
    nodes: Sequence[PlanNode],
    start: int,
    hidden_counts: Mapping[str, int],
    total_ops: int,
    *,
    dominance_limit: float,
) -> tuple[str, ...] | None:
    """Return the longest legal child-segment run starting at ``start``.

    Parameters
    ----------
    trace:
        Trace being optimized.
    context:
        Rendering context.
    analysis:
        Shared collapse analysis.
    nodes:
        Plan-node sequence.
    start:
        Candidate start index.
    hidden_counts:
        Renderer-faithful hidden counts.
    total_ops:
        Total rendered operation units.
    dominance_limit:
        Maximum segment dominance.

    Returns
    -------
    tuple[str, ...] | None
        Legal run, or ``None``.
    """

    addresses: list[str] = []
    parent: str | None = None
    for node in nodes[start:]:
        address = _plan_module_address(node)
        if address is None:
            break
        module = trace.modules[address] if address in trace.modules else None
        if int(getattr(module, "num_calls", 1) or 1) > 1:
            # Rendered segment absorption is address-global: every call of a
            # member address is hidden by the one segment node, while this
            # condensation only replaces the consecutive plan nodes of a
            # single pass. Absorbing a multi-call member would therefore
            # leave sibling-pass boxes in the plan that the renderer hides,
            # making ``plan.total`` over-count the rendered graph.
            break
        node_parent = address.rsplit(".", 1)[0] if "." in address else "self"
        if parent is None:
            parent = node_parent
        if node_parent != parent:
            break
        addresses.append(address)
    if len(addresses) < 2 or parent is None:
        return None
    graph = analysis.child_flow_graphs.get(parent)
    return _longest_legal_segment_prefix(
        addresses,
        graph,
        analysis,
        hidden_counts,
        total_ops,
        dominance_limit=dominance_limit,
        vis_mode=context.vis_mode,
    )


def _plan_module_address(node: PlanNode) -> str | None:
    """Return the pass-free module address represented by ``node`` if any."""

    if isinstance(node, ModuleBox):
        return node.call.rsplit(":", 1)[0]
    if isinstance(node, RepeatFold):
        return node.rep.call.rsplit(":", 1)[0]
    return None


def _segment_is_legal(
    addresses: tuple[str, ...],
    graph: ChildCondensedFlowGraph | None,
) -> bool:
    """Return whether ``addresses`` may render as one segment box.

    A segment box asserts adjacency (collapse reference, "Segment boxes"): its
    members are consecutive siblings in flow order, each joined directly to
    the next. The run-fold chain-interval contract alone also admits one-hop
    connectors between members -- a skipped sibling, a parent-owned op, or an
    external round trip -- and a segment absorbs none of them, so the rendered
    box would draw edges out to the connector and back in: a cycle the network
    does not have, around siblings that never ran back to back.
    """

    if len(addresses) < 2 or graph is None:
        return False
    adjacency = _flow_adjacency(graph)
    if _segment_adjacent_prefix_length(addresses, adjacency) < len(addresses):
        return False
    return _run_fold_is_chain_interval(addresses, adjacency)


def _segment_adjacent_prefix_length(
    members: Sequence[str],
    adjacency: FlowAdjacency,
) -> int:
    """Return the longest prefix of ``members`` forming a direct sibling chain.

    The prefix is flow-consecutive and every member feeds the next through a
    direct condensed edge. Both properties hold for every shorter prefix, so
    the result is an exact ceiling on segment-legal prefix lengths.
    """

    if not members:
        return 0
    first = adjacency.flow_index.get(members[0])
    if first is None:
        return 0
    length = 1
    for offset in range(1, len(members)):
        left, right = members[offset - 1], members[offset]
        if adjacency.flow_index.get(right) != first + offset:
            break
        if (left, right) not in adjacency.edge_set:
            break
        length += 1
    return length


def _longest_legal_segment_prefix(
    members: Sequence[str],
    graph: ChildCondensedFlowGraph | None,
    analysis: CollapseAnalysis,
    hidden_counts: Mapping[str, int],
    total_ops: int,
    *,
    dominance_limit: float,
    vis_mode: str = "unrolled",
) -> tuple[str, ...] | None:
    """Return the longest prefix of ``members`` that is a legal segment run.

    Selects exactly the prefix a forward ``members[:end]`` scan would keep
    last, but reduces each gate to its cheapest exact form so the search costs
    one linear pass plus the chain-interval probes it cannot avoid:

    * a member with two or more landmark edges poisons every prefix that
      reaches it, so one forward scan yields a hard prefix ceiling instead of
      a rescan of the whole prefix per candidate;
    * hidden-unit currency accumulates over prefixes, so one incremental pass
      settles the dominance gate for every prefix length at once;
    * only the longest passing prefix is ever returned, so the chain-interval
      probe walks down from the longest surviving candidate and stops at the
      first pass rather than testing every prefix;
    * ends whose probe provably fails from prefix-independent graph facts
      (``_segment_prefix_candidate_ends``) are pruned before probing, so an
      all-illegal component costs one linear pass instead of one probe per end.

    Parameters
    ----------
    members:
        Candidate member addresses in flow order.
    graph:
        Child-condensed flow graph for the parent.
    analysis:
        Shared collapse analysis.
    hidden_counts:
        Renderer-faithful hidden counts.
    total_ops:
        Total rendered operation units.
    dominance_limit:
        Maximum segment dominance.
    vis_mode:
        Render currency for hidden-unit counting.

    Returns
    -------
    tuple[str, ...] | None
        Longest legal run, or ``None``.
    """

    if len(members) < 2:
        return None
    if graph is None or not graph.edges:
        # Chain-interval legality demands exactly one external entry edge, so
        # an absent or edge-less parent graph fails every candidate prefix.
        # Returning early keeps the walk-down from probing each end of a long
        # component against a graph that can never pass.
        return None
    signals = analysis.signals
    # Adjacency ceiling: a prefix that stops being a direct sibling chain stays
    # broken for every longer prefix, so ``_segment_is_legal`` fails all of
    # them; capping here keeps a non-adjacent component (DenseNet's
    # denseblocks around their transitions) from probing every end.
    limit = _segment_adjacent_prefix_length(members, _flow_adjacency(graph))
    # Landmark ceiling: ``any(...)`` over a prefix short-circuits at the first
    # landmark member, and every longer prefix still contains it, so the first
    # landmark index is the exclusive bound on candidate lengths.
    for index, address in enumerate(members[:limit]):
        if signals[address].landmark_edges >= 2:
            limit = index
            break
    if limit < 2:
        return None
    candidate_ends = _segment_prefix_candidate_ends(members, graph, limit)
    if not candidate_ends:
        return None
    limit = candidate_ends[0]

    # Dominance gate, resolved for every prefix length in one pass. Hidden
    # units are counted in the active render currency -- concrete
    # pass-qualified ops for unrolled graphs, deduplicated layer labels for
    # rolled graphs -- matching the optimizer's total-unit normalization. Both
    # that covered-op currency and the no-coverage fallback accumulate member
    # by member, so recounting a whole prefix per candidate is redundant work
    # over already-seen members.
    dominant: set[int] = set()
    if total_ops > 0:
        rolled = vis_mode == "rolled"
        covered: dict[str, None] = {}
        rolled_bases: dict[str, None] = {}
        fallback = 0
        for end in range(1, limit + 1):
            address = members[end - 1]
            signal = signals[address]
            for label in signal.subtree_ops:
                text = str(label)
                covered[text] = None
                if rolled:
                    rolled_bases[text.rsplit(":", 1)[0]] = None
            fallback += hidden_counts.get(address, signal.hidden_ops)
            if end < 2:
                continue
            if covered:
                hidden = len(rolled_bases) if rolled else len(covered)
            else:
                hidden = fallback
            if hidden / total_ops > dominance_limit:
                dominant.add(end)

    for end in candidate_ends:
        if end in dominant:
            continue
        candidate = tuple(members[:end])
        if _segment_is_legal(candidate, graph):
            return candidate
    return None


def _segment_prefix_candidate_ends(
    members: Sequence[str],
    graph: ChildCondensedFlowGraph,
    limit: int,
) -> list[int]:
    """Return candidate prefix lengths not excluded by prefix-independent facts.

    Each rule prunes only candidate ends whose from-scratch chain-interval
    probe provably returns False, so the walk-down still reaches the exact
    prefix the naive forward scan would keep. The connector-facing rules lean
    on one containment fact: for every prefix, ``_chain_connector_nodes`` only
    ever admits out-neighbors of members and their external source/sink
    counterparts, so a node outside that superset can never be hidden as a
    connector.

    * A member-to-member edge that is not one forward step is an internal edge
      no expected set contains, poisoning every prefix that includes both
      endpoints.
    * An entry edge into an interior member from a never-connector external
      source makes the single-entry check fail for every prefix that includes
      the target; two such edges into the first member fail every prefix.
    * An exit edge to a never-connector external node is legal only while its
      source member is the prefix's last member.
    * A prefix with no potential entry edge at all (or none from later members)
      fails the exactly-one-entry check; likewise for exits. Potential coverage
      deliberately overcounts -- connector concealment is ignored -- so only
      provably empty ends are pruned.

    Parameters
    ----------
    members:
        Candidate member addresses in flow order.
    graph:
        Child-condensed flow graph for the parent.
    limit:
        Exclusive bound on candidate prefix lengths from the landmark scan.

    Returns
    -------
    list[int]
        Surviving candidate ends in descending order.
    """

    positions = {address: index for index, address in enumerate(members[:limit])}
    edges = set(graph.edges)
    possible_connectors: set[str] = set()
    for source, target in edges:
        if source in positions and target not in positions:
            possible_connectors.add(target)
            paired = _paired_external_connector(target)
            if paired is not None:
                possible_connectors.add(paired)
    first = members[0]
    first_entry_edges = 0
    entry_cover = [0] * (limit + 2)
    exit_cover = [0] * (limit + 2)

    def cover(diff: list[int], low: int, high: int) -> None:
        """Add one clamped ``[low, high]`` interval to a difference array."""

        low = max(low, 2)
        high = min(high, limit)
        if low <= high:
            diff[low] += 1
            diff[high + 1] -= 1

    for source, target in edges:
        source_pos = positions.get(source)
        target_pos = positions.get(target)
        if source_pos is not None and target_pos is not None:
            if source != target and target_pos != source_pos + 1:
                limit = min(limit, max(source_pos, target_pos))
            if source_pos > target_pos:
                cover(entry_cover, target_pos + 1, source_pos)
            elif target_pos > source_pos:
                cover(exit_cover, source_pos + 1, target_pos)
            continue
        if target_pos is not None:
            cover(entry_cover, target_pos + 1, limit)
            if source not in possible_connectors:
                if target == first:
                    first_entry_edges += 1
                    if first_entry_edges >= 2:
                        return []
                else:
                    limit = min(limit, target_pos)
        elif source_pos is not None:
            cover(exit_cover, source_pos + 1, limit)
            if target not in possible_connectors:
                limit = min(limit, source_pos + 1)
    candidates: list[int] = []
    entries_available = 0
    exits_available = 0
    for end in range(2, limit + 1):
        entries_available += entry_cover[end]
        exits_available += exit_cover[end]
        if entries_available > 0 and exits_available > 0:
            candidates.append(end)
    candidates.reverse()
    return candidates


def _instantiate_best_point(
    trace: Trace,
    context: RenderContext,
    analysis: CollapseAnalysis,
    child_addresses: Mapping[str, tuple[str, ...]],
    hidden_counts: Mapping[str, int],
    structural_digests: Mapping[str, str],
    expanded_cache: dict[
        str,
        tuple[ChildCondensedFlowGraph | None, tuple[str, ...], tuple[str, ...]],
    ],
    output_shape_cache: dict[tuple[str, str], tuple[int, ...] | None],
    best: tuple[
        float,
        float,
        int,
        _DecisionPoint,
        dict[_MemoKey, tuple[_DecisionPoint, ...]],
        OptimizerWeights,
        bool,
    ],
    prefer_instantiated_nodes: bool = False,
    source_graph: SourceGraph | None = None,
) -> tuple[_FrontierPoint, CollapsePlan]:
    """Instantiate a winning DP point and its renderer-faithful plan.

    Parameters
    ----------
    trace:
        Trace being optimized.
    context:
        Rendering context.
    analysis:
        Shared collapse analysis.
    child_addresses:
        Renderer-tree child addresses.
    hidden_counts:
        Renderer-faithful hidden counts.
    structural_digests:
        Structural memo digests.
    expanded_cache:
        Shared expanded-structure cache.
    output_shape_cache:
        Plan-local cache shared by optimizer states.
    best:
        Winner tuple from :func:`_select_best_decision`.
    prefer_instantiated_nodes:
        Whether to use decision nodes directly instead of renderer planning.
    source_graph:
        Optional schedule-local normalized source graph.

    Returns
    -------
    tuple[_FrontierPoint, CollapsePlan]
        Concrete DP point and renderer-faithful plan for its selected modules
        and run folds.
    """

    _, winning_g, _, decision_point, winning_memo, winning_weights, allow_folds = best
    winning_state = _OptimizerState(
        trace=trace,
        context=context,
        analysis=analysis,
        child_addresses=child_addresses,
        hidden_counts=hidden_counts,
        structural_digests=structural_digests,
        expanded_cache=expanded_cache,
        role_components_cache={},
        child_segments_cache={},
        single_member_expanded_cache={},
        box_cost_cache={},
        branch_salience_cache={},
        output_shape_cache=output_shape_cache,
        weights=winning_weights,
        g_star=winning_g,
        total_ops=_optimizer_total_units(trace, context),
        allow_folds=allow_folds,
        allow_segments=prefer_instantiated_nodes,
        max_salience_floor=None,
        rendered_own_units=_rendered_own_unit_map(trace, context),
    )
    instantiated_point = _instantiate_module("self", decision_point.k, winning_state, winning_memo)
    repeat_folds = _fold_mapping(instantiated_point.folds)
    collapse_fn = _collapse_fn_from_selected(instantiated_point.selected)
    if prefer_instantiated_nodes:
        plan = CollapsePlan(nodes=instantiated_point.nodes, context=context)
    else:
        plan = _collapse_plan_for_source_or_trace(
            trace,
            collapse_fn,
            repeat_folds,
            context,
            source_graph,
        )
    return instantiated_point, plan


def _collapse_plan_for_source_or_trace(
    trace: Trace,
    collapse_fn: Callable[[Module], bool] | None,
    repeat_folds: Mapping[str, ModuleRepeatFold] | None,
    context: RenderContext,
    source_graph: SourceGraph | None,
) -> CollapsePlan:
    """Build a plan from a shared source graph when one is available.

    Parameters
    ----------
    trace:
        Trace being projected.
    collapse_fn:
        Active collapse predicate.
    repeat_folds:
        Active repeat-fold mapping.
    context:
        Resolved rendering context.
    source_graph:
        Optional schedule-local normalized source graph.

    Returns
    -------
    CollapsePlan
        Renderer-faithful structural plan.
    """

    if source_graph is None:
        return collapse_plan_for_trace(trace, collapse_fn, repeat_folds, context)
    return collapse_plan_for_source_graph(source_graph, collapse_fn, repeat_folds)


def _collapse_fn_from_selected(selected: frozenset[str]) -> Callable[[Module], bool]:
    """Return a module-collapse predicate for selected addresses."""

    def collapse_fn(module: Module) -> bool:
        """Return whether ``module`` is selected."""

        return module.address in selected

    return collapse_fn


def _select_best_decision(
    trace: Trace,
    context: RenderContext,
    analysis: CollapseAnalysis,
    child_addresses: Mapping[str, tuple[str, ...]],
    hidden_counts: Mapping[str, int],
    structural_digests: Mapping[str, str],
    expanded_cache: dict[
        str,
        tuple[ChildCondensedFlowGraph | None, tuple[str, ...], tuple[str, ...]],
    ],
    output_shape_cache: dict[tuple[str, str], tuple[int, ...] | None],
    weights: OptimizerWeights,
    allow_folds: bool,
    allow_segments: bool = False,
    k_min: int | None = None,
    k_max: int | None = None,
    objective: Literal["auto", "max"] = "auto",
    require_segments: bool = False,
) -> (
    tuple[
        float,
        float,
        int,
        _DecisionPoint,
        dict[_MemoKey, tuple[_DecisionPoint, ...]],
        OptimizerWeights,
        bool,
    ]
    | None
):
    """Return the best surviving global decision for one fold-gating pass.

    "Best" is relative to the beam-capped frontiers described in the module
    docstring; globally better cuts can be pruned before scoring.

    Parameters
    ----------
    trace:
        Trace being optimized.
    context:
        Rendering context.
    analysis:
        Shared collapse analysis.
    child_addresses:
        Renderer-tree child addresses.
    hidden_counts:
        Renderer-faithful hidden counts.
    structural_digests:
        Structural memo digests.
    expanded_cache:
        Shared expanded-structure cache.
    output_shape_cache:
        Plan-local cache shared by optimizer states.
    weights:
        Optimizer weights for this pass.
    allow_folds:
        Whether folded role-component treatments may enter the DP frontier.

    Returns
    -------
    tuple[...] | None
        Best decision tuple, or ``None`` when no frontier was produced.
    """

    best: (
        tuple[
            float,
            float,
            int,
            _DecisionPoint,
            dict[_MemoKey, tuple[_DecisionPoint, ...]],
            OptimizerWeights,
            bool,
        ]
        | None
    ) = None
    total_units = _optimizer_total_units(trace, context)
    rendered_own_units = _rendered_own_unit_map(trace, context)
    for g_star in _g_star_candidates(trace, analysis, context, hidden_counts):
        state = _OptimizerState(
            trace=trace,
            context=context,
            analysis=analysis,
            child_addresses=child_addresses,
            hidden_counts=hidden_counts,
            structural_digests=structural_digests,
            expanded_cache=expanded_cache,
            role_components_cache={},
            child_segments_cache={},
            single_member_expanded_cache={},
            box_cost_cache={},
            branch_salience_cache={},
            output_shape_cache=output_shape_cache,
            weights=weights,
            g_star=g_star,
            total_ops=total_units,
            allow_folds=allow_folds,
            allow_segments=allow_segments,
            max_salience_floor=None,
            rendered_own_units=rendered_own_units,
        )
        memo: dict[_MemoKey, tuple[_DecisionPoint, ...]] = {}
        frontier = _frontier_for_module("self", state, memo)
        for point in frontier:
            if k_min is not None and point.k < k_min:
                continue
            if k_max is not None and point.k > k_max:
                continue
            if require_segments and -1 not in point.priority:
                continue
            score = (
                _global_max_q(point, weights)
                if objective == "max"
                else _global_q(point, trace, weights)
            )
            if context.vis_mode == "rolled" and not point.box_costs and hidden_counts:
                score = round(score + 10.0, 6)
            candidate = (score, g_star, point.k, point, memo, weights, allow_folds)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    return best


def _frontier_can_reach_band(
    trace: Trace,
    context: RenderContext,
    analysis: CollapseAnalysis,
    child_addresses: Mapping[str, tuple[str, ...]],
    hidden_counts: Mapping[str, int],
    structural_digests: Mapping[str, str],
    expanded_cache: dict[
        str,
        tuple[ChildCondensedFlowGraph | None, tuple[str, ...], tuple[str, ...]],
    ],
    output_shape_cache: dict[tuple[str, str], tuple[int, ...] | None],
    weights: OptimizerWeights,
    band_high: int,
    source_graph: SourceGraph | None = None,
) -> bool:
    """Return whether module boxes or run folds can reach the readable band.

    Parameters
    ----------
    trace:
        Trace being optimized.
    context:
        Rendering context.
    analysis:
        Shared collapse analysis.
    child_addresses:
        Renderer-tree child addresses.
    hidden_counts:
        Renderer-faithful hidden counts.
    structural_digests:
        Structural memo digests.
    expanded_cache:
        Shared expanded-structure cache.
    output_shape_cache:
        Plan-local cache shared by optimizer states.
    weights:
        Optimizer weights for the fold-enabled pass.
    band_high:
        Maximum readable auto node count.
    source_graph:
        Optional schedule-local normalized source graph.

    Returns
    -------
    bool
        True when a non-segment frontier point can already render within the
        auto band.
    """

    total_units = _optimizer_total_units(trace, context)
    rendered_own_units = _rendered_own_unit_map(trace, context)
    for g_star in _g_star_candidates(trace, analysis, context, hidden_counts):
        state = _OptimizerState(
            trace=trace,
            context=context,
            analysis=analysis,
            child_addresses=child_addresses,
            hidden_counts=hidden_counts,
            structural_digests=structural_digests,
            expanded_cache=expanded_cache,
            role_components_cache={},
            child_segments_cache={},
            single_member_expanded_cache={},
            box_cost_cache={},
            branch_salience_cache={},
            output_shape_cache=output_shape_cache,
            weights=weights,
            g_star=g_star,
            total_ops=total_units,
            allow_folds=True,
            allow_segments=False,
            max_salience_floor=None,
            rendered_own_units=rendered_own_units,
        )
        memo: dict[_MemoKey, tuple[_DecisionPoint, ...]] = {}
        for point in _frontier_for_module("self", state, memo):
            if point.k < 1 or point.k > band_high:
                continue
            instantiated = _instantiate_module("self", point.k, state, memo)
            plan = _collapse_plan_for_source_or_trace(
                trace,
                _collapse_fn_from_selected(instantiated.selected),
                _fold_mapping(instantiated.folds),
                context,
                source_graph,
            )
            if 1 <= count(plan) <= band_high:
                return True
    return False


def build_role_components(
    trace: Trace,
    parent_address: str,
    child_addresses: Sequence[str],
    analysis: CollapseAnalysis | None = None,
    hidden_counts: Mapping[str, int] | None = None,
) -> tuple[RoleComponent, ...]:
    """Return role components for a parent's flow-ordered children.

    Parameters
    ----------
    trace:
        Trace owning the module hierarchy.
    parent_address:
        Parent module address.
    child_addresses:
        Child addresses, preferably in flow order.
    analysis:
        Optional shared collapse analysis.
    hidden_counts:
        Optional renderer-faithful hidden masses for role grouping.

    Returns
    -------
    tuple[RoleComponent, ...]
        Connected components in deterministic flow order.
    """

    resolved_analysis = analyze_collapse(trace) if analysis is None else analysis
    children = tuple(child for child in child_addresses if child in trace.modules)
    resolved_hidden = hidden_counts or {}
    # The _same_role relation depends only on per-child data: same class_name
    # and |log2(1 + n_left) - log2(1 + n_right)| <= 1.5. The connected
    # components of a within-tolerance relation on a line are exactly the
    # maximal sorted runs with consecutive gap <= 1.5 (IEEE subtraction is
    # monotone, so no pair straddling a gap > 1.5 can be within tolerance),
    # so this O(N log N) partition matches the all-pairs union sweep exactly.
    buckets: dict[str, list[tuple[float, int]]] = {}
    for index, child in enumerate(children):
        module = cast("Module", trace.modules[child])
        class_name = str(getattr(module, "class_name", ""))
        mass = math.log2(
            1 + _faithful_hidden_count(resolved_analysis.signals[child], resolved_hidden)
        )
        buckets.setdefault(class_name, []).append((mass, index))
    component_indices: list[list[int]] = []
    for entries in buckets.values():
        entries.sort()
        run = [entries[0][1]]
        previous_mass = entries[0][0]
        for mass, index in entries[1:]:
            if mass - previous_mass > 1.5:
                component_indices.append(run)
                run = []
            run.append(index)
            previous_mass = mass
        component_indices.append(run)
    return tuple(
        RoleComponent(tuple(children[index] for index in sorted(indices)))
        for indices in sorted(component_indices, key=min)
    )


def _role_components_for_children(
    state: _OptimizerState,
    parent_address: str,
    child_addresses: Sequence[str],
) -> tuple[RoleComponent, ...]:
    """Return cached role components for a concrete parent child sequence."""

    child_key = tuple(child_addresses)
    key = (parent_address, child_key)
    cached = state.role_components_cache.get(key)
    if cached is not None:
        return cached
    components = build_role_components(
        state.trace,
        parent_address,
        child_key,
        state.analysis,
        state.hidden_counts,
    )
    state.role_components_cache[key] = components
    return components


# _floor_fallback_selection retired (memo D5(iv)): the frontier-empty path
# now serves the deterministic significance-greedy fallback in
# collapse_fallback.linear_fallback_plan; _top_level_fallback_addresses
# moved there with it.


def _frontier_for_module(
    address: str,
    state: _OptimizerState,
    memo: dict[_MemoKey, tuple[_DecisionPoint, ...]],
) -> tuple[_DecisionPoint, ...]:
    """Return the DP frontier for one module address."""

    signal = state.analysis.signals.get(address)
    if signal is None:
        return ()
    key = _memo_key(state, signal)
    if key in memo:
        return memo[key]
    points: list[_DecisionPoint] = []
    if address != "self" and _eligible_module_box(state, address, signal):
        box_cost = _cached_box_cost(state.trace, signal, state)
        points.append(
            _DecisionPoint(
                # M2a (collapse memo D1): a selected box renders one box PER
                # CALL plus the atomic own-output ops the renderer keeps, so
                # the scored k is that realized count from the cached render
                # units -- k=1 here was the phantom-k pricing lie (the DP
                # optimized a number that is not what gets drawn). Rolled
                # contexts render one merged box and genuinely charge 1.
                k=_module_box_k(state, address),
                cost=box_cost,
                decision=_ModuleDecision("box"),
                box_costs=(box_cost,),
                priority=(0,),
            )
        )
    points.extend(_expanded_points(address, state, memo))
    frontier = _prune_frontier(points)
    memo[key] = frontier
    return frontier


def _module_box_k(state: _OptimizerState, address: str) -> int:
    """Return the honest rendered-unit count for selecting ``address`` as a box.

    Delegates to :func:`_module_box_plan_nodes` -- the exact function that
    REALIZES the box in the plan -- so the scored k and the realized plan
    count agree by construction (the three-way parity gate pins this).
    """

    return len(_module_box_plan_nodes(state.trace, state.context, address))


def _expanded_points(
    address: str,
    state: _OptimizerState,
    memo: dict[_MemoKey, tuple[_DecisionPoint, ...]],
) -> tuple[_DecisionPoint, ...]:
    """Return expanded-frontier points for one module."""

    graph, child_addresses, own_ops = _expanded_structure(address, state)
    if state.allow_segments and _own_ops_segment_is_legal(state, own_ops):
        base = _DecisionPoint(
            k=1,
            cost=state.weights.segment_intrinsic,
            decision=_ModuleDecision("expand", ()),
            box_costs=(state.weights.segment_intrinsic,),
            priority=(-1,),
        )
    else:
        base = _DecisionPoint(
            k=len(own_ops),
            cost=0.0,
            decision=_ModuleDecision("expand", ()),
            box_costs=(),
            priority=(2,),
        )
    if not child_addresses:
        return (base,) if base.k <= K_CAP else ()
    segments = _child_segments_for_parent(state, child_addresses)
    accumulated: tuple[_DecisionPoint, ...] = (base,)
    for segment in segments:
        segment_points = _sequence_points(address, segment, graph, state, memo)
        accumulated = _merge_module_segment_frontiers(accumulated, segment_points)
    return accumulated


def _expanded_structure(
    address: str,
    state: _OptimizerState,
) -> tuple[ChildCondensedFlowGraph | None, tuple[str, ...], tuple[str, ...]]:
    """Return child-flow graph, ordered children, and own ops for expansion."""

    cached = state.expanded_cache.get(address)
    if cached is not None:
        return cached
    graph = state.analysis.child_flow_graphs.get(address)
    raw_child_addresses = state.child_addresses.get(address, ())
    if raw_child_addresses and (
        graph is None or any(child not in graph.flow_children for child in raw_child_addresses)
    ):
        graph = _cheap_synthetic_child_condensed_flow_graph(address, raw_child_addresses, state)
    child_addresses = tuple(_flow_ordered_child_addresses(list(raw_child_addresses), graph))
    signal = state.analysis.signals[address]
    own_ops = tuple(graph.parent_owned_ops if graph is not None else ())
    if state.context.vis_mode == "rolled":
        own_ops = state.rendered_own_units.get(address, ())
    if not child_addresses:
        if state.context.vis_mode == "rolled":
            own_ops = state.rendered_own_units.get(address, ())
        else:
            own_ops = tuple(
                label
                for label in signal.subtree_ops
                if not getattr(state.trace.ops[label], "is_buffer", False)
            )
    elif address == "self":
        if state.context.vis_mode == "rolled":
            own_ops = state.rendered_own_units.get(address, ())
        else:
            child_ops = {
                label
                for child in child_addresses
                for label in state.analysis.signals.get(child, signal).subtree_ops
            }
            own_ops = tuple(
                op.label
                for op in state.trace.ops
                if op.label not in child_ops and not getattr(op, "is_buffer", False)
            )
    result = (graph, child_addresses, tuple(own_ops))
    state.expanded_cache[address] = result
    return result


def _synthetic_parent_owned_ops(
    parent_address: str,
    child_sets: Mapping[str, set[str]],
    state: _OptimizerState,
) -> tuple[str, ...]:
    """Return the parent's own non-buffer ops for a synthetic child graph.

    M1 (collapse memo D1): the parent's own ops come from the SAME subtree
    set-difference the root path uses honestly. Stamping () here made every
    synthetic-graph parent claim zero own ops, so the optimizer scored a
    node count that was not what gets drawn (phantom k) on albert-class and
    branchy models.
    """

    parent_signal = state.analysis.signals.get(parent_address)
    if parent_signal is None:
        return ()
    child_ops = {label for labels in child_sets.values() for label in labels}
    return tuple(
        label
        for label in parent_signal.subtree_ops
        if label not in child_ops and not getattr(state.trace.ops[label], "is_buffer", False)
    )


def _cross_owner_edges(
    state: _OptimizerState,
    owner_by_label: Mapping[str, str],
) -> set[tuple[str, str]]:
    """Return owner-to-owner dataflow edges implied by op-level children.

    Returns
    -------
    set
        Directed edges between owning children, plus ``external_source:``
        pseudo-nodes for dataflow entering from outside every child.
    """

    edges: set[tuple[str, str]] = set()
    for op in state.trace.ops:
        source = owner_by_label.get(op.label)
        for child_label in getattr(op, "children", ()) or ():
            target_label = str(child_label)
            target = owner_by_label.get(target_label)
            if source is None and target is None:
                continue
            if source is None:
                assert target is not None
                edges.add((f"external_source:{op.label}", target))
            elif target is None:
                continue
            elif source != target:
                edges.add((source, target))
    return edges


def _cheap_synthetic_child_condensed_flow_graph(
    parent_address: str,
    child_addresses: Sequence[str],
    state: _OptimizerState,
) -> ChildCondensedFlowGraph:
    """Build a lightweight child graph without trace substring lookups."""

    op_order = {op.label: index for index, op in enumerate(state.trace.ops)}
    child_sets = {
        child: set(state.analysis.signals[child].subtree_ops)
        for child in child_addresses
        if child in state.analysis.signals
    }
    parent_owned_ops = _synthetic_parent_owned_ops(parent_address, child_sets, state)
    flow_children = tuple(
        sorted(
            child_sets,
            key=lambda child: (
                min((op_order.get(label, 10**12) for label in child_sets[child]), default=10**12),
                child,
            ),
        )
    )
    owner_by_label: dict[str, str] = {}
    for child, labels in child_sets.items():
        for label in labels:
            owner_by_label[label] = child
    edges = _cross_owner_edges(state, owner_by_label)
    for left, right in zip(flow_children[:-1], flow_children[1:], strict=True):
        edges.add((left, right))
    ordered_nodes = (
        *flow_children,
        *sorted({node for edge in edges for node in edge if ":" in node}),
    )
    ordered = {node: index for index, node in enumerate(ordered_nodes)}
    sorted_edges = tuple(
        sorted(
            edges,
            key=lambda edge: (
                ordered.get(edge[0], 10**9),
                ordered.get(edge[1], 10**9),
                edge,
            ),
        )
    )
    return ChildCondensedFlowGraph(
        parent=parent_address,
        flow_children=flow_children,
        parent_owned_ops=parent_owned_ops,
        nodes=ordered_nodes,
        edges=sorted_edges,
        child_external_endpoint_counts=_external_endpoint_counts(sorted_edges, flow_children),
        interval_flags={},
    )


def _external_endpoint_counts(
    edges: Sequence[tuple[str, str]],
    flow_children: Sequence[str],
) -> dict[str, tuple[int, int]]:
    """Return simple external endpoint counts for a lightweight child graph."""

    child_set = set(flow_children)
    entries: dict[str, set[str]] = {child: set() for child in flow_children}
    exits: dict[str, set[str]] = {child: set() for child in flow_children}
    for source, target in edges:
        if target in child_set and source not in child_set:
            entries[target].add(source)
        if source in child_set and target not in child_set:
            exits[source].add(target)
    return {child: (len(entries[child]), len(exits[child])) for child in flow_children}


def _sequence_points(
    parent_address: str,
    child_addresses: Sequence[str],
    graph: ChildCondensedFlowGraph | None,
    state: _OptimizerState,
    memo: dict[_MemoKey, tuple[_DecisionPoint, ...]],
) -> tuple[_DecisionPoint, ...]:
    """Return child-sequence DP points constrained by role components."""

    components = _role_components_for_children(state, parent_address, child_addresses)
    component_choices = [
        _component_treatment_points(component, graph, state, memo) for component in components
    ]
    frontier: tuple[_DecisionPoint, ...] = (
        _DecisionPoint(
            k=0,
            cost=0.0,
            decision=_SegmentDecision(()),
            box_costs=(),
            priority=(),
        ),
    )
    for choices in component_choices:
        frontier = _merge_segment_component_frontiers(frontier, choices)
    return frontier


def _component_treatment_points(
    component: RoleComponent,
    graph: ChildCondensedFlowGraph | None,
    state: _OptimizerState,
    memo: dict[_MemoKey, tuple[_DecisionPoint, ...]],
) -> tuple[_DecisionPoint, ...]:
    """Enumerate BOXES, EXPANDED, and FOLDED treatments for one role component."""

    choices: list[_DecisionPoint] = []
    if state.allow_segments:
        segmented = _component_segmented(component, graph, state)
        if segmented is not None:
            choices.append(segmented)
    boxes = _component_boxes(component, state)
    if boxes is not None:
        choices.append(boxes)
    expanded = _component_expanded(component, state, memo)
    if expanded:
        choices.extend(expanded)
    if state.allow_folds:
        folded = _component_folded(component, graph, state)
        if folded is not None:
            choices.append(folded)
    return _prune_frontier(choices)


def _own_ops_segment_is_legal(state: _OptimizerState, own_ops: tuple[str, ...]) -> bool:
    """Return whether parent-owned ops may be replaced by one op segment."""

    if len(own_ops) < 3:
        return False
    if state.total_ops > 0 and len(own_ops) / state.total_ops > 0.75:
        return False
    op_by_label = {str(op.label): op for op in state.trace.ops}
    previous_stack: tuple[str, ...] | None = None
    seen_stacks: dict[tuple[str, ...], tuple[str, ...]] = {}
    for label in own_ops:
        op = op_by_label.get(label)
        if op is None:
            return False
        if getattr(op, "is_input", False) or getattr(op, "is_output", False):
            return False
        if state.context.vis_mode != "rolled":
            # Module-call containment (round-24 C1): a multi-call module's
            # own ops span several rendered call clusters, so replacing them
            # with ONE segment node would nest later calls' work under one
            # call's cluster. Refuse reuse-crossing sequences here; the
            # condense pass re-segments the raw ops per call where
            # beneficial. Same rule as ``_legal_plan_op_segment_run``.
            stack = _effective_render_module_stack(op)
            if previous_stack is not None and _crosses_module_call_boundary(previous_stack, stack):
                return False
            pass_free = tuple(entry.rsplit(":", 1)[0] for entry in stack)
            if seen_stacks.setdefault(pass_free, stack) != stack:
                return False
            previous_stack = stack
    return True


def _component_segmented(
    component: RoleComponent,
    graph: ChildCondensedFlowGraph | None,
    state: _OptimizerState,
) -> _DecisionPoint | None:
    """Return a first-class child segment treatment for one role component."""

    run = _legal_component_segment_run(component.members, graph, state)
    if run is None:
        return None
    run_set = set(run)
    costs = [_segment_run_cost(run, state)]
    node_count = 1
    for address in component.members:
        if address in run_set:
            continue
        signal = state.analysis.signals[address]
        if not _eligible_module_box(state, address, signal):
            return None
        costs.append(_cached_box_cost(state.trace, signal, state))
        node_count += 1
    return _DecisionPoint(
        k=node_count,
        cost=round(sum(costs), 6),
        decision=_ComponentDecision(
            "segmented",
            run_indices=(tuple(range(len(run))),),
        ),
        box_costs=tuple(costs),
        priority=(-1,),
    )


def _segment_run_cost(run: tuple[str, ...], state: _OptimizerState) -> float:
    """Return the normalized cost for one child segment run."""

    member_costs = [
        _cached_box_cost(state.trace, state.analysis.signals[member], state) for member in run
    ]
    return round(sum(member_costs) / len(member_costs) + state.weights.segment_intrinsic, 6)


def _legal_component_segment_run(
    members: Sequence[str],
    graph: ChildCondensedFlowGraph | None,
    state: _OptimizerState,
) -> tuple[str, ...] | None:
    """Return the longest legal component segment run."""

    if len(members) < 2 or graph is None:
        return None
    return _longest_legal_segment_prefix(
        members,
        graph,
        state.analysis,
        state.hidden_counts,
        state.total_ops,
        dominance_limit=0.75,
    )


def _component_boxes(component: RoleComponent, state: _OptimizerState) -> _DecisionPoint | None:
    """Return the all-boxes treatment for a component when legal."""

    costs: list[float] = []
    total_k = 0
    for address in component.members:
        signal = state.analysis.signals[address]
        if not _eligible_module_box(state, address, signal):
            return None
        cost = _cached_box_cost(state.trace, signal, state)
        costs.append(cost)
        # M2a: one box per rendered CALL plus kept atomic ops, per member --
        # scoring one unit per member address was the phantom-k pricing lie.
        total_k += _module_box_k(state, address)
    return _DecisionPoint(
        k=total_k,
        cost=round(sum(costs), 6),
        decision=_ComponentDecision("boxes"),
        box_costs=tuple(costs),
        priority=(0,),
    )


def _component_expanded(
    component: RoleComponent,
    state: _OptimizerState,
    memo: dict[_MemoKey, tuple[_DecisionPoint, ...]],
) -> tuple[_DecisionPoint, ...]:
    """Return independently expanded member combinations for one component."""

    if len(component.members) == 1:
        return _single_member_expanded(component.members[0], state, memo)
    member_frontiers = [
        tuple(
            point
            for point in _frontier_for_module(address, state, memo)
            # M2b: recognize boxes by decision kind, never by k == 1.
            if not (isinstance(point.decision, _ModuleDecision) and point.decision.kind == "box")
        )
        for address in component.members
    ]
    if any(not frontier for frontier in member_frontiers):
        return ()
    points: tuple[_DecisionPoint, ...] = (
        _DecisionPoint(
            k=0,
            cost=0.0,
            decision=_ComponentDecision("expanded", ()),
            box_costs=(),
            priority=(2,),
        ),
    )
    for frontier in member_frontiers:
        points = _merge_component_member_frontiers(points, frontier)
    return points


def _single_member_expanded(
    address: str,
    state: _OptimizerState,
    memo: dict[_MemoKey, tuple[_DecisionPoint, ...]],
) -> tuple[_DecisionPoint, ...]:
    """Return expanded treatment points for a one-member role component."""

    signal = state.analysis.signals.get(address)
    if signal is None:
        return ()
    key = _memo_key(state, signal)
    cached = state.single_member_expanded_cache.get(key)
    if cached is not None:
        return cached
    points: list[_DecisionPoint] = []
    for point in _frontier_for_module(address, state, memo):
        # M2b: recognize boxes by decision kind, never by k == 1.
        if isinstance(point.decision, _ModuleDecision) and point.decision.kind == "box":
            continue
        points.append(
            _DecisionPoint(
                k=point.k,
                cost=point.cost,
                decision=_ComponentDecision("expanded", (point.k,)),
                box_costs=point.box_costs,
                priority=point.priority,
            )
        )
    result = tuple(points)
    state.single_member_expanded_cache[key] = result
    return result


def _component_folded(
    component: RoleComponent,
    graph: ChildCondensedFlowGraph | None,
    state: _OptimizerState,
) -> _DecisionPoint | None:
    """Return the maximal-run folded treatment for a component when useful."""

    if len(component.members) < RUN_FOLD_MIN_LENGTH or graph is None:
        return None
    runs = _maximal_legal_runs(component.members, graph, state)
    if not runs:
        return None
    run_members = {address for run in runs for address in run}
    costs: list[float] = []
    run_by_first = {run[0]: run for run in runs}
    run_indices: list[tuple[int, ...]] = []
    skipped: set[str] = set()
    node_count = 0
    for member_index, address in enumerate(component.members):
        if address in skipped:
            continue
        run = run_by_first.get(address)
        if run is None:
            signal = state.analysis.signals[address]
            if not _eligible_module_box(state, address, signal):
                return None
            cost = _cached_box_cost(state.trace, signal, state)
            costs.append(cost)
            node_count += 1
            continue
        member_costs = [
            _cached_box_cost(state.trace, state.analysis.signals[member], state) for member in run
        ]
        fold_cost = round(sum(member_costs) / len(member_costs) + state.weights.fold_intrinsic, 6)
        costs.append(fold_cost)
        node_count += 2
        run_indices.append(tuple(range(member_index, member_index + len(run))))
        skipped.update(run_members & set(run))
    return _DecisionPoint(
        k=node_count,
        cost=round(sum(costs), 6),
        decision=_ComponentDecision("folded", run_indices=tuple(run_indices)),
        box_costs=tuple(costs),
        priority=(1,),
    )


def _maximal_legal_runs(
    members: Sequence[str],
    graph: ChildCondensedFlowGraph,
    state: _OptimizerState,
) -> tuple[tuple[str, ...], ...]:
    """Partition component members into maximal legal fold repeats."""

    # B2 (collapse memo item 7): grow the cheap prefix (flow adjacency,
    # box eligibility, equal shapes -- O(1) amortized per extension), then
    # resolve the longest legal member-uniform window with ONE signature
    # pass over the component and a longest-first legality scan. Output is
    # identical to the historical grow-every-window enumeration, which paid
    # a full-window legality check plus full-window signature recomputation
    # at EVERY width (24,165 legality calls measured on densenet201).
    runs: list[tuple[str, ...]] = []
    fingerprints = fingerprints_for(state.trace, state.analysis)
    index = 0
    while index < len(members):
        candidate: list[str] = []
        for address in members[index:]:
            if candidate and not _flow_adjacent(candidate[-1], address, graph):
                break
            if not _eligible_module_box(state, address, state.analysis.signals[address]):
                break
            if candidate and not _cached_module_output_shapes_equal(state, candidate[-1], address):
                break
            candidate.append(address)
        best = longest_uniform_legal_run(tuple(candidate), graph, fingerprints)
        if best:
            runs.append(best)
            index += len(best)
        else:
            index += 1
    return tuple(runs)


def _module_render_box_units(
    trace: Trace,
    context: RenderContext,
) -> Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    """Return per-call rendered collapse units for every module address.

    For each pass-free module address the value is ``(box_calls, kept_ops)``:

    - ``box_calls``: the pass-qualified module calls the renderer draws as one
      collapsed box each when the address is selected. A call is included only
      when it hides at least one rendered node, so a pure atomic module (whose
      only content is its own atomic exit op) has no box calls at all: the
      renderer drops the innermost atomic box and keeps the op visible.
    - ``kept_ops``: one pass-free render label per concrete op occurrence that
      stays visible when the address is selected, because atomic module
      collapse drops the innermost module from the op's owner stack.

    Both tuples enumerate in render order, so occurrences of one op layer
    appear in pass order. The map is meaningful for unrolled contexts; rolled
    selection always renders one merged box per address.

    Parameters
    ----------
    trace:
        Trace being optimized.
    context:
        Render context for the selected plan.

    Returns
    -------
    Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]]
        Rendered box calls and kept op occurrences keyed by module address.
    """

    revision = _collapse_graph_revision(trace)
    cache_entry = _BOX_UNITS_CACHE.get(trace)
    if cache_entry is None or cache_entry[0] != revision:
        cached_by_trace: dict[
            RenderContext,
            Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]],
        ] = {}
        _BOX_UNITS_CACHE[trace] = (revision, cached_by_trace)
    else:
        cached_by_trace = cache_entry[1]
    cached = cached_by_trace.get(context)
    if cached is not None:
        return cached
    from ._render_common import BoundaryNode
    from ._render_edges import _is_buffer_visible
    from ._render_flow import _entries_to_plot_for_context
    from ._render_nodes import _normalize_buffer_visibility

    show_buffer_layers = _normalize_buffer_visibility(context.show_buffer_layers)
    box_calls: dict[str, dict[str, None]] = {}
    kept_ops: dict[str, list[str]] = {}
    for node in _entries_to_plot_for_context(trace, context.vis_mode).values():
        if isinstance(node, BoundaryNode):
            continue
        if node.is_buffer and not _is_buffer_visible(node, show_buffer_layers):
            continue
        modules = [str(module_call) for module_call in (getattr(node, "modules", ()) or ())]
        if getattr(node, "is_atomic_module", False) and modules:
            innermost_address = modules[-1].rsplit(":", 1)[0]
            remaining = modules[:-1]
            if innermost_address not in {call.rsplit(":", 1)[0] for call in remaining}:
                base = str(node.layer_label).rsplit(":", 1)[0]
                kept_ops.setdefault(innermost_address, []).append(base)
            modules = remaining
        for module_call in modules:
            box_calls.setdefault(module_call.rsplit(":", 1)[0], {})[module_call] = None
    units: Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
        address: (tuple(box_calls.get(address, ())), tuple(kept_ops.get(address, ())))
        for address in set(box_calls) | set(kept_ops)
    }
    cached_by_trace[context] = units
    return units


def _module_box_plan_nodes(
    trace: Trace,
    context: RenderContext,
    address: str,
) -> tuple[PlanNode, ...]:
    """Return plan nodes rendered for one selected module box.

    Parameters
    ----------
    trace:
        Trace being optimized.
    context:
        Render context for the selected plan.
    address:
        Pass-free module address selected as a collapsed box.

    Returns
    -------
    tuple[PlanNode, ...]
        One module box per rendered call of ``address`` plus the raw atomic
        own-output op occurrences that the renderer keeps visible because
        atomic module collapse drops the innermost module. Multi-call modules
        contribute one box per call, matching the per-call render of the
        address collapse predicate; counting them once per address is the
        round-22 plan/schedule node-count lie.
    """

    if context.vis_mode == "rolled":
        return (ModuleBox(f"{address}:1"),)
    units = _module_render_box_units(trace, context).get(address)
    if units is None:
        # Address absent from the rendered universe (for example fully
        # invisible content): preserve the legacy single-box shape rather
        # than claiming zero rendered nodes for a selected module.
        return (ModuleBox(f"{address}:1"),)
    calls, kept = units
    nodes: list[PlanNode] = [ModuleBox(module_call) for module_call in calls]
    nodes.extend(RawOp(label) for label in kept)
    if not nodes:
        return (ModuleBox(f"{address}:1"),)
    return tuple(nodes)


def _instantiate_module(
    address: str,
    k: int,
    state: _OptimizerState,
    memo: dict[_MemoKey, tuple[_DecisionPoint, ...]],
) -> _FrontierPoint:
    """Instantiate memoized decisions for the concrete module address."""

    decision_point = _decision_point_for_k(address, k, state, memo)
    decision = cast(_ModuleDecision, decision_point.decision)
    if decision.kind == "box":
        signal = state.analysis.signals[address]
        box_cost = _cached_box_cost(state.trace, signal, state)
        box_nodes = _module_box_plan_nodes(state.trace, state.context, address)
        return _FrontierPoint(
            k=len(box_nodes),
            cost=box_cost,
            nodes=box_nodes,
            selected=frozenset({address}),
            folds=(),
            box_costs=(box_cost,),
        )
    graph, child_addresses, own_ops = _expanded_structure(address, state)
    if (
        state.allow_segments
        and decision_point.k == 1
        and _own_ops_segment_is_legal(state, tuple(own_ops))
    ):
        nodes: list[PlanNode] = [OpSegment(tuple(str(op).rsplit(":", 1)[0] for op in own_ops))]
    else:
        nodes = [RawOp(op) for op in own_ops]
    selected: set[str] = set()
    folds: list[ModuleRepeatFold] = []
    box_costs: list[float] = []
    segments = _child_segments_for_parent(state, child_addresses)
    for segment, segment_decision in zip(segments, decision.segments, strict=True):
        segment_point = _instantiate_segment(address, segment, graph, segment_decision, state, memo)
        nodes.extend(segment_point.nodes)
        selected.update(segment_point.selected)
        folds.extend(segment_point.folds)
        box_costs.extend(segment_point.box_costs)
    return _FrontierPoint(
        k=decision_point.k,
        cost=decision_point.cost,
        nodes=tuple(nodes),
        selected=frozenset(selected),
        folds=tuple(folds),
        box_costs=tuple(box_costs),
    )


def _decision_point_for_k(
    address: str,
    k: int,
    state: _OptimizerState,
    memo: dict[_MemoKey, tuple[_DecisionPoint, ...]],
) -> _DecisionPoint:
    """Return the memoized decision point for ``address`` and node count ``k``."""

    frontier = _frontier_for_module(address, state, memo)
    for point in frontier:
        if point.k == k:
            return point
    raise ValueError(f"no v2 collapse decision for {address!r} at k={k}")


def _instantiate_segment(
    parent_address: str,
    child_addresses: Sequence[str],
    graph: ChildCondensedFlowGraph | None,
    decision: _SegmentDecision,
    state: _OptimizerState,
    memo: dict[_MemoKey, tuple[_DecisionPoint, ...]],
) -> _FrontierPoint:
    """Instantiate one segmented child sequence."""

    components = _role_components_for_children(state, parent_address, child_addresses)
    nodes: list[PlanNode] = []
    selected: set[str] = set()
    folds: list[ModuleRepeatFold] = []
    box_costs: list[float] = []
    total_cost = 0.0
    total_k = 0
    for component, component_decision in zip(components, decision.components, strict=True):
        point = _instantiate_component(component, graph, component_decision, state, memo)
        nodes.extend(point.nodes)
        selected.update(point.selected)
        folds.extend(point.folds)
        box_costs.extend(point.box_costs)
        total_cost += point.cost
        total_k += point.k
    return _FrontierPoint(
        k=total_k,
        cost=round(total_cost, 6),
        nodes=tuple(nodes),
        selected=frozenset(selected),
        folds=tuple(folds),
        box_costs=tuple(box_costs),
    )


def _instantiate_component(
    component: RoleComponent,
    graph: ChildCondensedFlowGraph | None,
    decision: _ComponentDecision,
    state: _OptimizerState,
    memo: dict[_MemoKey, tuple[_DecisionPoint, ...]],
) -> _FrontierPoint:
    """Instantiate a concrete role-component treatment."""

    if decision.kind == "boxes":
        return _instantiate_component_boxes(component, state)
    if decision.kind == "expanded":
        return _instantiate_component_expanded(component, decision, state, memo)
    if decision.kind == "segmented":
        return _instantiate_component_segmented(component, decision, state)
    return _instantiate_component_folded(component, graph, decision, state)


def _instantiate_component_segmented(
    component: RoleComponent,
    decision: _ComponentDecision,
    state: _OptimizerState,
) -> _FrontierPoint:
    """Instantiate a first-class child segment component treatment."""

    if not decision.run_indices:
        raise ValueError("segmented component requires member indices")
    indices = decision.run_indices[0]
    run = tuple(component.members[index] for index in indices)
    run_set = set(run)
    segment_cost = _segment_run_cost(run, state)
    nodes: list[PlanNode] = [ChildSegment(run)]
    selected: set[str] = set(run)
    box_costs: list[float] = [segment_cost]
    total_cost = segment_cost
    for address in component.members:
        if address in run_set:
            continue
        signal = state.analysis.signals[address]
        cost = _cached_box_cost(state.trace, signal, state)
        nodes.extend(_module_box_plan_nodes(state.trace, state.context, address))
        selected.add(address)
        box_costs.append(cost)
        total_cost += cost
    return _FrontierPoint(
        k=len(nodes),
        cost=round(total_cost, 6),
        nodes=tuple(nodes),
        selected=frozenset(selected),
        folds=(),
        box_costs=tuple(box_costs),
    )


def _instantiate_component_boxes(
    component: RoleComponent,
    state: _OptimizerState,
) -> _FrontierPoint:
    """Instantiate an all-box role component."""

    nodes: list[PlanNode] = []
    selected: set[str] = set()
    box_costs: list[float] = []
    for address in component.members:
        signal = state.analysis.signals[address]
        cost = _cached_box_cost(state.trace, signal, state)
        nodes.extend(_module_box_plan_nodes(state.trace, state.context, address))
        selected.add(address)
        box_costs.append(cost)
    return _FrontierPoint(
        k=len(nodes),
        cost=round(sum(box_costs), 6),
        nodes=tuple(nodes),
        selected=frozenset(selected),
        folds=(),
        box_costs=tuple(box_costs),
    )


def _instantiate_component_expanded(
    component: RoleComponent,
    decision: _ComponentDecision,
    state: _OptimizerState,
    memo: dict[_MemoKey, tuple[_DecisionPoint, ...]],
) -> _FrontierPoint:
    """Instantiate an expanded role component."""

    nodes: list[PlanNode] = []
    selected: set[str] = set()
    folds: list[ModuleRepeatFold] = []
    box_costs: list[float] = []
    total_cost = 0.0
    total_k = 0
    for address, member_k in zip(component.members, decision.member_ks, strict=True):
        point = _instantiate_module(address, member_k, state, memo)
        nodes.extend(point.nodes)
        selected.update(point.selected)
        folds.extend(point.folds)
        box_costs.extend(point.box_costs)
        total_cost += point.cost
        total_k += point.k
    return _FrontierPoint(
        k=total_k,
        cost=round(total_cost, 6),
        nodes=tuple(nodes),
        selected=frozenset(selected),
        folds=tuple(folds),
        box_costs=tuple(box_costs),
    )


def _instantiate_component_folded(
    component: RoleComponent,
    graph: ChildCondensedFlowGraph | None,
    decision: _ComponentDecision,
    state: _OptimizerState,
) -> _FrontierPoint:
    """Instantiate a maximal-run folded role component."""

    if graph is None:
        raise ValueError("folded component requires a child flow graph")
    run_index_sets = {indices[0]: indices for indices in decision.run_indices}
    skipped: set[int] = set()
    nodes: list[PlanNode] = []
    selected: set[str] = set()
    folds: list[ModuleRepeatFold] = []
    box_costs: list[float] = []
    total_cost = 0.0
    total_k = 0
    for member_index, address in enumerate(component.members):
        if member_index in skipped:
            continue
        indices = run_index_sets.get(member_index)
        if indices is None:
            point = _instantiate_component_boxes(RoleComponent((address,)), state)
            nodes.extend(point.nodes)
            selected.update(point.selected)
            box_costs.extend(point.box_costs)
            total_cost += point.cost
            total_k += point.k
            continue
        run = tuple(component.members[index] for index in indices)
        fold = _make_run_fold(state.trace, run)
        member_costs = [
            _cached_box_cost(state.trace, state.analysis.signals[member], state) for member in run
        ]
        fold_cost = round(sum(member_costs) / len(member_costs) + state.weights.fold_intrinsic, 6)
        # The renderer folds hidden members on every pass but still draws the
        # representative once per call: pass 1 renders as the fold (box plus
        # one ellipsis) and every later pass renders a plain module box.
        # Charging a flat 2 regardless of the representative's call count is
        # the round-22 fold leg of the plan/schedule node-count lie.
        rep_plan_nodes = _module_box_plan_nodes(state.trace, state.context, fold.representative)
        rep_boxes = [node for node in rep_plan_nodes if isinstance(node, ModuleBox)]
        rep_kept = [node for node in rep_plan_nodes if isinstance(node, RawOp)]
        if not rep_boxes:
            rep_boxes = [ModuleBox(f"{fold.representative}:1")]
        nodes.append(
            RepeatFold(
                rep=rep_boxes[0],
                members=fold.addresses,
                ellipsis=EllipsisNode(fold.addresses[1:]),
            )
        )
        nodes.extend(rep_boxes[1:])
        nodes.extend(rep_kept)
        selected.update(run)
        folds.append(fold)
        box_costs.append(fold_cost)
        total_cost += fold_cost
        total_k += 1 + len(rep_boxes) + len(rep_kept)
        skipped.update(indices)
    return _FrontierPoint(
        k=total_k,
        cost=round(total_cost, 6),
        nodes=tuple(nodes),
        selected=frozenset(selected),
        folds=tuple(folds),
        box_costs=tuple(box_costs),
    )


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


def _prune_frontier(points: Sequence[Any]) -> tuple[Any, ...]:
    """Keep per-count best points under the deterministic beam cap."""

    # Memo D5(iii)+(v): every frontier record passes this chokepoint, so it
    # meters the allocation cap (memory defence) and the generous watchdog;
    # either trips FallbackDegrade, caught at the selection entry.
    charge_frontier_allocation(len(points))
    best_by_count: dict[int, tuple[Any, tuple[float, int, tuple[Any, ...], tuple[Any, ...]]]] = {}
    for point in points:
        if point.k > K_CAP:
            continue
        _retain_best_frontier_point(best_by_count, point)
    return _frontier_from_best_by_count(best_by_count)


def _point_sort_key(point: Any) -> tuple[float, int, tuple[Any, ...], tuple[Any, ...]]:
    """Return deterministic ordering key for frontier points."""

    if isinstance(point, _DecisionPoint):
        return (point.cost, point.k, point.priority, ())
    fold_addresses = tuple(fold.representative for fold in point.folds)
    return (point.cost, point.k, tuple(sorted(point.selected)), fold_addresses)


def _eligible_box(trace: Trace, address: str, signal: ModuleCollapseSignals) -> bool:
    """Return whether a module may render as a collapsed box."""

    _ = trace, address
    return signal.eligible and signal.landmark_edges < 2


def _eligible_module_box(
    state: _OptimizerState,
    address: str,
    signal: ModuleCollapseSignals,
) -> bool:
    """Return whether a module box is legal under the current optimizer state."""

    if not _eligible_box(state.trace, address, signal):
        return False
    floor = state.max_salience_floor
    if floor is None:
        return True
    return _max_box_salience_score(address, signal, state) < floor


def _max_box_salience_score(
    address: str,
    signal: ModuleCollapseSignals,
    state: _OptimizerState,
) -> float:
    """Return the max-mode salience floor score for hiding ``address``."""

    salience = _branch_salience(address, state)
    uniqueness = 1.0 / max(float(signal.peer_count), 1.0)
    return round(salience * uniqueness, 6)


def _box_cost(trace: Trace, signal: ModuleCollapseSignals, state: _OptimizerState) -> float:
    """Return normalized v2 box cost for one module."""

    n = _faithful_hidden_count(signal, state.hidden_counts)
    log_mass = math.log2(1 + n)
    grain_norm = math.log2(1 + 64) ** 2
    grain = (log_mass - state.g_star) ** 2 / grain_norm
    landmark = min(max(signal.landmark_edges, 0) / 3.0, 1.0)
    trunk = 1.0 if _is_trunk_collapse(trace, signal) else 0.0
    module = cast("Module", trace.modules[signal.address])
    generic = 1.0 if str(getattr(module, "class_name", "")) in GENERIC_CONTAINER_CLASSES else 0.0
    dominance = _dominance(n, state.total_ops)
    salience = _branch_salience(signal.address, state) / max(float(signal.peer_count), 1.0)
    cost = (
        state.weights.w_grain * grain
        + state.weights.w_landmark * landmark
        + state.weights.w_trunk * trunk
        + state.weights.w_generic * generic
        + state.weights.w_dom * dominance
        + state.weights.w_sal * salience
    )
    return round(cost, 6)


def _cached_box_cost(
    trace: Trace,
    signal: ModuleCollapseSignals,
    state: _OptimizerState,
) -> float:
    """Return a cached normalized v2 box cost for one module."""

    cached = state.box_cost_cache.get(signal.address)
    if cached is not None:
        return cached
    cost = _box_cost(trace, signal, state)
    state.box_cost_cache[signal.address] = cost
    return cost


def _dominance(hidden_count: int, total_ops: int) -> float:
    """Return dominance ramp above sixty percent of the trace."""

    if total_ops <= 0:
        return 0.0
    fraction = hidden_count / total_ops
    if fraction <= 0.6:
        return 0.0
    return min((fraction - 0.6) / 0.4, 1.0)


def _branch_salience(
    address: str,
    state: _OptimizerState,
    seen: frozenset[str] = frozenset(),
) -> float:
    """Return normalized child-level branch salience for a module.

    This uses the R1a child-condensed flow artifact with the prescribed simpler
    proxy: ``W`` is the largest set of direct child modules that share the same
    non-child upstream and downstream neighbors and have no flow edges among
    themselves. That catches ASPP/Inception-style parallel fans without paying
    to preserve ordinary sequential stacks. The score propagates through
    immediate children so a thin wrapper around a salient fan is also expensive
    to hide.
    """

    if address in seen:
        return 0.0
    cached = state.branch_salience_cache.get(address)
    if cached is not None:
        return cached
    graph = state.analysis.child_flow_graphs.get(address)
    own_salience = 0.0
    if graph is not None:
        width = _parallel_flow_width(graph)
        own_salience = min(max((width - 1) / 4.0, 0.0), 1.0)
    child_salience = max(
        (
            _branch_salience(child, state, seen | {address})
            for child in state.child_addresses.get(address, ())
            if child in state.analysis.signals
        ),
        default=0.0,
    )
    salience = max(own_salience, child_salience)
    state.branch_salience_cache[address] = salience
    return salience


def _parallel_flow_width(graph: ChildCondensedFlowGraph) -> int:
    """Return max parallel width visible in a child-condensed flow graph.

    The primary width is the task-prescribed child signature proxy. Some
    containers, notably ModuleList-backed branch fans, appear in the R1a graph
    as parent-owned op chains rather than direct child module nodes; for those,
    the fallback width is the largest non-external merge fan-in.
    """

    child_set = set(graph.flow_children)
    neighbors: dict[str, set[str]] = {child: set() for child in graph.flow_children}
    upstreams: dict[str, set[str]] = {child: set() for child in graph.flow_children}
    downstreams: dict[str, set[str]] = {child: set() for child in graph.flow_children}
    for source, target in graph.edges:
        if source in child_set and target in child_set:
            neighbors[source].add(target)
            neighbors[target].add(source)
        if target in child_set and source not in child_set:
            upstreams[target].add(source)
        if source in child_set and target not in child_set:
            downstreams[source].add(target)
    groups: dict[tuple[tuple[str, ...], tuple[str, ...]], list[str]] = {}
    for child in graph.flow_children:
        signature = (tuple(sorted(upstreams[child])), tuple(sorted(downstreams[child])))
        groups.setdefault(signature, []).append(child)
    max_width = 1 if graph.flow_children else 0
    for members in groups.values():
        # Greedy scan in member order; a candidate joins when no already-kept
        # member is a flow neighbor. Checking the candidate's neighbor set
        # against the kept set reproduces the pairwise edge test exactly while
        # costing degree instead of kept-set size per candidate.
        independent: set[str] = set()
        for child in members:
            if not (neighbors[child] & independent):
                independent.add(child)
        max_width = max(max_width, len(independent))
    return max(max_width, _parent_owned_merge_width(graph))


def _parent_owned_merge_width(graph: ChildCondensedFlowGraph) -> int:
    """Return the largest non-external merge fan-in in a condensed graph."""

    node_set = set(graph.nodes)
    incoming: dict[str, set[str]] = {node: set() for node in graph.nodes}
    for source, target in graph.edges:
        if target not in node_set or target.startswith("external_sink:"):
            continue
        if source.startswith("external_source:"):
            continue
        incoming[target].add(source)
    return max((len(sources) for sources in incoming.values()), default=0)


def _faithful_hidden_count(
    signal: ModuleCollapseSignals,
    hidden_counts: Mapping[str, int],
) -> int:
    """Return renderer-faithful hidden count with signal fallback."""

    return max(int(hidden_counts.get(signal.address, signal.hidden_ops)), 0)


def _global_q(
    point: _DecisionPoint | _FrontierPoint, trace: Trace, weights: OptimizerWeights
) -> float:
    """Return global selection objective for one realized cut."""

    mean_cost = sum(point.box_costs) / len(point.box_costs) if point.box_costs else 0.0
    max_cost = max(point.box_costs) if point.box_costs else 0.0
    return round(
        mean_cost
        + weights.w_max * max_cost
        + weights.w_k * _k_preference(point.k, trace)
        + _band_cost(point.k, trace),
        6,
    )


def _global_max_q(point: _DecisionPoint | _FrontierPoint, weights: OptimizerWeights) -> float:
    """Return max-mode objective for a realized cut.

    Parameters
    ----------
    point:
        Candidate DP point.
    weights:
        Optimizer weights.

    Returns
    -------
    float
        Rounded objective value, favoring lower-cost condensed cuts inside the
        active max ladder bound.
    """

    mean_cost = sum(point.box_costs) / len(point.box_costs) if point.box_costs else 0.0
    max_cost = max(point.box_costs) if point.box_costs else 0.0
    return round(mean_cost + weights.w_max * max_cost + 0.01 * point.k, 6)


def _k_preference(k: int, trace: Trace) -> float:
    """Return linear node-count preference normalized to the readable band."""

    lo = 8
    hi = _readable_band_high(trace)
    return (k - lo) / max(hi - lo, 1)


def _band_cost(k: int, trace: Trace) -> float:
    """Return quadratic node-band cost for auto mode."""

    lo = 8
    hi = _readable_band_high(trace)
    target = min(28, hi - 2)
    width = max((hi - lo) / 2.0, 1.0)
    slope = 2.0 if k < lo or k > hi else 1.0
    return slope * ((k - target) / width) ** 2


def _g_star_candidates(
    trace: Trace,
    analysis: CollapseAnalysis,
    context: RenderContext,
    hidden_counts: Mapping[str, int],
) -> tuple[float, ...]:
    """Return deterministic grain target candidates."""

    hi = _readable_band_high(trace)
    target = min(28, hi - 2)
    total_ops = _optimizer_total_units(trace, context)
    raw = [
        math.log2(1 + total_ops / max(target, 1)),
        math.log2(1 + total_ops / max(target / 2.0, 1.0)),
        math.log2(1 + total_ops / 12.0),
        math.log2(1 + total_ops / 20.0),
    ]
    masses = sorted(
        math.log2(
            1
            + max(
                hidden_counts.get(signal.address, signal.hidden_ops)
                if context.vis_mode == "rolled"
                else signal.hidden_ops,
                0,
            )
        )
        for signal in analysis.signals.values()
        if (
            hidden_counts.get(signal.address, signal.hidden_ops)
            if context.vis_mode == "rolled"
            else signal.hidden_ops
        )
        > 0
    )
    if masses:
        raw.append(masses[len(masses) // 2])
    distinct: list[float] = []
    for value in raw:
        rounded = round(value, 6)
        if rounded not in distinct:
            distinct.append(rounded)
    return tuple(distinct[:5])


def _same_role(
    trace: Trace,
    left: str,
    right: str,
    analysis: CollapseAnalysis,
    hidden_counts: Mapping[str, int],
) -> bool:
    """Return whether two siblings belong to the same role component."""

    left_module = cast("Module", trace.modules[left])
    right_module = cast("Module", trace.modules[right])
    if str(getattr(left_module, "class_name", "")) != str(getattr(right_module, "class_name", "")):
        return False
    left_n = _faithful_hidden_count(analysis.signals[left], hidden_counts)
    right_n = _faithful_hidden_count(analysis.signals[right], hidden_counts)
    return abs(math.log2(1 + left_n) - math.log2(1 + right_n)) <= 1.5


def _optimizer_total_units(trace: Trace, context: RenderContext) -> int:
    """Return the raw rendered universe size used for optimizer normalization."""

    if context.vis_mode != "rolled":
        return max(len(trace.ops), 1)
    return max(count(collapse_plan_for_trace(trace, None, None, context)), 1)


def _root_own_unit_count(
    trace: Trace,
    context: RenderContext,
    analysis: CollapseAnalysis,
    child_addresses: Mapping[str, tuple[str, ...]],
) -> int:
    """Return how many rendered units the root owns outside every top-level child.

    The K_CAP cliff variable (collapse memo D4): when this exceeds ``K_CAP``
    every frontier candidate overshoots the cap and the DP frontier empties.
    Rolled contexts count merged rendered units; unrolled contexts count the
    same non-buffer op set-difference ``_expanded_structure`` scores.
    """

    if context.vis_mode == "rolled":
        return len(_rendered_own_unit_map(trace, context).get("self", ()))
    top_level = child_addresses.get("self", ()) or _top_level_fallback_addresses(trace)
    child_ops = {
        label
        for child in top_level
        for label in getattr(analysis.signals.get(child), "subtree_ops", ())
    }
    return sum(
        1 for op in trace.ops if op.label not in child_ops and not getattr(op, "is_buffer", False)
    )


def _max_sibling_group_width(trace: Trace, context: RenderContext) -> int:
    """Return W: the widest rendered sibling group of the full universe.

    Definition (estimator v1, memo D5(ii)): for every parent (the root
    ``self`` included), the group width is its direct child-module count
    plus the rendered units the parent owns directly; W is the maximum.
    Deterministic and cheap -- one analysis read plus the cached
    own-unit/child maps -- and monotone in the fan width that drives the
    measured cost (the KV-flood models' width lives in root own units).
    """

    analysis = analyze_collapse(trace)
    child_addresses = _child_address_map(trace)
    root_own = _root_own_unit_count(trace, context, analysis, child_addresses)
    width = root_own + len(child_addresses.get("self", ()))
    for address, children in child_addresses.items():
        if address != "self":
            width = max(width, len(children))
    return max(width, 1)


# _warn_budget_fallback + its warn-once set live in _collapse_disclosures
# (the human-facing collapse disclosure home).


def _budget_fallback_result(
    trace: Trace,
    context: RenderContext,
    source_graph: SourceGraph | None,
    mode: str,
    budget: SelectionBudget,
) -> OptimizerResult:
    """Build, stamp, and disclose the deterministic fallback result.

    ``budget.fired`` carries the cause: ``"admission"`` (estimator refusal)
    or ``"watchdog"``/``"allocation"`` (mid-run abandon).
    """

    cause = budget.fired or "admission"
    selected, repeat_folds, plan = linear_fallback_plan(
        trace,
        context,
        source_graph,
        _collapse_plan_for_source_or_trace,
        COLLAPSE_OPTIMIZER_MAX_OPS,
    )
    _assert_visible_plan(count(plan), "collapse budget fallback")
    analysis = analyze_collapse(trace)
    child_addresses = _child_address_map(trace)
    root_own_units = _root_own_unit_count(trace, context, analysis, child_addresses)
    result = OptimizerResult(
        selected=selected,
        repeat_folds=repeat_folds,
        plan=plan,
        visible_count=count(plan),
        analyze_ms=analysis.elapsed_ms,
        select_ms=(time.perf_counter() - budget.started_at) * 1000.0,
        g_star=None,
        reason=(
            f"linear_fallback ({cause}): estimator predicted "
            f"{budget.predicted_ms:.0f} ms at U={budget.universe_count}, "
            f"W={budget.max_sibling_width}; reduce the rendered graph with "
            "module= focus, vis_call_depth, or rolled mode"
        ),
        planner="linear_fallback",
        k_cap_exhausted=root_own_units > K_CAP,
        root_own_units=root_own_units,
        estimator=EstimatorDiagnostics(
            universe_count=budget.universe_count,
            max_sibling_width=budget.max_sibling_width,
            predicted_ms=budget.predicted_ms,
            actual_ms=(time.perf_counter() - budget.started_at) * 1000.0,
            tier="linear_fallback",
            budget_fired=budget.fired if cause != "admission" else None,
        ),
    )
    _warn_budget_fallback(trace, mode, result)
    return result


def _child_address_map(trace: Trace) -> dict[str, tuple[str, ...]]:
    """Return optimizer child addresses for every recorded module."""

    children_by_parent: dict[str, list[str]] = {
        module.address: [
            str(child)
            for child in getattr(module, "address_children", ()) or ()
            if child in trace.modules
        ]
        for module in trace.modules
    }
    for candidate in trace.modules:
        candidate_address = str(candidate.address)
        nearest = _nearest_recorded_parent(trace, candidate_address)
        if nearest is None or nearest not in children_by_parent:
            continue
        if candidate_address == nearest or candidate_address in children_by_parent[nearest]:
            continue
        children_by_parent[nearest].append(candidate_address)
    return {
        parent: tuple(dict.fromkeys(children)) for parent, children in children_by_parent.items()
    }


def _rendered_own_unit_map(trace: Trace, context: RenderContext) -> dict[str, tuple[str, ...]]:
    """Return rendered raw units owned directly by each optimizer module.

    Parameters
    ----------
    trace:
        Trace being optimized.
    context:
        Render context defining the raw rendered node universe.

    Returns
    -------
    dict[str, tuple[str, ...]]
        Rendered raw node labels keyed by their pass-free innermost module
        owner, with module-less boundary/raw nodes assigned to ``"self"``.
    """

    if context.vis_mode != "rolled":
        return {}
    from .node_universe import build_node_universe
    from .source_graph import build_source_graph

    units: dict[str, list[str]] = {"self": []}
    emissions = build_node_universe(build_source_graph(trace, context), None, None).emissions
    for emission in emissions:
        if emission.kind in {"hidden_run_member", "run_fold_ellipsis", "module_box"}:
            continue
        node = emission.node
        modules = list(getattr(node, "modules", ()) or ()) if node is not None else []
        if getattr(node, "is_atomic_module", False) and modules:
            modules = modules[:-1]
        owner = str(modules[-1]).rsplit(":", 1)[0] if modules else "self"
        units.setdefault(owner, []).append(emission.op_label or emission.name)
    return {address: tuple(dict.fromkeys(labels)) for address, labels in units.items()}


def _structural_digest_map(
    trace: Trace,
    child_addresses: Mapping[str, tuple[str, ...]],
    analysis: CollapseAnalysis,
) -> dict[str, str]:
    """Return renderer-tree structural digests for memo reuse."""

    depths = {address: str(address).count(".") for address in analysis.signals}
    digests: dict[str, str] = {}
    for address in sorted(depths, key=lambda item: (-depths[item], item)):
        module = cast("Module", trace.modules[address])
        signal = analysis.signals[address]
        child_parts = tuple(
            digests[child] for child in child_addresses.get(address, ()) if child in digests
        )
        payload = repr(
            (
                str(getattr(module, "class_name", "")),
                signal.structural_digest,
                signal.own_func_names,
                child_parts,
            )
        )
        digests[address] = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return digests


def _nearest_recorded_parent(trace: Trace, address: str) -> str | None:
    """Return the nearest recorded ancestor for ``address``."""

    parent = getattr(trace.modules[address], "address_parent", None)
    while parent is not None and parent not in trace.modules:
        if "." not in str(parent):
            return "self"
        parent = str(parent).rsplit(".", 1)[0]
    return cast(str | None, parent)


def _child_segments_for_parent(
    state: _OptimizerState,
    children: Sequence[str],
) -> tuple[tuple[str, ...], ...]:
    """Pre-segment long child lists at boundary-classifier cliffs."""

    child_key = tuple(children)
    cached = state.child_segments_cache.get(child_key)
    if cached is not None:
        return cached
    if not child_key:
        return ()
    if len(child_key) <= K_CAP:
        cached_segments = (child_key,)
        state.child_segments_cache[child_key] = cached_segments
        return cached_segments
    segment_lists: list[list[str]] = [[]]
    for child in child_key:
        if segment_lists[-1] and _boundary_cliff(state, segment_lists[-1][-1], child):
            segment_lists.append([])
        segment_lists[-1].append(child)
    result = tuple(tuple(segment) for segment in segment_lists if segment)
    state.child_segments_cache[child_key] = result
    return result


def _cached_output_shape_tuple(
    state: _OptimizerState,
    address: str,
    source: Literal["module", "boundary"],
) -> tuple[int, ...] | None:
    """Return one cached output-shape view for an optimizer run.

    Parameters
    ----------
    state:
        Optimizer state owning the plan-local cache.
    address:
        Pass-free module address.
    source:
        Shape metadata view required by the consuming classifier.

    Returns
    -------
    tuple[int, ...] | None
        Output shape as integers, or ``None`` when unavailable.
    """

    key = (source, address)
    if key not in state.output_shape_cache:
        shape = (
            _module_output_shape_tuple(state.trace, address)
            if source == "module"
            else _output_shape_tuple_for_address(state.trace, address)
        )
        state.output_shape_cache[key] = shape
    return state.output_shape_cache[key]


def _cached_module_output_shapes_equal(
    state: _OptimizerState,
    left: str,
    right: str,
) -> bool:
    """Return whether two cached primary module output shapes are equal.

    Parameters
    ----------
    state:
        Optimizer state owning the plan-local cache.
    left:
        First module address.
    right:
        Second module address.

    Returns
    -------
    bool
        True only when both shapes are known and exactly equal.
    """

    left_shape = _cached_output_shape_tuple(state, left, "module")
    right_shape = _cached_output_shape_tuple(state, right, "module")
    return left_shape is not None and left_shape == right_shape


def _cached_run_span_allows_fold(
    state: _OptimizerState,
    addresses: tuple[str, ...],
) -> bool:
    """Return whether a cached first-to-last shape span is safe to fold.

    Parameters
    ----------
    state:
        Optimizer state owning the plan-local cache.
    addresses:
        Consecutive sibling addresses in the candidate run.

    Returns
    -------
    bool
        True when the run preserves the existing shape-span constraints.
    """

    first = _cached_output_shape_tuple(state, addresses[0], "module")
    last = _cached_output_shape_tuple(state, addresses[-1], "module")
    if first is None or last is None:
        return True
    if len(first) != len(last):
        return False
    first_spatial = _shape_spatial_dims(first)
    last_spatial = _shape_spatial_dims(last)
    if first_spatial is not None and last_spatial is not None and first_spatial != last_spatial:
        return False
    first_channels = _shape_channel_dim(first)
    last_channels = _shape_channel_dim(last)
    if first_channels is None or last_channels is None:
        return True
    smaller = min(first_channels, last_channels)
    larger = max(first_channels, last_channels)
    return smaller > 0 and larger <= smaller * 2


def _boundary_cliff(state: _OptimizerState, left: str, right: str) -> bool:
    """Return whether adjacent children should split a long-parent segment."""

    if not _cached_run_span_allows_fold(state, (left, right)):
        return True
    left_shape = _cached_output_shape_tuple(state, left, "boundary")
    right_shape = _cached_output_shape_tuple(state, right, "boundary")
    if left_shape is None or right_shape is None:
        return False
    left_spatial = _shape_spatial_dims(left_shape)
    right_spatial = _shape_spatial_dims(right_shape)
    if left_spatial is not None and right_spatial is not None and left_spatial != right_spatial:
        return True
    left_channel = _shape_channel_dim(left_shape)
    right_channel = _shape_channel_dim(right_shape)
    if left_channel is None or right_channel is None:
        return False
    smaller = min(left_channel, right_channel)
    larger = max(left_channel, right_channel)
    return smaller <= 0 or larger / smaller >= 2.0


def _output_shape_tuple_for_address(trace: Trace, address: str) -> tuple[int, ...] | None:
    """Return module output shape tuple for long-parent boundary splitting."""

    pass_address = f"{address}:1"
    if pass_address not in trace.modules:
        return None
    layer = trace.modules[pass_address]
    shape = getattr(layer, "shape", None) or getattr(layer, "out_shape", None)
    if not shape:
        return None
    try:
        return tuple(int(dim) for dim in shape)
    except (TypeError, ValueError):
        return None


def _flow_adjacent(
    left: str,
    right: str,
    graph: ChildCondensedFlowGraph,
) -> bool:
    """Return whether two addresses are adjacent in parent flow order."""

    indexes = {address: index for index, address in enumerate(graph.flow_children)}
    return indexes.get(right) == indexes.get(left, -10) + 1


def _memo_key(state: _OptimizerState, signal: ModuleCollapseSignals) -> _MemoKey:
    """Return the R3a structural memo key for a signal."""

    landmark_bucket = min(signal.landmark_edges, 3)
    if state.context.vis_mode == "rolled":
        return _MemoKey(
            digest=state.structural_digests.get(signal.address, signal.structural_digest),
            landmark_bucket=landmark_bucket,
            trunk=_is_trunk_collapse(state.trace, signal),
            num_calls=int(signal.num_calls),
            rolled_mass=_faithful_hidden_count(signal, state.hidden_counts),
        )
    return _MemoKey(
        digest=state.structural_digests.get(signal.address, signal.structural_digest),
        landmark_bucket=landmark_bucket,
        trunk=_is_trunk_collapse(state.trace, signal),
    )


def _fold_mapping(folds: Sequence[ModuleRepeatFold]) -> dict[str, ModuleRepeatFold]:
    """Return renderer fold mapping keyed by every folded address."""

    mapping: dict[str, ModuleRepeatFold] = {}
    for fold in sorted(folds, key=lambda item: item.representative):
        if any(address in mapping for address in fold.addresses):
            continue
        for address in fold.addresses:
            mapping[address] = fold
    return mapping
