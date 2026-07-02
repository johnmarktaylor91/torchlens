"""Tree-cut dynamic-programming optimizer for v2 auto collapse."""

from __future__ import annotations

import hashlib
import math
import time
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import product
from typing import TYPE_CHECKING, Any, Literal, cast

from .auto_collapse import (
    GENERIC_CONTAINER_CLASSES,
    RUN_FOLD_MIN_LENGTH,
    ChildCondensedFlowGraph,
    CollapseAnalysis,
    ModuleCollapseSignals,
    ModuleRunFold,
    _flow_ordered_child_addresses,
    _is_trunk_collapse,
    _make_run_fold,
    _module_output_shapes_equal,
    _readable_band_high,
    _rendered_module_hidden_counts,
    _run_fold_is_chain_interval,
    _run_fold_is_legal,
    _run_span_allows_fold,
    _shape_channel_dim,
    _shape_spatial_dims,
    analyze_collapse,
)
from .collapse_plan import (
    ChildSegment,
    CollapsePlan,
    ModuleBox,
    OpSegment,
    PlanNode,
    RawOp,
    RenderContext,
    RunFold,
    SegmentDescriptor,
    count,
    plan_from_v1,
)
from .collapse_plan import EllipsisNode

if TYPE_CHECKING:
    from ..data_classes.module import Module
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace
    from .auto_collapse import ChildCondensedFlowGraph


K_CAP = 64
FRONTIER_CAP = 32


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
    run_folds:
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
    """

    selected: frozenset[str]
    run_folds: Mapping[str, ModuleRunFold]
    plan: CollapsePlan
    visible_count: int
    analyze_ms: float
    select_ms: float
    g_star: float | None
    declined: bool = False
    reason: str | None = None
    segments: Mapping[str, SegmentDescriptor] | None = None
    level: str | None = None


@dataclass(frozen=True)
class _FrontierPoint:
    """One DP frontier point for a subtree."""

    k: int
    cost: float
    nodes: tuple[PlanNode, ...]
    selected: frozenset[str]
    folds: tuple[ModuleRunFold, ...]
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
    segments: tuple["_SegmentDecision", ...] = ()


@dataclass(frozen=True)
class _SegmentDecision:
    """Memoized choices for one segmented child sequence."""

    components: tuple["_ComponentDecision", ...]


@dataclass(frozen=True)
class _ComponentDecision:
    """Memoized role-component treatment."""

    kind: Literal["boxes", "folded", "expanded"]
    member_ks: tuple[int, ...] = ()
    run_indices: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class _OptimizerState:
    """Immutable shared state for one g-star DP run."""

    trace: "Trace"
    context: RenderContext
    analysis: CollapseAnalysis
    child_addresses: Mapping[str, tuple[str, ...]]
    hidden_counts: Mapping[str, int]
    structural_digests: Mapping[str, str]
    expanded_cache: dict[
        str,
        tuple["ChildCondensedFlowGraph | None", tuple[str, ...], tuple[str, ...]],
    ]
    role_components_cache: dict[tuple[str, tuple[str, ...]], tuple[RoleComponent, ...]]
    child_segments_cache: dict[tuple[str, ...], tuple[tuple[str, ...], ...]]
    box_cost_cache: dict[str, float]
    branch_salience_cache: dict[str, float]
    weights: OptimizerWeights
    g_star: float
    total_ops: int
    allow_folds: bool
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
    object, dict[tuple[RenderContext, str], OptimizerResult]
] = weakref.WeakKeyDictionary()


def select_collapse_plan(
    trace: "Trace",
    context: RenderContext,
    weights: OptimizerWeights | None = None,
    mode: Literal["auto", "max"] = "auto",
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

    Returns
    -------
    OptimizerResult
        Selected collapse result, or a declined result when unsupported.
    """

    cache_key = (context, mode)
    cached_by_context = _RESULT_CACHE.setdefault(trace, {})
    cached = cached_by_context.get(cache_key)
    if cached is not None:
        return cached
    if mode == "max":
        result = _select_max_plan(trace, context, weights)
        cached_by_context[cache_key] = result
        return result
    resolved_weights = OptimizerWeights() if weights is None else weights
    analysis = analyze_collapse(trace)
    start = time.perf_counter()
    hidden_counts = _rendered_module_hidden_counts(trace, context)
    child_addresses = _child_address_map(trace)
    structural_digests = _structural_digest_map(trace, child_addresses, analysis)
    expanded_cache: dict[
        str,
        tuple["ChildCondensedFlowGraph | None", tuple[str, ...], tuple[str, ...]],
    ] = {}
    best = _select_best_decision(
        trace=trace,
        context=context,
        analysis=analysis,
        child_addresses=child_addresses,
        hidden_counts=hidden_counts,
        structural_digests=structural_digests,
        expanded_cache=expanded_cache,
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
            best=best,
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
            weights=replace(resolved_weights, fold_intrinsic=0.05),
            allow_folds=True,
        )
        first_pass_point = None
        first_pass_plan = None
    if best is None:
        plan = _floor_fallback_plan(trace, context)
        result = OptimizerResult(
            selected=frozenset(),
            run_folds={},
            plan=plan,
            visible_count=count(plan),
            analyze_ms=analysis.elapsed_ms,
            select_ms=(time.perf_counter() - start) * 1000.0,
            g_star=None,
            reason="floor_fallback: no optimizer frontier was produced",
        )
        cached_by_context[cache_key] = result
        return result
    if first_pass_point is None or first_pass_plan is None:
        instantiated_point, plan = _instantiate_best_point(
            trace=trace,
            context=context,
            analysis=analysis,
            child_addresses=child_addresses,
            hidden_counts=hidden_counts,
            structural_digests=structural_digests,
            expanded_cache=expanded_cache,
            best=best,
        )
    else:
        instantiated_point = first_pass_point
        plan = first_pass_plan
    _, winning_g, _, _, _, _, _ = best
    run_folds = _fold_mapping(instantiated_point.folds)
    assert count(plan) > 0, "v2 collapse plan produced no visible nodes"
    result = OptimizerResult(
        selected=instantiated_point.selected,
        run_folds=run_folds,
        plan=plan,
        visible_count=count(plan),
        analyze_ms=analysis.elapsed_ms,
        select_ms=(time.perf_counter() - start) * 1000.0,
        g_star=winning_g,
    )
    cached_by_context[cache_key] = result
    return result


def _select_max_plan(
    trace: "Trace",
    context: RenderContext,
    weights: OptimizerWeights | None,
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

    Returns
    -------
    OptimizerResult
        Max-mode result with segment descriptors, or an L3 auto fallback.
    """

    auto = select_collapse_plan(trace, context, weights, mode="auto")
    if auto.declined:
        return auto
    analysis = analyze_collapse(trace)
    hidden_counts = _rendered_module_hidden_counts(trace, context)
    total_ops = _optimizer_total_units(trace, context)
    auto_count = count(auto.plan)
    levels = (
        ("L0", 12, 0.60),
        ("L1", 12, 0.75),
        ("L2", min(20, auto_count), 0.75),
    )
    for level, k_hi, dominance_limit in levels:
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
                reason=None,
            )
    return replace(
        auto,
        segments={},
        level="L3",
        reason=f"fallback_to_auto: no legal max plan in [3,{min(20, auto_count)}]",
    )


def _condense_plan_with_child_segments(
    trace: "Trace",
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
    index = 0
    while index < len(plan.nodes):
        node = plan.nodes[index]
        if isinstance(node, RawOp) and isinstance(node.op, str) and node.op in hidden_raw_ops:
            index += 1
            continue
        op_run = _legal_plan_op_segment_run(
            trace,
            plan.nodes,
            index,
            total_ops,
            dominance_limit=dominance_limit,
        )
        if op_run:
            descriptor = _make_op_segment_descriptor(trace, context, op_run)
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
            hidden_raw_ops.update(covered_ops)
            nodes.append(ChildSegment(run))
            index += len(run)
            continue
        nodes.append(node)
        index += 1
    return CollapsePlan(nodes=tuple(nodes), context=context), segments


def _plan_respects_max_dominance(
    trace: "Trace",
    analysis: CollapseAnalysis,
    plan: CollapsePlan,
    segments: Mapping[str, SegmentDescriptor],
    dominance_limit: float,
) -> bool:
    """Return whether visible max boxes and segments obey the dominance cap."""

    total_ops = _optimizer_total_units(trace, plan.context)
    if total_ops <= 0:
        return True
    segment_by_member = {
        tuple(segment.members if segment.kind == "child" else segment.ops): segment
        for segment in segments.values()
    }
    for node in plan.nodes:
        if isinstance(node, ModuleBox):
            address = node.call.rsplit(":", 1)[0]
            signal = analysis.signals.get(address)
            if signal is not None and signal.hidden_ops / total_ops > dominance_limit:
                return False
        elif isinstance(node, ChildSegment):
            segment = segment_by_member.get(tuple(node.members))
            if segment is not None and segment.num_ops / total_ops > dominance_limit:
                return False
        elif isinstance(node, OpSegment):
            segment = segment_by_member.get(tuple(node.ops))
            if segment is not None and segment.num_ops / total_ops > dominance_limit:
                return False
    return True


def _legal_plan_op_segment_run(
    trace: "Trace",
    nodes: Sequence[PlanNode],
    start: int,
    total_ops: int,
    *,
    dominance_limit: float,
) -> tuple[str, ...] | None:
    """Return a legal consecutive raw-op segment run at ``start``.

    Parameters
    ----------
    trace:
        Trace being optimized.
    nodes:
        Plan-node sequence.
    start:
        Candidate start index.
    total_ops:
        Total rendered operation units.
    dominance_limit:
        Maximum segment dominance.

    Returns
    -------
    tuple[str, ...] | None
        Operation labels for a legal segment, or ``None``.
    """

    labels: list[str] = []
    op_by_label = {str(op.label).rsplit(":", 1)[0]: op for op in trace.ops}
    op_order = {str(op.label).rsplit(":", 1)[0]: index for index, op in enumerate(trace.ops)}
    previous_index: int | None = None
    for node in nodes[start:]:
        if not isinstance(node, RawOp) or not isinstance(node.op, str):
            break
        label = node.op
        current_index = op_order.get(label)
        if current_index is None:
            break
        op = op_by_label[label]
        if getattr(op, "is_input", False) or getattr(op, "is_output", False):
            break
        if previous_index is not None and current_index != previous_index + 1:
            break
        labels.append(label)
        previous_index = current_index
    if len(labels) < 3:
        return None
    if total_ops > 0:
        max_len = max(3, int(total_ops * dominance_limit))
        labels = labels[:max_len]
    return tuple(labels) if len(labels) >= 3 else None


def _legal_plan_child_segment_run(
    trace: "Trace",
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

    _ = context
    addresses: list[str] = []
    parent: str | None = None
    for node in nodes[start:]:
        address = _plan_module_address(node)
        if address is None:
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
    best: tuple[str, ...] | None = None
    for end in range(2, len(addresses) + 1):
        candidate = tuple(addresses[:end])
        if any(analysis.signals[address].landmark_edges >= 2 for address in candidate):
            continue
        if not _segment_is_legal(candidate, graph):
            continue
        hidden = _child_segment_hidden_units(analysis, candidate, hidden_counts)
        if total_ops > 0 and hidden / total_ops > dominance_limit:
            continue
        best = candidate
    return best


def _child_segment_hidden_units(
    analysis: CollapseAnalysis,
    addresses: tuple[str, ...],
    hidden_counts: Mapping[str, int],
) -> int:
    """Return rendered hidden-unit count for a candidate child segment."""

    covered_ops = _child_segment_covered_ops(analysis, addresses)
    if covered_ops:
        return len(covered_ops)
    return sum(
        hidden_counts.get(address, analysis.signals[address].hidden_ops) for address in addresses
    )


def _plan_module_address(node: PlanNode) -> str | None:
    """Return the pass-free module address represented by ``node`` if any."""

    if isinstance(node, ModuleBox):
        return node.call.rsplit(":", 1)[0]
    if isinstance(node, RunFold):
        return node.rep.call.rsplit(":", 1)[0]
    return None


def _segment_is_legal(
    addresses: tuple[str, ...],
    graph: "ChildCondensedFlowGraph | None",
) -> bool:
    """Return whether ``addresses`` satisfy chain-interval segment legality."""

    if len(addresses) < 2 or graph is None:
        return False
    return _run_fold_is_chain_interval(addresses, graph)


def _make_child_segment_descriptor(
    trace: "Trace",
    context: RenderContext,
    addresses: tuple[str, ...],
    covered_ops: tuple[str, ...],
) -> SegmentDescriptor:
    """Build a renderer descriptor for one child segment."""

    _ = context
    num_ops = len(covered_ops) or sum(
        int(getattr(trace.modules[address], "num_layers", 0) or 0) for address in addresses
    )
    num_params = sum(
        int(getattr(trace.modules[address], "num_params", 0) or 0) for address in addresses
    )
    owner = _segment_owner_key(trace, addresses)
    label = _child_segment_label(addresses, num_ops, num_params)
    name = f"{addresses[0].replace('.', '_')}__segment__{addresses[-1].replace('.', '_')}pass1"
    return SegmentDescriptor(
        name=name,
        kind="child",
        label=label,
        members=addresses,
        ops=covered_ops,
        owner=owner,
        num_ops=num_ops,
        num_params=num_params,
    )


def _child_segment_covered_ops(
    analysis: CollapseAnalysis,
    addresses: tuple[str, ...],
) -> tuple[str, ...]:
    """Return pass-free raw op labels covered by a child segment."""

    labels: list[str] = []
    for address in addresses:
        signal = analysis.signals.get(address)
        if signal is None:
            continue
        labels.extend(str(label).rsplit(":", 1)[0] for label in signal.subtree_ops)
    return tuple(dict.fromkeys(labels))


def _make_op_segment_descriptor(
    trace: "Trace",
    context: RenderContext,
    labels: tuple[str, ...],
) -> SegmentDescriptor:
    """Build a renderer descriptor for one operation segment."""

    _ = context
    owner = _op_segment_owner_key(trace, labels)
    label = _op_segment_label(labels)
    name = f"{labels[0].replace(':', 'pass')}__segment__{labels[-1].replace(':', 'pass')}pass1"
    return SegmentDescriptor(
        name=name,
        kind="op",
        label=label,
        ops=labels,
        owner=owner,
        num_ops=len(labels),
        num_params=0,
    )


def _op_segment_owner_key(trace: "Trace", labels: tuple[str, ...]) -> str | None:
    """Return the lowest common rendered module cluster for operation labels."""

    module_stacks: list[list[str]] = []
    for label in labels:
        op = _trace_op_for_render_label(trace, label)
        modules = [str(module) for module in getattr(op, "modules", ()) or ()]
        if getattr(op, "is_atomic_module", False) and modules:
            modules = modules[:-1]
        module_stacks.append(modules)
    if not module_stacks or any(not stack for stack in module_stacks):
        return None
    common: str | None = None
    for values in zip(*module_stacks, strict=False):
        pass_free = {value.rsplit(":", 1)[0] for value in values}
        if len(pass_free) != 1:
            break
        common = values[0]
    return common


def _trace_op_for_render_label(trace: "Trace", label: str) -> "Op":
    """Return the trace op for a pass-free rendered label."""

    try:
        return trace.ops[f"{label}:1"]
    except (KeyError, IndexError):
        return trace.ops[label]


def _op_segment_label(labels: tuple[str, ...]) -> str:
    """Return a class-free range label for an operation segment."""

    return f"{labels[0]} ... {labels[-1]} -- {len(labels)} ops"


def _segment_owner_key(trace: "Trace", addresses: tuple[str, ...]) -> str | None:
    """Return the lowest rendered module cluster that owns ``addresses``."""

    parent = addresses[0].rsplit(".", 1)[0] if "." in addresses[0] else "self"
    if parent == "self" or parent not in trace.modules:
        return None
    parent_key = f"{parent}:1"
    return parent_key if parent_key in trace.modules else parent


def _child_segment_label(addresses: tuple[str, ...], num_ops: int, num_params: int) -> str:
    """Return a class-free range label for a child segment."""

    first = addresses[0]
    last = addresses[-1]
    prefix = first.rsplit(".", 1)[0] if "." in first else ""
    first_leaf = first.rsplit(".", 1)[-1]
    last_leaf = last.rsplit(".", 1)[-1]
    range_text = f"{prefix}.{first_leaf}-{last_leaf}" if prefix else f"{first_leaf}-{last_leaf}"
    return f"{range_text} -- {len(addresses)} blocks, {num_ops} ops, {_format_param_count(num_params)} params"


def _format_param_count(value: int) -> str:
    """Return a compact parameter count for segment labels."""

    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def _instantiate_best_point(
    trace: "Trace",
    context: RenderContext,
    analysis: CollapseAnalysis,
    child_addresses: Mapping[str, tuple[str, ...]],
    hidden_counts: Mapping[str, int],
    structural_digests: Mapping[str, str],
    expanded_cache: dict[
        str,
        tuple["ChildCondensedFlowGraph | None", tuple[str, ...], tuple[str, ...]],
    ],
    best: tuple[
        float,
        float,
        int,
        _DecisionPoint,
        dict[_MemoKey, tuple[_DecisionPoint, ...]],
        OptimizerWeights,
        bool,
    ],
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
    best:
        Winner tuple from :func:`_select_best_decision`.

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
        box_cost_cache={},
        branch_salience_cache={},
        weights=winning_weights,
        g_star=winning_g,
        total_ops=_optimizer_total_units(trace, context),
        allow_folds=allow_folds,
        rendered_own_units=_rendered_own_unit_map(trace, context),
    )
    instantiated_point = _instantiate_module("self", decision_point.k, winning_state, winning_memo)
    run_folds = _fold_mapping(instantiated_point.folds)
    collapse_fn = _collapse_fn_from_selected(instantiated_point.selected)
    plan = plan_from_v1(trace, collapse_fn, run_folds, context)
    return instantiated_point, plan


def _collapse_fn_from_selected(selected: frozenset[str]) -> Callable[["Module"], bool]:
    """Return a module-collapse predicate for selected addresses."""

    def collapse_fn(module: "Module") -> bool:
        """Return whether ``module`` is selected."""

        return module.address in selected

    return collapse_fn


def _select_best_decision(
    trace: "Trace",
    context: RenderContext,
    analysis: CollapseAnalysis,
    child_addresses: Mapping[str, tuple[str, ...]],
    hidden_counts: Mapping[str, int],
    structural_digests: Mapping[str, str],
    expanded_cache: dict[
        str,
        tuple["ChildCondensedFlowGraph | None", tuple[str, ...], tuple[str, ...]],
    ],
    weights: OptimizerWeights,
    allow_folds: bool,
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
    """Return the best global decision for one fold-gating pass.

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
            box_cost_cache={},
            branch_salience_cache={},
            weights=weights,
            g_star=g_star,
            total_ops=total_units,
            allow_folds=allow_folds,
            rendered_own_units=rendered_own_units,
        )
        memo: dict[_MemoKey, tuple[_DecisionPoint, ...]] = {}
        frontier = _frontier_for_module("self", state, memo)
        for point in frontier:
            score = _global_q(point, trace, weights)
            if context.vis_mode == "rolled" and not point.box_costs and hidden_counts:
                score = round(score + 10.0, 6)
            candidate = (score, g_star, point.k, point, memo, weights, allow_folds)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    return best


def build_role_components(
    trace: "Trace",
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
    parent_index = {child: index for index, child in enumerate(children)}
    parent = {child: child for child in children}

    def find(address: str) -> str:
        """Return the union-find representative for ``address``."""

        current = address
        while parent[current] != current:
            parent[current] = parent[parent[current]]
            current = parent[current]
        return current

    def union(left: str, right: str) -> None:
        """Union two child addresses, preserving earliest flow representative."""

        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if parent_index[left_root] <= parent_index[right_root]:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    for left_index, left in enumerate(children):
        for right in children[left_index + 1 :]:
            if _same_role(trace, left, right, resolved_analysis, hidden_counts or {}):
                union(left, right)
    grouped: dict[str, list[str]] = {}
    for child in children:
        grouped.setdefault(find(child), []).append(child)
    return tuple(
        RoleComponent(tuple(members))
        for _, members in sorted(
            grouped.items(),
            key=lambda item: min(parent_index[member] for member in item[1]),
        )
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


def _declined_result(context: RenderContext, reason: str) -> OptimizerResult:
    """Return a declined optimizer result."""

    return OptimizerResult(
        selected=frozenset(),
        run_folds={},
        plan=CollapsePlan(nodes=(), context=context),
        visible_count=0,
        analyze_ms=0.0,
        select_ms=0.0,
        g_star=None,
        declined=True,
        reason=reason,
    )


def _floor_fallback_plan(trace: "Trace", context: RenderContext) -> CollapsePlan:
    """Return the conservative full-op plan used when the DP frontier is empty.

    Parameters
    ----------
    trace:
        Trace being optimized.
    context:
        Rendering context.

    Returns
    -------
    CollapsePlan
        Renderer-faithful uncollapsed plan. This is intentionally the safest
        legal fallback: it preserves all visible structure instead of returning
        a zero-node cut.
    """

    plan = plan_from_v1(trace, None, None, context)
    assert count(plan) > 0, "collapse floor fallback produced no visible nodes"
    return plan


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
    if address != "self" and _eligible_box(state.trace, address, signal):
        box_cost = _cached_box_cost(state.trace, signal, state)
        points.append(
            _DecisionPoint(
                k=1,
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


def _expanded_points(
    address: str,
    state: _OptimizerState,
    memo: dict[_MemoKey, tuple[_DecisionPoint, ...]],
) -> tuple[_DecisionPoint, ...]:
    """Return expanded-frontier points for one module."""

    graph, child_addresses, own_ops = _expanded_structure(address, state)
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
) -> tuple["ChildCondensedFlowGraph | None", tuple[str, ...], tuple[str, ...]]:
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
        parent_owned_ops=(),
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
    graph: "ChildCondensedFlowGraph | None",
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
    graph: "ChildCondensedFlowGraph | None",
    state: _OptimizerState,
    memo: dict[_MemoKey, tuple[_DecisionPoint, ...]],
) -> tuple[_DecisionPoint, ...]:
    """Enumerate BOXES, EXPANDED, and FOLDED treatments for one role component."""

    choices: list[_DecisionPoint] = []
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


def _component_boxes(component: RoleComponent, state: _OptimizerState) -> _DecisionPoint | None:
    """Return the all-boxes treatment for a component when legal."""

    costs: list[float] = []
    for address in component.members:
        signal = state.analysis.signals[address]
        if not _eligible_box(state.trace, address, signal):
            return None
        cost = _cached_box_cost(state.trace, signal, state)
        costs.append(cost)
    return _DecisionPoint(
        k=len(component.members),
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
            if not (
                point.k == 1
                and isinstance(point.decision, _ModuleDecision)
                and point.decision.kind == "box"
            )
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

    points: list[_DecisionPoint] = []
    for point in _frontier_for_module(address, state, memo):
        if (
            point.k == 1
            and isinstance(point.decision, _ModuleDecision)
            and point.decision.kind == "box"
        ):
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
    return tuple(points)


def _component_folded(
    component: RoleComponent,
    graph: "ChildCondensedFlowGraph | None",
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
            if not _eligible_box(state.trace, address, signal):
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
    graph: "ChildCondensedFlowGraph",
    state: _OptimizerState,
) -> tuple[tuple[str, ...], ...]:
    """Partition component members into maximal legal fold runs."""

    runs: list[tuple[str, ...]] = []
    index = 0
    while index < len(members):
        best: tuple[str, ...] = ()
        candidate: list[str] = []
        for address in members[index:]:
            if candidate and not _flow_adjacent(candidate[-1], address, graph):
                break
            if not _eligible_box(state.trace, address, state.analysis.signals[address]):
                break
            if candidate and not _module_output_shapes_equal(state.trace, candidate[-1], address):
                break
            candidate.append(address)
            run = tuple(candidate)
            if len(run) >= RUN_FOLD_MIN_LENGTH and _run_fold_is_legal(run, graph):
                best = run
        if best:
            runs.append(best)
            index += len(best)
        else:
            index += 1
    return tuple(runs)


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
        return _FrontierPoint(
            k=1,
            cost=box_cost,
            nodes=(ModuleBox(f"{address}:1"),),
            selected=frozenset({address}),
            folds=(),
            box_costs=(box_cost,),
        )
    graph, child_addresses, own_ops = _expanded_structure(address, state)
    nodes: list[PlanNode] = [RawOp(op) for op in own_ops]
    selected: set[str] = set()
    folds: list[ModuleRunFold] = []
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
    graph: "ChildCondensedFlowGraph | None",
    decision: _SegmentDecision,
    state: _OptimizerState,
    memo: dict[_MemoKey, tuple[_DecisionPoint, ...]],
) -> _FrontierPoint:
    """Instantiate one segmented child sequence."""

    components = _role_components_for_children(state, parent_address, child_addresses)
    nodes: list[PlanNode] = []
    selected: set[str] = set()
    folds: list[ModuleRunFold] = []
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
    graph: "ChildCondensedFlowGraph | None",
    decision: _ComponentDecision,
    state: _OptimizerState,
    memo: dict[_MemoKey, tuple[_DecisionPoint, ...]],
) -> _FrontierPoint:
    """Instantiate a concrete role-component treatment."""

    if decision.kind == "boxes":
        return _instantiate_component_boxes(component, state)
    if decision.kind == "expanded":
        return _instantiate_component_expanded(component, decision, state, memo)
    return _instantiate_component_folded(component, graph, decision, state)


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
        nodes.append(ModuleBox(f"{address}:1"))
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
    folds: list[ModuleRunFold] = []
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
    graph: "ChildCondensedFlowGraph | None",
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
    folds: list[ModuleRunFold] = []
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
        nodes.append(
            RunFold(
                rep=ModuleBox(f"{fold.representative}:1"),
                members=fold.addresses,
                ellipsis=EllipsisNode(fold.addresses[1:]),
            )
        )
        selected.update(run)
        folds.append(fold)
        box_costs.append(fold_cost)
        total_cost += fold_cost
        total_k += 2
        skipped.update(indices)
    return _FrontierPoint(
        k=total_k,
        cost=round(total_cost, 6),
        nodes=tuple(nodes),
        selected=frozenset(selected),
        folds=tuple(folds),
        box_costs=tuple(box_costs),
    )


def _merge_frontiers(
    left_points: Sequence[_FrontierPoint],
    right_points: Sequence[_FrontierPoint],
) -> tuple[_FrontierPoint, ...]:
    """Merge two frontier sequences by standard node-count knapsack."""

    merged: list[_FrontierPoint] = []
    for left, right in product(left_points, right_points):
        k = left.k + right.k
        if k > K_CAP:
            continue
        merged.append(
            _FrontierPoint(
                k=k,
                cost=round(left.cost + right.cost, 6),
                nodes=(*left.nodes, *right.nodes),
                selected=frozenset((*left.selected, *right.selected)),
                folds=(*left.folds, *right.folds),
                box_costs=(*left.box_costs, *right.box_costs),
            )
        )
    return _prune_frontier(merged)


def _merge_module_segment_frontiers(
    left_points: Sequence[_DecisionPoint],
    right_points: Sequence[_DecisionPoint],
) -> tuple[_DecisionPoint, ...]:
    """Merge module expand points with child-segment decisions."""

    merged: list[_DecisionPoint] = []
    for left, right in product(left_points, right_points):
        k = left.k + right.k
        if k > K_CAP:
            continue
        left_decision = cast(_ModuleDecision, left.decision)
        right_decision = cast(_SegmentDecision, right.decision)
        merged.append(
            _DecisionPoint(
                k=k,
                cost=round(left.cost + right.cost, 6),
                decision=_ModuleDecision(
                    "expand",
                    (*left_decision.segments, right_decision),
                ),
                box_costs=(*left.box_costs, *right.box_costs),
                priority=(*left.priority, *right.priority),
            )
        )
    return _prune_frontier(merged)


def _merge_segment_component_frontiers(
    left_points: Sequence[_DecisionPoint],
    right_points: Sequence[_DecisionPoint],
) -> tuple[_DecisionPoint, ...]:
    """Merge segment points with one role-component treatment frontier."""

    merged: list[_DecisionPoint] = []
    for left, right in product(left_points, right_points):
        k = left.k + right.k
        if k > K_CAP:
            continue
        left_decision = cast(_SegmentDecision, left.decision)
        right_decision = cast(_ComponentDecision, right.decision)
        merged.append(
            _DecisionPoint(
                k=k,
                cost=round(left.cost + right.cost, 6),
                decision=_SegmentDecision((*left_decision.components, right_decision)),
                box_costs=(*left.box_costs, *right.box_costs),
                priority=(*left.priority, *right.priority),
            )
        )
    return _prune_frontier(merged)


def _merge_component_member_frontiers(
    left_points: Sequence[_DecisionPoint],
    right_points: Sequence[_DecisionPoint],
) -> tuple[_DecisionPoint, ...]:
    """Merge expanded role-component points with one member frontier."""

    merged: list[_DecisionPoint] = []
    for left, right in product(left_points, right_points):
        k = left.k + right.k
        if k > K_CAP:
            continue
        left_decision = cast(_ComponentDecision, left.decision)
        merged.append(
            _DecisionPoint(
                k=k,
                cost=round(left.cost + right.cost, 6),
                decision=_ComponentDecision(
                    "expanded",
                    (*left_decision.member_ks, right.k),
                ),
                box_costs=(*left.box_costs, *right.box_costs),
                priority=(*left.priority, *right.priority),
            )
        )
    return _prune_frontier(merged)


def _prune_frontier(points: Sequence[Any]) -> tuple[Any, ...]:
    """Keep deterministic Pareto frontier points, capped for R3a."""

    best_by_count: dict[int, Any] = {}
    for point in points:
        if point.k > K_CAP:
            continue
        incumbent = best_by_count.get(point.k)
        if incumbent is None or _point_sort_key(point) < _point_sort_key(incumbent):
            best_by_count[point.k] = point
    ordered = sorted(best_by_count.values(), key=_point_sort_key)
    return tuple(sorted(ordered[:FRONTIER_CAP], key=lambda point: point.k))


def _point_sort_key(point: Any) -> tuple[float, int, tuple[Any, ...], tuple[Any, ...]]:
    """Return deterministic ordering key for frontier points."""

    if isinstance(point, _DecisionPoint):
        return (point.cost, point.k, point.priority, ())
    fold_addresses = tuple(fold.representative for fold in point.folds)
    return (point.cost, point.k, tuple(sorted(point.selected)), fold_addresses)


def _eligible_box(trace: "Trace", address: str, signal: ModuleCollapseSignals) -> bool:
    """Return whether a module may render as a collapsed box."""

    _ = trace, address
    return signal.eligible and signal.landmark_edges < 2


def _box_cost(trace: "Trace", signal: ModuleCollapseSignals, state: _OptimizerState) -> float:
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
    trace: "Trace",
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
    child_edges = {
        (source, target)
        for source, target in graph.edges
        if source in child_set and target in child_set
    }
    upstreams: dict[str, set[str]] = {child: set() for child in graph.flow_children}
    downstreams: dict[str, set[str]] = {child: set() for child in graph.flow_children}
    for source, target in graph.edges:
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
        independent: list[str] = []
        for child in members:
            if all(
                (child, other) not in child_edges and (other, child) not in child_edges
                for other in independent
            ):
                independent.append(child)
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
    point: _DecisionPoint | _FrontierPoint, trace: "Trace", weights: OptimizerWeights
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


def _k_preference(k: int, trace: "Trace") -> float:
    """Return linear node-count preference normalized to the readable band."""

    lo = 8
    hi = _readable_band_high(trace)
    return (k - lo) / max(hi - lo, 1)


def _band_cost(k: int, trace: "Trace") -> float:
    """Return quadratic node-band cost for auto mode."""

    lo = 8
    hi = _readable_band_high(trace)
    target = min(28, hi - 2)
    width = max((hi - lo) / 2.0, 1.0)
    slope = 2.0 if k < lo or k > hi else 1.0
    return slope * ((k - target) / width) ** 2


def _g_star_candidates(
    trace: "Trace",
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
    trace: "Trace",
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


def _optimizer_total_units(trace: "Trace", context: RenderContext) -> int:
    """Return the raw rendered universe size used for optimizer normalization."""

    if context.vis_mode != "rolled":
        return max(len(trace.ops), 1)
    return max(count(plan_from_v1(trace, None, None, context)), 1)


def _child_address_map(trace: "Trace") -> dict[str, tuple[str, ...]]:
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


def _rendered_own_unit_map(trace: "Trace", context: RenderContext) -> dict[str, tuple[str, ...]]:
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
    from .rendering import rendered_node_universe_from_v1

    units: dict[str, list[str]] = {"self": []}
    emissions = rendered_node_universe_from_v1(
        trace,
        collapse_fn=None,
        run_folds=None,
        context=context,
    )
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
    trace: "Trace",
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


def _nearest_recorded_parent(trace: "Trace", address: str) -> str | None:
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
        if segment_lists[-1] and _boundary_cliff(state.trace, segment_lists[-1][-1], child):
            segment_lists.append([])
        segment_lists[-1].append(child)
    result = tuple(tuple(segment) for segment in segment_lists if segment)
    state.child_segments_cache[child_key] = result
    return result


def _boundary_cliff(trace: "Trace", left: str, right: str) -> bool:
    """Return whether adjacent children should split a long-parent segment."""

    if not _run_span_allows_fold(trace, (left, right)):
        return True
    left_shape = _output_shape_tuple_for_address(trace, left)
    right_shape = _output_shape_tuple_for_address(trace, right)
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


def _output_shape_tuple_for_address(trace: "Trace", address: str) -> tuple[int, ...] | None:
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
    graph: "ChildCondensedFlowGraph",
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


def _fold_mapping(folds: Sequence[ModuleRunFold]) -> dict[str, ModuleRunFold]:
    """Return renderer fold mapping keyed by every folded address."""

    mapping: dict[str, ModuleRunFold] = {}
    for fold in sorted(folds, key=lambda item: item.representative):
        if any(address in mapping for address in fold.addresses):
            continue
        for address in fold.addresses:
            mapping[address] = fold
    return mapping
