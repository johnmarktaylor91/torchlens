"""Smart module-collapse scoring and selection for graph rendering."""

from __future__ import annotations

import hashlib
import math
import os
import re
import time
import weakref
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from .._literals import CollapseLiteral, FoldRunsLiteral, VisModeLiteral
from .collapse_plan import RenderContext, count, plan_from_v1

if TYPE_CHECKING:
    from ..data_classes.module import Module
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace


GENERIC_CONTAINER_CLASSES = frozenset({"Sequential", "ModuleList", "ModuleDict", "ParameterList"})
_COUNT_MISMATCH_WARNING_EMITTED = False
JUNCTION_FUNC_NAMES = frozenset({"__add__", "add", "cat", "concat", "concatenate"})
_INDEXED_CHILD_RE = re.compile(r"^(?P<stem>.*?)(?:\.?\d+|_?\d+[a-z]?)$")
RUN_FOLD_MIN_LENGTH = 3


@dataclass(frozen=True)
class ModuleRunFold:
    """Consecutive structurally-identical module run selected for render folding.

    Parameters
    ----------
    representative:
        First module address in the folded run.
    addresses:
        Consecutive sibling module addresses included in the run.
    class_name:
        Module class name shared by the folded run (named in the elision label).
    num_layers:
        Aggregate recursive layer count across the run.
    num_params:
        Aggregate recursive parameter count across the run.
    num_params_trainable:
        Aggregate trainable parameter count across the run.
    num_params_frozen:
        Aggregate frozen parameter count across the run.
    shape_summary:
        Short first-to-last output-shape summary when shapes vary, else ``None``.
    hidden_member_composition:
        Metadata describing hidden members represented by the ellipsis.
    """

    representative: str
    addresses: tuple[str, ...]
    class_name: str
    num_layers: int
    num_params: int
    num_params_trainable: int
    num_params_frozen: int
    shape_summary: str | None
    hidden_member_composition: Mapping[str, int]

    @property
    def multiplicity(self) -> int:
        """Return the number of folded sibling modules."""

        return len(self.addresses)


@dataclass(frozen=True)
class ModuleCollapseSignals:
    """Precomputed structural signals for one module.

    Parameters
    ----------
    address:
        Primary module address.
    subtree_ops:
        Pass-qualified operation labels in module scope.
    own_func_names:
        Function names for ops directly owned by the module, in call order.
    internal_edges:
        Distinct op-graph edges with both endpoints in the module.
    input_edges:
        Distinct op-graph edges entering the module from outside.
    output_edges:
        Distinct op-graph edges leaving the module.
    landmark_edges:
        Boundary-crossing edges that enter or leave non-boundary internal
        operations and therefore hide a meaningful cross-module junction.
    passthrough_edges:
        Internal output junctions that combine module input with internal work.
    output_junctions:
        External multi-parent children fed by module outputs.
    params:
        Number of recursive parameters for the module.
    depth:
        Address-tree depth.
    num_calls:
        Number of module calls.
    structural_digest:
        Trace-local structural digest.
    peer_count:
        Number of modules in the same address-keyed peer group.
    hidden_ops:
        Rendered op count hidden by collapsing this module.
    eligible:
        Whether renderer-faithful hard gating allows collapse.
    """

    address: str
    subtree_ops: tuple[str, ...]
    own_func_names: tuple[str, ...]
    internal_edges: int
    input_edges: int
    output_edges: int
    landmark_edges: int
    passthrough_edges: int
    output_junctions: tuple[str, ...]
    params: int
    depth: int
    num_calls: int
    structural_digest: str
    peer_count: int
    hidden_ops: int
    eligible: bool


@dataclass(frozen=True)
class FlowIntervalFlags:
    """Blocker flags for an interval between flow-adjacent children.

    Parameters
    ----------
    landmark:
        Whether landmark edges cross the interval.
    passthrough:
        Whether passthrough-style parent-owned flow crosses the interval.
    """

    landmark: bool
    passthrough: bool


@dataclass(frozen=True)
class ChildCondensedFlowGraph:
    """Child-condensed flow graph for one parent module call.

    Parameters
    ----------
    parent:
        Parent module address.
    flow_children:
        Direct child module addresses ordered by first executed op.
    parent_owned_ops:
        Parent-owned operation labels in execution order.
    nodes:
        Condensed graph nodes: child subtrees plus parent-owned ops.
    edges:
        Condensed op-flow edges between nodes.
    child_external_endpoint_counts:
        Per-child ``(entries, exits)`` counts against the condensed graph.
    interval_flags:
        Flags keyed by flow-child address pairs.
    """

    parent: str
    flow_children: tuple[str, ...]
    parent_owned_ops: tuple[str, ...]
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    child_external_endpoint_counts: Mapping[str, tuple[int, int]]
    interval_flags: Mapping[tuple[str, str], FlowIntervalFlags]


@dataclass(frozen=True)
class CollapseAnalysis:
    """Trace-local module-collapse analysis.

    Parameters
    ----------
    signals:
        Signals keyed by module address.
    scores:
        Canonical rounded scores keyed by module address.
    peer_groups:
        Structural or relaxed peer groups keyed by signature and sibling parent
        address.
    child_flow_graphs:
        Child-condensed flow graph artifacts keyed by parent module address.
    elapsed_ms:
        Signal and score computation time in milliseconds.
    """

    signals: Mapping[str, ModuleCollapseSignals]
    scores: Mapping[str, float]
    peer_groups: Mapping[tuple[str, str | None], tuple[str, ...]]
    child_flow_graphs: Mapping[str, ChildCondensedFlowGraph]
    elapsed_ms: float


_ANALYSIS_CACHE: weakref.WeakKeyDictionary[Any, CollapseAnalysis] = weakref.WeakKeyDictionary()


def analyze_collapse(trace: "Trace") -> CollapseAnalysis:
    """Return cached module-collapse signals and canonical scores for ``trace``.

    Parameters
    ----------
    trace:
        Trace to analyze.

    Returns
    -------
    CollapseAnalysis
        Cached signal, digest, peer, and score data.
    """

    cached = _ANALYSIS_CACHE.get(trace)
    if cached is not None:
        return cached
    start = time.perf_counter()
    signals_without_peers = _compute_signal_skeleton(trace)
    digests = _compute_structural_digests(trace, signals_without_peers)
    peer_groups = _group_structural_peers(trace, digests)
    child_flow_graphs = _compute_child_condensed_flow_graphs(trace, signals_without_peers)
    peer_count_by_address: dict[str, int] = {}
    for group in peer_groups.values():
        for address in group:
            peer_count_by_address[address] = max(peer_count_by_address.get(address, 1), len(group))
    signals = {
        address: ModuleCollapseSignals(
            address=signal.address,
            subtree_ops=signal.subtree_ops,
            own_func_names=signal.own_func_names,
            internal_edges=signal.internal_edges,
            input_edges=signal.input_edges,
            output_edges=signal.output_edges,
            landmark_edges=signal.landmark_edges,
            passthrough_edges=signal.passthrough_edges,
            output_junctions=signal.output_junctions,
            params=signal.params,
            depth=signal.depth,
            num_calls=signal.num_calls,
            structural_digest=digests[address],
            peer_count=peer_count_by_address.get(address, 1),
            hidden_ops=signal.hidden_ops,
            eligible=signal.eligible,
        )
        for address, signal in signals_without_peers.items()
    }
    scores = _signal_size_scores(signals)
    analysis = CollapseAnalysis(
        signals=signals,
        scores=scores,
        peer_groups=peer_groups,
        child_flow_graphs=child_flow_graphs,
        elapsed_ms=(time.perf_counter() - start) * 1000.0,
    )
    _ANALYSIS_CACHE[trace] = analysis
    return analysis


def _child_condensed_flow_graphs(trace: "Trace") -> Mapping[str, ChildCondensedFlowGraph]:
    """Return cached child-condensed flow graphs for tests and v2 downstream work.

    Parameters
    ----------
    trace:
        Trace to analyze.

    Returns
    -------
    Mapping[str, ChildCondensedFlowGraph]
        Flow graph artifacts keyed by parent module address.
    """

    return analyze_collapse(trace).child_flow_graphs


def collapse_order(
    trace: "Trace",
    weights: Mapping[str, float] | None = None,
    mode: Literal["auto", "max"] = "auto",
) -> list[tuple[str, float]]:
    """Return v2 collapse diagnostics sorted by score for a policy.

    Parameters
    ----------
    trace:
        Trace to rank.
    weights:
        Ignored legacy parameter retained for call compatibility.
    mode:
        ``"auto"`` or ``"max"`` landmark policy.

    Returns
    -------
    list[tuple[str, float]]
        ``(module_address, rounded_score)`` sorted by ``(-score, address)``.
    """

    _ = weights
    if mode not in {"auto", "max"}:
        raise ValueError("mode must be 'auto' or 'max'.")
    analysis = analyze_collapse(trace)
    scores = _v2_selected_module_scores(trace, analysis, mode=mode)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _signal_size_scores(signals: Mapping[str, ModuleCollapseSignals]) -> dict[str, float]:
    """Return non-selector signal-size diagnostics for cached analysis."""

    max_hidden = max((signal.hidden_ops for signal in signals.values()), default=0)
    if max_hidden <= 0:
        return {address: 0.0 for address in signals}
    return {
        address: round(signal.hidden_ops / max_hidden, 6) if signal.eligible else 0.0
        for address, signal in signals.items()
    }


def _v2_selected_module_scores(
    trace: "Trace",
    analysis: CollapseAnalysis,
    *,
    mode: Literal["auto", "max"],
) -> dict[str, float]:
    """Return same-shape scores derived from the v2 selected module set."""

    from .collapse_optimizer import select_collapse_plan

    result = select_collapse_plan(trace, RenderContext(), mode=mode)
    selected = result.selected if not result.declined else frozenset()
    hidden_max = max(
        (
            analysis.signals[address].hidden_ops
            for address in selected
            if address in analysis.signals
        ),
        default=0,
    )
    scores = {address: 0.0 for address in analysis.signals}
    for address in selected:
        signal = analysis.signals.get(address)
        if signal is None or hidden_max <= 0:
            continue
        scores[address] = round(max(signal.hidden_ops / hidden_max, 1e-6), 6)
    return scores


def resolve_collapse_fn(
    trace: "Trace",
    collapse: CollapseLiteral,
    vis_mode: VisModeLiteral,
    context: RenderContext | None = None,
) -> Callable[["Module"], bool] | None:
    """Resolve a public collapse option to a renderer predicate.

    Parameters
    ----------
    trace:
        Trace being rendered.
    collapse:
        Public collapse mode.
    vis_mode:
        Current visualization mode.
    context:
        Render context for v2 instrumentation. Defaults preserve v1 behavior.

    Returns
    -------
    Callable[[Module], bool] | None
        Collapse predicate, or ``None`` for ``"none"``.
    """

    resolved_context = RenderContext(vis_mode=vis_mode) if context is None else context
    if isinstance(collapse, float):
        if not 0.0 <= collapse <= 1.0:
            raise ValueError("collapse float level must be in [0.0, 1.0].")
        if collapse == 0.0:
            return None
        from .collapse_optimizer import select_collapse_level

        result = select_collapse_level(trace, resolved_context, collapse)
        if not result.declined:

            def v2_collapse_fn(module: "Module") -> bool:
                """Return whether ``module`` is selected by the v2 optimizer."""

                return module.address in result.selected

            setattr(v2_collapse_fn, "_torchlens_v2_run_folds", result.run_folds)
            setattr(v2_collapse_fn, "_torchlens_v2_segments", result.segments or {})
            setattr(v2_collapse_fn, "_torchlens_v2_plan", result.plan)
            setattr(v2_collapse_fn, "_torchlens_v2_result", result)
            setattr(v2_collapse_fn, "_torchlens_v2_mode", "level")
            return v2_collapse_fn
    if collapse == "none":
        return None
    if collapse not in {"auto", "max"}:
        raise ValueError("collapse must be 'none', 'auto', 'max', or a float in [0.0, 1.0].")
    if collapse in {"auto", "max"}:
        from .collapse_optimizer import select_collapse_plan

        result = select_collapse_plan(trace, resolved_context, mode=collapse)
        if not result.declined:

            def v2_collapse_fn(module: "Module") -> bool:
                """Return whether ``module`` is selected by the v2 optimizer."""

                return module.address in result.selected

            setattr(v2_collapse_fn, "_torchlens_v2_run_folds", result.run_folds)
            setattr(v2_collapse_fn, "_torchlens_v2_segments", result.segments or {})
            setattr(v2_collapse_fn, "_torchlens_v2_plan", result.plan)
            setattr(v2_collapse_fn, "_torchlens_v2_result", result)
            setattr(v2_collapse_fn, "_torchlens_v2_mode", collapse)
            return v2_collapse_fn
    return None


def resolve_run_folds(
    trace: "Trace",
    collapse_fn: Callable[["Module"], bool] | None,
    context: RenderContext | None = None,
    fold_runs: FoldRunsLiteral = None,
) -> dict[str, ModuleRunFold]:
    """Return render-time folds for consecutive collapsed sibling runs.

    Parameters
    ----------
    trace:
        Trace being rendered.
    collapse_fn:
        Active collapse predicate. ``None`` disables run folding.
    context:
        Render context for v2 instrumentation. Defaults preserve v1 behavior.
    fold_runs:
        Run-fold policy override. ``None`` preserves the current band-gated
        policy, ``True`` folds every eligible repeated run, and ``False``
        disables run folding.

    Returns
    -------
    dict[str, ModuleRunFold]
        Mapping from each folded module address to its run descriptor.
    """

    resolved_context = RenderContext() if context is None else context
    if fold_runs not in {None, True, False}:
        raise ValueError("fold_runs must be None, True, or False.")
    if fold_runs is False:
        return {}
    if collapse_fn is None and fold_runs is not True:
        return {}
    eligibility_collapse_fn = collapse_fn if collapse_fn is not None else _always_collapse_module
    render_collapse_fn = collapse_fn
    v2_run_folds = getattr(collapse_fn, "_torchlens_v2_run_folds", None)
    if fold_runs is None and v2_run_folds is not None:
        return dict(v2_run_folds)
    projected_count = count(plan_from_v1(trace, render_collapse_fn, None, resolved_context))
    if fold_runs is None and projected_count <= _readable_band_high(trace):
        _assert_plan_count(
            trace,
            render_collapse_fn,
            None,
            resolved_context,
            projected_count,
        )
        return {}
    hidden_member_contributions = _run_fold_hidden_member_contributions(
        trace,
        render_collapse_fn,
        resolved_context,
    )
    analysis = analyze_collapse(trace)
    candidate_folds: list[ModuleRunFold] = []
    candidate_addresses: set[str] = set()
    for parent_address, child_addresses in _sibling_address_groups(trace).items():
        graph = _flow_graph_for_sibling_group(
            trace,
            str(parent_address),
            child_addresses,
            analysis,
        )
        flow_addresses = _flow_ordered_child_addresses(child_addresses, graph)
        for run in _iter_collapsible_runs(trace, flow_addresses, eligibility_collapse_fn):
            if not _run_fold_is_legal(run, graph):
                continue
            fold = _make_run_fold(trace, run)
            candidate_folds.append(fold)
            candidate_addresses.update(run)
        for run in _iter_collapsible_child_path_runs(
            trace,
            flow_addresses,
            eligibility_collapse_fn,
        ):
            if any(address in candidate_addresses for address in run):
                continue
            run_parent = _common_parent_address(run)
            run_graph = _flow_graph_for_sibling_group(
                trace,
                str(run_parent),
                list(run),
                analysis,
            )
            if not _run_fold_is_legal(run, run_graph):
                continue
            fold = _make_run_fold(trace, run)
            candidate_folds.append(fold)
            candidate_addresses.update(run)
        for run in _iter_collapsible_runs(
            trace,
            flow_addresses,
            eligibility_collapse_fn,
            allow_selected_descendant=True,
        ):
            if any(address in candidate_addresses for address in run):
                continue
            if not _run_fold_is_legal(run, graph):
                continue
            fold = _make_run_fold(trace, run)
            candidate_folds.append(fold)
            candidate_addresses.update(run)
    folds_by_address: dict[str, ModuleRunFold] = {}
    for fold in sorted(candidate_folds, key=lambda item: (-item.multiplicity, item.representative)):
        if any(address in folds_by_address for address in fold.addresses):
            continue
        for address in fold.addresses:
            folds_by_address[address] = fold
        projected_count += _run_fold_delta(fold, hidden_member_contributions)
        if fold_runs is None and projected_count <= _readable_band_high(trace):
            break
    if fold_runs is True:
        projected_count = count(
            plan_from_v1(trace, render_collapse_fn, folds_by_address, resolved_context)
        )
    _assert_plan_count(
        trace,
        render_collapse_fn,
        folds_by_address,
        resolved_context,
        projected_count,
    )
    return folds_by_address


def _always_collapse_module(module: "Module") -> bool:
    """Return ``True`` for standalone run-fold eligibility checks.

    Parameters
    ----------
    module:
        Module being considered.

    Returns
    -------
    bool
        Always ``True``.
    """

    _ = module
    return True


def _sibling_address_groups(trace: "Trace") -> dict[str | None, list[str]]:
    """Return ordered sibling module addresses grouped by parent address.

    Parameters
    ----------
    trace:
        Trace owning the module hierarchy.

    Returns
    -------
    dict[str | None, list[str]]
        Module addresses keyed by their recorded parent address.
    """

    groups: dict[str | None, list[str]] = defaultdict(list)
    for module in trace.modules:
        if module.address == "self":
            continue
        groups[getattr(module, "address_parent", None)].append(module.address)
    return groups


def _flow_ordered_child_addresses(
    child_addresses: list[str],
    graph: ChildCondensedFlowGraph | None,
) -> list[str]:
    """Return child addresses in flow order with deterministic fallback.

    Parameters
    ----------
    child_addresses:
        Recorded sibling addresses.
    graph:
        Optional child-condensed flow graph for the sibling parent.

    Returns
    -------
    list[str]
        Sibling addresses ordered by first-op flow position when available.
    """

    if graph is None:
        return list(child_addresses)
    seen = set(graph.flow_children)
    ordered = [address for address in graph.flow_children if address in child_addresses]
    ordered.extend(address for address in child_addresses if address not in seen)
    return ordered


def _flow_graph_for_sibling_group(
    trace: "Trace",
    parent_address: str,
    child_addresses: list[str],
    analysis: CollapseAnalysis,
) -> ChildCondensedFlowGraph | None:
    """Return the child-flow graph for a concrete or synthetic sibling scope.

    Parameters
    ----------
    trace:
        Trace owning the modules.
    parent_address:
        Recorded parent address for the sibling group.
    child_addresses:
        Sibling child addresses.
    analysis:
        Cached collapse analysis for the trace.

    Returns
    -------
    ChildCondensedFlowGraph | None
        Precomputed graph when available, otherwise a graph synthesized for
        recorded non-module scopes such as ``ModuleList`` containers.
    """

    graph = analysis.child_flow_graphs.get(parent_address)
    if graph is not None:
        return graph
    if not child_addresses:
        return None
    return _synthetic_child_condensed_flow_graph(
        trace,
        parent_address,
        child_addresses,
        analysis.signals,
    )


def _synthetic_child_condensed_flow_graph(
    trace: "Trace",
    parent_address: str,
    child_addresses: list[str],
    signals: Mapping[str, ModuleCollapseSignals],
) -> ChildCondensedFlowGraph:
    """Build a child-condensed graph for recorded non-module sibling scopes.

    Parameters
    ----------
    trace:
        Trace owning the operation graph.
    parent_address:
        Synthetic parent address.
    child_addresses:
        Direct child addresses in the synthetic scope.
    signals:
        Precomputed module signals.

    Returns
    -------
    ChildCondensedFlowGraph
        Flow graph with external source/sink sentinels for boundary edges.
    """

    op_order = {op.label: index for index, op in enumerate(trace.ops)}
    child_sets = {
        child: set(signals[child].subtree_ops) for child in child_addresses if child in signals
    }
    flow_children = tuple(
        sorted(
            child_sets,
            key=lambda child: (
                _first_flow_op_order(trace, child_sets[child], op_order),
                child,
            ),
        )
    )
    owner_by_label: dict[str, str] = {}
    for child_address, labels in child_sets.items():
        for label in labels:
            owner_by_label[label] = child_address
    edges: set[tuple[str, str]] = set()
    for op in trace.ops:
        source = owner_by_label.get(op.label)
        for child_label in getattr(op, "children", ()) or ():
            child_op = cast("Op", trace.ops[child_label])
            target_label = child_op.label
            if not _is_forward_dataflow_edge(trace, op.label, target_label):
                continue
            target = owner_by_label.get(target_label)
            if source is None and target is None:
                continue
            if source is None:
                if target is None:
                    continue
                edges.add((f"external_source:{op.label}", target))
            elif target is None:
                edges.add((source, f"external_sink:{target_label}"))
            elif target != source:
                edges.add((source, target))
    ordered_nodes = (
        *flow_children,
        *sorted({node for edge in edges for node in edge if ":" in node}),
    )
    ordered = {node: index for index, node in enumerate(ordered_nodes)}
    sorted_edges = tuple(
        sorted(
            edges, key=lambda edge: (ordered.get(edge[0], 10**9), ordered.get(edge[1], 10**9), edge)
        )
    )
    return ChildCondensedFlowGraph(
        parent=parent_address,
        flow_children=flow_children,
        parent_owned_ops=(),
        nodes=ordered_nodes,
        edges=sorted_edges,
        child_external_endpoint_counts=_child_external_endpoint_counts(sorted_edges, flow_children),
        interval_flags=_flow_interval_flags(trace, flow_children, child_sets, sorted_edges),
    )


def module_collapse_score(module: "Module") -> float:
    """Return the canonical default collapse score for a module.

    Parameters
    ----------
    module:
        Module metadata entry.

    Returns
    -------
    float
        Rounded canonical score, or ``0.0`` for ineligible/unbound modules.
    """

    trace = module.trace
    if trace is None:
        return 0.0
    return dict(collapse_order(trace)).get(module.address, 0.0)


def _compute_dimless_structural_digests(trace: "Trace") -> dict[str, str]:
    """Compute module structural digests that ignore dimensions and parameters.

    Parameters
    ----------
    trace:
        Trace whose module hierarchy is being fingerprinted.

    Returns
    -------
    dict[str, str]
        Digest keyed by pass-free module address.
    """

    signals = _compute_signal_skeleton(trace)
    digests: dict[str, str] = {}
    modules = sorted(trace.modules, key=lambda module: module.address_depth, reverse=True)
    for module in modules:
        signal = signals[module.address]
        child_sigs = tuple(
            digests[child_address]
            for child_address in getattr(module, "address_children", ()) or ()
            if child_address in digests
        )
        payload = repr(
            (
                getattr(module, "class_name", ""),
                child_sigs,
                len(signal.subtree_ops),
                int(getattr(module, "num_layers", 0) or 0),
                _normalized_internal_topology(trace, signal.subtree_ops),
            )
        ).encode("utf-8")
        digests[module.address] = hashlib.sha1(payload).hexdigest()
    return digests


def _normalized_internal_topology(
    trace: "Trace",
    subtree_ops: tuple[str, ...],
) -> tuple[tuple[int, int], ...]:
    """Return dimension-free internal op-edge topology for ``subtree_ops``.

    Parameters
    ----------
    trace:
        Trace owning the operation graph.
    subtree_ops:
        Pass-qualified operation labels in module scope.

    Returns
    -------
    tuple[tuple[int, int], ...]
        Internal edges expressed as subtree-order indices.
    """

    index_by_label = {label: index for index, label in enumerate(subtree_ops)}
    subtree = set(subtree_ops)
    edges: set[tuple[int, int]] = set()
    for parent_label in subtree_ops:
        parent = cast("Op", trace.ops[parent_label])
        parent_index = index_by_label[parent.label]
        for child_label in getattr(parent, "children", ()) or ():
            if child_label not in subtree:
                continue
            edges.add((parent_index, index_by_label[child_label]))
    return tuple(sorted(edges))


def _iter_collapsible_runs(
    trace: "Trace",
    child_addresses: list[str],
    collapse_fn: Callable[["Module"], bool],
    run_stem: str | None = None,
    allow_selected_descendant: bool = False,
) -> Iterator[tuple[str, ...]]:
    """Yield flow-consecutive same-class runs with equal adjacent output shapes.

    Parameters
    ----------
    trace:
        Trace owning the modules.
    child_addresses:
        Direct children for one parent module in flow order.
    collapse_fn:
        Active collapse predicate.
    run_stem:
        Optional precomputed sibling-run stem for descendant-path folds.
    allow_selected_descendant:
        Whether selected descendants allow a sibling ancestor to stand in as
        the folded member.

    Yields
    ------
    tuple[str, ...]
        One run of at least :data:`RUN_FOLD_MIN_LENGTH` addresses.
    """

    current_key: tuple[str, str] | None = None
    current_descendant_only_num_layers: int | None = None
    current_has_direct_selection = False
    current_run: list[str] = []
    for address in child_addresses:
        module = cast("Module", trace.modules[address])
        directly_selected = collapse_fn(module)
        descendant_selected = allow_selected_descendant and bool(
            _selected_descendants(trace, address, collapse_fn)
        )
        selected = directly_selected or descendant_selected
        if not selected:
            if allow_selected_descendant:
                continue
            if len(current_run) >= RUN_FOLD_MIN_LENGTH:
                yield tuple(current_run)
            current_key = None
            current_descendant_only_num_layers = None
            current_has_direct_selection = False
            current_run = []
            continue
        key = (
            str(getattr(module, "class_name", "")),
            run_stem or _indexed_parent_stem(address),
        )
        num_layers = int(getattr(module, "num_layers", 0) or 0)
        descendant_only_depth_matches = (
            directly_selected
            or current_has_direct_selection
            or current_descendant_only_num_layers in {None, num_layers}
        )
        extends_current = (
            key == current_key
            and bool(current_run)
            and _module_output_shapes_equal(trace, current_run[-1], address)
            and descendant_only_depth_matches
        )
        if extends_current:
            current_run.append(address)
            current_has_direct_selection = current_has_direct_selection or directly_selected
            if not current_has_direct_selection:
                current_descendant_only_num_layers = num_layers
            continue
        if len(current_run) >= RUN_FOLD_MIN_LENGTH:
            yield tuple(current_run)
        current_key = key
        current_descendant_only_num_layers = None if directly_selected else num_layers
        current_has_direct_selection = directly_selected
        current_run = [address]
    if len(current_run) >= RUN_FOLD_MIN_LENGTH:
        yield tuple(current_run)


def _iter_collapsible_child_path_runs(
    trace: "Trace",
    sibling_addresses: list[str],
    collapse_fn: Callable[["Module"], bool],
) -> Iterator[tuple[str, ...]]:
    """Yield repeated selected child paths under consecutive sibling parents.

    Parameters
    ----------
    trace:
        Trace owning the modules.
    sibling_addresses:
        Ordered direct children for one parent module.
    collapse_fn:
        Active collapse predicate.

    Yields
    ------
    tuple[str, ...]
        One run of selected descendant modules sharing the same relative path.
    """

    relative_paths = sorted(
        {
            selected_address.removeprefix(f"{sibling}.")
            for sibling in sibling_addresses
            for selected_address in _selected_descendants(trace, sibling, collapse_fn)
            if selected_address.startswith(f"{sibling}.")
        }
    )
    for relative_path in relative_paths:
        candidate_addresses = [
            f"{sibling}.{relative_path}" if f"{sibling}.{relative_path}" in trace.modules else ""
            for sibling in sibling_addresses
        ]
        current_stem: str | None = None
        current_candidates: list[str] = []
        for sibling, candidate_address in zip(
            sibling_addresses,
            candidate_addresses,
            strict=True,
        ):
            stem = _indexed_parent_stem(sibling)
            if stem == current_stem:
                if candidate_address:
                    current_candidates.append(candidate_address)
                continue
            if current_candidates:
                yield from _iter_collapsible_runs(
                    trace,
                    current_candidates,
                    collapse_fn,
                    current_stem,
                )
            current_stem = stem
            current_candidates = [candidate_address] if candidate_address else []
        if current_candidates:
            yield from _iter_collapsible_runs(
                trace,
                current_candidates,
                collapse_fn,
                current_stem,
            )


def _indexed_parent_stem(address: str) -> str:
    """Return a stem that keeps long indexed sibling runs together.

    Parameters
    ----------
    address:
        Module address.

    Returns
    -------
    str
        Address stem before a trailing numeric component or suffix.
    """

    if "." in address:
        parent, name = address.rsplit(".", 1)
    else:
        parent, name = "", address
    match = _INDEXED_CHILD_RE.match(name)
    if match is None:
        stem = name
    else:
        stem = match.group("stem").rstrip("._") or ""
    return f"{parent}.{stem}" if parent and stem else parent or stem or name


def _common_parent_address(addresses: tuple[str, ...]) -> str | None:
    """Return the shared parent address for a run.

    Parameters
    ----------
    addresses:
        Candidate module addresses.

    Returns
    -------
    str | None
        Shared parent address, or ``None`` when the run is empty or mixed.
    """

    parents = {address.rsplit(".", 1)[0] if "." in address else "self" for address in addresses}
    if len(parents) != 1:
        return None
    return next(iter(parents))


def _module_output_shapes_equal(trace: "Trace", left: str, right: str) -> bool:
    """Return whether two modules have exactly equal known output shapes.

    Parameters
    ----------
    trace:
        Trace owning the modules.
    left:
        First module address.
    right:
        Second module address.

    Returns
    -------
    bool
        True only when both primary output shapes are known and all dimensions
        match exactly.
    """

    left_shape = _module_output_shape_tuple(trace, left)
    right_shape = _module_output_shape_tuple(trace, right)
    return left_shape is not None and left_shape == right_shape


def _run_fold_is_legal(
    addresses: tuple[str, ...],
    graph: ChildCondensedFlowGraph | None,
) -> bool:
    """Return whether a candidate run satisfies the v2 legality grammar.

    Parameters
    ----------
    addresses:
        Candidate run addresses in flow order.
    graph:
        Child-condensed flow graph for the run's parent.

    Returns
    -------
    bool
        True for legal chain intervals or legal parallel-fan bundles.
    """

    if len(addresses) < RUN_FOLD_MIN_LENGTH or graph is None:
        return False
    if not _run_is_flow_consecutive(addresses, graph):
        return False
    return _run_fold_is_chain_interval(addresses, graph) or _run_fold_is_parallel_fan(
        addresses,
        graph,
    )


def _run_is_flow_consecutive(
    addresses: tuple[str, ...],
    graph: ChildCondensedFlowGraph,
) -> bool:
    """Return whether ``addresses`` are adjacent in graph flow-child order.

    Parameters
    ----------
    addresses:
        Candidate run addresses.
    graph:
        Child-condensed flow graph for the parent.

    Returns
    -------
    bool
        True when the candidate is a contiguous interval in flow order.
    """

    flow_index = {address: index for index, address in enumerate(graph.flow_children)}
    indexes = [flow_index.get(address) for address in addresses]
    if any(index is None for index in indexes):
        return False
    first = cast(int, indexes[0])
    return indexes == list(range(first, first + len(addresses)))


def _run_fold_is_chain_interval(
    addresses: tuple[str, ...],
    graph: ChildCondensedFlowGraph,
) -> bool:
    """Return whether a run satisfies the chain-interval legality contract.

    Parameters
    ----------
    addresses:
        Candidate run addresses in flow order.
    graph:
        Child-condensed flow graph for the parent.

    Returns
    -------
    bool
        True when members form one path with one external entry, one external
        exit, and no flagged interior boundary crossing.
    """

    member_set = set(addresses)
    edges = set(graph.edges)
    internal_edges = {
        (source, target)
        for source, target in edges
        if source in member_set and target in member_set and source != target
    }
    expected_edges = set(zip(addresses[:-1], addresses[1:], strict=True))
    connector_nodes = _chain_connector_nodes(addresses, edges)
    if connector_nodes is None:
        return False
    direct_expected_edges = expected_edges & internal_edges
    if internal_edges - direct_expected_edges:
        return False
    entries = [
        (source, target)
        for source, target in edges
        if target in member_set and source not in member_set and source not in connector_nodes
    ]
    exits = [
        (source, target)
        for source, target in edges
        if source in member_set and target not in member_set and target not in connector_nodes
    ]
    if len(entries) != 1 or entries[0][1] != addresses[0]:
        return False
    if len(exits) != 1 or exits[0][0] != addresses[-1]:
        return False
    return True


def _chain_connector_nodes(
    addresses: tuple[str, ...],
    edges: set[tuple[str, str]],
) -> set[str] | None:
    """Return external one-hop connectors for a chain run if it forms a path.

    Parameters
    ----------
    addresses:
        Candidate run addresses in flow order.
    edges:
        Condensed graph edges.

    Returns
    -------
    set[str] | None
        External connector nodes used between members, or ``None`` if any
        consecutive pair is not connected by exactly one path step.
    """

    member_set = set(addresses)
    connectors: set[str] = set()
    for left, right in zip(addresses[:-1], addresses[1:], strict=True):
        if (left, right) in edges:
            continue
        pair_connectors = {
            target
            for source, target in edges
            if source == left and target not in member_set and (target, right) in edges
        }
        paired_connectors = {
            (target, paired)
            for source, target in edges
            for paired in (_paired_external_connector(target),)
            if source == left
            and target not in member_set
            and paired is not None
            and (paired, right) in edges
        }
        if paired_connectors:
            pair_connectors.update(connector for pair in paired_connectors for connector in pair)
        if len(pair_connectors) != 1:
            if len(pair_connectors) != 2 or not any(
                _paired_external_connector(connector) in pair_connectors
                for connector in pair_connectors
            ):
                return None
        connectors.update(pair_connectors)
    return connectors


def _paired_external_connector(node: str) -> str | None:
    """Return the source/sink counterpart for an external connector node.

    Parameters
    ----------
    node:
        Condensed external connector node name.

    Returns
    -------
    str | None
        Paired connector name, or ``None`` when ``node`` is not an external
        source/sink sentinel.
    """

    if node.startswith("external_sink:"):
        return f"external_source:{node.removeprefix('external_sink:')}"
    if node.startswith("external_source:"):
        return f"external_sink:{node.removeprefix('external_source:')}"
    return None


def _run_fold_is_parallel_fan(
    addresses: tuple[str, ...],
    graph: ChildCondensedFlowGraph,
) -> bool:
    """Return whether a run satisfies the parallel-fan legality contract.

    Parameters
    ----------
    addresses:
        Candidate run addresses in flow order.
    graph:
        Child-condensed flow graph for the parent.

    Returns
    -------
    bool
        True when members have no mutual edges and identical external source
        and sink sets.
    """

    member_set = set(addresses)
    source_sets: list[frozenset[str]] = []
    sink_sets: list[frozenset[str]] = []
    for address in addresses:
        sources: set[str] = set()
        sinks: set[str] = set()
        for source, target in graph.edges:
            if source == address and target in member_set:
                return False
            if target == address and source in member_set:
                return False
            if target == address and source not in member_set:
                sources.add(source)
            if source == address and target not in member_set:
                sinks.add(target)
        source_sets.append(frozenset(sources))
        sink_sets.append(frozenset(sinks))
    return (
        bool(source_sets[0])
        and bool(sink_sets[0])
        and all(sources == source_sets[0] for sources in source_sets[1:])
        and all(sinks == sink_sets[0] for sinks in sink_sets[1:])
    )


def _selected_descendants(
    trace: "Trace",
    address: str,
    collapse_fn: Callable[["Module"], bool],
) -> tuple[str, ...]:
    """Return selected descendant module addresses under ``address``.

    Parameters
    ----------
    trace:
        Trace owning the modules.
    address:
        Parent module address.
    collapse_fn:
        Active collapse predicate.

    Returns
    -------
    tuple[str, ...]
        Selected descendant addresses in lexical order.
    """

    prefix = f"{address}."
    return tuple(
        module.address
        for module in trace.modules
        if module.address.startswith(prefix) and collapse_fn(module)
    )


def _make_run_fold(trace: "Trace", addresses: tuple[str, ...]) -> ModuleRunFold:
    """Build aggregate metadata for one folded run.

    Parameters
    ----------
    trace:
        Trace owning the modules.
    addresses:
        Consecutive sibling addresses in the run.

    Returns
    -------
    ModuleRunFold
        Aggregate run-fold descriptor.
    """

    modules = [cast("Module", trace.modules[address]) for address in addresses]
    return ModuleRunFold(
        representative=addresses[0],
        addresses=addresses,
        class_name=str(getattr(modules[0], "class_name", "") or "blocks"),
        num_layers=sum(int(getattr(module, "num_layers", 0) or 0) for module in modules),
        num_params=sum(int(getattr(module, "num_params", 0) or 0) for module in modules),
        num_params_trainable=sum(
            int(getattr(module, "num_params_trainable", 0) or 0) for module in modules
        ),
        num_params_frozen=sum(
            int(getattr(module, "num_params_frozen", 0) or 0) for module in modules
        ),
        shape_summary=_run_shape_summary(trace, addresses),
        hidden_member_composition=_hidden_member_composition(trace, addresses),
    )


def _hidden_member_composition(trace: "Trace", addresses: tuple[str, ...]) -> Mapping[str, int]:
    """Return residual/passthrough composition for hidden run members.

    Parameters
    ----------
    trace:
        Trace owning the modules.
    addresses:
        Folded run addresses, including the representative.

    Returns
    -------
    Mapping[str, int]
        Counts for hidden members with and without residual or join-style
        passthrough operations.
    """

    analysis = analyze_collapse(trace)
    composition = {
        "hidden_with_residual_join": 0,
        "hidden_without_residual_join": 0,
    }
    for address in addresses[1:]:
        signal = analysis.signals.get(address)
        has_join = signal is not None and (
            signal.passthrough_edges > 0
            or any(
                _op_func_name(cast("Op", trace.ops[label])) in JUNCTION_FUNC_NAMES
                for label in signal.subtree_ops
            )
        )
        key = "hidden_with_residual_join" if has_join else "hidden_without_residual_join"
        composition[key] += 1
    return composition


def _run_shape_summary(trace: "Trace", addresses: tuple[str, ...]) -> str | None:
    """Return a compact first-to-last output shape summary for a folded run.

    Parameters
    ----------
    trace:
        Trace owning the modules.
    addresses:
        Consecutive sibling addresses in the run.

    Returns
    -------
    str | None
        Shape summary when first and last output shapes differ, else ``None``.
    """

    shapes = [_module_output_shape(trace, address) for address in addresses]
    first = shapes[0]
    last = shapes[-1]
    if first is None or last is None or first == last:
        return None
    return f"{first}->{last}"


def _run_span_allows_fold(trace: "Trace", addresses: tuple[str, ...]) -> bool:
    """Return whether first-to-last tensor shape span is safe to fold.

    Parameters
    ----------
    trace:
        Trace owning the modules.
    addresses:
        Consecutive sibling addresses in the candidate run.

    Returns
    -------
    bool
        True when the run does not cross a spatial-resolution boundary and
        does not span more than a 2x channel-width change. Unknown shapes are
        treated as foldable because the structural key is the primary guard.
    """

    first = _module_output_shape_tuple(trace, addresses[0])
    last = _module_output_shape_tuple(trace, addresses[-1])
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


def _module_output_shape_tuple(trace: "Trace", address: str) -> tuple[int, ...] | None:
    """Return the primary output shape tuple for a module address.

    Parameters
    ----------
    trace:
        Trace owning the module.
    address:
        Pass-free module address.

    Returns
    -------
    tuple[int, ...] | None
        Output shape as integers, or ``None`` when unavailable.
    """

    module_output_layer = _module_call_output_op(trace, f"{address}:1")
    if module_output_layer is None:
        return None
    shape = getattr(module_output_layer, "shape", None)
    if shape is None:
        shape = getattr(module_output_layer, "out_shape", None)
    if not shape:
        return None
    try:
        return tuple(int(dim) for dim in shape)
    except (TypeError, ValueError):
        return None


def _module_call_output_op(trace: "Trace", call_label: str) -> "Op | None":
    """Return the primary output Op for a module-call label.

    Parameters
    ----------
    trace:
        Trace owning the module call.
    call_label:
        Pass-qualified module-call label such as ``"encoder:1"``.

    Returns
    -------
    Op | None
        Last output op for the module call, or ``None`` when unavailable.
    """

    try:
        module_call = trace.module_calls[call_label]
    except (KeyError, IndexError):
        return None
    if not module_call.output_ops:
        return None
    try:
        return cast("Op", trace.ops[module_call.output_ops[-1]])
    except (KeyError, IndexError):
        return None


def _shape_spatial_dims(shape: tuple[int, ...]) -> tuple[int, ...] | None:
    """Return spatial dimensions for common image/video tensor shapes.

    Parameters
    ----------
    shape:
        Output tensor shape.

    Returns
    -------
    tuple[int, ...] | None
        Spatial dimensions, or ``None`` for non-spatial ranks.
    """

    if len(shape) == 4:
        return shape[2:]
    if len(shape) == 5:
        return shape[2:]
    return None


def _shape_channel_dim(shape: tuple[int, ...]) -> int | None:
    """Return the channel-like dimension for common tensor shapes.

    Parameters
    ----------
    shape:
        Output tensor shape.

    Returns
    -------
    int | None
        Channel dimension, or ``None`` when no stable convention applies.
    """

    if len(shape) in {2, 4, 5}:
        return shape[1]
    if len(shape) == 3:
        return shape[2]
    return None


def _module_output_shape(trace: "Trace", address: str) -> str | None:
    """Return the primary output shape string for a module address.

    Parameters
    ----------
    trace:
        Trace owning the module.
    address:
        Pass-free module address.

    Returns
    -------
    str | None
        Formatted shape string, or ``None`` when unavailable.
    """

    shape = _module_output_shape_tuple(trace, address)
    if shape is None:
        return None
    return str(tuple(shape))


def _compute_signal_skeleton(trace: "Trace") -> dict[str, ModuleCollapseSignals]:
    """Compute all non-peer module signals in one shared traversal."""

    op_labels_by_module: dict[str, list[str]] = defaultdict(list)
    own_func_names_by_module: dict[str, list[str]] = defaultdict(list)
    internal_edges: dict[str, set[tuple[str, str]]] = defaultdict(set)
    input_edges: dict[str, set[tuple[str, str]]] = defaultdict(set)
    output_edges: dict[str, set[tuple[str, str]]] = defaultdict(set)

    ops = list(trace.ops)
    op_by_label = {op.label: op for op in ops}
    stack_by_label = {op.label: _module_address_stack(op) for op in ops}

    for op in ops:
        stack = stack_by_label[op.label]
        for address in stack:
            op_labels_by_module[address].append(op.label)
        if stack:
            own_func_names_by_module[stack[-1]].append(_op_func_name(op))

    for parent in ops:
        parent_stack = stack_by_label[parent.label]
        parent_set = set(parent_stack)
        for child_label in parent.children:
            child = op_by_label.get(child_label)
            if child is None:
                child = cast("Op", trace.ops[child_label])
                op_by_label[child_label] = child
                op_by_label[child.label] = child
                stack_by_label[child.label] = _module_address_stack(child)
            child_stack = stack_by_label[child.label]
            child_set = set(child_stack)
            edge = (parent.label, child.label)
            for address in parent_set & child_set:
                internal_edges[address].add(edge)
            for address in child_set - parent_set:
                input_edges[address].add(edge)
            for address in parent_set - child_set:
                output_edges[address].add(edge)

    signals: dict[str, ModuleCollapseSignals] = {}
    for module in trace.modules:
        address = module.address
        subtree_ops = tuple(dict.fromkeys(op_labels_by_module.get(address, ())))
        hidden_ops = max(len(subtree_ops) - 1, 0)
        signals[address] = ModuleCollapseSignals(
            address=address,
            subtree_ops=subtree_ops,
            own_func_names=tuple(own_func_names_by_module.get(address, ())),
            internal_edges=len(internal_edges.get(address, ())),
            input_edges=len(input_edges.get(address, ())),
            output_edges=len(output_edges.get(address, ())),
            landmark_edges=_count_landmark_edges(
                trace,
                module,
                subtree_ops,
                input_edges.get(address, set()) | output_edges.get(address, set()),
            ),
            passthrough_edges=_count_passthrough_edges(trace, module, subtree_ops),
            output_junctions=_output_junctions(
                trace,
                module,
                subtree_ops,
                output_edges.get(address, set()),
            ),
            params=int(getattr(module, "num_params", 0) or 0),
            depth=int(getattr(module, "address_depth", 0) or 0),
            num_calls=int(getattr(module, "num_calls", 1) or 1),
            structural_digest="",
            peer_count=1,
            hidden_ops=hidden_ops,
            eligible=_gate_module(module, hidden_ops, signals),
        )
    return signals


def _module_address_stack(op: "Op") -> tuple[str, ...]:
    """Return pass-free module addresses enclosing an op."""

    return tuple(str(module).rsplit(":", 1)[0] for module in getattr(op, "modules", ()) or ())


def _compute_child_condensed_flow_graphs(
    trace: "Trace",
    signals: Mapping[str, ModuleCollapseSignals],
) -> dict[str, ChildCondensedFlowGraph]:
    """Compute child-condensed flow graphs for every parent module.

    Parameters
    ----------
    trace:
        Trace owning the module hierarchy.
    signals:
        Precomputed module signal skeletons.

    Returns
    -------
    dict[str, ChildCondensedFlowGraph]
        Flow graph artifacts keyed by parent module address.
    """

    graphs: dict[str, ChildCondensedFlowGraph] = {}
    op_order = {op.label: index for index, op in enumerate(trace.ops)}
    for module in trace.modules:
        parent = module.address
        child_addresses = tuple(
            str(child)
            for child in getattr(module, "address_children", ()) or ()
            if child in trace.modules
        )
        if not child_addresses:
            graphs[parent] = ChildCondensedFlowGraph(
                parent=parent,
                flow_children=(),
                parent_owned_ops=(),
                nodes=(),
                edges=(),
                child_external_endpoint_counts={},
                interval_flags={},
            )
            continue
        child_sets = {
            child: set(signals[child].subtree_ops) for child in child_addresses if child in signals
        }
        flow_children = tuple(
            sorted(
                child_sets,
                key=lambda child: (
                    min((op_order[label] for label in child_sets[child]), default=10**12),
                    child,
                ),
            )
        )
        parent_ops = tuple(
            label
            for label in signals.get(parent, _empty_signal(parent)).subtree_ops
            if _condensed_owner_for_op(label, parent, flow_children, child_sets) == label
            and not _is_buffer_op_label(trace, label)
        )
        nodes = (*flow_children, *parent_ops)
        edges = _condensed_edges(trace, parent, flow_children, child_sets, set(parent_ops))
        endpoint_counts = _child_external_endpoint_counts(edges, flow_children)
        interval_flags = _flow_interval_flags(trace, flow_children, child_sets, edges)
        graphs[parent] = ChildCondensedFlowGraph(
            parent=parent,
            flow_children=flow_children,
            parent_owned_ops=parent_ops,
            nodes=nodes,
            edges=edges,
            child_external_endpoint_counts=endpoint_counts,
            interval_flags=interval_flags,
        )
    return graphs


def _empty_signal(address: str) -> ModuleCollapseSignals:
    """Return an empty signal used for missing parent bookkeeping.

    Parameters
    ----------
    address:
        Module address.

    Returns
    -------
    ModuleCollapseSignals
        Empty signal with no subtree operations.
    """

    return ModuleCollapseSignals(
        address=address,
        subtree_ops=(),
        own_func_names=(),
        internal_edges=0,
        input_edges=0,
        output_edges=0,
        landmark_edges=0,
        passthrough_edges=0,
        output_junctions=(),
        params=0,
        depth=0,
        num_calls=1,
        structural_digest="",
        peer_count=1,
        hidden_ops=0,
        eligible=False,
    )


def _first_flow_op_order(
    trace: "Trace",
    op_labels: set[str],
    op_order: Mapping[str, int],
) -> int:
    """Return first non-buffer op order for a child subtree.

    Parameters
    ----------
    trace:
        Trace owning the operation graph.
    op_labels:
        Operation labels in the child subtree.
    op_order:
        Deterministic operation-order index keyed by op label.

    Returns
    -------
    int
        First non-buffer operation index, falling back to any operation index
        when the subtree has no non-buffer ops.
    """

    non_buffer_orders = [
        op_order[label] for label in op_labels if not _is_buffer_op_label(trace, label)
    ]
    if non_buffer_orders:
        return min(non_buffer_orders)
    return min((op_order[label] for label in op_labels), default=10**12)


def _is_buffer_op_label(trace: "Trace", op_label: str) -> bool:
    """Return whether ``op_label`` identifies a buffer/source op.

    Parameters
    ----------
    trace:
        Trace owning the operation graph.
    op_label:
        Operation label to inspect.

    Returns
    -------
    bool
        True when the label exists and represents a buffer op.
    """

    if op_label not in trace.ops:
        return False
    return bool(getattr(cast("Op", trace.ops[op_label]), "is_buffer", False))


def _is_forward_dataflow_edge(trace: "Trace", source_label: str, target_label: str) -> bool:
    """Return whether an op edge is real forward tensor dataflow.

    Parameters
    ----------
    trace:
        Trace owning the operation graph.
    source_label:
        Source operation label.
    target_label:
        Target operation label.

    Returns
    -------
    bool
        True for non-buffer endpoint edges. Registered-buffer provenance and
        write-version edges are excluded from the child-condensed dataflow
        artifact.
    """

    return not _is_buffer_op_label(trace, source_label) and not _is_buffer_op_label(
        trace,
        target_label,
    )


def _condensed_owner_for_op(
    op_label: str,
    parent: str,
    flow_children: Sequence[str],
    child_sets: Mapping[str, set[str]],
) -> str:
    """Return the condensed node that owns an op within ``parent``.

    Parameters
    ----------
    op_label:
        Operation label.
    parent:
        Parent module address.
    flow_children:
        Direct children in flow order.
    child_sets:
        Child subtree operation labels.

    Returns
    -------
    str
        Child address when the op belongs to a child subtree; otherwise the op label.
    """

    _ = parent
    for child in flow_children:
        if op_label in child_sets.get(child, set()):
            return child
    return op_label


def _condensed_edges(
    trace: "Trace",
    parent: str,
    flow_children: Sequence[str],
    child_sets: Mapping[str, set[str]],
    parent_ops: set[str],
) -> tuple[tuple[str, str], ...]:
    """Return condensed edges within one parent module subtree.

    Parameters
    ----------
    trace:
        Trace owning the operation graph.
    parent:
        Parent module address.
    flow_children:
        Direct children in flow order.
    child_sets:
        Child subtree operation labels.
    parent_ops:
        Parent-owned operation labels.

    Returns
    -------
    tuple[tuple[str, str], ...]
        Deterministically sorted condensed edges.
    """

    parent_subtree = set().union(*child_sets.values()) if child_sets else set()
    parent_subtree.update(parent_ops)
    order = {node: index for index, node in enumerate((*flow_children, *sorted(parent_ops)))}
    edges: set[tuple[str, str]] = set()
    for label in sorted(
        parent_subtree, key=lambda item: int(getattr(trace.ops[item], "step_index", 0))
    ):
        op = cast("Op", trace.ops[label])
        source = _condensed_owner_for_op(label, parent, flow_children, child_sets)
        for parent_label in getattr(op, "parents", ()) or ():
            parent_op = cast("Op", trace.ops[parent_label])
            normalized_parent_label = parent_op.label
            if normalized_parent_label in parent_subtree:
                continue
            if not _is_forward_dataflow_edge(trace, normalized_parent_label, label):
                continue
            edges.add((f"external_source:{normalized_parent_label}", source))
        for child_label in getattr(op, "children", ()) or ():
            child = cast("Op", trace.ops[child_label])
            normalized_child_label = child.label
            if not _is_forward_dataflow_edge(trace, label, normalized_child_label):
                continue
            if normalized_child_label not in parent_subtree:
                edges.add((source, f"external_sink:{normalized_child_label}"))
                continue
            target = _condensed_owner_for_op(
                normalized_child_label,
                parent,
                flow_children,
                child_sets,
            )
            if source != target:
                edges.add((source, target))
    return tuple(
        sorted(edges, key=lambda edge: (order.get(edge[0], 10**9), order.get(edge[1], 10**9), edge))
    )


def _child_external_endpoint_counts(
    edges: Sequence[tuple[str, str]],
    flow_children: Sequence[str],
) -> dict[str, tuple[int, int]]:
    """Return per-child external entry and exit endpoint counts.

    Parameters
    ----------
    edges:
        Condensed graph edges.
    flow_children:
        Direct children in flow order.

    Returns
    -------
    dict[str, tuple[int, int]]
        Mapping from child address to ``(entries, exits)``.
    """

    child_set = set(flow_children)
    entries: dict[str, set[str]] = {child: set() for child in flow_children}
    exits: dict[str, set[str]] = {child: set() for child in flow_children}
    for source, target in edges:
        if target in child_set and source != target:
            entries[target].add(source)
        if source in child_set and source != target:
            exits[source].add(target)
    return {child: (len(entries[child]), len(exits[child])) for child in flow_children}


def _flow_interval_flags(
    trace: "Trace",
    flow_children: Sequence[str],
    child_sets: Mapping[str, set[str]],
    edges: Sequence[tuple[str, str]],
) -> dict[tuple[str, str], FlowIntervalFlags]:
    """Return landmark and passthrough flags for child-flow intervals.

    Parameters
    ----------
    trace:
        Trace owning the operation graph.
    flow_children:
        Direct children in flow order.
    child_sets:
        Child subtree operation labels.
    edges:
        Condensed graph edges.

    Returns
    -------
    dict[tuple[str, str], FlowIntervalFlags]
        Flags keyed by adjacent child pairs in flow order.
    """

    if len(flow_children) < 2:
        return {}
    child_index = {child: index for index, child in enumerate(flow_children)}
    edge_set = set(edges)
    flags: dict[tuple[str, str], FlowIntervalFlags] = {}
    for left, right in zip(flow_children[:-1], flow_children[1:], strict=True):
        left_index = child_index[left]
        right_index = child_index[right]
        crossing_edges = [
            edge
            for edge in edge_set
            if edge[0] in child_index
            and edge[1] in child_index
            and child_index[edge[0]] <= left_index
            and child_index[edge[1]] >= right_index
        ]
        passthrough = any(
            edge[0] not in child_index or edge[1] not in child_index
            for edge in edge_set
            if _edge_touches_interval(edge, child_index, left_index, right_index)
        )
        landmark = any(
            _child_has_junction_op(trace, child_sets.get(child, set()))
            for child in flow_children[left_index : right_index + 1]
        ) or bool(crossing_edges)
        flags[(left, right)] = FlowIntervalFlags(landmark=landmark, passthrough=passthrough)
    return flags


def _edge_touches_interval(
    edge: tuple[str, str],
    child_index: Mapping[str, int],
    left_index: int,
    right_index: int,
) -> bool:
    """Return whether a condensed edge touches an interval boundary.

    Parameters
    ----------
    edge:
        Condensed edge.
    child_index:
        Child address to flow index.
    left_index:
        Left child index of the interval.
    right_index:
        Right child index of the interval.

    Returns
    -------
    bool
        True when the edge is adjacent to the interval.
    """

    source, target = edge
    source_index = child_index.get(source)
    target_index = child_index.get(target)
    return source_index in {left_index, right_index} or target_index in {left_index, right_index}


def _child_has_junction_op(trace: "Trace", op_labels: set[str]) -> bool:
    """Return whether a child subtree contains a junction operation.

    Parameters
    ----------
    trace:
        Trace owning the operation graph.
    op_labels:
        Operation labels in the child subtree.

    Returns
    -------
    bool
        True when a known fan-in/fan-out junction op is present.
    """

    return any(
        _op_func_name(cast("Op", trace.ops[label])) in JUNCTION_FUNC_NAMES for label in op_labels
    )


def _op_func_name(op: "Op") -> str:
    """Return a stable operation function name for digesting."""

    return str(getattr(op, "func_name", None) or getattr(op, "layer_type", "") or "")


def _count_landmark_edges(
    trace: "Trace",
    module: "Module",
    subtree_ops: tuple[str, ...],
    boundary_edges: set[tuple[str, str]],
) -> int:
    """Return boundary-crossing junction edges for a module.

    Parameters
    ----------
    trace:
        Trace that owns the operation graph.
    module:
        Candidate module being scored.
    subtree_ops:
        Pass-qualified operation labels in the module subtree.
    boundary_edges:
        Distinct edges crossing the module boundary.

    Returns
    -------
    int
        Count of boundary edges that would hide or visually skip a junction
        across the collapsed module boundary. Fully internal junctions and
        ordinary module I/O edges are intentionally not counted because they are
        safely represented by the collapsed module box.
    """

    subtree = set(subtree_ops)
    input_layers = {_base_label(label) for label in getattr(module, "input_layers", ()) or ()}
    output_layers = {_base_label(label) for label in getattr(module, "output_layers", ()) or ()}
    landmarks: set[tuple[str, str]] = set()
    for parent_label, child_label in boundary_edges:
        parent = cast("Op", trace.ops[parent_label])
        child = cast("Op", trace.ops[child_label])
        if getattr(parent, "is_buffer", False) or getattr(child, "is_buffer", False):
            continue
        parent_inside = parent.label in subtree
        child_inside = child.label in subtree
        if parent_inside == child_inside:
            continue
        parent_base = _base_label(parent.label)
        child_base = _base_label(child.label)
        if child_inside and parent_base in input_layers:
            continue
        if parent_inside and parent_base in output_layers:
            continue
        if child_inside and child_base in output_layers:
            continue
        if getattr(parent, "is_output", False) or getattr(child, "is_output", False):
            continue
        if not _boundary_edge_preserves_junction(trace, parent, child, subtree):
            continue
        landmarks.add((parent.label, child.label))
    return len(landmarks)


def _boundary_edge_preserves_junction(
    trace: "Trace",
    parent: "Op",
    child: "Op",
    subtree: set[str],
) -> bool:
    """Return whether a boundary edge is part of a cross-boundary junction.

    Parameters
    ----------
    trace:
        Trace that owns the operation graph.
    parent:
        Parent endpoint of the boundary edge.
    child:
        Child endpoint of the boundary edge.
    subtree:
        Pass-qualified operation labels in the candidate module subtree.

    Returns
    -------
    bool
        True when collapsing the subtree would obscure a junction whose visible
        endpoints span the module boundary.
    """

    parent_inside = parent.label in subtree
    child_inside = child.label in subtree
    if parent_inside == child_inside:
        return False
    internal = parent if parent_inside else child
    external = child if parent_inside else parent
    if _is_junction_op(external):
        return True
    if not _is_junction_op(internal):
        return False
    return _has_external_parent(trace, internal, subtree) and _has_external_child(
        trace,
        internal,
        subtree,
    )


def _is_junction_op(op: "Op") -> bool:
    """Return whether an operation is a fan-in or fan-out junction."""

    return _op_func_name(op) in JUNCTION_FUNC_NAMES


def _has_external_parent(trace: "Trace", op: "Op", subtree: set[str]) -> bool:
    """Return whether an operation has a non-buffer parent outside ``subtree``."""

    for parent_label in getattr(op, "parents", ()) or ():
        parent = cast("Op", trace.ops[parent_label])
        if parent.label not in subtree and not getattr(parent, "is_buffer", False):
            return True
    return False


def _has_external_child(trace: "Trace", op: "Op", subtree: set[str]) -> bool:
    """Return whether an operation has a non-buffer child outside ``subtree``."""

    for child_label in getattr(op, "children", ()) or ():
        child = cast("Op", trace.ops[child_label])
        if child.label not in subtree and not getattr(child, "is_buffer", False):
            return True
    return False


def _base_label(label: str) -> str:
    """Return a pass-free operation label.

    Parameters
    ----------
    label:
        Operation label that may include a pass suffix.

    Returns
    -------
    str
        Operation label without the trailing pass suffix.
    """

    return str(label).rsplit(":", 1)[0]


def _count_passthrough_edges(
    trace: "Trace",
    module: "Module",
    subtree_ops: tuple[str, ...],
) -> int:
    """Return internal output joins fed directly by module inputs.

    Parameters
    ----------
    trace:
        Trace that owns the operation graph.
    module:
        Candidate module being scored.
    subtree_ops:
        Pass-qualified operation labels in the module subtree.

    Returns
    -------
    int
        Number of module-output Ops that merge an external module input with
        internal computation. These joins are useful orientation landmarks for
        ``collapse="auto"`` but may be hidden by ``collapse="max"``.
    """

    subtree = set(subtree_ops)
    input_layers = {_base_label(label) for label in getattr(module, "input_layers", ()) or ()}
    output_layers = {_base_label(label) for label in getattr(module, "output_layers", ()) or ()}
    passthrough_edges = 0
    for label in subtree_ops:
        op = cast("Op", trace.ops[label])
        if _base_label(op.label) not in output_layers:
            continue
        if _op_func_name(op) not in JUNCTION_FUNC_NAMES:
            continue
        has_internal_parent = False
        has_input_parent = False
        for parent_label in op.parents:
            parent = cast("Op", trace.ops[parent_label])
            if parent.label in subtree:
                has_internal_parent = True
            elif _base_label(parent.label) in input_layers:
                has_input_parent = True
        if has_internal_parent and has_input_parent:
            passthrough_edges += 1
    return passthrough_edges


def _output_junctions(
    trace: "Trace",
    module: "Module",
    subtree_ops: tuple[str, ...],
    output_edges: set[tuple[str, str]],
) -> tuple[str, ...]:
    """Return external multi-parent junction children fed by module outputs.

    Parameters
    ----------
    trace:
        Trace that owns the operation graph.
    module:
        Candidate module being scored.
    subtree_ops:
        Pass-qualified operation labels in the module subtree.
    output_edges:
        Distinct edges leaving the module subtree.

    Returns
    -------
    tuple[str, ...]
        Pass-free labels for external multi-parent children fed by this module.
    """

    subtree = set(subtree_ops)
    output_layers = {_base_label(label) for label in getattr(module, "output_layers", ()) or ()}
    junctions: set[str] = set()
    for parent_label, child_label in output_edges:
        parent = cast("Op", trace.ops[parent_label])
        child = cast("Op", trace.ops[child_label])
        if parent.label not in subtree:
            continue
        if _base_label(parent.label) not in output_layers:
            continue
        if len(getattr(child, "parents", ()) or ()) < 2:
            continue
        junctions.add(_base_label(child.label))
    return tuple(sorted(junctions))


def _gate_module(
    module: "Module",
    hidden_ops: int,
    partial_signals: Mapping[str, ModuleCollapseSignals],
) -> bool:
    """Return whether a module mirrors renderer collapse eligibility."""

    if module.address in {"", "self"}:
        return False
    if int(getattr(module, "num_layers", 0) or 0) <= 1:
        return False
    child_addresses = list(getattr(module, "address_children", ()) or ())
    if len(child_addresses) == 1:
        child_signal = partial_signals.get(child_addresses[0])
        if child_signal is not None and child_signal.hidden_ops == hidden_ops:
            return False
    return True


def _compute_structural_digests(
    trace: "Trace",
    signals: Mapping[str, ModuleCollapseSignals],
) -> dict[str, str]:
    """Compute structural digests bottom-up for every module."""

    digests: dict[str, str] = {}
    modules = sorted(trace.modules, key=lambda module: module.address_depth, reverse=True)
    for module in modules:
        signal = signals[module.address]
        child_sigs = tuple(
            digests[child_address]
            for child_address in getattr(module, "address_children", ()) or ()
            if child_address in digests
        )
        payload = repr(
            (
                getattr(module, "class_name", ""),
                signal.own_func_names,
                child_sigs,
                round(math.log10(1 + max(signal.params, 0))),
            )
        ).encode("utf-8")
        digests[module.address] = hashlib.sha1(payload).hexdigest()
    return digests


def _group_structural_peers(
    trace: "Trace",
    digests: Mapping[str, str],
) -> dict[tuple[str, str | None], tuple[str, ...]]:
    """Group trace-local structural peers by exact and relaxed sibling signatures."""

    groups: dict[tuple[str, str | None], list[str]] = defaultdict(list)
    for module in trace.modules:
        parent = str(getattr(module, "address_parent", None))
        exact_key = (f"exact:{digests[module.address]}", _peer_scope_key(trace, module))
        class_key = (f"class:{getattr(module, 'class_name', '')}", parent)
        stem_key = (f"stem:{_sibling_stem(module.address)}", parent)
        groups[exact_key].append(module.address)
        groups[class_key].append(module.address)
        groups[stem_key].append(module.address)
    return {
        key: tuple(sorted(dict.fromkeys(addresses)))
        for key, addresses in groups.items()
        if len(set(addresses)) >= 2 and key[0] not in {"class:", "stem:"}
    }


def _sibling_stem(address: str) -> str:
    """Return a relaxed sibling-address stem for stage-like module names."""

    name = address.rsplit(".", 1)[-1]
    match = _INDEXED_CHILD_RE.match(name)
    if match is None:
        return name
    stem = match.group("stem").rstrip("._")
    return stem or name


def _peer_scope_key(trace: "Trace", module: "Module") -> str | None:
    """Return the sibling scope key used for repeated structural peers.

    Parameters
    ----------
    trace:
        Trace that owns the module hierarchy.
    module:
        Module whose peer grouping scope is being resolved.

    Returns
    -------
    str | None
        Stable scope key shared by repeated siblings or cousins under repeated
        parents.
    """

    parent_address = getattr(module, "address_parent", None)
    if parent_address is None:
        return None
    try:
        parent = cast("Module", trace.modules[parent_address])
    except KeyError:
        return str(parent_address)
    grandparent = getattr(parent, "address_parent", None)
    parent_class = str(getattr(parent, "class_name", ""))
    return f"{grandparent}:{parent_class}"


def _readable_band_high(trace: "Trace") -> int:
    """Return the high watermark for a readable auto-collapsed render.

    Parameters
    ----------
    trace:
        Trace being rendered.

    Returns
    -------
    int
        Upper readable node-count budget for auto collapse.
    """

    return 25 if len(trace.ops) > 100 else 40


def _is_trunk_collapse(trace: "Trace", signal: ModuleCollapseSignals) -> bool:
    """Return whether a module would collapse nearly the whole input-output trunk."""

    visible_after = max(1, len(trace.ops) - signal.hidden_ops)
    if visible_after >= 4:
        return False
    op_set = set(signal.subtree_ops)
    has_input = any(cast("Op", trace.ops[label]).is_input for label in op_set)
    has_output = any(cast("Op", trace.ops[label]).is_output for label in op_set)
    return has_input or has_output


def _rendered_module_hidden_counts(trace: "Trace", context: RenderContext) -> dict[str, int]:
    """Return rendered-node counts hidden by selecting each module alone.

    Parameters
    ----------
    trace:
        Trace being rendered.
    context:
        Render context used by the caller's render.

    Returns
    -------
    dict[str, int]
        Per-module rendered hidden contribution. A module replacing ``n``
        rendered nodes with one box contributes ``n - 1``.
    """

    from .rendering import (
        BoundaryNode,
        _entries_to_plot_for_context,
        _is_buffer_visible,
        _normalize_buffer_visibility,
    )

    absorbed_counts: dict[str, int] = defaultdict(int)
    show_buffer_layers = _normalize_buffer_visibility(context.show_buffer_layers)
    entries_to_plot = _entries_to_plot_for_context(trace, context.vis_mode)
    for node in entries_to_plot.values():
        if isinstance(node, BoundaryNode):
            continue
        if node.is_buffer and not _is_buffer_visible(node, show_buffer_layers):
            continue
        modules = list(getattr(node, "modules", ()) or ())
        if getattr(node, "is_atomic_module", False) and modules:
            modules = modules[:-1]
        addresses = tuple(dict.fromkeys(str(module).rsplit(":", 1)[0] for module in modules))
        for address in addresses:
            absorbed_counts[address] += 1
    return {
        address: max(absorbed_count - 1, 0)
        for address, absorbed_count in absorbed_counts.items()
        if absorbed_count > 1
    }


def _assert_plan_count(
    trace: "Trace",
    collapse_fn: Callable[["Module"], bool] | None,
    run_folds: Mapping[str, ModuleRunFold] | None,
    context: RenderContext,
    running_count: int,
) -> None:
    """Assert that incremental count maintenance matches full planning.

    Parameters
    ----------
    trace:
        Trace being rendered.
    collapse_fn:
        Active collapse predicate.
    run_folds:
        Active run-fold mapping.
    context:
        Render context used for planning.
    running_count:
        Incrementally maintained rendered node count.
    """

    planned_count = count(plan_from_v1(trace, collapse_fn, run_folds, context))
    if running_count == planned_count:
        return
    message = (
        "incremental collapse count mismatch: "
        f"running_count={running_count}, planned_count={planned_count}"
    )
    if _strict_count_checks_enabled():
        raise AssertionError(message)
    _warn_count_mismatch_once(message)


def _strict_count_checks_enabled() -> bool:
    """Return whether collapse count mismatches should fail loudly.

    Returns
    -------
    bool
        True under pytest or when ``TORCHLENS_COLLAPSE_STRICT=1`` is set.
    """

    return os.environ.get("TORCHLENS_COLLAPSE_STRICT") == "1" or "PYTEST_CURRENT_TEST" in os.environ


def _warn_count_mismatch_once(message: str) -> None:
    """Emit a single production warning for incremental count mismatches.

    Parameters
    ----------
    message:
        Diagnostic mismatch message.
    """

    global _COUNT_MISMATCH_WARNING_EMITTED
    if _COUNT_MISMATCH_WARNING_EMITTED:
        return
    _COUNT_MISMATCH_WARNING_EMITTED = True
    warnings.warn(
        f"{message}; using authoritative CollapsePlan count for rendering.",
        RuntimeWarning,
        stacklevel=3,
    )


def _run_fold_hidden_member_contributions(
    trace: "Trace",
    collapse_fn: Callable[["Module"], bool] | None,
    context: RenderContext,
) -> dict[str, int]:
    """Return pre-fold rendered contribution under each module address.

    Parameters
    ----------
    trace:
        Trace being rendered.
    collapse_fn:
        Active collapse predicate before run folding, or ``None`` for an
        uncollapsed render.
    context:
        Render context used for the caller's render.

    Returns
    -------
    dict[str, int]
        Count of currently rendered node contributions contained by each
        module address. Boundary nodes with no module ancestry are excluded.
    """

    from .rendering import rendered_node_universe_from_v1

    contributions: dict[str, int] = defaultdict(int)
    emissions = rendered_node_universe_from_v1(
        trace,
        collapse_fn=collapse_fn,
        run_folds=None,
        context=context,
    )
    for emission in emissions:
        if emission.kind in {"hidden_run_member", "run_fold_ellipsis"}:
            continue
        addresses = _emission_module_ancestors(emission)
        for address in addresses:
            contributions[address] += 1
    return contributions


def _emission_module_ancestors(emission: Any) -> tuple[str, ...]:
    """Return pass-free module ancestors for a rendered emission.

    Parameters
    ----------
    emission:
        Diagnostic rendered-node emission from the renderer.

    Returns
    -------
    tuple[str, ...]
        Pass-free module addresses enclosing the emitted node.
    """

    addresses: list[str] = []
    if emission.module_address is not None:
        addresses.append(emission.module_address)
    node = emission.node
    if node is None:
        return tuple(dict.fromkeys(addresses))
    addresses.extend(str(module).rsplit(":", 1)[0] for module in getattr(node, "modules", ()) or ())
    return tuple(dict.fromkeys(addresses))


def _run_fold_delta(fold: ModuleRunFold, hidden_member_contributions: Mapping[str, int]) -> int:
    """Return rendered-node delta for accepting ``fold``.

    Parameters
    ----------
    fold:
        Candidate run fold.
    hidden_member_contributions:
        Pre-fold contribution count keyed by module address.

    Returns
    -------
    int
        Incremental count change after replacing all member contributions with
        the representative box plus one ellipsis node.
    """

    removed = sum(hidden_member_contributions.get(address, 0) for address in fold.addresses)
    return 2 - removed
