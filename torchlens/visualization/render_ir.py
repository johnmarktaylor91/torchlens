"""Resolved render-IR adapters for TorchLens graph visualization."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from .request import RenderContext

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from ..data_classes.module import Module
    from ..data_classes.trace import Trace
    from ._render_common import RenderedNodeEmission
    from ._render_edges import _SegmentLookup
    from .auto_collapse import ModuleRepeatFold
    from .node_spec import NodeSpec
    from .node_universe import NodeUnit
    from .renderers.base import RendererCapabilities


@dataclass(frozen=True)
class RenderIRNode:
    """Resolved render node independent of Graphviz mutation.

    Parameters
    ----------
    name:
        Rendered node identifier.
    kind:
        Semantic node kind from the renderer-faithful node universe.
    owner_cluster:
        Pass-free module cluster that owns the node when known.
    source_label:
        Source op/layer label or module call backing the node.
    hidden_originals:
        Original module addresses hidden by this rendered node.
    """

    name: str
    kind: Literal[
        "raw_op",
        "module_box",
        "boundary",
        "run_fold_ellipsis",
        "hidden_run_member",
        "grad_fn",
        "grad_fn_call",
    ]
    owner_cluster: str | None
    source_label: str | None
    hidden_originals: tuple[str, ...] = ()
    label_spans: tuple[str, ...] = ()
    node_calls: tuple[Any, ...] = ()
    owned_node_args: tuple[tuple[str, dict[str, Any]], ...] = ()
    node_color: str = "black"
    # S5 contract (C4): typed NodeSpec | None (was Any).
    node_spec: NodeSpec | None = None
    region_path: tuple[str, ...] = ()


@dataclass(frozen=True)
class _NodeDecisionInputs:
    """Per-draw inputs shared by every render-node presentation decision."""

    universe: Any
    repeat_folds: Mapping[str, ModuleRepeatFold] | None
    segment_lookup: _SegmentLookup
    sibling_counts: Mapping[str, int] | None
    suppressed_args: Mapping[int, frozenset[str]]


@dataclass(frozen=True)
class RenderIREdge:
    """Resolved forward edge independent of DOT emission.

    Parameters
    ----------
    source_unit:
        Rendered source endpoint after collapse/repeat-fold projection.
    target_unit:
        Rendered target endpoint after collapse/repeat-fold projection.
    source_originals:
        Original TorchLens source labels represented by the edge.
    target_originals:
        Original TorchLens target labels represented by the edge.
    owner_cluster:
        Cluster key where the edge is emitted, or ``None`` for top level.
    occurrence_key:
        Stable renderer occurrence key for the represented edge.
    projection_reason:
        Reason endpoint projection changed the original source/target units.
    """

    source_unit: str
    target_unit: str
    source_originals: tuple[str, ...]
    target_originals: tuple[str, ...]
    owner_cluster: str | None
    occurrence_key: tuple[Any, ...]
    projection_reason: Literal[
        "direct",
        "source_projected",
        "target_projected",
        "both_projected",
        "run_fold_ellipsis",
    ]
    tail_name: str | None = None
    head_name: str | None = None
    attrs: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class RenderIRRegion:
    """One nested, renderer-neutral region selected for DOT emission.

    Parameters
    ----------
    key:
        Stable region key.
    parent_key:
        Parent region key, or ``None`` for a top-level region.
    kind:
        Semantic region category.
    label:
        Resolved display label.
    style:
        Ordered Graphviz-compatible region attributes.
    node_names:
        Rendered nodes owned by the region in emission order.
    edge_indexes:
        Indexes into :attr:`RenderIR.edges` emitted inside the cluster.
    """

    key: str
    parent_key: str | None
    kind: Literal["module", "container", "backward_pass", "combined", "intervening_grad_fn"]
    label: str
    style: tuple[tuple[str, str], ...]
    node_names: tuple[str, ...]
    edge_indexes: tuple[int, ...]


@dataclass(frozen=True)
class RenderIROrderingConstraint:
    """Declarative sibling-ordering constraint resolved before backend layout.

    Parameters
    ----------
    kind:
        Constraint family.
    source_label:
        Source TorchLens label for diagnostics.
    source_name:
        Rendered source node identifier.
    targets:
        Rendered target identifiers in required order.
    target_labels:
        Source labels corresponding to ``targets``.
    lca_key:
        Region key where the constraint is emitted, or ``-1`` at top level.
    """

    kind: Literal["sibling_order"]
    source_label: str
    source_name: str
    targets: tuple[str, ...]
    target_labels: tuple[str, ...]
    lca_key: str | int


@dataclass(frozen=True)
class RenderIRRankGroup:
    """One stacking (rank) channel group: nodes pinned to one Graphviz rank.

    Parameters
    ----------
    kind:
        Constraint kind; v1 emits ``"stack"`` only (the ``stack_by``
        encoding channel; the license/cohort semantics live in
        ``visualization._stacking``).
    key:
        Repr of the shared annotation value (rank key).
    members:
        Rendered node names pinned to the shared rank.
    """

    kind: str
    key: str
    members: tuple[str, ...]


@dataclass(frozen=True)
class RenderIRDotStatement:
    """Immutable backend-ready DOT statement without TorchLens host objects."""

    kind: Literal["node", "edge", "attr", "subgraph", "raw"]
    args: tuple[Any, ...] = ()
    attrs: tuple[tuple[str, Any], ...] = ()
    children: tuple[RenderIRDotStatement, ...] = ()


@dataclass(frozen=True)
class RenderIR:
    """Resolved render description slice used before DOT emission.

    Parameters
    ----------
    context:
        Render context used to resolve visibility.
    nodes:
        Resolved nodes in deterministic renderer emission order.
    edges:
        Resolved forward edges in deterministic renderer traversal order.
    regions:
        Nested region records in deterministic emission order.
    ordering_constraints:
        Declarative backend layout constraints.
    """

    context: RenderContext
    nodes: tuple[RenderIRNode, ...]
    edges: tuple[RenderIREdge, ...]
    regions: tuple[RenderIRRegion, ...]
    ordering_constraints: tuple[RenderIROrderingConstraint, ...] = ()
    dot_statements: tuple[RenderIRDotStatement, ...] = ()
    # Stacking channel (L5 M3): rank=same groups resolved at the prepass.
    stack_rank_groups: tuple[RenderIRRankGroup, ...] = ()

    def required_capabilities(self) -> RendererCapabilities:
        """Return backend features required to render this IR exactly.

        Returns
        -------
        RendererCapabilities
            Capability set inferred from decision-complete records.
        """

        from .renderers.base import RendererCapabilities

        return RendererCapabilities(
            nested_regions=bool(self.regions),
            ordering_constraints=bool(self.ordering_constraints),
            html_labels=any(_statement_uses_html(statement) for statement in self.dot_statements),
            layout_execution=True,
        )


def _statement_uses_html(statement: RenderIRDotStatement) -> bool:
    """Return whether a statement tree contains a Graphviz HTML label.

    Parameters
    ----------
    statement:
        Statement tree to inspect.

    Returns
    -------
    bool
        Whether a label uses Graphviz HTML syntax.
    """

    attrs: Mapping[str, Any] = dict(statement.attrs)
    label = attrs.get("label")
    return (isinstance(label, str) and label.startswith("<")) or any(
        _statement_uses_html(child) for child in statement.children
    )


def projected_antiparallel_endpoint_pairs(render_ir: RenderIR) -> frozenset[tuple[str, str]]:
    """Return projected IR edge endpoints with an opposite projected edge.

    Parameters
    ----------
    render_ir:
        Render IR whose projected edge endpoints should be inspected.

    Returns
    -------
    frozenset[tuple[str, str]]
        Directed endpoint pairs that should receive explicit anti-parallel styling.
    """

    projected_edges = [edge for edge in render_ir.edges if edge.projection_reason != "direct"]
    projected_pairs = {(edge.source_unit, edge.target_unit) for edge in projected_edges}
    anti_parallel_pairs: set[tuple[str, str]] = set()
    for source_unit, target_unit in projected_pairs:
        if source_unit != target_unit and (target_unit, source_unit) in projected_pairs:
            anti_parallel_pairs.add((source_unit, target_unit))
    return frozenset(anti_parallel_pairs)


def build_render_ir(
    trace: Trace,
    *,
    collapse_fn: Callable[[Module], bool] | None,
    repeat_folds: Mapping[str, ModuleRepeatFold] | None,
    context: RenderContext | None = None,
    universe: Any | None = None,
    segments: Mapping[str, Any] | None = None,
    segment_lookup: _SegmentLookup | None = None,
    suppressed_args: Mapping[int, frozenset[str]] | None = None,
) -> RenderIR:
    """Build the first render-IR slice from current renderer-faithful emissions.

    Parameters
    ----------
    trace:
        Trace being rendered.
    collapse_fn:
        Active collapse predicate.
    repeat_folds:
        Active repeat-fold descriptors.
    context:
        Render context. Defaults to :class:`RenderContext`.

    Returns
    -------
    RenderIR
        Node-level render IR whose emission order matches the current renderer.
    """

    resolved_context = RenderContext() if context is None else context
    if segment_lookup is None:
        from ._render_edges import _build_segment_lookup

        segment_lookup = _build_segment_lookup(segments)
    if universe is None:
        from .node_universe import build_node_universe
        from .source_graph import build_source_graph

        universe = build_node_universe(
            build_source_graph(trace, resolved_context), collapse_fn, repeat_folds
        )
    encoding = getattr(resolved_context, "encoding", None)
    if encoding is not None:
        # PHASE A of the encoding channel (L5): a data prepass over the
        # already-chosen visible-node universe, before any per-node spec
        # resolution -- collect values once, validate, compute the domain.
        from ._encoding import populate_encoding_state

        populate_encoding_state(encoding, trace, universe)
    if suppressed_args is None and not getattr(resolved_context, "show_redundant_args", False):
        # Checked suppression (L5 M4, DEFAULT-ON): the trace-bearing prepass
        # proves which constructor-arg rows duplicate captured shapes on
        # THIS trace; unprovable or mismatching args stay visible.
        from ._arg_suppression import compute_suppressed_arg_keys

        suppressed_args = compute_suppressed_arg_keys(trace, universe)
    suppressed_args = suppressed_args or {}
    from ._render_nodes import _atomic_module_sibling_counts

    sibling_counts = _atomic_module_sibling_counts(trace)
    decision_inputs = _NodeDecisionInputs(
        universe=universe,
        repeat_folds=repeat_folds,
        segment_lookup=segment_lookup,
        sibling_counts=sibling_counts,
        suppressed_args=suppressed_args,
    )
    nodes = tuple(
        _node_from_unit(
            trace,
            unit,
            resolved_context,
            decision_inputs,
        )
        for unit in universe.units
    )
    edges = _build_forward_edges_from_universe(universe)
    regions = _build_regions(trace, nodes, edges)
    _warn_if_render_exceeds_disclosure_ceiling(len(nodes), len(edges))
    stack_rank_groups: tuple[RenderIRRankGroup, ...] = ()
    if encoding is not None:
        stack_rank_groups = tuple(
            RenderIRRankGroup(kind="stack", key=key, members=members)
            for key, members in getattr(encoding, "stack_groups", ()) or ()
        )
    return RenderIR(
        context=resolved_context,
        nodes=nodes,
        edges=edges,
        regions=regions,
        stack_rank_groups=stack_rank_groups,
    )


#: Disclosure ceiling for one render (R60): draw() has no hard node/edge cap
#: anywhere — a hard refusal is a public-behavior decision (it would break
#: legitimate giant renders such as model-catalog sweeps) — but past this size
#: Graphviz layout time and memory grow super-linearly, so the user gets an
#: actionable warning instead of an unexplained multi-minute hang.
RENDER_DISCLOSURE_NODE_CEILING = 10_000
RENDER_DISCLOSURE_EDGE_CEILING = 40_000


def _warn_if_render_exceeds_disclosure_ceiling(num_nodes: int, num_edges: int) -> None:
    """Warn once per render when the resolved graph is Graphviz-hostile.

    Parameters
    ----------
    num_nodes:
        Resolved visible node count.
    num_edges:
        Resolved visible edge count.
    """

    if num_nodes <= RENDER_DISCLOSURE_NODE_CEILING and num_edges <= RENDER_DISCLOSURE_EDGE_CEILING:
        return
    import warnings

    from ..errors._base import TorchLensWarning
    from ..utils.display import user_stacklevel

    warnings.warn(
        f"TorchLens is rendering {num_nodes} nodes / {num_edges} edges; Graphviz "
        "layout beyond ~10k nodes can take minutes and gigabytes. Consider "
        "collapse='auto' / collapse='max', show_containers=False, or drawing a "
        "focused subgraph.",
        TorchLensWarning,
        stacklevel=user_stacklevel(),
    )


def build_backward_render_ir(
    trace: Trace,
    *,
    vis_mode: Literal["rolled", "unrolled"],
    pass_filter: set[int] | None,
    dot_statements: tuple[RenderIRDotStatement, ...],
) -> RenderIR:
    """Normalize a captured backward graph into the package-wide render IR.

    Parameters
    ----------
    trace:
        Trace containing captured grad-function handles and calls.
    vis_mode:
        Rolled handle mode or unrolled call mode.
    pass_filter:
        Optional one-based backward-pass indexes to retain.
    dot_statements:
        Fully resolved backend statements in byte-stable emission order.

    Returns
    -------
    RenderIR
        Renderer-neutral backward nodes, edges, regions, and resolved statements.
    """

    nodes, visible_by_pass = _normalize_backward_nodes(trace, vis_mode, pass_filter)
    edges = _normalize_backward_edges(trace, vis_mode, pass_filter, visible_by_pass)
    regions = _normalize_backward_regions(nodes, visible_by_pass)
    _warn_if_render_exceeds_disclosure_ceiling(len(nodes), len(edges))
    return RenderIR(
        context=RenderContext(vis_mode=vis_mode),
        nodes=nodes,
        edges=edges,
        regions=regions,
        dot_statements=dot_statements,
    )


def build_combined_render_ir(
    trace: Trace,
    forward_ir: RenderIR,
    *,
    pass_filter: set[int] | None,
    intervening_node_names: tuple[str, ...],
    dot_statements: tuple[RenderIRDotStatement, ...],
) -> RenderIR:
    """Merge normalized forward and backward sources into one render IR.

    Parameters
    ----------
    trace:
        Trace containing the paired forward and backward structures.
    forward_ir:
        Normalized unrolled forward graph.
    pass_filter:
        Optional one-based backward-pass indexes to retain.
    intervening_node_names:
        Grad-function node names assigned to the special intervening region.
    dot_statements:
        Fully resolved backend statements in byte-stable emission order.

    Returns
    -------
    RenderIR
        One IR containing both source normalizations.
    """

    backward_nodes, visible_by_pass = _normalize_backward_nodes(trace, "rolled", pass_filter)
    backward_edges = _normalize_backward_edges(trace, "rolled", pass_filter, visible_by_pass)
    correspondence_edges = _normalize_correspondence_edges(trace, pass_filter)
    regions = list(forward_ir.regions)
    if intervening_node_names:
        regions.append(
            RenderIRRegion(
                key="__intervening__",
                parent_key=None,
                kind="intervening_grad_fn",
                label="intervening grad_fns",
                style=(),
                node_names=intervening_node_names,
                edge_indexes=(),
            )
        )
    return replace(
        forward_ir,
        nodes=forward_ir.nodes + backward_nodes,
        edges=forward_ir.edges + backward_edges + correspondence_edges,
        regions=tuple(regions),
        dot_statements=dot_statements,
    )


def _normalize_backward_nodes(
    trace: Trace,
    vis_mode: Literal["rolled", "unrolled"],
    pass_filter: set[int] | None,
) -> tuple[tuple[RenderIRNode, ...], dict[int, list[tuple[Any, Any]]]]:
    """Normalize visible grad-function handles or calls into IR nodes."""

    from ._render_leaf import (
        _backward_dot_call_node_name,
        _backward_dot_node_name,
        _grad_fn_call_matches_backward_filter,
        _grad_fn_matches_backward_filter,
    )

    visible_by_pass: defaultdict[int, list[tuple[Any, Any]]] = defaultdict(list)
    nodes: list[RenderIRNode] = []
    for grad_fn in trace.grad_fns:
        if vis_mode == "rolled":
            if not _grad_fn_matches_backward_filter(grad_fn, pass_filter):
                continue
            nodes.append(
                RenderIRNode(
                    name=_backward_dot_node_name(grad_fn),
                    kind="grad_fn",
                    owner_cluster=None,
                    source_label=grad_fn.type,
                )
            )
            continue
        for call in grad_fn.calls.values():
            if not _grad_fn_call_matches_backward_filter(call, pass_filter):
                continue
            pass_index = getattr(call, "backward_pass_index", None)
            if pass_index is None:
                continue
            visible_by_pass[int(pass_index)].append((grad_fn, call))
            nodes.append(
                RenderIRNode(
                    name=_backward_dot_call_node_name(grad_fn, call),
                    kind="grad_fn_call",
                    owner_cluster=f"backward_pass_{pass_index}",
                    source_label=grad_fn.type,
                    region_path=(f"backward_pass_{pass_index}",),
                )
            )
    return tuple(nodes), dict(sorted(visible_by_pass.items()))


def _normalize_backward_edges(
    trace: Trace,
    vis_mode: Literal["rolled", "unrolled"],
    pass_filter: set[int] | None,
    visible_by_pass: dict[int, list[tuple[Any, Any]]],
) -> tuple[RenderIREdge, ...]:
    """Normalize visible grad-function dependencies into IR edges."""

    from ._render_leaf import (
        _backward_dot_call_node_name,
        _backward_dot_node_name,
        _grad_fn_matches_backward_filter,
    )

    edges: list[RenderIREdge] = []
    if vis_mode == "rolled":
        visible_ids = {
            grad_fn.grad_fn_object_id
            for grad_fn in trace.grad_fns
            if _grad_fn_matches_backward_filter(grad_fn, pass_filter)
        }
        for grad_fn in trace.grad_fns:
            if grad_fn.grad_fn_object_id not in visible_ids:
                continue
            for next_id in grad_fn.next_grad_fn_ids:
                if next_id not in visible_ids:
                    continue
                head = trace.grad_fn_logs[next_id]
                tail_name = _backward_dot_node_name(grad_fn)
                head_name = _backward_dot_node_name(head)
                edges.append(
                    _normalized_grad_edge(
                        tail_name,
                        head_name,
                        grad_fn,
                        head,
                        len(edges),
                    )
                )
        return tuple(edges)
    for pass_index, calls in visible_by_pass.items():
        calls_by_id: defaultdict[int, list[tuple[Any, Any]]] = defaultdict(list)
        for grad_fn, call in calls:
            calls_by_id[grad_fn.grad_fn_object_id].append((grad_fn, call))
        for grad_fn, call in calls:
            for next_id in grad_fn.next_grad_fn_ids:
                for head, head_call in calls_by_id.get(next_id, ()):
                    edges.append(
                        _normalized_grad_edge(
                            _backward_dot_call_node_name(grad_fn, call),
                            _backward_dot_call_node_name(head, head_call),
                            grad_fn,
                            head,
                            len(edges),
                            owner_cluster=f"backward_pass_{pass_index}",
                        )
                    )
    return tuple(edges)


def _normalized_grad_edge(
    tail_name: str,
    head_name: str,
    tail: Any,
    head: Any,
    index: int,
    owner_cluster: str | None = None,
) -> RenderIREdge:
    """Create one normalized backward dependency edge."""

    from ._render_leaf import _backward_edge_attrs

    return RenderIREdge(
        source_unit=tail_name,
        target_unit=head_name,
        source_originals=(tail_name,),
        target_originals=(head_name,),
        owner_cluster=owner_cluster,
        occurrence_key=("grad", index, tail_name, head_name),
        projection_reason="direct",
        tail_name=tail_name,
        head_name=head_name,
        attrs=tuple(_backward_edge_attrs(tail, head).items()),
    )


def _normalize_correspondence_edges(
    trace: Trace, pass_filter: set[int] | None
) -> tuple[RenderIREdge, ...]:
    """Normalize visible forward-to-grad-function correspondence edges."""

    from ._render_leaf import _backward_dot_node_name, _grad_fn_matches_backward_filter

    edges: list[RenderIREdge] = []
    for grad_fn in trace.grad_fns:
        if not grad_fn.has_op or not _grad_fn_matches_backward_filter(grad_fn, pass_filter):
            continue
        op = grad_fn.op
        if op is None:
            continue
        head_name = _backward_dot_node_name(grad_fn)
        edges.append(
            RenderIREdge(
                source_unit=op.layer_label,
                target_unit=head_name,
                source_originals=(op.layer_label,),
                target_originals=(head_name,),
                owner_cluster=None,
                occurrence_key=("correspondence", len(edges), op.layer_label, head_name),
                projection_reason="direct",
                tail_name=op.layer_label,
                head_name=head_name,
            )
        )
    return tuple(edges)


def _normalize_backward_regions(
    nodes: tuple[RenderIRNode, ...],
    visible_by_pass: dict[int, list[tuple[Any, Any]]],
) -> tuple[RenderIRRegion, ...]:
    """Normalize unrolled backward-pass groups into IR regions."""

    return tuple(
        RenderIRRegion(
            key=f"backward_pass_{pass_index}",
            parent_key=None,
            kind="backward_pass",
            label=f"backward pass {pass_index}",
            style=(),
            node_names=tuple(
                node.name for node in nodes if node.owner_cluster == f"backward_pass_{pass_index}"
            ),
            edge_indexes=(),
        )
        for pass_index in visible_by_pass
    )


def _build_forward_edges_from_universe(universe: Any) -> tuple[RenderIREdge, ...]:
    """Build forward IR edges from projected universe occurrences.

    Parameters
    ----------
    universe:
        Shared presentation-free node universe.

    Returns
    -------
    tuple[RenderIREdge, ...]
        Projected forward edges in source traversal order.
    """

    edges: list[RenderIREdge] = []
    for occurrence in universe.projected_edges:
        raw_source = occurrence.source_label.replace(":", "pass")
        raw_target = occurrence.target_label.replace(":", "pass")
        reason = _projection_reason(
            raw_source,
            raw_target,
            occurrence.source_unit,
            occurrence.target_unit,
        )
        if occurrence.source_unit == occurrence.target_unit:
            continue
        edges.append(
            RenderIREdge(
                source_unit=occurrence.source_unit,
                target_unit=occurrence.target_unit,
                source_originals=(occurrence.source_label,),
                target_originals=(occurrence.target_label,),
                owner_cluster=None,
                occurrence_key=occurrence.occurrence_key,
                projection_reason=reason,
            )
        )
    return tuple(edges)


def _node_from_unit(
    trace: Trace,
    unit: NodeUnit,
    context: RenderContext,
    inputs: _NodeDecisionInputs,
) -> RenderIRNode:
    """Decorate one structural node-universe unit as a render-IR node."""

    emission = unit.emission

    hidden_originals: tuple[str, ...] = ()
    if emission.fold is not None and emission.kind in {"module_box", "run_fold_ellipsis"}:
        hidden_originals = tuple(emission.fold.addresses)
    source_label = emission.op_label or emission.call or emission.boundary_kind
    owner_cluster = emission.module_address
    if emission.kind == "module_box" and emission.call is not None:
        from ._render_edges import _collapsed_module_owner_key

        address, _, call_index = emission.call.partition(":")
        owner_cluster = _collapsed_module_owner_key(
            trace,
            address,
            call_index or "1",
            context.vis_mode,
        )
    elif emission.kind == "run_fold_ellipsis" and emission.fold is not None:
        from ._render_edges import _run_fold_ellipsis_owner_key

        owner_cluster = _run_fold_ellipsis_owner_key(trace, emission.fold, context.vis_mode)
    node_calls: tuple[Any, ...] = ()
    owned_node_args: tuple[tuple[str, dict[str, Any]], ...] = ()
    node_color = "black"
    label_spans: tuple[str, ...] = ()
    node_spec: NodeSpec | None = None
    region_path: tuple[str, ...] = ()
    if emission.node is not None:
        node_calls, owned_node_args, node_color, node_spec, label_spans = _resolve_node_decision(
            trace,
            emission,
            context,
            inputs,
        )
        modules = list(emission.node.modules)
        if emission.kind == "module_box":
            modules = modules[: context.vis_call_depth - 1]
        if context.vis_mode == "rolled":
            modules = list(dict.fromkeys(module.split(":")[0] for module in modules))
        region_path = tuple(modules)
        if region_path:
            owner_cluster = region_path[-1]
    return RenderIRNode(
        name=emission.name,
        kind=emission.kind,
        owner_cluster=owner_cluster,
        source_label=source_label,
        hidden_originals=hidden_originals,
        label_spans=label_spans,
        node_calls=node_calls,
        owned_node_args=owned_node_args,
        node_color=node_color,
        node_spec=node_spec,
        region_path=region_path,
    )


def _resolve_node_decision(
    trace: Trace,
    emission: RenderedNodeEmission,
    context: RenderContext,
    inputs: _NodeDecisionInputs,
) -> tuple[
    tuple[Any, ...],
    tuple[tuple[str, dict[str, Any]], ...],
    str,
    NodeSpec | None,
    tuple[str, ...],
]:
    """Resolve one visible node's complete presentation decision.

    Parameters
    ----------
    trace:
        Trace that owns the source node.
    emission:
        Visible structural emission being decorated.
    context:
        Fully resolved render request.
    inputs:
        Per-draw structural maps shared across node decisions.

    Returns
    -------
    tuple
        Recorded top-level calls, owned node arguments, edge color, and structured label spans.
    """
    from collections import defaultdict

    from ._render_common import _RenderIRDecisionBuilder
    from ._render_edges import _segment_for_node
    from ._render_flow import _collapsed_container_leaf_nodes
    from ._render_nodes import (
        _build_collapsed_module_node,
        _build_layer_node,
        _normalize_buffer_visibility,
    )
    from .themes import resolve_theme

    node = emission.node
    if node is None:
        return (), (), "black", None, ()
    if _segment_for_node(node, inputs.segment_lookup) is not None:
        return (), (), "black", None, ()
    recorder = _RenderIRDecisionBuilder()
    module_nodes: dict[str, Any] = defaultdict(dict)
    show_buffers = _normalize_buffer_visibility(context.show_buffer_layers)
    collapsed_containers = _collapsed_container_leaf_nodes(
        trace,
        inputs.universe.source_graph.entries_to_plot,
        vis_mode=context.vis_mode,
        show_containers=context.show_containers,
        container_max_inline=context.container_max_inline,
        pending_nodes=[],
    )
    theme = resolve_theme(context.theme, for_paper=context.for_paper)
    resolved_specs: list[Any] = []
    if emission.kind == "module_box":
        _build_collapsed_module_node(
            trace,
            node,
            recorder,
            module_nodes,
            set(),
            context.vis_mode,
            context.vis_call_depth,
            emission.call,
            context.overrides,
            context.node_mode,
            context.collapsed_node_spec_fn,
            theme,
            inputs.repeat_folds,
            context.collapse_fn,
            resolved_specs,
        )
        color = "black"
    else:
        color = _build_layer_node(
            trace,
            node,
            recorder,
            show_buffers,
            context.vis_mode,
            context.overrides,
            context.node_mode,
            context.node_spec_fn,
            theme,
            context.node_overlay,
            list(context.node_label_fields) if context.node_label_fields is not None else None,
            context.show_containers,
            collapsed_containers,
            context.show_input_transform_summary,
            resolved_specs,
            inputs.sibling_counts,
            encoding=getattr(context, "encoding", None),
            suppressed_args=inputs.suppressed_args,
            show_saved_for_backward=getattr(context, "show_saved_for_backward", False),
        )
    owned = tuple(
        (owner, dict(args))
        for owner, payload in module_nodes.items()
        for args in payload.get("nodes", ())
    )
    spec = resolved_specs[0] if resolved_specs else None
    spans = tuple(str(line) for line in spec.lines) if spec is not None else ()
    return tuple(recorder.calls), owned, color, spec, spans


def _projection_reason(
    source_name: str,
    target_name: str,
    source_unit: str,
    target_unit: str,
) -> Literal["direct", "source_projected", "target_projected", "both_projected"]:
    """Return whether endpoint projection changed either rendered endpoint.

    Parameters
    ----------
    source_name:
        Original rendered source node name.
    target_name:
        Original rendered target node name.
    source_unit:
        Projected source endpoint.
    target_unit:
        Projected target endpoint.

    Returns
    -------
    Literal["direct", "source_projected", "target_projected", "both_projected"]
        Projection reason before repeat-fold ellipsis projection is applied.
    """

    source_projected = source_unit != source_name
    target_projected = target_unit != target_name
    if source_projected and target_projected:
        return "both_projected"
    if source_projected:
        return "source_projected"
    if target_projected:
        return "target_projected"
    return "direct"


def _build_regions(
    trace: Trace,
    nodes: tuple[RenderIRNode, ...],
    edges: tuple[RenderIREdge, ...],
) -> tuple[RenderIRRegion, ...]:
    """Build preliminary nested module regions from resolved ownership.

    Parameters
    ----------
    nodes:
        Resolved IR nodes.
    edges:
        Resolved IR edges.

    Returns
    -------
    tuple[RenderIRRegion, ...]
        Module regions sorted by stable key. Later forward-DOT assembly enriches
        them with the final style and container members before serialization.
    """

    cluster_nodes: defaultdict[str, list[str]] = defaultdict(list)
    cluster_edges: defaultdict[str, list[int]] = defaultdict(list)
    for node in nodes:
        if node.owner_cluster is not None:
            cluster_nodes[node.owner_cluster].append(node.name)
    for index, edge in enumerate(edges):
        if edge.owner_cluster is not None:
            cluster_edges[edge.owner_cluster].append(index)
    region_keys = set(cluster_nodes) | set(cluster_edges)
    if not region_keys:
        region_keys = {
            module.address
            for module in trace.modules
            if getattr(module, "address", "self") != "self"
        }
    return tuple(
        RenderIRRegion(
            key=key,
            parent_key=_region_parent_key(key, region_keys),
            kind="module",
            label=key,
            style=(),
            node_names=tuple(cluster_nodes.get(key, ())),
            edge_indexes=tuple(cluster_edges.get(key, ())),
        )
        for key in sorted(region_keys)
    )


def _region_parent_key(key: str, keys: set[str]) -> str | None:
    """Return the nearest emitted module parent for a region key.

    Parameters
    ----------
    key:
        Region key using module-address syntax.
    keys:
        All emitted region keys.

    Returns
    -------
    str | None
        Nearest ancestor key present in ``keys``.
    """

    address, separator, call = key.partition(":")
    parts = address.split(".")
    for length in range(len(parts) - 1, 0, -1):
        parent_address = ".".join(parts[:length])
        parent = f"{parent_address}{separator}{call}" if separator else parent_address
        if parent in keys:
            return parent
    return None


def finalize_forward_regions(
    render_ir: RenderIR,
    trace: Trace,
    *,
    vis_mode: str,
    module_payloads: dict[str, Any],
    container_regions: tuple[Any, ...],
    captured_edges: tuple[Any, ...],
    overrides: Any,
    rankdir: str | None = None,
) -> RenderIR:
    """Attach final DOT region decisions to a forward render IR.

    Parameters
    ----------
    render_ir:
        IR built before edge emission.
    trace:
        Trace supplying module labels and types.
    vis_mode:
        Active rolled or unrolled mode.
    module_payloads:
        Legacy edge-emission payloads, consumed once to complete region membership.
    container_regions:
        Resolved opt-in container region descriptors.
    captured_edges:
        Rendered edge occurrences used to assign edge ownership.
    overrides:
        Resolved visualization overrides.
    rankdir:
        Graphviz rank direction; enables member-geometry caption placement
        (vizmech item 15). ``None`` keeps the geometry-free defaults.

    Returns
    -------
    RenderIR
        The same IR with immutable, decision-complete nested region records.
    """

    from ._render_flow import _get_max_call_depth
    from ._render_utils import compute_module_penwidth, make_module_cluster_attrs

    captured_by_occurrence = {edge.occurrence_key: edge for edge in captured_edges}
    edges = tuple(
        replace(
            edge,
            owner_cluster=(
                str(captured_by_occurrence[edge.occurrence_key].module_key)
                if captured_by_occurrence.get(edge.occurrence_key) is not None
                and captured_by_occurrence[edge.occurrence_key].module_key != -1
                else None
            ),
            tail_name=(
                captured_by_occurrence[edge.occurrence_key].tail_name
                if edge.occurrence_key in captured_by_occurrence
                else edge.source_unit
            ),
            head_name=(
                captured_by_occurrence[edge.occurrence_key].head_name
                if edge.occurrence_key in captured_by_occurrence
                else edge.target_unit
            ),
            attrs=(
                captured_by_occurrence[edge.occurrence_key].attrs
                if edge.occurrence_key in captured_by_occurrence
                else ()
            ),
        )
        for edge in render_ir.edges
    )
    # Bucket every per-region lookup ONCE, the way ``_build_regions`` already does,
    # so the region loop below stays linear. Re-scanning ``render_ir.nodes`` /
    # ``edges`` / the module tree inside the loop is a linear number of full
    # collection scans, i.e. quadratic time on module-rich models.
    region_keys = set(module_payloads)
    region_nodes: defaultdict[str, list[str]] = defaultdict(list)
    region_subtree_nodes: defaultdict[str, list[str]] = defaultdict(list)
    for node in render_ir.nodes:
        region_keys.update(node.region_path)
        if node.region_path:
            region_nodes[node.region_path[-1]].append(node.name)
        for path_key in node.region_path:
            region_subtree_nodes[path_key].append(node.name)
    region_edge_indexes: defaultdict[str, list[int]] = defaultdict(list)
    for index, edge in enumerate(edges):
        if edge.owner_cluster is not None:
            region_edge_indexes[edge.owner_cluster].append(index)
    # Caption-placement geometry works on the EMITTED endpoint names (the
    # captured tail/head names), never the pre-replacement unit ids -- the
    # two spaces differ (pass suffixes), and a name mismatch silently
    # degrades every caption decision to the geometry-free default.
    emitted_edge_endpoints = tuple(
        (edge.tail_name or edge.source_unit, edge.head_name or edge.target_unit) for edge in edges
    )
    node_depths = (
        _node_flow_depths(render_ir, emitted_edge_endpoints) if rankdir is not None else {}
    )
    module_children, top_modules = _region_module_hierarchy(trace, vis_mode)
    max_depth = _get_max_call_depth(top_modules, module_payloads, module_children)
    call_depths = _region_call_depths(top_modules, module_children)
    regions: list[RenderIRRegion] = []
    for key in sorted(region_keys):
        address = key.split(":", 1)[0]
        module = trace.modules[address]
        label = _region_display_label(trace, module, key, address, vis_mode)
        payload = module_payloads.get(key, {})
        node_names = tuple(str(args.get("name", "")) for args in payload.get("nodes", ()))
        # ``node_names`` on the right-hand side of the historical ``+=`` was the
        # payload-derived prefix only (the tuple is fully built before rebinding),
        # so IR nodes are deduplicated against the prefix and NOT against each
        # other. ``payload_names`` reproduces exactly that scope.
        payload_names = set(node_names)
        node_names += tuple(name for name in region_nodes.get(key, ()) if name not in payload_names)
        caption_end, caption_pierced = _region_caption_plan(
            direct_names=node_names,
            # The DIRECT set is what is DRAWN inside this cluster (pruned leaf
            # child modules hoist their nodes into the parent's payload).
            subtree_names=tuple(region_subtree_nodes.get(key, ())) + node_names,
            node_depths=node_depths,
            edge_endpoints=emitted_edge_endpoints,
        )
        attrs = make_module_cluster_attrs(
            title=label,
            module_type=module.class_name,
            line_style="solid" if payload.get("has_input_ancestor") else "dashed",
            penwidth=compute_module_penwidth(
                call_depths.get(key, _module_depth(key) - 1), max_depth
            ),
            caption_end=caption_end,
            caption_pierced=caption_pierced,
            rankdir=rankdir,
        )
        for attr_name, attr_value in overrides.module.items():
            attrs[attr_name] = str(attr_value(trace, key) if callable(attr_value) else attr_value)
        regions.append(
            RenderIRRegion(
                key=key,
                parent_key=_region_parent_key(key, region_keys),
                kind="module",
                label=label,
                style=tuple(attrs.items()),
                node_names=node_names,
                edge_indexes=tuple(region_edge_indexes.get(key, ())),
            )
        )
    for container in container_regions:
        attrs = make_module_cluster_attrs(
            title=container.title,
            module_type=container.kind,
            line_style="dotted",
            penwidth=1.0,
            fillcolor="white",
        )
        owner = str(container.owner_key)
        regions.append(
            RenderIRRegion(
                key=f"container:{container.cluster_id}",
                parent_key=owner if owner in region_keys else None,
                kind="container",
                label=container.title,
                style=tuple(attrs.items()),
                node_names=container.node_names,
                edge_indexes=(),
            )
        )
    return replace(render_ir, edges=edges, regions=tuple(regions))


def _region_display_label(trace: Trace, module: Any, key: str, address: str, vis_mode: str) -> str:
    """Return the cluster display label for one module region."""

    from ._render_leaf import _collapsed_module_rolling_suffix

    if vis_mode == "unrolled" and getattr(module, "num_calls", 1) > 1:
        return key
    if vis_mode == "rolled" and getattr(module, "num_calls", 1) > 1:
        return f"{address} (x{module.num_calls}{_collapsed_module_rolling_suffix(trace, address)})"
    return address


def _node_flow_depths(
    render_ir: RenderIR, edge_endpoints: tuple[tuple[str, str], ...]
) -> dict[str, int]:
    """Return a BFS flow depth per node name over the emitted dataflow edges.

    A cheap emit-time stand-in for the layout ranks dot will compute: enough
    to tell whether a cluster's direct members sit at its entry or exit end
    (vizmech item 15). ``edge_endpoints`` are the EMITTED tail/head names
    (unit ids differ from node names on pass-suffixed graphs). Cycles are
    handled by the visited guard; unreached nodes default to depth 0.
    """

    children: defaultdict[str, list[str]] = defaultdict(list)
    in_degree: defaultdict[str, int] = defaultdict(int)
    names = {node.name for node in render_ir.nodes}
    for source, target in edge_endpoints:
        if source in names and target in names:
            children[source].append(target)
            in_degree[target] += 1
    depths: dict[str, int] = {}
    frontier = deque(sorted(name for name in names if in_degree[name] == 0))
    for name in frontier:
        depths[name] = 0
    while frontier:
        current = frontier.popleft()
        for child in children[current]:
            if child not in depths:
                depths[child] = depths[current] + 1
                frontier.append(child)
    return depths


def _region_caption_plan(
    *,
    direct_names: Any,
    subtree_names: Any,
    node_depths: dict[str, int],
    edge_endpoints: tuple[tuple[str, str], ...],
) -> tuple[str | None, bool]:
    """Plan a cluster caption: which flow end, and is that end pierced.

    Two forces (vizmech item 15 + the widened audit's finding):

    - FINDABILITY: the caption belongs at the end where the cluster's DIRECT
      members sit (the measured defect was captions up to 1121.9 pt from
      anything they name).
    - LEGIBILITY: boundary-crossing edge splines pierce the border near the
      end their inside endpoint occupies; a caption centered on a pierced
      end gets struck through (found by the widened audit on the first axis
      sweep). The end with FEWER crossings wins ties against the member end,
      and a pierced choice is reported so the caller can justify the caption
      into the corner instead of the spline channel.

    Returns ``(end, pierced)`` where ``end`` is ``"entry"``/``"exit"``/
    ``None`` (no usable geometry) and ``pierced`` says whether the chosen
    end has any crossing edges.
    """

    if not node_depths:
        return None, False
    direct_depths = [node_depths[name] for name in direct_names if name in node_depths]
    subtree = {name for name in subtree_names if name in node_depths}
    subtree_depths = [node_depths[name] for name in subtree]
    if not subtree_depths:
        return None, False
    if not direct_depths:
        # A region whose own direct set is empty (its ops live in pruned
        # leaf child regions that never render as clusters) effectively
        # DRAWS its whole subtree directly; the subtree distribution is the
        # honest member geometry.
        direct_depths = subtree_depths
    low, high = min(subtree_depths), max(subtree_depths)
    if high == low:
        return None, False
    midpoint = (low + high) / 2.0
    mean_direct = sum(direct_depths) / len(direct_depths)
    member_end = "entry" if mean_direct <= midpoint else "exit"

    crossings = {"entry": 0, "exit": 0}
    for source, target in edge_endpoints:
        source_in = source in subtree
        target_in = target in subtree
        if source_in == target_in:
            continue
        inside = source if source_in else target
        inside_depth = node_depths.get(inside, low)
        crossings["entry" if inside_depth <= midpoint else "exit"] += 1

    other_end = "exit" if member_end == "entry" else "entry"
    chosen = other_end if crossings[other_end] < crossings[member_end] else member_end
    return chosen, crossings[chosen] > 0


def _module_depth(key: str) -> int:
    """Return a module-address nesting depth for a region key.

    Parameters
    ----------
    key:
        Module region key, optionally pass-qualified.

    Returns
    -------
    int
        One-based module depth.
    """

    return len(key.split(":", 1)[0].split("."))


def _region_module_hierarchy(
    trace: Trace, vis_mode: str
) -> tuple[defaultdict[str, list[str]], list[str]]:
    """Return the module hierarchy used by legacy DOT subgraph emission.

    Parameters
    ----------
    trace:
        Trace containing recorded module calls.
    vis_mode:
        Active rolled or unrolled visualization mode.

    Returns
    -------
    tuple[defaultdict[str, list[str]], list[str]]
        Child mapping and top-level module keys in DOT emission order.
    """

    children: defaultdict[str, list[str]] = defaultdict(list)
    if vis_mode == "unrolled":
        for call_label, module_pass in trace.modules._pass_dict.items():
            children[call_label] = list(module_pass.call_children)
        return children, list(trace.modules["self"].ops[0].call_children)
    for module in trace.modules:
        if module.address != "self":
            children[module.address] = list(module.call_children)
    return children, list(trace.modules["self"].call_children)


def _region_call_depths(
    top_modules: list[str],
    children: Mapping[str, list[str]],
) -> dict[str, int]:
    """Return legacy BFS subgraph depths for every reachable module key.

    One breadth-first sweep replaces the historical per-key sweep. Both report
    the depth at which breadth-first traversal from ``top_modules``, in the same
    child order, first reaches a key -- the shortest root distance -- so every
    recorded depth matches the per-key search exactly. Keys the sweep never
    reaches are simply absent; callers keep the historical
    ``_module_depth(key) - 1`` fallback for those.

    Parameters
    ----------
    top_modules:
        Top-level emitted module keys.
    children:
        Module child mapping.

    Returns
    -------
    dict[str, int]
        Zero-based nesting depth per reachable key, used for module border widths.
    """

    depths: dict[str, int] = {}
    pending: deque[tuple[str, int]] = deque((candidate, 0) for candidate in top_modules)
    while pending:
        candidate, depth = pending.popleft()
        if candidate in depths:
            continue
        depths[candidate] = depth
        pending.extend((child, depth + 1) for child in children.get(candidate, ()))
    return depths
