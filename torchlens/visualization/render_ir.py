"""Resolved render-IR adapters for TorchLens graph visualization."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from .request import RenderContext

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from ..data_classes.module import Module
    from ..data_classes.trace import Trace
    from .auto_collapse import ModuleRepeatFold
    from .rendering import RenderedNodeEmission


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
    kind: Literal["raw_op", "module_box", "boundary", "run_fold_ellipsis", "hidden_run_member"]
    owner_cluster: str | None
    source_label: str | None
    hidden_originals: tuple[str, ...] = ()
    label_spans: tuple[str, ...] = ()
    node_calls: tuple[Any, ...] = ()
    owned_node_args: tuple[tuple[str, dict[str, Any]], ...] = ()
    node_color: str = "black"
    node_spec: Any | None = None


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


@dataclass(frozen=True)
class RenderIRCluster:
    """Resolved ownership slice for nodes and edges in one cluster.

    Parameters
    ----------
    key:
        Renderer cluster key.
    node_names:
        Rendered nodes owned by the cluster.
    edge_indexes:
        Indexes into :attr:`RenderIR.edges` emitted inside the cluster.
    """

    key: str
    node_names: tuple[str, ...]
    edge_indexes: tuple[int, ...]


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
    clusters:
        Cluster ownership records derived from resolved node and edge owners.
    node_emissions:
        Legacy node emission adapter kept during migration.
    """

    context: RenderContext
    nodes: tuple[RenderIRNode, ...]
    edges: tuple[RenderIREdge, ...]
    clusters: tuple[RenderIRCluster, ...]
    node_emissions: tuple["RenderedNodeEmission", ...]


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
    trace: "Trace",
    *,
    collapse_fn: "Callable[[Module], bool] | None",
    repeat_folds: "Mapping[str, ModuleRepeatFold] | None",
    context: RenderContext | None = None,
    universe: Any | None = None,
    segments: "Mapping[str, Any] | None" = None,
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
    if universe is None:
        from .node_universe import build_node_universe
        from .source_graph import build_source_graph

        universe = build_node_universe(
            build_source_graph(trace, resolved_context), collapse_fn, repeat_folds
        )
    emissions = universe.emissions
    nodes = tuple(
        _node_from_emission(trace, emission, resolved_context, universe, repeat_folds, segments)
        for emission in emissions
    )
    edges = _build_forward_edges_from_universe(universe)
    clusters = _build_clusters(nodes, edges)
    return RenderIR(
        context=resolved_context,
        nodes=nodes,
        edges=edges,
        clusters=clusters,
        node_emissions=emissions,
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


def _node_from_emission(
    trace: "Trace",
    emission: "RenderedNodeEmission",
    context: RenderContext,
    universe: Any,
    repeat_folds: "Mapping[str, ModuleRepeatFold] | None",
    segments: "Mapping[str, Any] | None",
) -> RenderIRNode:
    """Convert a legacy node emission into a render-IR node."""

    hidden_originals: tuple[str, ...] = ()
    if emission.fold is not None and emission.kind in {"module_box", "run_fold_ellipsis"}:
        hidden_originals = tuple(emission.fold.addresses)
    source_label = emission.op_label or emission.call or emission.boundary_kind
    owner_cluster = emission.module_address
    if emission.kind == "module_box" and emission.call is not None:
        from .rendering import _collapsed_module_owner_key

        address, _, call_index = emission.call.partition(":")
        owner_cluster = _collapsed_module_owner_key(
            trace,
            address,
            call_index or "1",
            context.vis_mode,
        )
    elif emission.kind == "run_fold_ellipsis" and emission.fold is not None:
        from .rendering import _run_fold_ellipsis_owner_key

        owner_cluster = _run_fold_ellipsis_owner_key(trace, emission.fold, context.vis_mode)
    node_calls: tuple[Any, ...] = ()
    owned_node_args: tuple[tuple[str, dict[str, Any]], ...] = ()
    node_color = "black"
    label_spans: tuple[str, ...] = ()
    node_spec: Any | None = None
    if emission.node is not None:
        node_calls, owned_node_args, node_color, node_spec, label_spans = _resolve_node_decision(
            trace, emission, context, universe, repeat_folds, segments
        )
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
    )


def _resolve_node_decision(
    trace: "Trace",
    emission: "RenderedNodeEmission",
    context: RenderContext,
    universe: Any,
    repeat_folds: "Mapping[str, ModuleRepeatFold] | None",
    segments: "Mapping[str, Any] | None",
) -> tuple[
    tuple[Any, ...], tuple[tuple[str, dict[str, Any]], ...], str, Any | None, tuple[str, ...]
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
    universe:
        Presentation-free universe that selected the node.

    Returns
    -------
    tuple
        Recorded top-level calls, owned node arguments, edge color, and structured label spans.
    """
    from collections import defaultdict

    from .rendering import (
        _ForwardDotRecorder,
        _build_collapsed_module_node,
        _build_layer_node,
        _collapsed_container_leaf_nodes,
        _normalize_buffer_visibility,
        _segment_for_node,
        resolve_theme,
    )

    node = emission.node
    if node is None:
        return (), (), "black", None, ()
    if _segment_for_node(node, segments) is not None:
        return (), (), "black", None, ()
    recorder = _ForwardDotRecorder()
    module_nodes: dict[str, Any] = defaultdict(dict)
    show_buffers = _normalize_buffer_visibility(context.show_buffer_layers)
    collapsed_containers = _collapsed_container_leaf_nodes(
        trace,
        universe.source_graph.entries_to_plot,
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
            repeat_folds,
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


def _build_clusters(
    nodes: tuple[RenderIRNode, ...],
    edges: tuple[RenderIREdge, ...],
) -> tuple[RenderIRCluster, ...]:
    """Build cluster ownership records from IR nodes and edges.

    Parameters
    ----------
    nodes:
        Resolved IR nodes.
    edges:
        Resolved IR edges.

    Returns
    -------
    tuple[RenderIRCluster, ...]
        Cluster records sorted by cluster key.
    """

    cluster_nodes: defaultdict[str, list[str]] = defaultdict(list)
    cluster_edges: defaultdict[str, list[int]] = defaultdict(list)
    for node in nodes:
        if node.owner_cluster is not None:
            cluster_nodes[node.owner_cluster].append(node.name)
    for index, edge in enumerate(edges):
        if edge.owner_cluster is not None:
            cluster_edges[edge.owner_cluster].append(index)
    return tuple(
        RenderIRCluster(
            key=key,
            node_names=tuple(cluster_nodes.get(key, ())),
            edge_indexes=tuple(cluster_edges.get(key, ())),
        )
        for key in sorted(set(cluster_nodes) | set(cluster_edges))
    )
