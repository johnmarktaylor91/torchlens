"""Resolved render-IR adapters for TorchLens graph visualization."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from .collapse_plan import RenderContext

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from ..data_classes.module import Module
    from ..data_classes.trace import Trace
    from .auto_collapse import ModuleRunFold
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


@dataclass(frozen=True)
class RenderIREdge:
    """Resolved forward edge independent of DOT emission.

    Parameters
    ----------
    source_unit:
        Rendered source endpoint after collapse/run-fold projection.
    target_unit:
        Rendered target endpoint after collapse/run-fold projection.
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
    run_folds: "Mapping[str, ModuleRunFold] | None",
    context: RenderContext | None = None,
) -> RenderIR:
    """Build the first render-IR slice from current renderer-faithful emissions.

    Parameters
    ----------
    trace:
        Trace being rendered.
    collapse_fn:
        Active collapse predicate.
    run_folds:
        Active run-fold descriptors.
    context:
        Render context. Defaults to :class:`RenderContext`.

    Returns
    -------
    RenderIR
        Node-level render IR whose emission order matches the current renderer.
    """

    resolved_context = RenderContext() if context is None else context
    from .rendering import rendered_node_universe_from_v1

    emissions = rendered_node_universe_from_v1(
        trace,
        collapse_fn=collapse_fn,
        run_folds=run_folds,
        context=resolved_context,
    )
    nodes = tuple(_node_from_emission(trace, emission, resolved_context) for emission in emissions)
    edges = _build_forward_edges(trace, collapse_fn, run_folds, resolved_context)
    clusters = _build_clusters(nodes, edges)
    return RenderIR(
        context=resolved_context,
        nodes=nodes,
        edges=edges,
        clusters=clusters,
        node_emissions=emissions,
    )


def _node_from_emission(
    trace: "Trace",
    emission: "RenderedNodeEmission",
    context: RenderContext,
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
    return RenderIRNode(
        name=emission.name,
        kind=emission.kind,
        owner_cluster=owner_cluster,
        source_label=source_label,
        hidden_originals=hidden_originals,
    )


def _build_forward_edges(
    trace: "Trace",
    collapse_fn: "Callable[[Module], bool] | None",
    run_folds: "Mapping[str, ModuleRunFold] | None",
    context: RenderContext,
) -> tuple[RenderIREdge, ...]:
    """Build renderer-faithful forward edge records for the IR slice.

    Parameters
    ----------
    trace:
        Trace being rendered.
    collapse_fn:
        Active collapse predicate.
    run_folds:
        Active run-fold descriptors.
    context:
        Render context that controls visibility and endpoint projection.

    Returns
    -------
    tuple[RenderIREdge, ...]
        Forward edge records in deterministic renderer traversal order.
    """

    from .rendering import (
        RenderEdge as LegacyRenderEdge,
        _build_skip_filtered_edge_map,
        _collapsed_container_leaf_nodes,
        _collapsed_endpoint_for_emission,
        _edge_touches_run_fold,
        _entries_to_plot_for_context,
        _get_lowest_module_for_two_nodes,
        _get_node_by_label,
        _is_buffer_visible,
        _normalize_buffer_visibility,
        _render_node_label,
        _run_fold_ellipsis_node_name,
        _run_fold_graph_node_name,
        _run_fold_hidden_endpoint,
    )

    show_buffer_layers = _normalize_buffer_visibility(context.show_buffer_layers)
    entries_to_plot = _entries_to_plot_for_context(trace, context.vis_mode)
    skipped_labels: set[str] = set()
    edge_map: dict[str, list[LegacyRenderEdge]] = {}
    if run_folds or context.skip_fn is not None:
        edge_map, skipped_labels = _build_skip_filtered_edge_map(
            trace,
            entries_to_plot,
            vis_mode=context.vis_mode,
            show_buffer_layers=show_buffer_layers,
            skip_fn=context.skip_fn,
        )
    collapsed_container_nodes = _collapsed_container_leaf_nodes(
        trace,
        entries_to_plot,
        vis_mode=context.vis_mode,
        show_containers=context.show_containers,
        container_max_inline=12,
        pending_nodes=[],
    )
    edges: list[RenderIREdge] = []
    seen: set[tuple[str, str, tuple[Any, ...]]] = set()
    for parent_node in entries_to_plot.values():
        parent_render_label = _render_node_label(parent_node, context.vis_mode)
        if parent_render_label in skipped_labels:
            continue
        if parent_node.is_buffer and not _is_buffer_visible(parent_node, show_buffer_layers):
            continue
        render_edges = edge_map.get(parent_render_label)
        if render_edges is None:
            render_edges = [
                LegacyRenderEdge(
                    target=_get_node_by_label(trace, child_label, context.vis_mode),
                    metadata_child=None,
                    occurrence_key=("edge", parent_node.layer_label, child_label),
                )
                for child_label in parent_node.children
            ]
        for render_edge in render_edges:
            child_node = render_edge.target
            if child_node.is_buffer and not _is_buffer_visible(child_node, show_buffer_layers):
                continue
            source_unit = _projected_endpoint(
                trace,
                parent_node,
                context,
                collapse_fn,
                run_folds,
                collapsed_container_nodes,
            )
            target_unit = _projected_endpoint(
                trace,
                child_node,
                context,
                collapse_fn,
                run_folds,
                collapsed_container_nodes,
            )
            source_fold = _run_fold_hidden_endpoint(
                _collapsed_endpoint_for_emission(
                    trace,
                    parent_node,
                    vis_mode=context.vis_mode,
                    vis_call_depth=1000,
                    collapse_fn=collapse_fn,
                    run_folds=run_folds,
                ),
                run_folds,
            )
            target_fold = _run_fold_hidden_endpoint(
                _collapsed_endpoint_for_emission(
                    trace,
                    child_node,
                    vis_mode=context.vis_mode,
                    vis_call_depth=1000,
                    collapse_fn=collapse_fn,
                    run_folds=run_folds,
                ),
                run_folds,
            )
            if source_fold is not None and target_fold is source_fold:
                continue
            fold = source_fold or target_fold
            projection_reason: Literal[
                "direct",
                "source_projected",
                "target_projected",
                "both_projected",
                "run_fold_ellipsis",
            ] = _projection_reason(
                _render_node_label(parent_node, context.vis_mode).replace(":", "pass"),
                _render_node_label(child_node, context.vis_mode).replace(":", "pass"),
                source_unit,
                target_unit,
            )
            if fold is not None:
                representative_name = _run_fold_graph_node_name(
                    f"{fold.representative}:1",
                    context.vis_mode,
                    {fold.representative: fold},
                )
                ellipsis_name = _run_fold_ellipsis_node_name(representative_name)
                if source_fold is not None:
                    source_unit = ellipsis_name
                if target_fold is not None:
                    target_unit = ellipsis_name
                projection_reason = "run_fold_ellipsis"
            if source_unit == target_unit and not _edge_touches_run_fold(
                source_unit, target_unit, run_folds, context.vis_mode
            ):
                continue
            dedupe_key = (source_unit, target_unit, render_edge.occurrence_key)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            module_key = _get_lowest_module_for_two_nodes(
                cast(Any, parent_node),
                cast(Any, child_node),
                source_unit
                != _render_node_label(parent_node, context.vis_mode).replace(":", "pass")
                and target_unit
                != _render_node_label(child_node, context.vis_mode).replace(":", "pass"),
                1000,
            )
            edges.append(
                RenderIREdge(
                    source_unit=source_unit,
                    target_unit=target_unit,
                    source_originals=(parent_node.layer_label,),
                    target_originals=(child_node.layer_label,),
                    owner_cluster=None if module_key == -1 else str(module_key),
                    occurrence_key=render_edge.occurrence_key,
                    projection_reason=projection_reason,
                )
            )
    return tuple(edges)


def _projected_endpoint(
    trace: "Trace",
    node: Any,
    context: RenderContext,
    collapse_fn: "Callable[[Module], bool] | None",
    run_folds: "Mapping[str, ModuleRunFold] | None",
    collapsed_container_nodes: "Mapping[str, str]",
) -> str:
    """Return the non-ellipsis projected endpoint name for one node.

    Parameters
    ----------
    trace:
        Trace being rendered.
    node:
        Graph node whose endpoint should be resolved.
    context:
        Render context that controls visibility.
    collapse_fn:
        Active collapse predicate.
    run_folds:
        Active run-fold descriptors.
    collapsed_container_nodes:
        Mapping from container leaf node names to collapsed container nodes.

    Returns
    -------
    str
        Rendered endpoint name before hidden-member ellipsis projection.
    """

    from .rendering import (
        _collapsed_endpoint_for_emission,
        _render_node_label,
        _run_fold_graph_node_name,
    )

    render_name = _render_node_label(node, context.vis_mode).replace(":", "pass")
    endpoint = _collapsed_endpoint_for_emission(
        trace,
        node,
        vis_mode=context.vis_mode,
        vis_call_depth=1000,
        collapse_fn=collapse_fn,
        run_folds=run_folds,
    )
    if endpoint is not None:
        return _run_fold_graph_node_name(endpoint, context.vis_mode, run_folds)
    return collapsed_container_nodes.get(render_name, render_name)


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
        Projection reason before run-fold ellipsis projection is applied.
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
