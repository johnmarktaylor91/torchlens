"""Presentation-free visible node universe for forward visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from .source_graph import SourceGraph

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from ..data_classes.module import Module
    from .auto_collapse import ModuleRepeatFold, SegmentDescriptor
    from ._render_common import RenderEdge, RenderedNodeEmission

NodeUnitKind = Literal[
    "raw_op",
    "module_box",
    "boundary",
    "repeat_ellipsis",
    "segment",
    "container_summary",
]


@dataclass(frozen=True)
class NodeUnit:
    """One visible structural unit and the source nodes it represents.

    Parameters
    ----------
    unit_id:
        Stable renderer-neutral identifier.
    kind:
        Structural role of the visible unit.
    emission:
        Temporary renderer adapter retained until Phase 3.
    source_nodes:
        Ordered source-node provenance represented by this unit.
    hidden_members:
        Source labels hidden behind the representative.
    counts_for_collapse:
        Whether this unit contributes one node to collapse planning.
    """

    unit_id: str
    kind: NodeUnitKind
    emission: "RenderedNodeEmission"
    source_nodes: tuple[Any, ...]
    hidden_members: tuple[str, ...]
    counts_for_collapse: bool = True


@dataclass(frozen=True)
class ProjectedEdgeOccurrence:
    """One source edge occurrence projected onto visible unit identifiers.

    Parameters
    ----------
    source_unit:
        Visible source unit identifier.
    target_unit:
        Visible target unit identifier.
    occurrence_key:
        Stable source edge occurrence key.
    source_label:
        Original source label.
    target_label:
        Original target label.
    """

    source_unit: str
    target_unit: str
    occurrence_key: tuple[Any, ...]
    source_label: str
    target_label: str


@dataclass(frozen=True)
class NodeUniverse:
    """Single presentation-free visible-unit truth for counting and rendering.

    Parameters
    ----------
    source_graph:
        Normalized source graph projected by this universe.
    units:
        Visible structural units in deterministic order.
    endpoint_projection:
        Raw render labels mapped to their visible representatives.
    projected_edges:
        Source edge occurrences projected through visible representatives.
    """

    source_graph: SourceGraph
    units: tuple[NodeUnit, ...]
    endpoint_projection: dict[str, str]
    projected_edges: tuple[ProjectedEdgeOccurrence, ...]

    @property
    def emissions(self) -> tuple["RenderedNodeEmission", ...]:
        """Return temporary renderer-compatible visible emissions.

        Returns
        -------
        tuple[RenderedNodeEmission, ...]
            Visible emissions in unit order.
        """

        return tuple(unit.emission for unit in self.units)


def build_node_universe(
    source_graph: SourceGraph,
    collapse_fn: "Callable[[Module], bool] | None",
    repeat_folds: "Mapping[str, ModuleRepeatFold] | None",
    segments: "Mapping[str, SegmentDescriptor] | None" = None,
    containers: Any = None,
) -> NodeUniverse:
    """Project one normalized source graph into visible structural units.

    Parameters
    ----------
    source_graph:
        Focus- and skip-normalized source graph.
    collapse_fn:
        Active module collapse predicate.
    repeat_folds:
        Active repeat-fold descriptors.
    segments:
        Active optimizer segments. Reserved for structural segment units.
    containers:
        Optional pre-resolved container state. Defaults to request policy.

    Returns
    -------
    NodeUniverse
        Visible units, endpoint projection, and projected edge occurrences.
    """

    del segments, containers
    from ._render_flow import (
        _base_rendered_node_emission,
        _collapsed_container_leaf_nodes,
        _enumerate_base_rendered_node_emissions,
        _enumerate_run_fold_ellipsis_emissions,
        _normalize_buffer_visibility,
        _render_node_label,
    )
    from ._render_leaf import _run_fold_graph_node_name

    trace = source_graph.trace
    request = source_graph.request
    show_buffers = _normalize_buffer_visibility(request.show_buffer_layers)
    collapsed_containers = _collapsed_container_leaf_nodes(
        trace,
        source_graph.entries_to_plot,
        vis_mode=request.vis_mode,
        show_containers=request.show_containers,
        container_max_inline=request.container_max_inline,
        pending_nodes=[],
    )
    base_emissions = _enumerate_base_rendered_node_emissions(
        trace,
        source_graph.entries_to_plot,
        skipped_labels=source_graph.skipped_labels,
        vis_mode=request.vis_mode,
        vis_call_depth=request.vis_call_depth,
        show_buffer_layers=show_buffers,
        collapse_fn=collapse_fn,
        repeat_folds=repeat_folds,
        show_containers=request.show_containers,
        collapsed_container_nodes=collapsed_containers,
    )
    ellipses = _enumerate_run_fold_ellipsis_emissions(
        trace,
        source_graph.entries_to_plot,
        edge_map=source_graph.edge_map,
        skipped_labels=source_graph.skipped_labels,
        vis_mode=request.vis_mode,
        vis_call_depth=request.vis_call_depth,
        show_buffer_layers=show_buffers,
        collapse_fn=collapse_fn,
        repeat_folds=repeat_folds,
        collapsed_container_nodes=collapsed_containers,
    )
    visible_by_id: dict[str, RenderedNodeEmission] = {}
    provenance: dict[str, list[Any]] = {}
    hidden: dict[str, list[str]] = {}
    endpoint_projection: dict[str, str] = {}
    for node in source_graph.entries_to_plot.values():
        raw_label = _render_node_label(node, request.vis_mode)
        emission = _base_rendered_node_emission(
            trace,
            node,
            vis_mode=request.vis_mode,
            vis_call_depth=request.vis_call_depth,
            collapse_fn=collapse_fn,
            repeat_folds=repeat_folds,
            show_containers=request.show_containers,
            collapsed_container_nodes=collapsed_containers,
        )
        if emission is None:
            continue
        if emission.kind == "hidden_run_member" and emission.fold is not None:
            unit_id = _run_fold_graph_node_name(
                emission.fold.representative,
                request.vis_mode,
                repeat_folds,
            )
        else:
            unit_id = emission.name
        endpoint_projection[raw_label] = unit_id
        provenance.setdefault(unit_id, []).append(node)
        if emission.kind == "hidden_run_member":
            hidden.setdefault(unit_id, []).append(raw_label)
        else:
            visible_by_id.setdefault(unit_id, emission)
    for emission in base_emissions:
        if emission.kind != "hidden_run_member":
            visible_by_id.setdefault(emission.name, emission)
    for emission in ellipses:
        visible_by_id.setdefault(emission.name, emission)
    ordered = [
        emission
        for emission in (*base_emissions, *ellipses)
        if emission.kind != "hidden_run_member"
    ]
    seen: set[str] = set()
    units: list[NodeUnit] = []
    for emission in ordered:
        if emission.name in seen:
            continue
        seen.add(emission.name)
        kind = cast(
            NodeUnitKind,
            "repeat_ellipsis" if emission.kind == "run_fold_ellipsis" else emission.kind,
        )
        units.append(
            NodeUnit(
                emission.name,
                kind,
                emission,
                tuple(provenance.get(emission.name, ())),
                tuple(hidden.get(emission.name, ())),
            )
        )
    projected_edges = _project_edges(source_graph.edge_map, endpoint_projection, request.vis_mode)
    return NodeUniverse(source_graph, tuple(units), endpoint_projection, projected_edges)


def _project_edges(
    edge_map: "Mapping[str, list[RenderEdge]]",
    projection: "Mapping[str, str]",
    vis_mode: str,
) -> tuple[ProjectedEdgeOccurrence, ...]:
    """Project normalized edge occurrences onto visible unit identifiers.

    Parameters
    ----------
    edge_map:
        Normalized source edge occurrences.
    projection:
        Raw-label to visible-unit mapping.
    vis_mode:
        Active render granularity.

    Returns
    -------
    tuple[ProjectedEdgeOccurrence, ...]
        Projected occurrences in source traversal order.
    """

    from ._render_flow import _render_node_label

    projected: list[ProjectedEdgeOccurrence] = []
    for source_label, edges in edge_map.items():
        source_unit = projection.get(source_label)
        if source_unit is None:
            continue
        for edge in edges:
            target_label = _render_node_label(edge.target, vis_mode)
            target_unit = projection.get(target_label)
            if target_unit is None:
                continue
            projected.append(
                ProjectedEdgeOccurrence(
                    source_unit, target_unit, edge.occurrence_key, source_label, target_label
                )
            )
    return tuple(projected)
