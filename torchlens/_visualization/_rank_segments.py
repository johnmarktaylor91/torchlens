"""Segment boxes for the rank backend (collapse memo D2, the ``__segment__`` half).

The node universe has no segment unit: the dot path draws a segment box while
it emits the member ops, and the render IR gives every member an empty node.
The rank backend positions IR nodes and edges only, so a segment plan reached
it with edges pointing at members that were never declared (and at a segment
box that had no node), which ``neato -n`` rejects. This module projects the
forward IR for the rank backend: one IR node per segment, carrying the same
node attributes the dot path queues, every member endpoint rerouted to it,
and edges that fall inside one segment dropped. The dot path is untouched.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from .render_ir import RenderIR, RenderIRDotStatement, RenderIRNode


def _segment_node(segment: Any, vis_mode: str, theme: Any) -> tuple[str | None, dict[str, Any]]:
    """Return the segment's owner cluster key and the node args the dot path queues."""

    from collections import defaultdict

    from ._render_common import _RenderIRDecisionBuilder
    from ._render_nodes import _queue_segment_node

    recorder = _RenderIRDecisionBuilder()
    module_nodes: dict[str, Any] = defaultdict(dict)
    _queue_segment_node(recorder, module_nodes, set(), segment, (vis_mode, theme))
    for owner, payload in module_nodes.items():
        for args in payload.get("nodes", ()):
            return owner, dict(args)
    statement = recorder.calls[0]
    return None, dict(statement.attrs)


def _member_segments(universe: Any, segment_lookup: Any) -> dict[str, Any]:
    """Return IR unit names mapped to the segment that absorbs them."""

    from ._render_edges import _segment_for_node

    members: dict[str, Any] = {}
    for unit in universe.units:
        node = unit.emission.node
        segment = _segment_for_node(node, segment_lookup) if node is not None else None
        if segment is not None:
            members[unit.unit_id] = segment
    return members


def _project_nodes(
    ir: RenderIR, member_segments: Mapping[str, Any], context: Any, vis_mode: str
) -> tuple[list[RenderIRNode], dict[str, str | None]]:
    """Return the IR nodes with each segment's members replaced by one node."""

    nodes: list[RenderIRNode] = []
    owners: dict[str, str | None] = {}
    for node in ir.nodes:
        segment = member_segments.get(node.name)
        if segment is None:
            nodes.append(node)
            continue
        if segment.name in owners:
            continue
        owner, args = _segment_node(segment, vis_mode, context.theme)
        owners[segment.name] = owner
        name = str(args.pop("name"))
        nodes.append(
            RenderIRNode(
                name=name,
                kind="raw_op",
                owner_cluster=owner,
                source_label=None,
                node_calls=(RenderIRDotStatement("node", (name,), tuple(args.items())),),
                region_path=node.region_path,
            )
        )
    return nodes, owners


def _reroute_edges(edges: Sequence[Any], target: Mapping[str, str]) -> list[Any]:
    """Return ``edges`` with member endpoints moved to their segment node."""

    rerouted = []
    seen: set[tuple[str, str]] = set()
    for edge in edges:
        source = target.get(edge.source_unit, edge.source_unit)
        head = target.get(edge.target_unit, edge.target_unit)
        if source == edge.source_unit and head == edge.target_unit:
            rerouted.append(edge)
            continue
        # A rerouted edge inside one segment, or a repeat of one already
        # rerouted, is what the dot path's segment-edge dedupe drops too.
        if source == head or (source, head) in seen:
            continue
        seen.add((source, head))
        rerouted.append(
            replace(
                edge,
                source_unit=source,
                target_unit=head,
                tail_name=target.get(edge.tail_name or "", edge.tail_name),
                head_name=target.get(edge.head_name or "", edge.head_name),
            )
        )
    return rerouted


def _project_regions(
    regions: Sequence[Any], target: Mapping[str, str], owners: Mapping[str, str | None]
) -> tuple[Any, ...]:
    """Return regions listing each segment node in place of its members."""

    owned_by_region: dict[str, list[str]] = {}
    for name, owner in owners.items():
        if owner is not None:
            owned_by_region.setdefault(owner, []).append(name)
    return tuple(
        replace(
            region,
            node_names=tuple(
                dict.fromkeys(
                    [name for name in region.node_names if name not in target]
                    + owned_by_region.get(region.key, [])
                )
            ),
        )
        for region in regions
    )


def project_segments_for_rank(ir: RenderIR, context: Any, vis_mode: str) -> RenderIR:
    """Return ``ir`` with segment members folded into one node per segment.

    Parameters
    ----------
    ir:
        Forward render IR built for the rank backend.
    context:
        Render context carrying the node universe the IR was built from, the
        active segments, their membership index and the theme (segment
        colors follow it).
    vis_mode:
        ``"unrolled"`` or ``"rolled"``.

    Returns
    -------
    RenderIR
        The same IR when no segment is active; otherwise members are replaced
        by their segment node in nodes, regions and edges.
    """

    if not context.segments:
        return ir
    member_segments = _member_segments(context.node_universe, context.segment_lookup)
    if not member_segments:
        return ir
    target = {unit: segment.name for unit, segment in member_segments.items()}
    nodes, owners = _project_nodes(ir, member_segments, context, vis_mode)
    return replace(
        ir,
        nodes=tuple(nodes),
        edges=tuple(_reroute_edges(ir.edges, target)),
        regions=_project_regions(ir.regions, target, owners),
    )
