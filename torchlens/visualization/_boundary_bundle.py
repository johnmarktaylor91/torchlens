"""Boundary bundle (collapse memo D3): one node for a flood of like model outputs.

In an overview render (``collapse`` of ``"auto"``, ``"max"`` or a level above
0, unrolled, no container display), three or more model-output leaves that
share their top-level container key, role, dtype and shape, and whose
producers all sit inside ONE top-level module, draw as a single node when that
module renders as one box. The node keeps the first member's id, counts 1 in
the plan and renders in both backends; a named single output
(``last_hidden_state``) stays individual. It is a render-only grouping: no
kwarg, field or node kind names it, and nothing can select or expand it.

The planner prices the bundle exactly: the root's own units drop every
member, the producer's box adds one unit per bundle, and every expanded
choice of the producer adds the members back (the bundle refuses when its
producer is not one box). The node universe applies the same rule from the
resolved collapse predicate, so scored k, plan count and render agree.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..data_classes.trace import Trace

#: Fewest like leaves that form a bundle (memo D3).
MIN_BUNDLE_LEAVES = 3
_DEFAULT_CALL_DEPTH = 1000


@dataclass(frozen=True)
class BundleGroup:
    """One candidate bundle of model-output leaves.

    Parameters
    ----------
    producer:
        Top-level module address that must render as one box.
    members:
        Output op labels in output order; the first is the representative.
    container:
        Top-level container key label shown on the node.
    shape:
        Shared leaf shape.
    """

    producer: str
    members: tuple[str, ...]
    container: str
    shape: tuple[int, ...]


def bundles_enabled(context: Any) -> bool:
    """Return whether ``context`` is an overview render that bundles outputs."""

    collapse = getattr(context, "collapse", "none")
    overview = collapse in ("auto", "max") or (
        isinstance(collapse, float) and not isinstance(collapse, bool) and collapse > 0.0
    )
    return bool(
        overview
        and getattr(context, "vis_mode", "unrolled") == "unrolled"
        and not getattr(context, "show_containers", False)
        and getattr(context, "module", None) is None
        and getattr(context, "skip_fn", None) is None
        and getattr(context, "vis_call_depth", _DEFAULT_CALL_DEPTH) == _DEFAULT_CALL_DEPTH
    )


def _module_address(call: object) -> str:
    """Return the pass-free address of a module call string."""

    return str(call).rsplit(":", 1)[0]


def _producer_module(trace: Trace, op: Any) -> str | None:
    """Return the top-level module holding ``op``'s single producer, if eligible."""

    from .auto_collapse import analyze_collapse

    parents = tuple(getattr(op, "parents", ()) or ())
    if len(parents) != 1:
        return None
    modules = list(getattr(trace.ops[parents[0]], "modules", ()) or ())
    if not modules:
        return None
    top = _module_address(modules[0])
    module = trace.modules[top] if top in trace.modules else None
    signal = analyze_collapse(trace).signals.get(top)
    if module is None or signal is None or int(getattr(module, "num_calls", 1) or 1) != 1:
        return None
    # A structurally unique module cannot fold or segment with peers, so it
    # renders either as its own box or expanded.
    return top if signal.peer_count == 1 else None


def bundle_groups(trace: Trace, context: Any) -> dict[str, tuple[BundleGroup, ...]]:
    """Return candidate bundles keyed by producing top-level module.

    Parameters
    ----------
    trace:
        Trace being planned or rendered.
    context:
        Render context; bundles exist only in overview renders.

    Returns
    -------
    dict[str, tuple[BundleGroup, ...]]
        Plan-independent candidates; a candidate forms only while its
        producer renders as one box.
    """

    if not bundles_enabled(context):
        return {}
    from ..ir.container import TupleIndex
    from ._render_leaf import _container_component_role

    grouped: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for label in getattr(trace, "output_layers", ()) or ():
        op = trace.ops[label]
        path = tuple(getattr(op, "container_path", ()) or ())
        if not path or getattr(op, "is_buffer", False):
            continue
        producer = _producer_module(trace, op)
        if producer is None:
            continue
        # The role is the last NAMED component (``keys``/``values``); tuple
        # positions are where a leaf sits, not what it is.
        named = [part for part in path[1:] if not isinstance(part, TupleIndex)]
        role = _container_component_role(named[-1]) if named else ""
        container = _container_component_role(path[0]) + (f".{role}" if role else "")
        key = (
            producer,
            container,
            str(getattr(op, "dtype", None)),
            tuple(getattr(op, "shape", ()) or ()),
        )
        grouped[key].append(str(op.label))
    by_producer: dict[str, list[BundleGroup]] = defaultdict(list)
    for (producer, container, _dtype, shape), members in grouped.items():
        if len(members) >= MIN_BUNDLE_LEAVES:
            by_producer[producer].append(BundleGroup(producer, tuple(members), container, shape))
    return {producer: tuple(groups) for producer, groups in by_producer.items()}


def bundle_member_labels(trace: Trace, context: Any) -> frozenset[str]:
    """Return every candidate member label (the root's own units drop them)."""

    return frozenset(
        label
        for groups in bundle_groups(trace, context).values()
        for group in groups
        for label in group.members
    )


def box_representatives(trace: Trace, context: Any, address: str) -> tuple[str, ...]:
    """Return the representative labels a box of ``address`` draws as bundles."""

    return tuple(group.members[0] for group in bundle_groups(trace, context).get(address, ()))


def expanded_members(trace: Trace, context: Any, address: str) -> tuple[str, ...]:
    """Return the member labels drawn one by one when ``address`` expands."""

    return tuple(
        label for group in bundle_groups(trace, context).get(address, ()) for label in group.members
    )


def shift_expanded_points(points: Sequence[Any], shift: int) -> tuple[Any, ...]:
    """Charge ``shift`` member units to every non-box frontier point."""

    if not shift:
        return tuple(points)
    return tuple(
        point
        if getattr(point.decision, "kind", None) == "box"
        else replace(point, k=point.k + shift)
        for point in points
    )


def bundle_member_nodes(
    trace: Trace,
    entries_to_plot: Mapping[str, Any],
    context: Any,
    collapse_fn: Callable[[Any], bool] | None,
) -> dict[str, str]:
    """Return rendered member names mapped to their representative's name.

    A candidate forms only when every member's producer is absorbed by its
    producer module's one box under ``collapse_fn``; otherwise it refuses and
    every member renders individually.
    """

    groups = bundle_groups(trace, context)
    if not groups:
        return {}
    from ._render_leaf import _collapse_address_for_node, _render_node_name

    by_label: dict[str, Any] = {}
    for node in entries_to_plot.values():
        for attr in ("layer_label", "label"):
            try:
                value = getattr(node, attr, None)
            except ValueError:  # a multi-pass aggregate has no single label
                value = None
            if value is not None:
                by_label.setdefault(str(value), node)
    hidden: dict[str, str] = {}
    for group in (group for groups_for in groups.values() for group in groups_for):
        found = [by_label.get(label) for label in group.members]
        nodes = [node for node in found if node is not None]
        if len(nodes) != len(found):
            continue
        box = f"{group.producer}:1"
        absorbed = all(
            _collapse_address_for_node(
                trace,
                trace.ops[node.parents[0]],
                vis_mode="unrolled",
                collapse_fn=collapse_fn,
                max_module_depth=_DEFAULT_CALL_DEPTH,
            )
            == box
            for node in nodes
        )
        if not absorbed:
            continue
        representative = _render_node_name(nodes[0], "unrolled")
        for node in nodes[1:]:
            hidden[_render_node_name(node, "unrolled")] = representative
    return hidden


def bundle_labels(
    trace: Trace,
    entries_to_plot: Mapping[str, Any],
    context: Any,
    collapse_fn: Callable[[Any], bool] | None,
) -> dict[str, str]:
    """Return representative node names mapped to their bundle's HTML label."""

    hidden = bundle_member_nodes(trace, entries_to_plot, context, collapse_fn)
    if not hidden:
        return {}
    from ._render_leaf import _render_node_name
    from .node_spec import render_lines_to_html

    counts: dict[str, int] = defaultdict(lambda: 1)
    for representative in hidden.values():
        counts[representative] += 1
    labels: dict[str, str] = {}
    for groups in bundle_groups(trace, context).values():
        for group in groups:
            name = _render_node_name(trace.ops[group.members[0]], "unrolled")
            if name in counts:
                shape = "x".join(str(dim) for dim in group.shape) or "scalar"
                labels[name] = render_lines_to_html(
                    [f"OUTPUTS: {group.container}", f"{counts[name]} tensors, {shape} each"]
                )
    return labels


def relabel_bundle_nodes(nodes: Sequence[Any], labels: Mapping[str, str]) -> tuple[Any, ...]:
    """Return render-IR nodes with each representative's label replaced."""

    if not labels:
        return tuple(nodes)
    relabeled = []
    for node in nodes:
        label = labels.get(node.name)
        if label is None:
            relabeled.append(node)
            continue
        calls = tuple(
            replace(call, attrs=tuple((k, label if k == "label" else v) for k, v in call.attrs))
            if call.kind == "node"
            else call
            for call in node.node_calls
        )
        relabeled.append(replace(node, node_calls=calls))
    return tuple(relabeled)
