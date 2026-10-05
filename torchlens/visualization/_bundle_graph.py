"""Graphviz construction helpers for ``tl.show_bundle_graph``.

These walk a bundle's forward supergraph (and, when present, its backward
graph) and add nodes, module clusters and edges to a Graphviz digraph. The
public entry point and its argument handling stay in
:mod:`torchlens._user_public_impls`.
"""

from __future__ import annotations

from typing import Any


def _bundle_node_display_label(graph_node_label: str, node: Any, vis_mode: str) -> str:
    """Return a compact Graphviz label for a bundle supergraph node.

    Parameters
    ----------
    graph_node_label:
        Canonical supergraph node name.
    node:
        Supergraph node-like object.
    vis_mode:
        Bundle visualization mode.

    Returns
    -------
    str
        Display label.
    """

    traces = ",".join(getattr(node, "traces", []))
    mode_suffix = " rolled" if vis_mode == "rolled" else ""
    op_type = getattr(node, "op_type", "") or "op"
    return f"{graph_node_label}\n{op_type}{mode_suffix}\n[{traces}]"


def _bundle_module_groups(bundle: Any) -> dict[str, list[str]]:
    """Return bundle supergraph nodes grouped by representative module path.

    Parameters
    ----------
    bundle:
        Bundle with a ``supergraph`` accessor.

    Returns
    -------
    dict[str, list[str]]
        Module path to canonical node names.
    """

    groups: dict[str, list[str]] = {}
    for graph_node_label in bundle.supergraph.topological_order:
        node = bundle.supergraph.nodes[graph_node_label]
        module_path = getattr(node, "module_path", None)
        if module_path:
            groups.setdefault(str(module_path), []).append(graph_node_label)
    return groups


def _add_bundle_forward_nodes(
    dot: Any,
    bundle: Any,
    vis_mode: str,
    node_styles: dict[str, Any] | None,
) -> None:
    """Add forward supergraph nodes to a Graphviz digraph.

    Parameters
    ----------
    dot:
        Graphviz digraph.
    bundle:
        Bundle to render.
    vis_mode:
        Bundle visualization mode.
    node_styles:
        Optional per-node style overrides.

    Returns
    -------
    None
        ``dot`` is mutated in place.
    """

    from ._render_utils import (
        compute_module_penwidth,
        make_module_cluster_attrs,
        merge_node_style,
    )

    base_style = {
        "shape": "box",
        "style": "filled,rounded",
        "fillcolor": "#F7F7F7",
        "color": "#333333",
    }
    module_groups = _bundle_module_groups(bundle)
    grouped_nodes = {node for nodes in module_groups.values() for node in nodes}
    for module_path, node_names in module_groups.items():
        first_node = bundle.supergraph.nodes[node_names[0]]
        cluster_name = "cluster_bundle_" + "".join(
            char if char.isalnum() else "_" for char in module_path
        )
        with dot.subgraph(name=cluster_name) as subgraph:
            subgraph.attr(
                **make_module_cluster_attrs(
                    title=module_path,
                    module_type=getattr(first_node, "module_type", None),
                    line_style="solid",
                    penwidth=compute_module_penwidth(0, 1),
                )
            )
            for graph_node_label in node_names:
                node = bundle.supergraph.nodes[graph_node_label]
                attrs = merge_node_style(base_style, node_styles, graph_node_label, node)
                subgraph.node(
                    f"fwd_{graph_node_label}",
                    label=_bundle_node_display_label(graph_node_label, node, vis_mode),
                    **attrs,
                )
    for graph_node_label in bundle.supergraph.topological_order:
        if graph_node_label in grouped_nodes:
            continue
        node = bundle.supergraph.nodes[graph_node_label]
        attrs = merge_node_style(base_style, node_styles, graph_node_label, node)
        dot.node(
            f"fwd_{graph_node_label}",
            label=_bundle_node_display_label(graph_node_label, node, vis_mode),
            **attrs,
        )


def _add_bundle_forward_edges(
    dot: Any,
    bundle: Any,
    edge_styles: dict[tuple[str, str], Any] | None,
) -> None:
    """Add forward supergraph edges to a Graphviz digraph.

    Parameters
    ----------
    dot:
        Graphviz digraph.
    bundle:
        Bundle to render.
    edge_styles:
        Optional per-edge style overrides.

    Returns
    -------
    None
        ``dot`` is mutated in place.
    """

    from ._render_utils import merge_edge_style

    base_style = {"color": "#555555", "fontcolor": "#555555"}
    for edge_key, traces in bundle.supergraph.edges.items():
        attrs = merge_edge_style(base_style, edge_styles, edge_key, {"traces": traces})
        dot.edge(
            f"fwd_{edge_key[0]}", f"fwd_{edge_key[1]}", label=",".join(sorted(traces)), **attrs
        )


def _add_bundle_backward_graph(dot: Any, bundle: Any) -> None:
    """Add per-member backward graph clusters to a Graphviz digraph.

    Parameters
    ----------
    dot:
        Graphviz digraph.
    bundle:
        Bundle to render.

    Returns
    -------
    None
        ``dot`` is mutated in place.
    """

    for member_name, member in bundle.members.items():
        with dot.subgraph(name=f"cluster_backward_{member_name}") as subgraph:
            subgraph.attr(label=f"{member_name} backward", color="#7A3E9D")
            grad_fns = list(getattr(member, "grad_fns", []))
            if not grad_fns:
                subgraph.node(
                    f"bwd_{member_name}_empty",
                    label="no backward graph",
                    shape="box",
                    style="dashed",
                    color="#7A3E9D",
                )
                continue
            visible_ids = {grad_fn_handle.grad_fn_object_id for grad_fn_handle in grad_fns}
            for grad_fn_handle in grad_fns:
                subgraph.node(
                    f"bwd_{member_name}_{grad_fn_handle.grad_fn_object_id}",
                    label=str(
                        getattr(
                            grad_fn_handle,
                            "label",
                            getattr(grad_fn_handle, "name", "grad_fn_handle"),
                        )
                    ),
                    shape="box",
                    style="filled,rounded",
                    fillcolor="#F4E8FA",
                    color="#7A3E9D",
                )
            for grad_fn_handle in grad_fns:
                for next_id in getattr(grad_fn_handle, "next_grad_fn_ids", []):
                    if next_id in visible_ids:
                        subgraph.edge(
                            f"bwd_{member_name}_{grad_fn_handle.grad_fn_object_id}",
                            f"bwd_{member_name}_{next_id}",
                            color="#7A3E9D",
                        )
