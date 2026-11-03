"""Module-scope placement for rendered edges and rolled endpoints.

An edge is drawn inside the deepest module cluster holding both endpoints. This
module finds that cluster and, in rolled mode, queues endpoints whose pass pairs
run under a different parent than the layer's first pass into their own frame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from .._data_classes.layer import Layer
from ._render_leaf import _base_node_for_metadata

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from ._render_common import GraphNode

#: One rendered edge's endpoints: (parent node, tail name), (child node, head name).
_RolledEdgeEndpoints = tuple[tuple["GraphNode", str], tuple["GraphNode", str]]


def _get_lowest_module_for_two_nodes(
    node1: Op | Layer,
    node2: Op | Layer,
    both_nodes_collapsed_modules: bool,
    vis_call_depth: int,
) -> str | int:
    """Find the deepest module subgraph that contains both nodes.

    Used to place an edge into the correct Graphviz cluster (subgraph).
    Edges between nodes in the same module cluster are drawn inside that
    cluster; edges crossing module boundaries are drawn at the level of
    the lowest common ancestor module.

    Returns -1 when no module contains both nodes (the edge belongs to the
    top-level graph, not any subgraph).

    Special handling:
    - ``is_atomic_module`` nodes are adjusted to their parent
      scope (they represent the module's output, rendered one level up).
    - Rolled mode: pass suffixes are stripped from module names so that all
      ops share the same cluster.
    - Both-collapsed case: when both nodes are collapsed module boxes, the
      containing module must be at least one level above the collapse depth.

    Args:
        node1: The first node.
        node2: The second node.
        both_nodes_collapsed_modules: Whether both nodes are collapsed module boxes.
        vis_call_depth: How many levels deep to visualize.

    Returns:
        Module name (str) for the containing cluster, or -1 for top-level.
    """
    node1_modules = node1.modules[:]
    node2_modules = node2.modules[:]

    if isinstance(node1, Layer) or isinstance(node2, Layer):
        node1_modules = [module.split(":")[0] for module in node1_modules]
        node2_modules = [module.split(":")[0] for module in node2_modules]

    if node1.is_atomic_module:
        node1_nested_modules = node1_modules[:-1]
    else:
        node1_nested_modules = node1_modules[:]

    if (
        (len(node1_modules) == 0)
        or (len(node2_modules) == 0)
        or (node1_modules[0] != node2_modules[0])
    ):
        return -1  # no submodule contains them both.

    if node1 == node2:
        if node1.is_atomic_module and (len(node1_modules) == 1):
            return -1
        elif node1.is_atomic_module and (len(node1_modules) > 1):
            module = node1_modules[-2]
        else:
            module = node1_modules[-1]
        return cast(str, module)

    if both_nodes_collapsed_modules:
        if (vis_call_depth == 1) or (len(node1_nested_modules) == 1):
            return -1
        if len(node1_modules) < vis_call_depth or len(node2_modules) < vis_call_depth:
            return -1
        if node1_modules[vis_call_depth - 1] == node2_modules[vis_call_depth - 1]:
            module = node1_modules[vis_call_depth - 2]
            return cast(str, module)

    module = node1_modules[0]
    for m in range(min(len(node1_modules), len(node2_modules))):
        if node1_modules[m] != node2_modules[m]:
            break
        module = node1_modules[m]

    return cast(str, module)


def _rolled_module_path(node: Op | Layer) -> list[str]:
    """Return a node's module path with the pass suffixes stripped."""

    return [module.split(":")[0] for module in node.modules]


def _passes_share_module_path(layer: Layer) -> bool:
    """Whether every pass of ``layer`` runs inside the layer's own module path."""

    own_path = _rolled_module_path(layer)
    return all(_rolled_module_path(op) == own_path for op in layer.ops.values())


def _rolled_edge_pass_scopes(
    parent: Layer,
    child: Layer,
    vis_call_depth: int,
) -> list[str]:
    """Return the pass-stripped innermost modules holding each pass pair of a rolled edge.

    A rolled edge stands for every (parent pass, child pass) pair it joins. When a
    layer's passes run under different parents (one module instance shared by two
    parents), the layer-level module path is only the first pass's, so the pair
    that runs in a later pass's parent has its own, different scope.
    """

    # ``Op.children`` holds lookup labels: the bare layer label for a single-pass
    # layer, the pass-qualified op label otherwise.
    child_ops = {
        (op.layer_label if op.num_passes == 1 else op.label): op for op in child.ops.values()
    }
    scopes: list[str] = []
    for parent_op in parent.ops.values():
        for child_label in parent_op.children:
            child_op = child_ops.get(child_label)
            if child_op is None:
                continue
            scope = _get_lowest_module_for_two_nodes(parent_op, child_op, False, vis_call_depth)
            if scope == -1:
                continue
            stripped = cast(str, scope).split(":")[0]
            if stripped not in scopes:
                scopes.append(stripped)
    return scopes


def _place_rolled_endpoints_in_pass_scope(
    endpoints: _RolledEdgeEndpoints,
    edge_module_key: str | int,
    vis_call_depth: int,
    module_edge_dict: dict[str, Any],
) -> None:
    """Put rolled endpoints into the module frame their pass pair runs in.

    Rolled frames are filled by the edges emitted inside them. An edge whose
    endpoints are drawn under different parents is emitted above both, so an
    endpoint whose only in-module edge is such a pass pair was left outside its
    own module's frame. This queues that endpoint as a node of the pass pair's
    module, the cluster an in-module edge would have pulled it into. Layers whose
    passes all share one module path are untouched.
    """

    parent = _base_node_for_metadata(endpoints[0][0])
    child = _base_node_for_metadata(endpoints[1][0])
    if not isinstance(parent, Layer) or not isinstance(child, Layer):
        return
    if _passes_share_module_path(parent) and _passes_share_module_path(child):
        return
    drawn_key = None if edge_module_key == -1 else str(edge_module_key)
    for scope in _rolled_edge_pass_scopes(parent, child, vis_call_depth):
        if drawn_key is not None and (scope == drawn_key or drawn_key.startswith(f"{scope}.")):
            continue
        queued = module_edge_dict[scope].setdefault("nodes", [])
        queued_names = {node_args.get("name") for node_args in queued}
        for endpoint, name in endpoints:
            base = _base_node_for_metadata(endpoint)
            if scope in _rolled_module_path(base) and name not in queued_names:
                queued.append({"name": name})
                queued_names.add(name)


__all__ = [
    "_RolledEdgeEndpoints",
    "_get_lowest_module_for_two_nodes",
    "_place_rolled_endpoints_in_pass_scope",
]
