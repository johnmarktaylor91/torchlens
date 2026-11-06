"""Backward-view style inventory and pass-row helpers (vizmech item 17).

Split from ``_render_leaf.py`` (the R43 file-size ratchet): the per-render
inventory of painted backward styles feeds the in-frame key and the
uniform-constant-row suppression; the pass-row helper spells the ``bwd N``
label line under that suppression.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, TypeVar, cast

from .._utils.display import int_list_to_compact_str

if TYPE_CHECKING:
    import graphviz

    from .._data_classes.grad_fn import GradFn
    from .._data_classes.trace import Trace
    from ._render_common import BackwardPassFilter

__all__ = [
    "BackwardStyleInventory",
    "_backward_dot_call_node_name",
    "_backward_dot_node_name",
    "_backward_pass_row",
    "backward_pass_cluster",
    "compute_backward_style_inventory",
    "ungroup_single_pass",
]

_IRNodes = TypeVar("_IRNodes", bound=tuple[Any, ...])
_IREdges = TypeVar("_IREdges", bound=tuple[Any, ...])


@dataclass(frozen=True)
class BackwardStyleInventory:
    """Per-render inventory of the backward styles ACTUALLY painted.

    Computed ONCE over the visible grad_fns (vizmech item 17): it feeds the
    in-frame backward key (a key row for an unused style would itself be an
    unexplained claim) and the uniform-row suppression (a row carrying the
    same degenerate value on every node is ink, not information). The grad
    row needs no inventory: a node without a stored gradient omits it.
    """

    has_higher_order: bool
    has_intervening: bool
    has_accumulation: bool
    has_custom: bool
    num_backward_passes: int
    suppress_order_row: bool
    suppress_bwd_row: bool


def compute_backward_style_inventory(
    trace: Trace,
    pass_filter: BackwardPassFilter = None,
) -> BackwardStyleInventory:
    """Build the style inventory over the pass-filtered visible grad_fns.

    Parameters
    ----------
    trace:
        Trace with a captured backward graph.
    pass_filter:
        Normalized backward-pass filter (visibility must match the render).
    """

    # Call-time import: the filter helper lives in ``_render_leaf``, which
    # imports THIS module at load time.
    # deferred: cycle _visualization
    from ._render_leaf import _grad_fn_matches_backward_filter

    visible = [
        grad_fn_handle
        for grad_fn_handle in trace.autograd.grad_fns
        if _grad_fn_matches_backward_filter(grad_fn_handle, pass_filter)
    ]
    orders = [getattr(grad_fn_handle, "order", None) for grad_fn_handle in visible]
    has_higher_order = any(order is not None and order > 1 for order in orders)
    num_passes = int(trace._num_backward_passes or 1)
    return BackwardStyleInventory(
        has_higher_order=has_higher_order,
        has_intervening=any(not grad_fn_handle.has_op for grad_fn_handle in visible),
        has_accumulation=any(grad_fn_handle.type == "accumulategrad" for grad_fn_handle in visible),
        has_custom=any(grad_fn_handle.is_custom for grad_fn_handle in visible),
        num_backward_passes=num_passes,
        suppress_order_row=not has_higher_order,
        suppress_bwd_row=num_passes <= 1,
    )


def _backward_pass_row(
    grad_fn_handle: GradFn,
    call: Any | None,
    pass_filter: BackwardPassFilter,
    inventory: BackwardStyleInventory | None,
) -> list[str]:
    """Return the ``bwd N`` label row (empty when suppressed or unknown).

    Suppressed as a uniform constant row on single-pass captures (vizmech
    item 17); otherwise rolled nodes list their passes compactly and
    unrolled calls carry their one pass index.
    """

    if inventory is not None and inventory.suppress_bwd_row:
        return []
    if call is None:
        pass_indices = sorted(
            {
                int(pass_index)
                for pass_index in (
                    getattr(grad_fn_call, "backward_pass_number", None)
                    for grad_fn_call in grad_fn_handle.calls.values()
                )
                if pass_index is not None
            }
        )
        if pass_filter is not None:
            pass_indices = [pass_index for pass_index in pass_indices if pass_index in pass_filter]
        if pass_indices:
            return [f"bwd {int_list_to_compact_str(pass_indices)}"]
        return []
    pass_index = getattr(call, "backward_pass_number", None)
    if pass_index is not None:
        return [f"bwd {pass_index}"]
    return []


def _backward_dot_node_name(grad_fn_handle: GradFn) -> str:
    """Return a DOT-safe node name for a grad_fn_handle log.

    Parameters
    ----------
    grad_fn_handle:
        GradFn to name.

    Returns
    -------
    str
        DOT-safe node identifier.
    """

    return f"grad_fn_{grad_fn_handle.grad_fn_object_id}"


def _backward_dot_call_node_name(grad_fn_handle: GradFn, call: Any) -> str:
    """Return a DOT-safe node name for one GradFnCall.

    Parameters
    ----------
    grad_fn_handle:
        GradFn owning the call.
    call:
        GradFnCall-like record.

    Returns
    -------
    str
        DOT-safe node identifier.
    """

    return (
        f"grad_fn_{grad_fn_handle.grad_fn_object_id}_"
        f"bwd{getattr(call, 'backward_pass_number', 0)}_call{getattr(call, 'call_index', 0)}"
    )


@contextmanager
def backward_pass_cluster(
    graph: graphviz.Digraph, pass_index: int, *, grouped: bool, color: str
) -> Iterator[graphviz.Digraph]:
    """Yield where one backward pass's nodes go: its dashed cluster, or the graph itself.

    A drawing of two or more passes boxes each pass as ``backward pass N`` in
    ``color``; a drawing of one pass needs no box or label (``grouped=False``).
    """

    if not grouped:
        yield graph
        return
    with graph.subgraph(name=f"cluster_backward_pass_{pass_index}") as subgraph:
        subgraph.attr(
            label=f"backward pass {pass_index}",
            color=color,
            fontcolor="black",
            style="rounded,dashed",
        )
        yield subgraph


def ungroup_single_pass(
    nodes: _IRNodes, edges: _IREdges, visible_by_pass: dict[int, list[tuple[Any, Any]]]
) -> tuple[_IRNodes, _IREdges, dict[int, list[tuple[Any, Any]]]]:
    """Return backward IR nodes and edges with no pass grouping when one pass is drawn.

    Mirrors :func:`backward_pass_cluster`: with fewer than two drawn passes there
    is no ``backward_pass_N`` cluster, so no node or edge names one and no region
    is built (the returned pass map is empty).
    """

    if len(visible_by_pass) > 1:
        return nodes, edges, visible_by_pass
    ungrouped_nodes = tuple(replace(node, owner_cluster=None, region_path=()) for node in nodes)
    ungrouped_edges = tuple(replace(edge, owner_cluster=None) for edge in edges)
    return cast(_IRNodes, ungrouped_nodes), cast(_IREdges, ungrouped_edges), {}
