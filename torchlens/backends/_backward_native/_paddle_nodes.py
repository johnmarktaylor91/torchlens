"""Paddle grad-node helpers for native backward capture, with no trace knowledge.

Reading nodes (``name()``, ``node_ptr()``, ``next_functions``), copying hook
gradients, walking the graph from the loss and matching an op's outgoing
gradient slots to its recorded inputs.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from ... import _state

__all__ = (
    "ACCUMULATION",
    "LIST_SLOT_NODES",
    "UNKNOWN",
    "Edge",
    "Unavailable",
    "alignments",
    "copy_grad",
    "copy_group",
    "frontier",
    "is_differentiable",
    "next_nodes",
    "node_name",
    "node_ptr",
    "route",
    "walk_nodes",
)


ACCUMULATION = "GradNodeAccumulation"
"""Paddle's name for the leaf node that accumulates a leaf tensor's gradient."""

UNKNOWN = object()
"""What an input the capture never labelled reaches: maybe a grad node, maybe no slot."""

LIST_SLOT_NODES = frozenset(
    {
        "ConcatGradNode",
        "StackGradNode",
        "AddNGradNode",
        "EinsumGradNode",
        "MeshgridGradNode",
        "BroadcastTensorsGradNode",
        "MultiDotGradNode",
        "UnstackGradNode",
    }
)
"""Nodes with a list-valued input slot, for nodes no recorded op describes."""


class Unavailable(Exception):
    """The pass cannot be recorded; the message is the refusal reason."""


def node_ptr(node: Any) -> int:
    return int(node.node_ptr())


def node_name(node: Any) -> str:
    return str(node.name())


def next_nodes(node: Any) -> list[Any]:
    return list(node.next_functions or ())


def copy_grad(value: Any) -> Any:
    """A detached copy of one gradient (``None`` stays ``None``)."""

    if value is None or not hasattr(value, "detach"):
        return None
    initialized = getattr(value, "_is_initialized", None)
    if callable(initialized) and not initialized():
        return None  # Paddle's placeholder for a slot that gets no gradient
    with _state.pause_logging():
        return value.detach().clone()


def copy_group(values: Any) -> tuple[Any, ...]:
    """Copy one hook argument: a slot is a tensor, ``None`` or a list of tensors."""

    return tuple(
        tuple(copy_grad(item) for item in slot)
        if isinstance(slot, (list, tuple))
        else copy_grad(slot)
        for slot in values
    )


def is_differentiable(tensor: Any) -> bool:
    import paddle

    if bool(getattr(tensor, "stop_gradient", True)):
        return False
    return bool(paddle.is_floating_point(tensor) or paddle.is_complex(tensor))


def walk_nodes(root: Any) -> dict[int, Any]:
    """Every grad node reachable from ``root``, in breadth-first discovery order."""

    found: dict[int, Any] = {}
    queue = [root]
    while queue:
        node = queue.pop(0)
        if node_ptr(node) in found:
            continue
        found[node_ptr(node)] = node
        queue.extend(item for item in next_nodes(node) if item is not None)
    return found


@dataclass(frozen=True)
class Edge:
    """One gradient slot leaving an op's backward: ``node``'s slot ``index`` to ``target``."""

    node: Any
    index: int
    target: Any


def frontier(site_node: Any, owned: set[int]) -> tuple[list[Edge], list[Any]]:
    """The op's outgoing slots in call order, and its inner nodes (expanded in place)."""

    edges: list[Edge] = []
    inner: list[Any] = []

    def visit(node: Any) -> None:
        for index, target in enumerate(next_nodes(node)):
            inner_node = (
                target is not None
                and node_ptr(target) not in owned
                and node_name(target) != ACCUMULATION
            )
            if inner_node and all(node_ptr(target) != node_ptr(seen) for seen in inner):
                inner.append(target)
                visit(target)
            else:
                edges.append(Edge(node, index, target))

    visit(site_node)
    return edges, inner


def matches(expected: Any, target: Any) -> bool:
    if expected is None or target is None:
        return expected is None and target is None
    if expected == ACCUMULATION:
        return node_name(target) == ACCUMULATION
    return node_ptr(expected) == node_ptr(target)


def choices(want: Any, edge: Edge | None) -> list[bool]:
    """Whether one input may consume the next edge (``True``) or owns no slot (``False``)."""

    if want is UNKNOWN:
        return [False] if edge is None else [True, False]
    if edge is not None and matches(want, edge.target):
        return [True]
    return [False] if want is None else []


def alignments(wants: list[Any], edges: list[Edge]) -> Iterator[tuple[int | None, ...]]:
    """Every way to give each input one edge or none, in order, using every edge once."""

    def walk(i: int, j: int) -> Iterator[tuple[int | None, ...]]:
        if i == len(wants):
            if j == len(edges):
                yield ()
            return
        edge = edges[j] if j < len(edges) else None
        for take in choices(wants[i], edge):
            for rest in walk(i + 1, j + 1 if take else j):
                yield (j if take else None, *rest)

    return walk(0, 0)


def route(edges: list[Edge], index: int | None) -> int | None:
    """The edge an input's gradient travels, or ``None`` when it carries none."""

    return None if index is None or edges[index].target is None else index
