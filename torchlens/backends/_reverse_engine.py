"""Reverse gradient engine over a captured graph, through a backend's replay adapter.

Gradient values for every recorded op, computed from the framework's own
per-op derivative (``adapter.vjp``) and summed over each op's consumers. The
engine returns values only and builds no records: backward records come only
from an observed backward pass (``backends/_backward_neutral.py``). It needs no
backward tape, so it works on every backend with a replay adapter; the
``tl.grad`` edit sites run on it.

The graph is a sequence of :class:`EngineNode` in forward graph order. Each
node names the op the adapter takes, its parent occurrences (repeats kept, in
call order) and its output slots; ``value(label)`` returns the saved value of
an output slot. :func:`graph_from_recipes` derives the nodes from the
adapter's own recipes. Labels no node produces are leaves (graph inputs,
parameters): they collect gradients and propagate nothing. A node the adapter
cannot describe is blocked, and a gradient reaching it is refused rather than
silently stopped there.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..errors import RunCapabilityUnavailableError

__all__ = (
    "EngineNode",
    "ReplayGraph",
    "graph_from_recipes",
    "reverse_gradients",
)

_DETECTION_STAGE = "reverse_engine"
_CODE = "run_capability_unavailable"  # RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE


@dataclass(frozen=True)
class EngineNode:
    """One recorded op as the engine sees it.

    Attributes
    ----------
    op:
        What the adapter functions take for this op.
    parents:
        Parent labels per occurrence, in call order, repeats kept.
    outputs:
        Output slot labels in emission order.
    blocked:
        Why the adapter cannot describe this op, or ``None``.
    """

    op: Any
    parents: tuple[str, ...]
    outputs: tuple[str, ...]
    blocked: str | None = None

    @property
    def label(self) -> str:
        """The op's label (its first output slot)."""

        return self.outputs[0]


@dataclass(frozen=True)
class ReplayGraph:
    """Nodes in forward graph order and the saved value of each output slot."""

    nodes: tuple[EngineNode, ...]
    value: Callable[[str], Any]

    def node_for(self, label: str) -> EngineNode | None:
        """Return the node with output slot ``label``, or ``None``."""

        for node in self.nodes:
            if label in node.outputs:
                return node
        return None


def graph_from_recipes(
    trace: Any, adapter: Any, ops: Iterable[Any], value: Callable[[str], Any]
) -> ReplayGraph:
    """Build the engine graph from ``adapter.recipe`` for each op in ``ops``.

    An op without a recipe keeps its place as a blocked node: a gradient that
    reaches it is refused, because the gradients upstream of it would be
    incomplete. Leave graph inputs out of ``ops`` to make them leaves.
    """

    nodes = []
    for op in ops:
        recipe = adapter.recipe(trace, op)
        if recipe is None:
            label = getattr(op, "label", None) or getattr(op, "layer_label", None)
            reason = adapter.can_replay(trace, op) or "the adapter has no recipe for it"
            nodes.append(EngineNode(op=op, parents=(), outputs=(str(label),), blocked=reason))
            continue
        nodes.append(EngineNode(op=op, parents=recipe.parent_labels, outputs=recipe.output_slots))
    return ReplayGraph(nodes=tuple(nodes), value=value)


def _check_forward_order(graph: ReplayGraph) -> None:
    """Refuse a node list where a parent is produced after its consumer."""

    produced: set[str] = set()
    all_outputs = {slot for node in graph.nodes for slot in node.outputs}
    for node in graph.nodes:
        late = [p for p in node.parents if p in all_outputs and p not in produced]
        if late:
            raise RunCapabilityUnavailableError(
                f"reverse gradients are unavailable: op {node.label!r} consumes "
                f"{late[0]!r} before it is produced",
                code=_CODE,
                detection_stage=_DETECTION_STAGE,
            )
        produced.update(node.outputs)


def _accumulate(adapter: Any, cotangents: dict[str, Any], label: str, contribution: Any) -> None:
    """Add ``contribution`` into the running cotangent of ``label``."""

    if contribution is None:
        return
    prior = cotangents.get(label)
    cotangents[label] = contribution if prior is None else adapter.add(prior, contribution)


def _node_contributions(
    trace: Any, adapter: Any, graph: ReplayGraph, node: EngineNode, incoming: Sequence[Any]
) -> tuple[Any, ...]:
    """One cotangent per parent occurrence of ``node`` from its incoming cotangents."""

    reason = node.blocked or adapter.can_replay(trace, node.op)
    if reason is not None:
        raise RunCapabilityUnavailableError(
            f"reverse gradients are unavailable: op {node.label!r} cannot be replayed ({reason})",
            code=_CODE,
            detection_stage=_DETECTION_STAGE,
        )
    filled = [
        adapter.zeros_like(graph.value(slot)) if cotangent is None else cotangent
        for slot, cotangent in zip(node.outputs, incoming, strict=True)
    ]
    parent_values = [graph.value(parent) for parent in node.parents]
    contributions = tuple(adapter.vjp(trace, node.op, parent_values, filled))
    if len(contributions) != len(node.parents):
        raise RunCapabilityUnavailableError(
            f"reverse gradients are unavailable: the adapter's derivative of op "
            f"{node.label!r} returned {len(contributions)} cotangents for "
            f"{len(node.parents)} parent occurrences",
            code=_CODE,
            detection_stage=_DETECTION_STAGE,
        )
    return contributions


def reverse_gradients(
    trace: Any,
    adapter: Any,
    graph: ReplayGraph,
    seeds: Mapping[str, Any],
) -> dict[str, Any]:
    """Propagate ``seeds`` backward over ``graph``; return the gradient of every reached label.

    Parameters
    ----------
    trace:
        The trace the adapter replays.
    adapter:
        The backend's replay adapter (``_replay_adapters.get_adapter(name)``).
    graph:
        Nodes in forward graph order and saved values.
    seeds:
        Cotangent per output slot label to start from (for a scalar loss, ones
        at the loss op).

    Returns
    -------
    dict[str, Any]
        Gradient per label that receives one: each op's incoming gradient is
        the sum of its consumers' contributions plus its seed.

    Raises
    ------
    RunCapabilityUnavailableError
        When a reached op is blocked or cannot be replayed, the adapter returns the wrong
        number of cotangents, or the nodes are not in forward order.
    """

    _check_forward_order(graph)
    cotangents: dict[str, Any] = {}
    for label, seed in seeds.items():
        _accumulate(adapter, cotangents, label, seed)
    for node in reversed(graph.nodes):
        incoming = [cotangents.get(slot) for slot in node.outputs]
        if all(item is None for item in incoming) or not (node.parents or node.blocked):
            continue
        contributions = _node_contributions(trace, adapter, graph, node, incoming)
        for parent, contribution in zip(node.parents, contributions, strict=True):
            _accumulate(adapter, cotangents, parent, contribution)
    return cotangents
