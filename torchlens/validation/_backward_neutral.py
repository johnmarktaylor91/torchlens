"""Backward validation for backends whose backward pass is recorded natively.

Checks one recorded backward pass against the framework itself, three ways:

1. node: every recorded node's framework derivative (``adapter.vjp``) at its
   recorded incoming gradient equals its recorded outgoing gradients;
2. sum: every op's recorded incoming gradient equals the sum of its
   consumers' recorded contributions (ops with no consumer in the graph are
   the pass's seeds and are checked by 3);
3. leaf: the recorded gradient at each leaf equals the framework's
   whole-program gradient.

Every comparison is the backend validator's own (``adapter.close``); this
module adds no tolerance. A native backward lane journals
``GradFnFired.grad_output_refs`` as one incoming gradient per output slot and
``grad_input_refs`` as one outgoing gradient per parent occurrence, in call
order; :func:`recorded_backward_from_trace` reads them back in that form.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..backends._reverse_engine import EngineNode, ReplayGraph

__all__ = [
    "BackwardCheckFinding",
    "BackwardCheckResult",
    "RecordedBackward",
    "check_backward_records",
    "recorded_backward_from_trace",
]


@dataclass(frozen=True)
class RecordedBackward:
    """What one backward pass recorded, keyed by op label.

    Attributes
    ----------
    calls:
        Per node label: ``(incoming per output slot, outgoing per parent occurrence)``.
    grads:
        Recorded incoming gradient per op label (``Op.grad`` for the pass).
    """

    calls: Mapping[str, tuple[tuple[Any, ...], tuple[Any, ...]]]
    grads: Mapping[str, Any]


@dataclass(frozen=True)
class BackwardCheckFinding:
    """One failed comparison: which check, which op, and what differed."""

    check: str
    op_label: str
    detail: str


@dataclass(frozen=True)
class BackwardCheckResult:
    """Outcome of :func:`check_backward_records`; passes only with no finding."""

    findings: tuple[BackwardCheckFinding, ...]
    nodes_checked: int
    sums_checked: int
    leaves_checked: int

    @property
    def passed(self) -> bool:
        """Whether every comparison held and at least one node was checked."""

        return not self.findings and self.nodes_checked > 0


def _same(adapter: Any, expected: Any, actual: Any) -> bool:
    """``None`` matches only ``None``; arrays compare with the backend's ``close``."""

    if expected is None or actual is None:
        return expected is None and actual is None
    return bool(adapter.close(expected, actual))


@dataclass
class _Findings:
    items: list[BackwardCheckFinding] = field(default_factory=list)

    def add(self, check: str, label: str, detail: str) -> None:
        self.items.append(BackwardCheckFinding(check, label, detail))


def _check_node(
    trace: Any,
    adapter: Any,
    graph: ReplayGraph,
    node: EngineNode,
    call: tuple[tuple[Any, ...], tuple[Any, ...]],
    findings: _Findings,
) -> None:
    """Check 1 for one node: framework derivative at the recorded incoming gradient."""

    incoming, outgoing = call
    if len(incoming) != len(node.outputs) or len(outgoing) != len(node.parents):
        findings.add("node", node.label, "recorded gradients do not match the op's arity")
        return
    filled = [
        adapter.zeros_like(graph.value(slot)) if item is None else item
        for slot, item in zip(node.outputs, incoming, strict=True)
    ]
    parent_values = [graph.value(parent) for parent in node.parents]
    expected = tuple(adapter.vjp(trace, node.op, parent_values, filled))
    for index, (want, got) in enumerate(zip(expected, outgoing, strict=True)):
        if not _same(adapter, want, got):
            parent = node.parents[index]
            findings.add("node", node.label, f"outgoing gradient to {parent!r} differs")


def _consumer_sums(adapter: Any, graph: ReplayGraph, recorded: RecordedBackward) -> dict[str, Any]:
    """Sum of the recorded contributions each label receives from its consumers.

    Summed in reverse graph order, the order a backward pass accumulates them.
    """

    sums: dict[str, Any] = {}
    for node in reversed(graph.nodes):
        call = recorded.calls.get(node.label)
        if call is None or len(call[1]) != len(node.parents):
            continue
        for parent, contribution in zip(node.parents, call[1], strict=True):
            if contribution is None:
                continue
            prior = sums.get(parent)
            sums[parent] = contribution if prior is None else adapter.add(prior, contribution)
    return sums


def _check_sums(
    adapter: Any, graph: ReplayGraph, recorded: RecordedBackward, findings: _Findings
) -> int:
    """Check 2: recorded incoming gradients equal the consumers' summed contributions."""

    sums = _consumer_sums(adapter, graph, recorded)
    checked = 0
    for label, total in sums.items():
        if label not in recorded.grads:
            findings.add("sum", label, "consumers sent a gradient the op did not record")
            continue
        checked += 1
        if not _same(adapter, total, recorded.grads[label]):
            findings.add("sum", label, "recorded gradient differs from its consumers' sum")
    for node in graph.nodes:
        call = recorded.calls.get(node.label)
        if call is None:
            continue
        for slot, incoming in zip(node.outputs, call[0], strict=False):
            if slot in recorded.grads and not _same(adapter, recorded.grads[slot], incoming):
                findings.add("sum", slot, "node's incoming gradient differs from the op's")
    return checked


def _check_leaves(
    adapter: Any, recorded: RecordedBackward, leaf_grads: Mapping[str, Any], findings: _Findings
) -> int:
    """Check 3: recorded leaf gradients equal the framework's whole-program gradient."""

    for label, expected in leaf_grads.items():
        if label not in recorded.grads:
            findings.add("leaf", label, "no gradient was recorded for this leaf")
        elif not _same(adapter, expected, recorded.grads[label]):
            findings.add("leaf", label, "recorded gradient differs from the framework's")
    return len(leaf_grads)


def check_backward_records(
    trace: Any,
    adapter: Any,
    graph: ReplayGraph,
    recorded: RecordedBackward,
    leaf_grads: Mapping[str, Any],
) -> BackwardCheckResult:
    """Check one recorded backward pass against the framework.

    Parameters
    ----------
    trace:
        The trace the adapter replays.
    adapter:
        The backend's replay adapter.
    graph:
        The captured graph (``backends._reverse_engine.graph_from_recipes``).
    recorded:
        The pass's recorded calls and gradients.
    leaf_grads:
        The framework's whole-program gradient per leaf label.

    Returns
    -------
    BackwardCheckResult
        Every finding; ``passed`` only when there is none and a node was checked.
    """

    findings = _Findings()
    nodes_checked = 0
    for node in graph.nodes:
        call = recorded.calls.get(node.label)
        if call is None or not (node.parents or node.blocked):
            continue
        nodes_checked += 1
        reason = node.blocked or adapter.can_replay(trace, node.op)
        if reason is not None:
            findings.add("node", node.label, f"cannot be replayed to check ({reason})")
            continue
        _check_node(trace, adapter, graph, node, call, findings)
    sums_checked = _check_sums(adapter, graph, recorded, findings)
    leaves_checked = _check_leaves(adapter, recorded, leaf_grads, findings)
    return BackwardCheckResult(tuple(findings.items), nodes_checked, sums_checked, leaves_checked)


def _as_tuple(value: Any) -> tuple[Any, ...]:
    """A journaled gradient group as a tuple (a lone value becomes a 1-tuple)."""

    if value is None:
        return ()
    return tuple(value) if isinstance(value, (list, tuple)) else (value,)


def _pass_grads(trace: Any, pass_number: int) -> dict[str, Any]:
    """Recorded incoming gradient per op label for one pass."""

    grads: dict[str, Any] = {}
    for op in trace._layer_list:
        for record in op._slot("_grad_records") or ():
            if record.backward_pass_number == pass_number and record.grad is not None:
                grads[op.layer_label] = record.grad
    return grads


def recorded_backward_from_trace(trace: Any, pass_number: int) -> RecordedBackward:
    """Read one pass's recorded calls and gradients off ``trace``'s backward records."""

    calls: dict[str, tuple[tuple[Any, ...], tuple[Any, ...]]] = {}
    for record in trace._grad_fn_logs.values():
        if record.op_label is None:
            continue
        for call in getattr(record.calls, "_dict", record.calls).values():
            if call.backward_pass_number == pass_number:
                calls[record.op_label] = (_as_tuple(call.grad_outputs), _as_tuple(call.grad_inputs))
    return RecordedBackward(calls=calls, grads=_pass_grads(trace, pass_number))
