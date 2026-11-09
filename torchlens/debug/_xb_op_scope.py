"""The ops of one paired module call on each side, read for op-level placement.

Scope: the ops recorded inside the call and its descendant calls, minus the ops of
descendant calls that the module level paired and found equal (those collapse to
their recorded outputs, which become seeds). Seeds are pairs established without
values: the call's recorded inputs and the collapsed children's outputs, position
by position. An op is *anchored* when it is a seed or an in-scope op with an
anchored parent; unanchored ops (constants, parameter views, literal producers and
what is built only from them) are leaves the walk ignores.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..backends._op_spelling import Spelling, op_spelling
from ._common import _op_from_label, _op_label, _ordered_ops
from ._xb_calls import CallInfo
from ._xb_pairing import CallPair

if TYPE_CHECKING:
    from torchlens._data_classes.trace import Trace


@dataclass(frozen=True)
class SideGraph:
    """The anchored ops of one side's call, in execution order.

    Parameters
    ----------
    trace:
        The side's trace.
    walk:
        Anchored in-scope op labels in execution order.
    parents, children:
        Anchored parents (in recorded order) and anchored in-scope children per op.
    order:
        Execution position of every op of the trace.
    spelling:
        Canonical spelling and family per in-scope op.
    """

    trace: Trace
    walk: tuple[str, ...]
    parents: Mapping[str, tuple[str, ...]]
    children: Mapping[str, tuple[str, ...]]
    order: Mapping[str, int]
    spelling: Mapping[str, Spelling]


def spelling_backend(trace: Trace) -> str:
    """The backend whose spelling list labels ``trace``'s ops (a test hook)."""

    return str(trace._provenance_backend)


def _call_records(trace: Trace) -> dict[str, Any]:
    """Every recorded ``ModuleCall`` of ``trace`` by label."""

    return {
        str(call.label): call for module in trace._module_logs for call in module.calls.values()
    }


def descendants(calls: Iterable[CallInfo], root: str) -> list[str]:
    """Labels of the calls nested under ``root`` (any depth)."""

    parent = {call.label: call.parent for call in calls}
    found = []
    for label in parent:
        current, steps = parent[label], 0
        while current is not None and steps <= len(parent):
            if current == root:
                found.append(label)
                break
            current, steps = parent.get(current), steps + 1
    return found


def collapsed_pairs(
    root_a: CallInfo, pairs: list[CallPair], rows: list[dict[str, Any]], calls: list[CallInfo]
) -> list[CallPair]:
    """The paired, equal descendant calls of side a's ``root_a``, outermost first."""

    inside = set(descendants(calls, root_a.label))
    equal = [
        pair
        for pair, row in zip(pairs, rows, strict=True)
        if pair.a is not None
        and pair.b is not None
        and pair.a.label in inside
        and row["outcome"] == "equal"
        and row["alignment"] in ("supported", "asserted")
    ]
    return sorted(equal, key=lambda pair: pair.a.depth if pair.a is not None else 0)


def seeds(inputs: list[tuple[str, str]], collapsed: list[CallPair]) -> list[tuple[str, str]]:
    """Seed pairs: the root's inputs that met at module level, then each collapsed child's outputs.

    ``inputs`` are the root's recorded input pairs chosen by
    :func:`~torchlens.debug._xb_values.input_plan`, the same ones the module row compared.
    """

    found = list(inputs)
    for pair in collapsed:
        if pair.a is not None and pair.b is not None:
            found += _positional(pair.a.outputs, pair.b.outputs)
    return found


def _positional(labels_a: tuple[str, ...], labels_b: tuple[str, ...]) -> list[tuple[str, str]]:
    """Pair two recorded boundaries position by position (equal lengths only)."""

    if len(labels_a) != len(labels_b):
        return []
    return list(zip(labels_a, labels_b, strict=True))


def _scope(trace: Trace, root: str, nested: list[str], collapsed: list[str]) -> set[str]:
    """Op labels inside ``root`` and ``nested`` calls, minus those inside ``collapsed`` ones."""

    records = _call_records(trace)

    def ops_of(labels: Iterable[str]) -> set[str]:
        """Pass-qualified op labels recorded inside the named calls."""

        out: set[str] = set()
        for label in labels:
            for op_label in getattr(records.get(label), "ops", None) or ():
                op = _op_from_label(trace, str(op_label))
                if op is not None:
                    out.add(_op_label(op))
        return out

    return ops_of([root, *nested]) - ops_of(collapsed)


def side_graph(
    trace: Trace,
    root: str,
    calls: list[CallInfo],
    collapsed: list[str],
    seeded: set[str],
) -> SideGraph:
    """Build one side's anchored op graph for the call labelled ``root``.

    Parameters
    ----------
    trace:
        The side's trace.
    root:
        Label of the placed call on this side.
    calls:
        Every call of the trace (for nesting).
    collapsed:
        Labels of this side's collapsed child calls (and their descendants).
    seeded:
        This side's seed op labels.
    """

    nested = descendants(calls, root)
    hidden = [*collapsed, *(d for label in collapsed for d in descendants(calls, label))]
    scope = _scope(trace, root, nested, hidden)
    ordered = _ordered_ops(trace)
    order = {_op_label(op): index for index, op in enumerate(ordered)}
    backend = spelling_backend(trace)
    parents = _anchored(trace, ordered, scope, seeded)
    walk = tuple(sorted(parents, key=order.__getitem__))
    spelling = {
        label: op_spelling(backend, str(getattr(_op_from_label(trace, label), "func_name", "")))
        for label in walk
    }
    return SideGraph(trace, walk, parents, _children(walk, parents), order, spelling)


def _anchored(
    trace: Trace, ordered: list[Any], scope: set[str], seeded: set[str]
) -> dict[str, tuple[str, ...]]:
    """In-scope ops reachable from the seeds, each with its anchored parents."""

    anchored, parents = set(seeded), {}
    for op in ordered:
        label = _op_label(op)
        if label not in scope:
            continue
        kept = tuple(p for p in _resolved(trace, op) if p in anchored)
        if kept:
            anchored.add(label)
            parents[label] = kept
    return parents


def _children(
    walk: tuple[str, ...], parents: dict[str, tuple[str, ...]]
) -> dict[str, tuple[str, ...]]:
    """Invert ``parents`` over ``walk`` (children in execution order)."""

    children: dict[str, list[str]] = {}
    for label in walk:
        for parent in parents[label]:
            children.setdefault(parent, []).append(label)
    return {key: tuple(value) for key, value in children.items()}


def _resolved(trace: Trace, op: Any) -> list[str]:
    """``op``'s parent labels, pass-qualified, in recorded order (duplicates dropped)."""

    found: list[str] = []
    for parent in getattr(op, "parents", None) or ():
        resolved = _op_from_label(trace, str(parent))
        if resolved is not None:
            label = _op_label(resolved)
            if label not in found:
                found.append(label)
    return found
