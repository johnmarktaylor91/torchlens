"""Pair the ops of two same-backend traces for ``tl.debug.compare`` and ``first_divergence``.

When the two traces' graph fingerprints (``trace.provenance.fingerprint``) are equal the
graphs have the same ops in the same order, so ops pair by label exactly as before. When
they differ, one inserted or deleted op shifts every later label, and a label join pairs
the wrong ops or none. Ops then pair by graph structure through the Bundle topology
matcher (``intervention/_topology/topology.py``), reused as is: same module and function,
already-paired parents, execution order. The matcher runs on light stand-ins for the ops
and is rerun with unpaired ops seen through (an op consuming an inserted op keeps the
inserted op's parents) until the pairs stop changing.

The evidence rule holds: a pair is established without looking at either value. An op the
matcher cannot pair is a coverage gap (``only-a`` / ``only-b``). When the ops sharing one
structural signature (module, function, paired parents) differ in number between the two
traces, the matcher could only guess which of them correspond, so none of them pairs and
each row says ``ambiguous``.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .._data_classes._module_edit_sites import replaced_site
from ..intervention._topology.topology import _find_alignment
from ._common import _op_label, _ordered_ops

if TYPE_CHECKING:
    from torchlens._data_classes.trace import Trace

#: ``reason`` of a row whose op was left unpaired because its structural twins differ in number.
AMBIGUOUS_NOTE = "ambiguous: ops with the same structure differ in number"
#: ``reason`` note on a compared row paired by structure under a different label.
STRUCTURE_NOTE = "paired by graph structure"
#: Rounds of re-pairing with unpaired ops seen through; each round only refines the last.
_MAX_ROUNDS = 8


@dataclass(frozen=True)
class _Node:
    """The fields of one op the topology matcher reads, under the label its children use."""

    layer_label: str
    module: Any
    func_name: Any
    parents: tuple[str, ...]


@dataclass(frozen=True)
class _Side:
    """One trace's ops as matcher nodes, plus the label maps between nodes and rows."""

    nodes: dict[str, _Node]
    row_label: dict[str, str]
    raw_parents: dict[str, tuple[str, ...]]


#: One set of node keys per trace, ``(trace_a's, trace_b's)``.
_Pair = tuple[set[str], set[str]]


def _lookup_label(op: Any) -> str:
    """The label an op's children list it by: per-pass only for multi-pass ops."""

    if int(getattr(op, "num_passes", 1) or 1) == 1:
        return str(op.layer_label)
    return _op_label(op)


def _side(trace: Trace, compared: dict[str, Any]) -> _Side:
    """Build a trace's matcher view; module-edit replacement ops are folded into their site."""

    nodes: dict[str, _Node] = {}
    row_label: dict[str, str] = {}
    raw_parents: dict[str, tuple[str, ...]] = {}
    folded: dict[str, str] = {}
    for op in _ordered_ops(trace):
        site = replaced_site(op, trace)
        if site is not op:
            folded[_lookup_label(op)] = _lookup_label(site)
            continue
        key = _lookup_label(op)
        raw_parents[key] = tuple(str(parent) for parent in getattr(op, "parents", ()) or ())
        nodes[key] = _Node(key, getattr(op, "module", None), getattr(op, "func_name", None), ())
        if _op_label(op) in compared:
            row_label[key] = _op_label(op)
    raw_parents = {
        key: tuple(dict.fromkeys(folded.get(parent, parent) for parent in parents))
        for key, parents in raw_parents.items()
    }
    return _Side(nodes, row_label, raw_parents)


def _visible_parents(side: _Side, key: str, hidden: set[str]) -> tuple[str, ...]:
    """A node's parents with every hidden (unpaired) parent replaced by its own parents."""

    found: dict[str, None] = {}
    pending, seen = deque(side.raw_parents.get(key, ())), set()
    while pending:
        parent = pending.popleft()
        if parent in seen:
            continue
        seen.add(parent)
        if parent in hidden:
            pending.extend(side.raw_parents.get(parent, ()))
        elif parent in side.nodes:
            found[parent] = None
    return tuple(found)


def _view(side: _Side, hidden: set[str], out: set[str]) -> list[_Node]:
    """Matcher nodes in execution order: ``out`` nodes dropped, ``hidden`` parents seen through."""

    unseen = hidden | out
    return [
        _Node(key, node.module, node.func_name, _visible_parents(side, key, unseen))
        for key, node in side.nodes.items()
        if key not in out
    ]


def _align(sides: tuple[_Side, _Side], hidden: _Pair, out: _Pair) -> dict[str, str]:
    """One run of the Bundle topology matcher over the two traces' current views."""

    nodes_a, nodes_b = _view(sides[0], hidden[0], out[0]), _view(sides[1], hidden[1], out[1])
    pairs = _find_alignment(nodes_a, nodes_b)  # type: ignore[arg-type]
    return {nodes_a[i].layer_label: nodes_b[j].layer_label for i, j in pairs}


def _settle(sides: tuple[_Side, _Side], out: _Pair) -> dict[str, str]:
    """Rerun the matcher, seeing through the ops the last run left unpaired, until stable."""

    pairs = _align(sides, (set(), set()), out)
    for _ in range(_MAX_ROUNDS):
        hidden = (set(sides[0].nodes) - set(pairs), set(sides[1].nodes) - set(pairs.values()))
        refined = _align(sides, hidden, out)
        if refined == pairs:
            break
        pairs = refined
    return pairs


def _signatures(side: _Side, to_a: dict[str, str]) -> dict[str, tuple[str, str, frozenset]]:
    """Each node's structural signature, its paired parents named by trace a's labels."""

    hidden = set(side.nodes) - set(to_a)
    return {
        key: (
            str(node.module),
            str(node.func_name),
            frozenset(to_a[p] for p in _visible_parents(side, key, hidden) if p in to_a),
        )
        for key, node in side.nodes.items()
    }


def _ambiguous(sides: tuple[_Side, _Side], pairs: dict[str, str]) -> _Pair:
    """Nodes whose signature is shared by a different number of nodes in each trace."""

    sig_a = _signatures(sides[0], {a: a for a in pairs})
    sig_b = _signatures(sides[1], {b: a for a, b in pairs.items()})
    count_a, count_b = Counter(sig_a.values()), Counter(sig_b.values())
    unequal = {sig for sig in set(count_a) & set(count_b) if count_a[sig] != count_b[sig]}
    return (
        {key for key, sig in sig_a.items() if sig in unequal},
        {key for key, sig in sig_b.items() if sig in unequal},
    )


def structural_pairs(
    trace_a: Trace, trace_b: Trace, ops_a: dict[str, Any], ops_b: dict[str, Any]
) -> tuple[dict[str, str], _Pair]:
    """Pair two traces' compared ops by graph structure.

    Parameters
    ----------
    trace_a, trace_b:
        The two completed traces.
    ops_a, ops_b:
        Their compared ops keyed by row label (``_compared_site_ops``).

    Returns
    -------
    tuple[dict[str, str], tuple[set[str], set[str]]]
        Row label in ``trace_a`` to row label in ``trace_b`` for every structural pair,
        and the row labels of each trace left unpaired as ambiguous.
    """

    sides = (_side(trace_a, ops_a), _side(trace_b, ops_b))
    out: _Pair = (set(), set())
    while True:
        pairs = _settle(sides, out)
        found_a, found_b = _ambiguous(sides, pairs)
        if found_a <= out[0] and found_b <= out[1]:
            break
        out = (out[0] | found_a, out[1] | found_b)
    rows = {
        sides[0].row_label[a]: sides[1].row_label[b]
        for a, b in pairs.items()
        if a in sides[0].row_label and b in sides[1].row_label
    }
    ambiguous = tuple(
        {side.row_label[key] for key in keys if key in side.row_label}
        for side, keys in zip(sides, out, strict=True)
    )
    return rows, (ambiguous[0], ambiguous[1])


def _pairing(
    trace_a: Trace, trace_b: Trace, ops_a: dict[str, Any], ops_b: dict[str, Any]
) -> tuple[dict[str, str], _Pair]:
    """Label pairs for equal graph fingerprints, else :func:`structural_pairs`."""

    fingerprint = trace_a._provenance_fingerprint
    if fingerprint is not None and fingerprint == trace_b._provenance_fingerprint:
        return {label: label for label in ops_a if label in ops_b}, (set(), set())
    return structural_pairs(trace_a, trace_b, ops_a, ops_b)


def _gap_note(label: str, ambiguous: set[str], side: str) -> str:
    """The ``reason`` of an unpaired row: ambiguous, or the side it exists on."""

    return AMBIGUOUS_NOTE if label in ambiguous else side


def paired_rows(
    trace_a: Trace, trace_b: Trace, ops_a: dict[str, Any], ops_b: dict[str, Any]
) -> Iterator[tuple[str, str, Any, Any, str]]:
    """Yield ``(row label, status, op in a or None, op in b or None, note)`` in row order.

    Rows follow ``trace_a``'s execution order; ops found only in ``trace_b`` follow in its
    execution order. Equal graph fingerprints pair by label (the note of an unpaired row
    is ``only-a`` / ``only-b``, of a pair ``""``); different fingerprints pair by graph
    structure (:func:`structural_pairs`), noting :data:`STRUCTURE_NOTE` on a pair whose
    labels differ and :data:`AMBIGUOUS_NOTE` on an op left unpaired as ambiguous.
    """

    pairs, ambiguous = _pairing(trace_a, trace_b, ops_a, ops_b)
    for label, op_a in ops_a.items():
        if label not in pairs:
            yield label, "only-a", op_a, None, _gap_note(label, ambiguous[0], "only-a")
            continue
        note = "" if pairs[label] == label else STRUCTURE_NOTE
        yield label, "present-in-both", op_a, ops_b[pairs[label]], note
    paired_b = set(pairs.values())
    for label, op_b in ops_b.items():
        if label not in paired_b:
            yield label, "only-b", None, op_b, _gap_note(label, ambiguous[1], "only-b")


def row_source(op_a: Any, op_b: Any) -> Any:
    """The op a row's ``source_line`` comes from: ``op_a`` unless only ``op_b`` exists."""

    return op_b if op_a is None else op_a
