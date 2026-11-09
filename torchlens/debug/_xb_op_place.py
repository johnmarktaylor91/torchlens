"""Op-level placement inside the first differing module call (slice 3).

Runs only when that call's inputs agree. It appends one fixed phrase to the row's
``reason``: the op (one per side) or the operation group whose entry agrees and
whose exit differs, plus disclosures (decomposition gaps closed by agreement, ties
broken by spelling, ambiguous candidates). The module verdict is never changed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ._xb_op_scope import SideGraph, collapsed_pairs, seeds, side_graph
from ._xb_op_walk import Placement, Walk, place
from ._xb_pairing import CallPair
from ._xb_values import EQUAL, ValueOptions, compare_leaf, input_scale, read_values

if TYPE_CHECKING:
    from torchlens._data_classes.trace import Trace

#: Families whose op outputs may legitimately differ by a channels-last layout.
_LAYOUT_FAMILIES = frozenset({"conv", "pool", "norm"})
_SHOWN = 6


def _judge(graphs: tuple[SideGraph, SideGraph], options: ValueOptions) -> Any:
    """A cached ``(label_a, label_b) -> agrees`` reader over both traces' recorded values."""

    cache: dict[tuple[int, str], Any] = {}

    def value(side: int, label: str) -> Any:
        """The recorded tensor of ``label`` on ``side`` (``None`` when unavailable)."""

        key = (side, label)
        if key not in cache:
            cache[key] = read_values(graphs[side].trace, (label,))[0][0]
        return cache[key]

    def family(side: int, label: str) -> str:
        """The op's family, ``other`` for ops outside the walk (seeds)."""

        found = graphs[side].spelling.get(label)
        return found.family if found is not None else "other"

    def judge(label_a: str, label_b: str) -> bool:
        """Whether the two recorded values agree under the row's tolerance policy."""

        a, b = value(0, label_a), value(1, label_b)
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            return False
        justified = (
            family(0, label_a) in _LAYOUT_FAMILIES and family(1, label_b) in _LAYOUT_FAMILIES
        )
        return compare_leaf(a, b, justified=justified, options=options).outcome == EQUAL

    return judge


def _listed(labels: Any) -> str:
    """At most :data:`_SHOWN` labels, comma-separated, with a count of the rest."""

    labels = list(labels)
    more = f" and {len(labels) - _SHOWN} more" if len(labels) > _SHOWN else ""
    return ", ".join(labels[:_SHOWN]) + more


def _phrase(found: Placement, walk: Walk) -> str:
    """The reason phrase for a placement and its disclosures."""

    a, b = found.ops_a, found.ops_b
    if found.exit == "none":
        lead = "op placement: none (the recorded outputs agree op by op)"
    elif not a and not b:
        lead = "op placement: none (no op inside the call is reachable from its paired inputs)"
    elif len(a) <= 1 and len(b) <= 1:
        side_a = f"a {a[0]}" if a else "a (no op)"
        side_b = f"b {b[0]}" if b else "b (no op)"
        lead = f"op: {side_a} against {side_b}; exit differs ({found.exit})"
    else:
        lead = (
            f"op group: a [{_listed(a)}] against b [{_listed(b)}]; entry agrees, exit differs "
            f"({found.exit})"
        )
    parts = [lead]
    if walk.gaps:
        parts.append(f"decomposition gaps (agree at their exit): {_listed(walk.gaps)}")
    if walk.spelled:
        parts.append(f"tie broken by spelling: {_listed(walk.spelled)}")
    if walk.ambiguous:
        parts.append(f"ambiguous candidates at: {_listed(walk.ambiguous)}")
    return "; ".join(parts)


def placement_reason(
    pair: CallPair,
    table: tuple[list[dict[str, Any]], list[CallPair]],
    facts: dict[str, Any],
    traces: tuple[Trace, Trace],
    tol: tuple[float | None, float | None],
) -> str:
    """The op placement phrase for the first differing module row ``pair``.

    Parameters
    ----------
    pair:
        The first differing paired call.
    table:
        Every module row and its pair (collapsed children come from it).
    facts:
        ``module_rows`` facts (``calls``, ``cross_backend``).
    traces:
        The two traces.
    tol:
        User ``rtol`` and ``atol`` overrides, or ``None``.

    Returns
    -------
    str
        The phrase to append to the row's reason.
    """

    rows, pairs = table
    calls_a, calls_b = facts["calls"]
    target = _localized(pair, rows, pairs, calls_a)
    if target is None:
        row = rows[pairs.index(pair)]
        state = "inputs already differ" if row["inputs_agree"] is False else "inputs not compared"
        return f"op placement: none ({state}, and no enclosing call has agreeing inputs)"
    prefix = "" if target is pair else f"op placement inside {_label(target)} (inputs agree): "
    return prefix + _placed_in(target, table, facts, traces, tol)


def _label(pair: CallPair) -> str:
    """Side a's call label of a pair."""

    return pair.a.label if pair.a is not None else ""


def _localized(
    pair: CallPair, rows: list[dict[str, Any]], pairs: list[CallPair], calls_a: list[Any]
) -> CallPair | None:
    """``pair`` when its inputs agree, else its nearest enclosing differing call with agreeing inputs.

    When the first differing call's inputs already differ, the difference arose earlier,
    in code between matched boundaries; inside the nearest enclosing call whose inputs
    agree and whose output differs it is "localized inside that module" (AGREED 2.2
    item 3), so the walk runs there. The module row itself is unchanged.
    """

    by_label = {p.a.label: (p, r) for p, r in zip(pairs, rows, strict=True) if p.a is not None}
    parent = {call.label: call.parent for call in calls_a}
    label: str | None = _label(pair)
    for _ in range(len(parent) + 1):
        found = by_label.get(label or "")
        if found is not None:
            candidate, row = found
            paired = row["alignment"] in ("supported", "asserted") and candidate.b is not None
            if paired and row["outcome"] == "different" and row["inputs_agree"] is True:
                return candidate
        label = parent.get(label or "")
        if label is None:
            return None
    return None


def _placed_in(
    pair: CallPair,
    table: tuple[list[dict[str, Any]], list[CallPair]],
    facts: dict[str, Any],
    traces: tuple[Trace, Trace],
    tol: tuple[float | None, float | None],
) -> str:
    """Walk the paired call ``pair`` (inputs agree) and phrase where the difference sits."""

    if pair.a is None or pair.b is None:
        return "op placement: none (unpaired call)"
    rows, pairs = table
    calls_a, calls_b = facts["calls"]
    collapsed = collapsed_pairs(pair.a, pairs, rows, calls_a)
    seeded = seeds(pair.a, pair.b, collapsed)
    graphs = (
        side_graph(
            traces[0],
            pair.a.label,
            calls_a,
            [p.a.label for p in collapsed if p.a is not None],
            {a for a, _b in seeded},
        ),
        side_graph(
            traces[1],
            pair.b.label,
            calls_b,
            [p.b.label for p in collapsed if p.b is not None],
            {b for _a, b in seeded},
        ),
    )
    options = ValueOptions(
        bool(facts["cross_backend"]), tol[0], tol[1], input_scale(traces[0], pair.a.inputs)
    )
    walk = Walk(graphs[0], graphs[1], _judge(graphs, options))
    outputs = list(zip(pair.a.outputs, pair.b.outputs, strict=False))
    return _phrase(place(walk, seeded, outputs), walk)
