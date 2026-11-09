"""The op walk inside one paired module call: structural candidates and brackets.

Rule 2.1 holds throughout: values never choose a pair that can then differ. A pair
is made by structure (the other-side op consuming exactly the partners of a
reference op's parents) and kept when values agree; a value search (the
reconvergence walk along a single-consumer chain, or one agreeing candidate among
tied ones) can only pair ops that agree. A difference is read only at a structural
exit: the next reference op whose parents are paired or open, against the
other-side op consuming the partners and the open region (spelling family breaks a
tie, disclosed; both open regions must be fed by the same paired ops), or the
call's recorded outputs. The open ops behind a differing
exit are the placement (one op per side is an op-level placement).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ._xb_op_scope import SideGraph

#: Longest single-consumer chain the reconvergence walk follows.
CHAIN_LIMIT = 6

Judge = Callable[[str, str], bool]


@dataclass
class Walk:
    """Pairs and disclosures built while walking side a's ops."""

    a: SideGraph
    b: SideGraph
    judge: Judge
    pa: dict[str, str] = field(default_factory=dict)
    pb: dict[str, str] = field(default_factory=dict)
    by_value: list[tuple[str, str]] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    ambiguous: list[str] = field(default_factory=list)
    spelled: list[str] = field(default_factory=list)
    handled: set[str] = field(default_factory=set)

    def pair(self, label_a: str, label_b: str, *, by_value: bool = False) -> None:
        """Record a pair (``by_value`` pairs were found by agreement and only agree)."""

        self.pa[label_a], self.pb[label_b] = label_b, label_a
        if by_value:
            self.by_value.append((label_a, label_b))


@dataclass(frozen=True)
class Placement:
    """Where the first difference inside the call sits.

    Parameters
    ----------
    ops_a, ops_b:
        The open ops behind the first differing exit, in execution order.
    exit:
        What closed the group (``"module outputs"`` or the exit op pair).
    """

    ops_a: tuple[str, ...]
    ops_b: tuple[str, ...]
    exit: str


def _choose(walk: Walk, ref: str, candidates: list[str]) -> str | None:
    """One candidate for ``ref``: unique, else unique by spelling (disclosed), else ``None``.

    Spelling only ranks candidates structure already admitted: the same canonical
    spelling first, then the same family. Still tied, one candidate that agrees in
    value pairs by value (agreement only); otherwise ``ref`` is ambiguous.
    """

    if len(candidates) <= 1:
        return candidates[0] if candidates else None
    chosen, pool = _by_spelling(walk, ref, candidates)
    if chosen is not None:
        return chosen
    agreeing = [label for label in pool if walk.judge(ref, label)]
    if len(agreeing) == 1:
        walk.pair(ref, agreeing[0], by_value=True)
    else:
        walk.ambiguous.append(ref)
    return None


def _by_spelling(walk: Walk, ref: str, candidates: list[str]) -> tuple[str | None, list[str]]:
    """The one candidate sharing ``ref``'s canonical spelling, else its family; else the narrowed pool."""

    pool = candidates
    for attribute in ("canonical", "family"):
        wanted = getattr(walk.a.spelling[ref], attribute)
        same = [label for label in pool if getattr(walk.b.spelling[label], attribute) == wanted]
        if len(same) == 1:
            walk.spelled.append(f"{ref} -> {same[0]}")
            return same[0], pool
        pool = same or pool
    return None, pool


def _chain(graph: SideGraph, start: str, paired: dict[str, str]) -> list[str]:
    """The unpaired single-consumer chain after ``start`` (each op's only anchored parent)."""

    chain, current = [], start
    for _ in range(CHAIN_LIMIT):
        kids = [kid for kid in graph.children.get(current, ()) if kid not in paired]
        if len(kids) != 1 or graph.parents[kids[0]] != (current,):
            break
        chain.append(kids[0])
        current = kids[0]
    return chain


def _reconverge(walk: Walk, ref: str, cand: str) -> None:
    """Close a decomposition gap by agreement along either side's chain, if one exists."""

    for index, label in enumerate(_chain(walk.b, cand, walk.pb)):
        if walk.judge(ref, label):
            walk.gaps += [cand, *_chain(walk.b, cand, walk.pb)[:index]]
            walk.pair(ref, label, by_value=True)
            return
    chain_a = _chain(walk.a, ref, walk.pa)
    for index, label in enumerate(chain_a):
        if walk.judge(label, cand):
            walk.gaps += [ref, *chain_a[:index]]
            walk.handled.update([ref, *chain_a[:index]])
            walk.pair(label, cand, by_value=True)
            return


def _step_paired(walk: Walk, ref: str) -> None:
    """``ref``'s parents are all paired: take its structural candidate."""

    partners = {walk.pa[parent] for parent in walk.a.parents[ref]}
    candidates = [
        label
        for label in walk.b.walk
        if label not in walk.pb and set(walk.b.parents[label]) == partners
    ]
    cand = _choose(walk, ref, candidates)
    if cand is not None:
        _take(walk, ref, cand)


def _take(walk: Walk, ref: str, cand: str) -> None:
    """Pair a structural candidate that agrees; otherwise try to close a decomposition gap."""

    if walk.judge(ref, cand):
        walk.pair(ref, cand)
    else:
        _reconverge(walk, ref, cand)


def _exit_candidates(walk: Walk, ref: str) -> list[str]:
    """Side-b ops bracketing the same open computation as ``ref``.

    A candidate consumes the partners of ``ref``'s paired parents plus open ops, and
    the region behind its open parents is fed by the partners of the paired ops that
    feed ``ref``'s open region. That shared entry is the structural anchor; with no
    paired op on either boundary nothing is offered.
    """

    open_a = [p for p in walk.a.parents[ref] if p not in walk.pa]
    partners = {walk.pa[p] for p in walk.a.parents[ref] if p in walk.pa}
    entry = {walk.pa[p] for p in _entry(walk.a, open_a, walk.pa)}
    if not partners and not entry:
        return []
    found = []
    for label in walk.b.walk:
        if label in walk.pb:
            continue
        ups = walk.b.parents[label]
        open_b = [p for p in ups if p not in walk.pb]
        if open_b and set(ups) - set(open_b) == partners:
            if _entry(walk.b, open_b, walk.pb) == entry:
                found.append(label)
    return found


def _entry(graph: SideGraph, opens: list[str], paired: dict[str, str]) -> set[str]:
    """The paired ops feeding the open region behind ``opens``."""

    found: set[str] = set()
    for label in opens:
        for op in region(graph, label, paired):
            found.update(p for p in graph.parents[op] if p in paired)
    return found


def region(graph: SideGraph, start: str, paired: dict[str, str]) -> tuple[str, ...]:
    """The unpaired anchored ops behind ``start`` (itself included), in execution order."""

    seen: set[str] = set()
    stack = [start]
    while stack:
        label = stack.pop()
        if label in seen or label in paired or label not in graph.parents:
            continue
        seen.add(label)
        stack.extend(graph.parents[label])
    return tuple(sorted(seen, key=graph.order.__getitem__))


def _exit_pair(walk: Walk, ref: str, open_a: list[str]) -> tuple[str, list[str]] | None:
    """``ref``'s exit candidate and its open parents, when they bracket ``open_a`` one to one."""

    cand = _choose(walk, ref, _exit_candidates(walk, ref))
    if cand is None:
        return None
    open_b = [p for p in walk.b.parents[cand] if p not in walk.pb]
    same_spelling = walk.a.spelling[ref].canonical == walk.b.spelling[cand].canonical
    if len(open_a) != len(open_b) or (len(open_a) > 1 and not same_spelling):
        return None
    return cand, open_b


def _close_bracket(walk: Walk, open_a: list[str], open_b: list[str]) -> None:
    """Pair agreeing bracket outputs; the ops behind them become disclosed decomposition gaps."""

    for out_a, out_b in zip(open_a, open_b, strict=True):
        walk.gaps += [label for label in region(walk.a, out_a, walk.pa) if label != out_a]
        walk.gaps += [label for label in region(walk.b, out_b, walk.pb) if label != out_b]
        walk.pair(out_a, out_b)


def _step_exit(walk: Walk, ref: str, open_a: list[str]) -> Placement | None:
    """``ref`` consumes open ops: find its exit pair and read the bracket outputs."""

    found = _exit_pair(walk, ref, open_a)
    if found is None:
        return None
    cand, open_b = found
    for out_a, out_b in zip(open_a, open_b, strict=True):
        if not walk.judge(out_a, out_b):
            return Placement(
                region(walk.a, out_a, walk.pa), region(walk.b, out_b, walk.pb), f"{ref} / {cand}"
            )
    _close_bracket(walk, open_a, open_b)
    _take(walk, ref, cand)
    return None


def place(walk: Walk, seeds: list[tuple[str, str]], outputs: list[tuple[str, str]]) -> Placement:
    """Walk side a in execution order and return the first differing group.

    Parameters
    ----------
    walk:
        Fresh walk over both sides.
    seeds:
        Pairs established without values (recorded inputs, collapsed children).
    outputs:
        The call's recorded outputs, position by position (the last exit).

    Returns
    -------
    Placement
        The first group whose entry agrees and whose exit differs; an empty group
        with exit ``"none"`` when the recorded outputs agree at op level.
    """

    for label_a, label_b in seeds:
        walk.pair(label_a, label_b)
    for ref in walk.a.walk:
        if ref in walk.pa or ref in walk.handled:
            continue
        open_a = [p for p in walk.a.parents[ref] if p not in walk.pa]
        if not open_a:
            _step_paired(walk, ref)
            continue
        found = _step_exit(walk, ref, open_a)
        if found is not None:
            return found
    return _output_placement(walk, outputs)


def _output_placement(walk: Walk, outputs: list[tuple[str, str]]) -> Placement:
    """The last exit: the first recorded output pair that differs, else no placement."""

    for out_a, out_b in outputs:
        if walk.pa.get(out_a) != out_b and not walk.judge(out_a, out_b):
            return Placement(
                region(walk.a, out_a, walk.pa), region(walk.b, out_b, walk.pb), "module outputs"
            )
    return Placement((), (), "none")
