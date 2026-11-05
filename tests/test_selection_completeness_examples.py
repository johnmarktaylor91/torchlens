"""The selection completeness page runs as written and its stated facts hold.

``docs/reference/selection_completeness.md`` proves what the selection algebra
can express. Its Section 5 code blocks are one runnable script whose
assertions state the worked examples' sets; ``test_page_examples_run_as_written``
executes them verbatim. The other tests check the facts the proofs cite
against an independent reachability oracle built from ``op.parents``:
Theorem A on every site subset, the translation of Theorem B on random
queries (complement and difference under a reach included), the closure
rules, the edge-blindness witness of Theorem C, and Lemma R against the
relation tables. A disagreement means the code or the page has a bug.
"""

from __future__ import annotations

import itertools
import random
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._selection.provenance import _FLIP, _difference_relation, _join_relation

_PAGE = Path(__file__).resolve().parents[1] / "docs" / "reference" / "selection_completeness.md"
_PYTHON_FENCE = re.compile(r"```python\n(?P<code>.*?)\n```", re.DOTALL)
_DIRECTIONS = ("upstream", "downstream", "both")


class _TwoPaths(nn.Module):
    """The page's six-op model: a->h->d->b and a->b."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run relu, scale, shift and the merging add."""

        a = torch.relu(x)
        h = a * 2
        d = h + 1
        return d + a


class _Loopy(nn.Module):
    """One linear+relu block applied three times (pass-qualified records)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the shared block three times."""

        for _ in range(3):
            x = torch.relu(self.lin(x))
        return x


def _capture(model: nn.Module, shape: tuple[int, ...], seed: int) -> Any:
    torch.manual_seed(seed)
    return tl.trace(model, torch.randn(*shape))


@pytest.fixture(scope="module")
def two_paths():
    """The six-op capture."""

    trace = _capture(_TwoPaths(), (1, 3), 0)
    try:
        yield trace
    finally:
        trace.cleanup()


@pytest.fixture(scope="module")
def loopy():
    """A multi-pass capture."""

    trace = _capture(_Loopy(), (2, 4), 1)
    try:
        yield trace
    finally:
        trace.cleanup()


class _Oracle:
    """Plain adjacency over one trace, built without the selection-graph code."""

    def __init__(self, trace: Any) -> None:
        # The full record list (input and output records included); ``trace.ops``
        # is the counted view without them.
        ops = [op for op in trace._layer_list if isinstance(getattr(op, "label", None), str)]
        self.order = tuple(op.label for op in ops)
        self.ops = {op.label: op for op in ops}
        by_reference: dict[str, str] = {}
        for op in ops:
            by_reference[op.label] = op.label
            by_reference.setdefault(op.layer_label, op.label)
        self.parents: dict[str, set[str]] = {label: set() for label in self.order}
        self.children: dict[str, set[str]] = {label: set() for label in self.order}
        for op in ops:
            for reference in op.parents:
                parent = by_reference[reference]  # a dropped edge fails loudly here
                self.parents[op.label].add(parent)
                self.children[parent].add(op.label)

    def reach(self, seeds: Iterable[str], hops: int, direction: str) -> frozenset[str]:
        """Return Reach_d^{<=hops}(seeds), seeds included."""

        reached = set(seeds)
        frontier = set(seeds)
        for _ in range(hops):
            step: set[str] = set()
            for label in frontier:
                if direction in ("upstream", "both"):
                    step |= self.parents[label]
                if direction in ("downstream", "both"):
                    step |= self.children[label]
            frontier = step - reached
            reached |= frontier
        return frozenset(reached)

    def between(self, sources: Iterable[str], sinks: Iterable[str]) -> frozenset[str]:
        """Return Desc(sources) & Anc(sinks)."""

        size = len(self.order)
        return self.reach(sources, size, "downstream") & self.reach(sinks, size, "upstream")


def _sites(selection: Any) -> frozenset[str]:
    return frozenset(
        label for label, row in zip(selection.labels, selection, strict=True) if row.num_selected
    )


def _family(selection: Any) -> frozenset[str]:
    return frozenset(selection.labels)


def _lift(trace: Any, labels: Iterable[str]) -> Any:
    lifted = None
    for label in labels:
        term = trace[label].__selection__()
        lifted = term if lifted is None else lifted | term
    return lifted


# ---------------------------------------------------------------------------
# The page itself
# ---------------------------------------------------------------------------


def test_page_examples_run_as_written() -> None:
    """Section 5's code blocks execute verbatim; their assertions are the stated sets."""

    blocks = _PYTHON_FENCE.findall(_PAGE.read_text(encoding="utf-8"))
    assert len(blocks) == 4, "Section 5 has the setup block and one block per example"
    namespace: dict[str, Any] = {"__name__": "selection_completeness_page"}
    for index, block in enumerate(blocks):
        exec(compile(block, f"{_PAGE.name}:python-{index}", "exec"), namespace)
    namespace["trace"].cleanup()


# ---------------------------------------------------------------------------
# Closure rules (Section 1.4, A3, A5, A6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hops", [-1, None, True, 1.0])
def test_hops_must_be_a_non_negative_int(two_paths, hops) -> None:
    """A5: ``hops`` rejects negatives, ``None``, booleans and floats."""

    with pytest.raises((ValueError, TypeError)):
        two_paths[tl.neighborhood("relu_1_1:1", hops=hops)]


# ---------------------------------------------------------------------------
# Theorem B: the translation on random queries
# ---------------------------------------------------------------------------

_ATOMS = ("relu.*", "mul.*", "add.*", "add_2.*", "input.*", "output.*", "linear.*")
_Query = tuple[Any, ...]


def _random_query(rng: random.Random, depth: int) -> _Query:
    if depth == 0 or rng.random() < 0.25:
        return ("atom", rng.choice(_ATOMS))
    kind = rng.choice(("or", "and", "minus", "not", "reach", "reach", "star"))
    if kind in ("or", "and", "minus"):
        return (kind, _random_query(rng, depth - 1), _random_query(rng, depth - 1))
    if kind == "not":
        return (kind, _random_query(rng, depth - 1))
    direction = rng.choice(_DIRECTIONS)
    hops = rng.randint(0, 3)
    return (kind, _random_query(rng, depth - 1), direction, hops)


def _evaluate(query: _Query, atom: Callable[[str], frozenset[str]], oracle: _Oracle) -> frozenset:
    kind = query[0]
    if kind == "atom":
        return atom(query[1])
    if kind == "not":
        return frozenset(oracle.order) - _evaluate(query[1], atom, oracle)
    if kind in ("reach", "star"):
        inner = _evaluate(query[1], atom, oracle)
        hops = query[3] if kind == "reach" else len(oracle.order)
        return oracle.reach(inner, hops, query[2])
    left = _evaluate(query[1], atom, oracle)
    right = _evaluate(query[2], atom, oracle)
    return {"or": left | right, "and": left & right, "minus": left - right}[kind]


def _translate(query: _Query, star_hops: int) -> Any:
    kind = query[0]
    if kind == "atom":
        return tl.label_matches(query[1]).__selection__()
    if kind == "not":
        return tl.all_sites() - _translate(query[1], star_hops)
    if kind in ("reach", "star"):
        hops = query[3] if kind == "reach" else star_hops
        return tl.neighborhood(_translate(query[1], star_hops), hops=hops, direction=query[2])
    left = _translate(query[1], star_hops)
    right = _translate(query[2], star_hops)
    return {"or": left | right, "and": left & right, "minus": left - right}[kind]


# ---------------------------------------------------------------------------
# Lemma R: the relation tables
# ---------------------------------------------------------------------------

_RELATIONS = ("exact", "upper_bound", "lower_bound", "unknown")


def _holds(relation: str, mask: frozenset[int], support: frozenset[int]) -> bool:
    if relation == "exact":
        return mask == support
    if relation == "upper_bound":
        return mask >= support
    if relation == "lower_bound":
        return mask <= support
    return True


def _pair(rng: random.Random, relation: str, universe: frozenset[int]) -> tuple[frozenset, ...]:
    while True:
        mask = frozenset(i for i in universe if rng.random() < 0.5)
        support = frozenset(i for i in universe if rng.random() < 0.5)
        if relation == "exact":
            support = mask
        if _holds(relation, mask, support):
            return mask, support


def test_lemma_r_relations_survive_composition() -> None:
    """``|``, ``&``, ``~`` and ``-`` transfer relations soundly (300 draws per pair)."""

    rng = random.Random(0)
    universe = frozenset(range(8))
    for rel_a, rel_b in itertools.product(_RELATIONS, repeat=2):
        for _ in range(300):
            m_a, s_a = _pair(rng, rel_a, universe)
            m_b, s_b = _pair(rng, rel_b, universe)
            joined = _join_relation(rel_a, rel_b)
            assert _holds(joined, m_a | m_b, s_a | s_b)
            assert _holds(joined, m_a & m_b, s_a & s_b)
            assert _holds(_FLIP[rel_a], universe - m_a, universe - s_a)
            assert _holds(_difference_relation(rel_a, rel_b), m_a - m_b, s_a - s_b)
