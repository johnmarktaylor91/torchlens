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


class _TwoPathsRewired(nn.Module):
    """Theorem C's second model: ``h`` reads the input instead of ``a``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the same functions in the same order with one edge moved."""

        a = torch.relu(x)
        h = x * 2
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


class _MaxSlots(nn.Module):
    """A multi-output function whose two outputs are both used."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Take the row maximum and its index."""

        values, indices = torch.max(x, dim=1)
        return values + indices.float()


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


def test_page_labels_match_the_capture(two_paths) -> None:
    """The labels the page names for the six-op model are the captured ones."""

    assert _Oracle(two_paths).order == (
        "input_1:1",
        "relu_1_1:1",
        "mul_1_2:1",
        "add_1_3:1",
        "add_2_4:1",
        "output_1:1",
    )


# ---------------------------------------------------------------------------
# Theorem A and Example 1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", range(7))
def test_theorem_a_every_family_and_mask_is_a_selection(two_paths, size) -> None:
    """Every (F, m) over the six sites is the value of the proof's sentence."""

    oracle = _Oracle(two_paths)
    labels = oracle.order
    for family in itertools.combinations(labels, size):
        for empties in itertools.product((False, True), repeat=size):
            sentence = None
            for label, empty in zip(family, empties, strict=True):
                if empty:
                    whole = two_paths[label].__selection__()
                    term = whole - whole
                else:
                    shape = tuple(oracle.ops[label].shape)
                    term = tl.units(label, [(0,) * len(shape)])
                sentence = term if sentence is None else sentence | term
            if sentence is None:
                sentence = _lift(two_paths, [labels[0]]) & _lift(two_paths, [labels[1]])
            resolved = two_paths[sentence]
            assert _family(resolved) == frozenset(family)
            expected = {label for label, empty in zip(family, empties, strict=True) if not empty}
            assert _sites(resolved) == expected
            assert all(row.num_selected in (0, 1) for row in resolved)


def test_example_1_every_subset_and_complement(two_paths) -> None:
    """All 64 site subsets lift to themselves; ``U - lift(S)`` has sites V minus S."""

    oracle = _Oracle(two_paths)
    universe = frozenset(oracle.order)
    for size in range(1, len(oracle.order) + 1):
        for subset in itertools.combinations(oracle.order, size):
            lifted = _lift(two_paths, subset)
            assert _sites(two_paths[lifted]) == frozenset(subset)
            complement = two_paths[tl.all_sites() - lifted]
            assert _family(complement) == universe
            assert _sites(complement) == universe - frozenset(subset)


# ---------------------------------------------------------------------------
# Closure rules (Section 1.4, A3, A5, A6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_name", ["two_paths", "loopy"])
def test_neighborhood_is_bounded_reach(fixture_name, request) -> None:
    """Every singleton seed, direction and hop count from 0 to |V| matches the oracle."""

    trace = request.getfixturevalue(fixture_name)
    oracle = _Oracle(trace)
    for label in oracle.order:
        for direction in _DIRECTIONS:
            for hops in range(len(oracle.order) + 1):
                region = trace[tl.neighborhood(label, hops=hops, direction=direction)]
                assert _sites(region) == _family(region)
                assert _sites(region) == oracle.reach([label], hops, direction)


@pytest.mark.parametrize("fixture_name", ["two_paths", "loopy"])
def test_between_is_any_source_to_any_sink(fixture_name, request) -> None:
    """``tl.between`` equals Desc(sources) & Anc(sinks) for every ordered pair."""

    trace = request.getfixturevalue(fixture_name)
    oracle = _Oracle(trace)
    for source, sink in itertools.product(oracle.order, repeat=2):
        region = trace[tl.between(source, sink)]
        assert _sites(region) == oracle.between([source], [sink])


def test_pass_qualified_unrolling(loopy) -> None:
    """Upstream closure from a second pass reaches the first; downstream does not."""

    oracle = _Oracle(loopy)
    relu_passes = [label for label in oracle.order if label.startswith("relu")]
    assert len(relu_passes) == 3, relu_passes
    first, second = relu_passes[0], relu_passes[1]
    hops = len(oracle.order)
    assert first in _sites(loopy[tl.neighborhood(second, hops=hops, direction="upstream")])
    assert first not in _sites(loopy[tl.neighborhood(second, hops=hops, direction="downstream")])


@pytest.mark.parametrize("hops", [-1, None, True, 1.0])
def test_hops_must_be_a_non_negative_int(two_paths, hops) -> None:
    """A5: ``hops`` rejects negatives, ``None``, booleans and floats."""

    with pytest.raises((ValueError, TypeError)):
        two_paths[tl.neighborhood("relu_1_1:1", hops=hops)]


def test_multi_output_functions_record_one_site_per_slot() -> None:
    """A1: ``torch.max(x, dim)`` gives one shaped record per output slot."""

    trace = _capture(_MaxSlots(), (2, 3), 2)
    try:
        slots = [op for op in trace._layer_list if op.label.startswith("max")]
        assert len(slots) == 2, [op.label for op in trace._layer_list]
        assert all(isinstance(op.shape, tuple) for op in slots)
    finally:
        trace.cleanup()


def test_where_reads_the_graph(two_paths) -> None:
    """A4: ``tl.where`` sees ``op.parents``, which is why it is outside the kernel."""

    resolved = two_paths[tl.where(lambda op: len(op.parents) >= 2)]
    assert _sites(resolved) == {"add_2_4:1"}


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


@pytest.mark.parametrize(("fixture_name", "seed"), [("two_paths", 0), ("loopy", 1)])
def test_theorem_b_translation_matches_the_query(fixture_name, seed, request) -> None:
    """``Sites(trace[tr(q)]) == [[q]]`` for random queries over all of Q."""

    trace = request.getfixturevalue(fixture_name)
    oracle = _Oracle(trace)
    atoms = {pattern: _sites(trace[tl.label_matches(pattern)]) for pattern in _ATOMS}
    rng = random.Random(seed)
    for _ in range(60):
        query = _random_query(rng, depth=3)
        resolved = trace[_translate(query, star_hops=len(oracle.order))]
        assert _sites(resolved) == _evaluate(query, atoms.__getitem__, oracle), query


# ---------------------------------------------------------------------------
# Theorem C: the edge-blindness witness
# ---------------------------------------------------------------------------


def test_theorem_c_kernel_without_closures_is_edge_blind() -> None:
    """M and M' share labels and K0 values; one-hop reach from ``a`` differs."""

    first = _capture(_TwoPaths(), (1, 3), 0)
    second = _capture(_TwoPathsRewired(), (1, 3), 0)
    try:
        assert _Oracle(first).order == _Oracle(second).order
        k0_sentences = [
            tl.label_matches("relu.*"),
            tl.label_matches("add.*") | tl.label_matches("mul.*"),
            tl.all_sites() - tl.label_matches("mul.*").__selection__(),
            ~tl.label_matches("add.*").__selection__(),
        ]
        for sentence in k0_sentences:
            assert first[sentence].labels == second[sentence].labels
            assert _sites(first[sentence]) == _sites(second[sentence])
        reach = tl.neighborhood("relu_1_1:1", hops=1, direction="downstream")
        assert _sites(first[reach]) == {"relu_1_1:1", "mul_1_2:1", "add_2_4:1"}
        assert _sites(second[reach]) == {"relu_1_1:1", "add_2_4:1"}
    finally:
        first.cleanup()
        second.cleanup()


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
