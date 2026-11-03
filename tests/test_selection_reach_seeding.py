"""Closure producers seed only from element-carrying sites; ``tl.all_sites``.

A selection is a touched-site family plus a per-site element mask, and
``x - y`` (or ``~x``) keeps every site of ``x`` touched even where it
removes every element. ``tl.neighborhood``, ``tl.between`` (both the
producer and the ``trace.between`` slice) and ``trace.subgraph`` seed only
from sites whose mask is nonempty, so a closure of ``U - X`` is the closure
of the sites outside ``X``. These tests pin that rule against an
independent reachability oracle built from ``op.parents``, check that
plain selections are unchanged, and check the completeness statement it
restores (every site query with complement or subtraction under a reach is
expressed by the kernel translation ``tl.neighborhood(U - tr(q), ...)``),
brute-forced over every site subset of a six-op model.

``tl.all_sites()`` is the named universal selector: the same result as
``tl.label_matches(".*")`` on every fixture, composing like any selector.
"""

from __future__ import annotations

import itertools
import runpy
from collections.abc import Iterable
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._selection.selectors import RegexSelector

_DIRECTIONS = ("upstream", "downstream", "both")


class _TwoPath(nn.Module):
    """The six-site model: two paths from ``a`` to ``b`` (a->h->d->b and a->b)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run relu, scale, shift, and the merging add."""

        a = torch.relu(x)
        h = a * 2
        d = h + 1
        return d + a


class _Branchy(nn.Module):
    """Two parallel linear branches merging into a head."""

    def __init__(self) -> None:
        super().__init__()
        self.lin1 = nn.Linear(4, 4)
        self.lin2 = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run both branches and the head."""

        return self.head(torch.relu(self.lin1(x)) + torch.tanh(self.lin2(x)))


class _Loopy(nn.Module):
    """One linear+relu applied three times (multi-pass layers)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the shared block three times."""

        for _ in range(3):
            x = torch.relu(self.lin(x))
        return x


@pytest.fixture(scope="module")
def two_path():
    """The six-site two-path capture."""

    torch.manual_seed(0)
    trace = tl.trace(_TwoPath(), torch.randn(1, 3))
    try:
        yield trace
    finally:
        trace.cleanup()


@pytest.fixture(scope="module")
def branchy():
    """A branchy capture with parameters."""

    torch.manual_seed(1)
    trace = tl.trace(_Branchy(), torch.randn(2, 4))
    try:
        yield trace
    finally:
        trace.cleanup()


@pytest.fixture(scope="module")
def loopy():
    """A multi-pass capture."""

    torch.manual_seed(2)
    trace = tl.trace(_Loopy(), torch.randn(2, 4))
    try:
        yield trace
    finally:
        trace.cleanup()


# ---------------------------------------------------------------------------
# Independent oracle: the dataflow relation read straight from op.parents.
# ---------------------------------------------------------------------------


class _Oracle:
    """Plain adjacency over one trace, built without the selection-graph code."""

    def __init__(self, trace) -> None:
        ops = [op for op in trace._layer_list if isinstance(getattr(op, "label", None), str)]
        self.order = tuple(op.label for op in ops)
        by_reference: dict[str, str] = {}
        for op in ops:
            by_reference[op.label] = op.label
            by_reference.setdefault(op.layer_label, op.label)
        self.parents: dict[str, set[str]] = {label: set() for label in self.order}
        self.children: dict[str, set[str]] = {label: set() for label in self.order}
        for op in ops:
            for reference in op.parents:
                parent = by_reference[reference]  # a dropped edge would fail loudly here
                self.parents[op.label].add(parent)
                self.children[parent].add(op.label)
        self.ops = {op.label: op for op in ops}

    def reach(self, seeds: Iterable[str], hops: int, direction: str) -> frozenset[str]:
        """Return Reach_d^{<=hops}(seeds), seeds included."""

        reached = set(seeds)
        for _ in range(hops):
            step: set[str] = set()
            for label in reached:
                if direction in ("upstream", "both"):
                    step |= self.parents[label]
                if direction in ("downstream", "both"):
                    step |= self.children[label]
            reached |= step
        return frozenset(reached)

    def between(self, sources: Iterable[str], sinks: Iterable[str]) -> frozenset[str]:
        """Return Desc(sources) & Anc(sinks)."""

        size = len(self.order)
        return self.reach(sources, size, "downstream") & self.reach(sinks, size, "upstream")


def _lift(trace, labels: Iterable[str]):
    """Return the whole-site union of the named ops, or ``None`` for no ops."""

    oracle_ops = {op.label: op for op in trace._layer_list}
    lifted = None
    for label in labels:
        term = oracle_ops[label].__selection__()
        lifted = term if lifted is None else lifted | term
    return lifted


def _six(oracle: _Oracle) -> tuple[str, ...]:
    """Return the two-path model's labels: input, a, h, d, b, output."""

    assert len(oracle.order) == 6, oracle.order
    inp, a, h, d, b, out = oracle.order
    assert oracle.parents[b] == {d, a} and oracle.parents[h] == {a}, oracle.parents
    return inp, a, h, d, b, out


def _sites(selection) -> frozenset[str]:
    """Return the labels of rows carrying at least one selected element."""

    return frozenset(
        label
        for label, entry in zip(selection.labels, selection, strict=True)
        if entry.num_selected
    )


def _family(selection) -> frozenset[str]:
    """Return the labels of every touched row (empty masks included)."""

    return frozenset(selection.labels)


# ---------------------------------------------------------------------------
# tl.all_sites
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_name", ["two_path", "branchy", "loopy"])
def test_all_sites_equals_label_matches_dot_star(fixture_name, request) -> None:
    """Every fixture: the same rows, masks and digest as label_matches('.*')."""

    trace = request.getfixturevalue(fixture_name)
    universal = trace[tl.all_sites()]
    spelled = trace[tl.label_matches(".*")]
    assert universal.labels == spelled.labels
    assert universal.digest == spelled.digest
    assert set(universal.labels) == set(_Oracle(trace).order)
    assert all(entry.num_selected == entry.mask.numel() for entry in universal)


@pytest.mark.smoke
def test_all_sites_spelling_and_payload() -> None:
    """A named regex selector: same kind and payload, its own repr."""

    selector = tl.all_sites()
    assert repr(selector) == "tl.all_sites()"
    assert isinstance(selector, RegexSelector)
    assert (selector.selector_kind, selector.selector_value) == ("regex", ".*")
    spec = selector.to_target_spec()
    assert (spec.selector_kind, spec.selector_value) == ("regex", ".*")
    assert "all_sites" in tl.__all__
    assert tl.all_sites() == tl.all_sites()


@pytest.mark.parametrize("fixture_name", ["two_path", "branchy", "loopy"])
def test_all_sites_composes_like_label_matches(fixture_name, request) -> None:
    """&, |, - and ~ give the same result as the label_matches spelling."""

    trace = request.getfixturevalue(fixture_name)
    other = tl.func("relu")
    pairs = [
        (tl.all_sites() & other, tl.label_matches(".*") & other),
        (tl.all_sites() | other, tl.label_matches(".*") | other),
        (tl.all_sites() - other, tl.label_matches(".*") - other),
        (other & tl.all_sites(), other & tl.label_matches(".*")),
    ]
    for named, spelled in pairs:
        assert trace[named].digest == trace[spelled].digest
    assert trace[tl.all_sites() & other].labels == trace[other].labels
    lifted = tl.all_sites().__selection__()
    lifted_spelled = tl.label_matches(".*").__selection__()
    relu = other.__selection__()
    assert trace[lifted - relu].digest == trace[lifted_spelled - relu].digest
    assert trace[~lifted].digest == trace[~lifted_spelled].digest
    assert trace[~lifted].is_empty
    assert len(trace[~lifted]) == len(trace[lifted])


def test_all_sites_example_runs() -> None:
    """The shipped example executes."""

    example = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "intervention"
        / "18_complement_under_reach.py"
    )
    runpy.run_path(str(example), run_name="__main__")


# ---------------------------------------------------------------------------
# G9: element-empty sites seed nothing.
# ---------------------------------------------------------------------------


def test_complement_under_reach_matches_the_query(two_path) -> None:
    """PROOF.md example 3: one hop downstream of the sites not downstream of a."""

    oracle = _Oracle(two_path)
    inp, a, _h, _d, _b, _out = _six(oracle)
    downstream_of_a = tl.neighborhood(oracle.ops[a], hops=len(oracle.order), direction="downstream")
    outside = tl.all_sites() - downstream_of_a
    resolved_outside = two_path[outside]
    # The operand keeps every site touched; only the input carries elements.
    assert _family(resolved_outside) == frozenset(oracle.order)
    assert _sites(resolved_outside) == {inp}
    region = two_path[tl.neighborhood(outside, hops=1, direction="downstream")]
    assert _family(region) == {inp, a}
    assert _sites(region) == _family(region)


@pytest.mark.parametrize("direction", _DIRECTIONS)
def test_neighborhood_of_a_difference_seeds_the_remaining_sites(two_path, direction) -> None:
    """x - y where y empties some of x's sites: only x's other sites seed."""

    oracle = _Oracle(two_path)
    _inp, a, h, _d, b, _out = _six(oracle)
    x = tl.neighborhood(oracle.ops[a], hops=1)  # {input, a, h, b}
    y = _lift(two_path, [h, b])
    difference = two_path[x - y]
    remaining = _sites(difference)
    assert _family(difference) - remaining == {h, b}
    for hops in range(len(oracle.order)):
        region = two_path[tl.neighborhood(x - y, hops=hops, direction=direction)]
        assert _family(region) == oracle.reach(remaining, hops, direction)
        assert _sites(region) == _family(region)


def test_between_endpoints_drop_element_empty_sites(two_path) -> None:
    """Sources and sinks built by subtraction contribute only nonempty sites."""

    oracle = _Oracle(two_path)
    _inp, a, h, d, b, _out = _six(oracle)
    sources = _lift(two_path, [a, h]) - _lift(two_path, [a])  # touches a, carries h
    region = two_path[tl.between(sources, oracle.ops[b])]
    assert _family(region) == oracle.between({h}, {b}) == {h, d, b}
    sinks = _lift(two_path, [d, b]) - _lift(two_path, [b])  # touches b, carries d
    region = two_path[tl.between(oracle.ops[a], sinks)]
    assert _family(region) == oracle.between({a}, {d}) == {a, h, d}
    view = two_path.between(sources, sinks)
    assert set(view.labels) == oracle.between({h}, {d}) == {h, d}


def test_subgraph_of_a_difference_presents_the_remaining_sites(two_path) -> None:
    """trace.subgraph takes the sites with a nonempty mask as its members."""

    oracle = _Oracle(two_path)
    h = _six(oracle)[2]
    outside = tl.all_sites() - _lift(two_path, [h])
    view = two_path.subgraph(outside)
    expected = tuple(label for label in oracle.order if label != h)
    assert view.labels == expected


def test_fully_emptied_regions_are_empty(two_path) -> None:
    """x - x seeds nothing: empty neighborhood, empty between, empty slice."""

    oracle = _Oracle(two_path)
    _inp, a, _h, _d, b, _out = _six(oracle)
    nothing = tl.all_sites() - tl.all_sites().__selection__()
    assert len(two_path[nothing]) == len(two_path[tl.all_sites()])
    assert len(two_path[tl.neighborhood(nothing, hops=3)]) == 0
    assert len(two_path[tl.between(nothing, oracle.ops[b])]) == 0
    assert len(two_path[tl.between(oracle.ops[a], nothing)]) == 0
    assert two_path.subgraph(nothing).empty


@pytest.mark.parametrize("fixture_name", ["two_path", "branchy", "loopy"])
def test_plain_selections_are_unchanged(fixture_name, request) -> None:
    """Whole and partial masks still seed their whole node, as before."""

    trace = request.getfixturevalue(fixture_name)
    oracle = _Oracle(trace)
    for label in oracle.order:
        op = oracle.ops[label]
        partial = tl.units(label, [(0,) * len(op.shape)])
        for direction in _DIRECTIONS:
            whole_region = trace[tl.neighborhood(op, hops=1, direction=direction)]
            partial_region = trace[tl.neighborhood(partial, hops=1, direction=direction)]
            expected = oracle.reach({label}, 1, direction)
            assert _family(whole_region) == expected
            assert _family(partial_region) == expected
        assert trace.subgraph(partial).labels == (label,)


# ---------------------------------------------------------------------------
# The completeness statement G9 restores (PROOF.md, Theorem B over all of Q).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("direction", "hops"), list(itertools.product(_DIRECTIONS, (1, 5))))
def test_reach_of_a_complement_is_expressible(two_path, direction, hops) -> None:
    """For every site subset S: Sites(neighborhood(U - lift(S))) = Reach(V \\ S).

    Before G9 the closure seeded from all of ``V`` whenever ``S`` was
    nonempty, so this held only for ``S = {}``.
    """

    oracle = _Oracle(two_path)
    assert len(oracle.order) == 6
    universe = tl.all_sites().__selection__()
    for size in range(len(oracle.order) + 1):
        for subset in itertools.combinations(oracle.order, size):
            lifted = _lift(two_path, subset)
            operand = universe if lifted is None else universe - lifted
            complement = set(oracle.order) - set(subset)
            region = two_path[tl.neighborhood(operand, hops=hops, direction=direction)]
            assert _sites(region) == oracle.reach(complement, hops, direction), subset


def test_between_of_complements_is_expressible(two_path) -> None:
    """For every site subset S: between(U - lift(S), U - lift(S)) is exact."""

    oracle = _Oracle(two_path)
    universe = tl.all_sites().__selection__()
    for size in range(len(oracle.order) + 1):
        for subset in itertools.combinations(oracle.order, size):
            lifted = _lift(two_path, subset)
            operand = universe if lifted is None else universe - lifted
            complement = set(oracle.order) - set(subset)
            region = two_path[tl.between(operand, operand)]
            assert _sites(region) == oracle.between(complement, complement), subset
