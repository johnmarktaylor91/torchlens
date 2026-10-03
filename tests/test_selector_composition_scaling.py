"""r-b4 R27-6a: chained selector composition stays flat and stack-safe.

``&``/``|`` built one binary composite per application, so
``functools.reduce(operator.or_, [tl.func(n) for n in five_hundred])`` made a
500-deep binary tree and blew the interpreter stack at trace entry. Chained
same-operator composition now accumulates one flat n-ary child tuple (the
documented composite shape), and the diagnostic walkers are iterative.
"""

from __future__ import annotations

import functools
import operator

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.selectors import CompositeSelector
from torchlens.ir.selector_eval import flatten_and_conjuncts, walk_selector


def test_chained_or_composition_stays_flat() -> None:
    """reduce(or_, ...) over many selectors builds one flat composite."""

    names = [f"func_{i}" for i in range(500)]
    predicate = functools.reduce(operator.or_, [tl.func(name) for name in names])
    assert isinstance(predicate, CompositeSelector)
    assert predicate.operator == "or"
    assert len(predicate.selectors) == 500
    assert len(list(walk_selector(predicate))) == 501


def test_chained_and_composition_stays_flat() -> None:
    """Chained ``&`` accumulates a flat conjunction."""

    predicate = tl.func("relu") & tl.in_module("encoder") & tl.func("linear")
    assert isinstance(predicate, CompositeSelector)
    assert predicate.operator == "and"
    assert len(predicate.selectors) == 3
    assert flatten_and_conjuncts((predicate,)) == tuple(predicate.selectors)


def test_mixed_operator_composition_still_nests() -> None:
    """Mixed ``&``/``|`` keeps its semantic nesting."""

    predicate = (tl.func("relu") | tl.func("gelu")) & tl.in_module("encoder")
    assert predicate.operator == "and"
    assert len(predicate.selectors) == 2
    inner = predicate.selectors[0]
    assert isinstance(inner, CompositeSelector) and inner.operator == "or"


def test_walk_selector_handles_deep_nested_trees() -> None:
    """A hand-built deeply NESTED tree (deserialized shape) walks iteratively."""

    node = tl.func("relu")
    for _ in range(2000):
        node = CompositeSelector("or", (node, tl.func("gelu")))
    assert len(list(walk_selector(node))) == 4001


def test_large_or_chain_traces_end_to_end() -> None:
    """The 500-way union predicate works as a live save predicate."""

    names = [f"name_{i}" for i in range(498)] + ["relu", "linear"]
    predicate = functools.reduce(operator.or_, [tl.func(name) for name in names])
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    log = tl.trace(model, torch.ones(1, 4), save=predicate)
    assert log["relu_1_2"].out is not None


def test_followed_by_composition_rules_still_enforced() -> None:
    """The followed_by OR-composition refusal is unchanged."""

    from torchlens.intervention.errors import SelectorCompositionError

    with pytest.raises(SelectorCompositionError):
        _ = tl.func("conv2d") | tl.followed_by(tl.func("relu"))
    combined = tl.func("conv2d") & tl.followed_by(tl.func("relu"))
    assert isinstance(combined, CompositeSelector)
