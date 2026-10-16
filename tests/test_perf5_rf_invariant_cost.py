"""The always-on receptive-field metadata invariants cost about n^2 on a chain, not n^3.

``check_geometric_metadata_invariants`` resolves one box per solved operation and
direction. Each box walks its path to the endpoint one hop at a time, and every
hop used to enumerate the whole widening index window (then sort and recompress
it), while each query also re-derived the graph index and every rule result on its
path. The window and clip maps now take a closed form when the index set is one
arithmetic progression, and the invariant's queries share graph facts and rule
results, so a chain of ``n`` convolutions does ``O(n)`` rule calls and enumerates
no index window.

The closed forms must return exactly the canonical set the enumeration returns;
the property tests below compare them with the enumeration on many shapes.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from contextlib import contextmanager
from math import gcd
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.receptive_field import _engine, _query
from torchlens.receptive_field._engine_forward import solve_projective
from torchlens.receptive_field._errors import ReceptiveFieldError
from torchlens.receptive_field._types import ReceptiveFieldDirection
from torchlens.receptive_field._validation import (
    _box_for_descriptor,
    _indeterminate_unit,
    check_geometric_metadata_invariants,
)

_IndexSet = _query._IndexSet
_Progression = _query._Progression

DEPTH = 40


class _Chain(nn.Module):
    """``depth`` stride-1 kernel-3 padding-1 one-channel smoothing convolutions."""

    def __init__(self, depth: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Conv1d(1, 1, 3, padding=1, bias=False) for _ in range(depth))
        with torch.no_grad():
            for layer in self.layers:
                layer.weight.copy_(torch.tensor([[[0.25, 0.5, 0.25]]]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _Mixed(nn.Module):
    """Strided, dilated and padded windows plus a merge, so the enumeration path also runs."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Conv2d(1, 2, 3, padding=2, dilation=2)
        self.b = nn.Conv2d(1, 2, 3, stride=2, padding=1)
        self.c = nn.Conv2d(2, 2, 5, stride=3, padding=1)
        self.d = nn.ConvTranspose2d(2, 2, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.c(torch.relu(self.a(x)))
        z = nn.functional.max_pool2d(self.b(x), 2)
        return self.d(y[..., : z.shape[-2], : z.shape[-1]] + z)


def _enumerated_window(
    output_set: Any, *, kernel: int, stride: int, padding: int, dilation: int
) -> Any:
    """The window map by enumeration, as it was before the closed form."""

    values = (
        stride * output_index - padding + dilation * tap
        for output_index in output_set.values()
        for tap in range(kernel)
    )
    return _IndexSet.from_values(values, exact=output_set.exact)


def _enumerated_transposed(
    output_set: Any, *, kernel: int, stride: int, padding: int, dilation: int, extent: int
) -> Any:
    """The transposed-convolution feasible-tap map by enumeration."""

    feasible = []
    divisor = gcd(stride, dilation)
    for output_index in output_set.values():
        shifted = output_index + padding
        if shifted % divisor:
            continue
        for tap in range(kernel):
            numerator = shifted - dilation * tap
            if numerator % stride:
                continue
            input_index = numerator // stride
            if 0 <= input_index < extent:
                feasible.append(input_index)
    return _IndexSet.from_values(feasible, exact=output_set.exact)


def _random_sets(rng: random.Random, count: int) -> Iterator[Any]:
    """Single progressions (the closed-form inputs) and a few multi-run sets."""

    for _ in range(count):
        if rng.random() < 0.85:
            start = rng.randint(-12, 40)
            step = rng.randint(1, 5)
            length = rng.randint(1, 20)
            values: list[int] = list(range(start, start + step * length, step))
        else:
            values = rng.sample(range(-10, 60), rng.randint(2, 12))
        yield _IndexSet.from_values(values, exact=rng.random() < 0.8)


def test_window_map_closed_form_equals_enumeration() -> None:
    """Every window image equals the enumerated canonical set, exactness included."""

    rng = random.Random(0)
    for output_set in _random_sets(rng, 3000):
        params = {
            "kernel": rng.randint(1, 7),
            "stride": rng.randint(1, 4),
            "padding": rng.randint(0, 4),
            "dilation": rng.randint(1, 4),
        }
        assert _query._standard_window_index_set(output_set, **params) == _enumerated_window(
            output_set, **params
        ), (output_set, params)


def test_transposed_map_closed_form_equals_enumeration() -> None:
    """Every transposed-convolution image equals the enumerated feasible-tap set."""

    rng = random.Random(1)
    for output_set in _random_sets(rng, 3000):
        params = {
            "kernel": rng.randint(1, 7),
            "stride": rng.choice((1, 1, 1, 2, 3)),
            "padding": rng.randint(0, 4),
            "dilation": rng.randint(1, 4),
        }
        extent = rng.randint(0, 50)
        mapped, exact = _query.map_transposed_convolution_index_set(
            0, output_set, input_extent=extent, **params
        )
        assert exact is True
        assert mapped == _enumerated_transposed(output_set, extent=extent, **params), (
            output_set,
            params,
            extent,
        )


def test_clip_closed_form_equals_enumeration() -> None:
    """Clipping one progression equals filtering and recompressing its values."""

    rng = random.Random(2)
    for index_set in _random_sets(rng, 3000):
        extent = rng.randint(-2, 60)
        expected = _IndexSet._from_sorted_unique(
            [value for value in index_set.values() if 0 <= value < extent], exact=index_set.exact
        )
        if index_set.progressions and (
            index_set.progressions[0].start >= 0 and index_set.progressions[-1].stop < extent
        ):
            expected = index_set
        assert index_set.clipped(extent) == expected, (index_set, extent)


@contextmanager
def _counters() -> Iterator[dict[str, int]]:
    """Count rule evaluations and enumerated index elements while the block runs."""

    counts = {"rule_calls": 0, "index_elements": 0}
    rule_uncached = _engine._rule_result_uncached
    values = _IndexSet.values
    from_sorted = _IndexSet.__dict__["_from_sorted_unique"].__func__

    def counted_rule(op: Any) -> Any:
        counts["rule_calls"] += 1
        return rule_uncached(op)

    def counted_values(self: Any) -> tuple[int, ...]:
        result = values(self)
        counts["index_elements"] += len(result)
        return result

    def counted_from_sorted(cls: Any, ordered: Any, *, exact: bool = True) -> Any:
        counts["index_elements"] += len(ordered)
        return from_sorted(cls, ordered, exact=exact)

    patch = pytest.MonkeyPatch()
    patch.setattr(_engine, "_rule_result_uncached", counted_rule)
    patch.setattr(_IndexSet, "values", counted_values)
    patch.setattr(_IndexSet, "_from_sorted_unique", classmethod(counted_from_sorted))
    try:
        yield counts
    finally:
        patch.undo()


def test_chain_invariants_do_linear_rule_calls_and_enumerate_no_window() -> None:
    """On a 40-deep chain the invariant evaluates each rule a few times, not once per path."""

    torch.manual_seed(0)
    trace = tl.trace(_Chain(DEPTH), torch.randn(1, 1, 2 * DEPTH + 101))
    _engine.solve(trace)
    solve_projective(trace, tuple(trace.output_ops))
    n_ops = len(trace.layer_list)
    with _counters() as counts:
        assert check_geometric_metadata_invariants(trace)
    # Base: about n^2 rule calls (one per hop of every query) and about n^3 / 3
    # enumerated window elements per direction.
    assert counts["rule_calls"] <= 4 * n_ops, counts
    assert counts["index_elements"] <= 4 * n_ops, counts


def _all_boxes(trace: tl.Trace) -> list[tuple[str, str, str, Any]]:
    """Every invariant box of both directions, each query run on its own."""

    rows = []
    ops_by_label = {op.label: op for op in trace.layer_list}
    solutions = [(ReceptiveFieldDirection.RECEPTIVE, _engine.solve(trace))]
    if trace.output_ops:
        solutions.append(
            (ReceptiveFieldDirection.PROJECTIVE, solve_projective(trace, tuple(trace.output_ops)))
        )
    for direction, solution in solutions:
        for (label, role), descriptor in solution.descriptors.items():
            if solution.states[(label, role)].taint is not None or descriptor.axes is None:
                continue
            owner = ops_by_label[label]
            try:
                box: Any = _box_for_descriptor(
                    solution,
                    owner,
                    descriptor,
                    _indeterminate_unit(owner, 0),
                    direction=direction,
                    endpoint=None,
                )
            except ReceptiveFieldError as error:  # the invariant skips these; record the type
                box = type(error).__name__
            rows.append((direction.value, label, role, box))
    return rows


@pytest.mark.parametrize("model_name", ["chain", "mixed"])
def test_shared_scope_boxes_equal_per_query_boxes(model_name: str) -> None:
    """Sharing graph facts and rule results across queries leaves every box unchanged."""

    torch.manual_seed(0)
    if model_name == "chain":
        trace = tl.trace(_Chain(12), torch.randn(1, 1, 37))
    else:
        trace = tl.trace(_Mixed(), torch.randn(1, 1, 23, 19))
    alone = _all_boxes(trace)
    with _query._shared_query_scope():
        shared = _all_boxes(trace)
        shared_again = _all_boxes(trace)
    assert alone == shared == shared_again
    assert any(not isinstance(box, str) for *_, box in alone)
    assert not _query._SHARED_QUERY_STATE.graphs and not _query._SHARED_QUERY_STATE.reach
    assert check_geometric_metadata_invariants(trace)
