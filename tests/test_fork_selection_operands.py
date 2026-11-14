"""Resolved selections work wherever a fork accepts a selection.

A selection resolved on a trace carries over to that trace's forks by site
identity. ``do()`` already bound one; these tests pin the other doors:

1. A lazy producer whose operand is a resolved selection, applied in a fork:
   ``t.fork().do(tl.top_fraction(cone, 0.07, by="abs"), tl.zero_ablate())``
   (value producers' ``within``, ``random_like`` operands, graph-producer
   endpoints). A selection from outside the fork lineage still refuses
   ``selection_trace_mismatch``.
2. ``trace[selection]`` returns a resolved selection (rebound onto a fork of
   its trace), so callers need not know which compositions resolve eagerly.
3. A chained read on a temporary fork refuses ``trace_reference_collected``
   (records hold their trace weakly by design) with a message naming the
   chain and the remedy.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.errors import RecordBindingError, SelectionError

#: Two layer4 units whose fields are traced back into layer2 and layer3.
UNITS = [(0, 1, 3, 3), (0, 2, 3, 3)]
REGIONS = ["layer2", "layer3"]


class _ThreeStageCNN(nn.Module):
    """Small CNN with ``layer2``/``layer3``/``layer4`` stages (distinct widths)."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(4, 6, 3, padding=1), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(6, 8, 3, padding=1), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(8, 5, 3, padding=1), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the four stages in order."""

        return self.layer4(self.layer3(self.layer2(self.stem(x))))


def _trace(seed: int) -> tl.Trace:
    """Capture the model on one seeded 8x8 input with default options."""

    torch.manual_seed(0)
    model = _ThreeStageCNN().eval()
    torch.manual_seed(seed)
    return tl.trace(model, torch.randn(1, 3, 8, 8))


@pytest.fixture
def trace():
    """One default capture of the three-stage CNN."""

    captured = _trace(1)
    yield captured
    captured.cleanup()


@pytest.fixture
def other():
    """An unrelated capture of the same model on another input."""

    captured = _trace(2)
    yield captured
    captured.cleanup()


def _cone(trace: tl.Trace) -> tl.Selection:
    """The receptive field of ``UNITS`` in layer2 and layer3, resolved on ``trace``."""

    return trace["layer4"].receptive_field.of(UNITS, within=REGIONS)


def test_value_producer_over_parent_selection_applies_on_fork(trace):
    """The card's one-liner runs and zeroes exactly the parent-ranked elements."""

    cone = _cone(trace)
    parent_out = {name: trace[name].out.clone() for name in (*REGIONS, "layer4")}
    picked = trace[tl.top_fraction(cone, 0.07, by="abs")]

    fork = trace.fork().do(tl.top_fraction(cone, 0.07, by="abs"), tl.zero_ablate())

    masks = {tuple(row.shape): row.mask for row in picked}
    for region in REGIONS:
        mask = masks[tuple(parent_out[region].shape)]
        assert mask.any(), f"{region}: the producer picked nothing"
        assert parent_out[region][mask].ne(0).any(), f"{region}: nothing to ablate"
        assert fork[region].out[mask].eq(0).all(), f"{region}: picked elements not zeroed"
        assert torch.equal(fork[region].out[~mask], parent_out[region][~mask])
    assert not torch.equal(fork["layer4"].out, parent_out["layer4"])
    for name, before in parent_out.items():
        assert torch.equal(trace[name].out, before), f"{name}: the parent changed"
    fork.cleanup()


def test_embedded_parent_operands_resolve_on_fork_as_on_parent(trace):
    """Value, random and graph producers over a parent selection agree across the lineage."""

    cone = _cone(trace)
    seeds = trace[tl.func("conv2d")]
    fork = trace.fork()
    grandchild = fork.fork()
    producers = [
        tl.top_fraction(cone, 0.1, by="abs"),
        tl.top_k(cone, 3),
        tl.threshold(cone, above=0.0),
        tl.random_like(like=tl.top_k(cone, 3), within=cone, seed=0),
        tl.random_like(like=trace[tl.top_k(cone, 3)], within=cone, seed=0),
        tl.neighborhood(seeds, 1),
        tl.between(trace[tl.in_module("layer2")], trace[tl.in_module("layer3")]),
    ]
    try:
        for producer in producers:
            expected = trace[producer]
            for target in (fork, grandchild):
                resolved = target[producer]
                assert resolved._trace is target
                assert resolved.digest == expected.digest, repr(producer)
    finally:
        grandchild.cleanup()
        fork.cleanup()


def test_embedded_operand_from_outside_the_lineage_refuses(trace, other):
    """An unrelated capture, a sibling fork and the reverse direction keep refusing."""

    foreign = other[tl.func("relu")]
    first, second = trace.fork(), trace.fork()
    on_first = first[tl.func("relu")]
    cases = [
        (trace, foreign),
        (trace.fork(), foreign),
        (second, on_first),
        (trace, on_first),
    ]
    try:
        for target, operand in cases:
            for producer in (
                tl.top_fraction(operand, 0.1, by="abs"),
                tl.random_like(like=operand, within=operand, seed=0),
                tl.neighborhood(operand, 1),
            ):
                with pytest.raises(SelectionError) as excinfo:
                    target[producer]
                assert excinfo.value.fields["code"] == "selection_trace_mismatch"
            with pytest.raises(SelectionError) as excinfo:
                target.do(tl.top_fraction(operand, 0.1, by="abs"), tl.zero_ablate())
            assert excinfo.value.fields["code"] == "selection_trace_mismatch"
    finally:
        for fork in (cases[1][0], first, second):
            fork.cleanup()


def test_trace_getitem_returns_resolved_selection(trace, other):
    """``trace[selection]`` returns it; a fork rebinds it; a stranger refuses."""

    cone = _cone(trace)
    assert trace[cone] is cone
    # Eager and lazy compositions both pass through the same door.
    stage = trace[tl.in_module("layer2")]
    eager = stage - cone
    assert isinstance(eager, tl.Selection)
    assert trace[eager] is eager

    fork = trace.fork()
    try:
        rebound = fork[cone]
        assert rebound is not cone and rebound._trace is fork
        assert rebound.digest == cone.digest
        assert all("carried over to fork" in row.provenance.source for row in rebound)
        assert all(row._trace_ref() is trace for row in cone), "the parent's rows moved"
        assert fork[rebound] is rebound
    finally:
        fork.cleanup()

    with pytest.raises(SelectionError) as excinfo:
        other[cone]
    assert excinfo.value.fields["code"] == "selection_trace_mismatch"
    assert "align_to" in str(excinfo.value)


def test_chained_read_on_temporary_fork_names_the_chain(trace):
    """A record read off a temporary fork refuses typed, naming the chain and the remedy."""

    cone = _cone(trace)
    op_label = trace[tl.func("relu")].labels[-1]
    for key in ("layer2", op_label):
        with pytest.raises(RecordBindingError) as excinfo:
            trace.fork().do(cone, tl.zero_ablate())[key].out  # noqa: B018 - the read refuses
        error = excinfo.value
        assert error.fields["code"] == "trace_reference_collected"
        assert "holds its Trace weakly" in str(error)
        assert "t.fork().do(...)[key].out" in str(error)
        assert "f = t.fork().do(...)" in error.fields["remedy"]

    # The remedy works: hold the fork while reading its records.
    fork = trace.fork().do(cone, tl.zero_ablate())
    assert fork["layer2"].out.shape == trace["layer2"].out.shape
    assert fork[op_label].out.shape == trace[op_label].out.shape
    fork.cleanup()
