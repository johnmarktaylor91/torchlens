"""Segment-run prefix search tests for the v2 collapse optimizer.

Moved verbatim from ``tests/test_collapse_optimizer.py`` (at its size cap).
The linearized ``_longest_legal_segment_prefix`` search is checked against the
naive forward prefix scan it replaced (a differential oracle kept verbatim),
and its probe/recount work is pinned flat as one role component grows.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest
import torch

import torchlens.visualization.collapse_optimizer as collapse_optimizer
from tests.test_collapse_optimizer import (
    SegmentPrefixTail,
    SequentialEncoderHead,
    UnevenReusedSiblings,
    UniformStack,
    UniqueWideFanModel,
    _trace,
)
from torchlens.visualization.collapse_optimizer import (
    _child_segment_covered_ops,
    _segment_is_legal,
    collapse_schedule,
    select_collapse_plan,
)
from torchlens.visualization.collapse_plan import RenderContext


def _reference_longest_legal_segment_prefix(
    members: Sequence[str],
    graph: Any,
    analysis: Any,
    hidden_counts: Mapping[str, int],
    total_ops: int,
    *,
    dominance_limit: float,
    vis_mode: str = "unrolled",
) -> tuple[str, ...] | None:
    """Naive forward prefix scan the optimized segment-run search must match.

    This is the pre-optimization algorithm, kept verbatim as the differential
    oracle: rescan the whole prefix for landmark members, probe chain-interval
    legality for every prefix, recount hidden units from scratch, and keep the
    last prefix that passed all three gates.
    """

    def hidden_units(addresses: tuple[str, ...]) -> int:
        covered_ops = _child_segment_covered_ops(analysis, addresses)
        if covered_ops:
            if vis_mode == "rolled":
                return len({str(label).rsplit(":", 1)[0] for label in covered_ops})
            return len(covered_ops)
        return sum(
            hidden_counts.get(address, analysis.signals[address].hidden_ops)
            for address in addresses
        )

    best: tuple[str, ...] | None = None
    for end in range(2, len(members) + 1):
        candidate = tuple(members[:end])
        if any(analysis.signals[address].landmark_edges >= 2 for address in candidate):
            continue
        if not _segment_is_legal(candidate, graph):
            continue
        hidden = hidden_units(candidate)
        if total_ops > 0 and hidden / total_ops > dominance_limit:
            continue
        best = candidate
    return best


class _SegmentPrefixProbe:
    """Differential + call-count probe around the segment-run prefix search."""

    def __init__(self) -> None:
        """Start an empty probe."""

        self.calls = 0
        self.candidates = 0
        self.legality_probes = 0
        self.hidden_recounts = 0

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Patch the optimizer to cross-check every prefix search."""

        module = "torchlens.visualization.collapse_optimizer"
        real_prefix = collapse_optimizer._longest_legal_segment_prefix
        real_legal = collapse_optimizer._segment_is_legal
        real_covered = collapse_optimizer._child_segment_covered_ops

        def counting_segment_is_legal(addresses: tuple[str, ...], graph: Any) -> bool:
            self.legality_probes += 1
            return real_legal(addresses, graph)

        def counting_covered_ops(analysis: Any, addresses: tuple[str, ...]) -> tuple[str, ...]:
            self.hidden_recounts += 1
            return real_covered(analysis, addresses)

        def checked_prefix(
            members: Sequence[str],
            graph: Any,
            analysis: Any,
            hidden_counts: Mapping[str, int],
            total_ops: int,
            **kwargs: Any,
        ) -> tuple[str, ...] | None:
            self.calls += 1
            self.candidates += max(len(members) - 1, 0)
            fast = real_prefix(members, graph, analysis, hidden_counts, total_ops, **kwargs)
            slow = _reference_longest_legal_segment_prefix(
                members, graph, analysis, hidden_counts, total_ops, **kwargs
            )
            assert fast == slow, (
                f"segment-run prefix diverged for {len(members)} members: {fast!r} != {slow!r}"
            )
            return fast

        monkeypatch.setattr(f"{module}._segment_is_legal", counting_segment_is_legal)
        monkeypatch.setattr(f"{module}._child_segment_covered_ops", counting_covered_ops)
        monkeypatch.setattr(f"{module}._longest_legal_segment_prefix", checked_prefix)


def _exercise_collapse_surfaces(
    model: torch.nn.Module,
    x: torch.Tensor,
    modes: tuple[str | float, ...] = ("auto", "max", 0.0, 0.25, 0.5, 0.75, 1.0),
) -> None:
    """Drive every collapse-planning surface that runs a segment-run search."""

    trace = _trace(model, x)
    try:
        for mode in modes:
            for vis_mode in ("unrolled", "rolled"):
                select_collapse_plan(trace, RenderContext(vis_mode=vis_mode), mode=mode)
        collapse_schedule(trace, RenderContext())
    finally:
        trace.cleanup()


@pytest.mark.heavy
@pytest.mark.parametrize(
    ("factory", "shape"),
    [
        (lambda: UniformStack(depth=24), (2, 8)),
        (lambda: UniqueWideFanModel(), (1, 4, 8, 8)),
        (lambda: SequentialEncoderHead(), (2, 4)),
        (lambda: SegmentPrefixTail(), (2, 8)),
        (lambda: UnevenReusedSiblings(), (2, 4)),
    ],
)
def test_segment_run_prefix_matches_naive_prefix_scan(
    monkeypatch: pytest.MonkeyPatch,
    factory: Callable[[], torch.nn.Module],
    shape: tuple[int, ...],
) -> None:
    """The linearized segment-run search returns the naive scan's exact answer."""

    probe = _SegmentPrefixProbe()
    probe.install(monkeypatch)
    _exercise_collapse_surfaces(factory(), torch.randn(*shape))

    assert probe.calls > 0, "probe never fired; the differential check was vacuous"


class FlatLinearChain(torch.nn.Module):
    """Long chain of sibling linears -- the worst case for segment-run search."""

    def __init__(self, depth: int = 64, width: int = 8) -> None:
        """Initialize the chain."""

        super().__init__()
        self.layers = torch.nn.Sequential(*[torch.nn.Linear(width, width) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the chain."""

        return self.layers(x)


@pytest.mark.heavy
@pytest.mark.serial
def test_segment_run_prefix_work_does_not_grow_with_component_size() -> None:
    """One long role component costs a bounded search, not one probe per prefix.

    A flat sibling chain puts every member in one role component, so the naive
    forward scan spent one chain-interval probe and one hidden-unit recount on
    every prefix -- quadratic in the chain length. The linearized search settles
    the landmark and dominance gates in one pass and probes down from the
    longest surviving candidate, so its work must stay flat as the chain grows.
    """

    counts: dict[int, tuple[int, int, int]] = {}
    for depth in (64, 128, 256):
        probe = _SegmentPrefixProbe()
        with pytest.MonkeyPatch.context() as monkeypatch:
            probe.install(monkeypatch)
            _exercise_collapse_surfaces(
                FlatLinearChain(depth=depth), torch.randn(1, 8), modes=("max",)
            )
        counts[depth] = (probe.candidates, probe.legality_probes, probe.hidden_recounts)

    for depth, (candidates, probes, recounts) in counts.items():
        assert probes <= candidates, f"depth {depth}: {probes} probes for {candidates} candidates"
        assert recounts <= candidates, f"depth {depth}: {recounts} recounts for {candidates}"
    assert counts[256][0] > 2 * counts[64][0], "the chain component never grew"
    assert counts[256][1] <= 2 * counts[64][1], (
        f"legality probes grew with component size: {counts[64][1]} -> {counts[256][1]}"
    )
    assert counts[256][2] <= 2 * counts[64][2] + 8, (
        f"hidden-unit recounts grew with component size: {counts[64][2]} -> {counts[256][2]}"
    )
    assert counts[256][1] * 10 < counts[256][0], (
        "probes did not fall far below the naive one-per-prefix count: "
        f"{counts[256][1]} probes for {counts[256][0]} candidate prefixes"
    )
