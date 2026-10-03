"""F11 perf gates: B1 fingerprint memoization + B2 de-quadratic enumeration.

Collapse memo items 6-7 (the collapse design memo, D6). These gates are
deterministic COUNT instruments, never wall-clock: B1 pins that one sweep
walks each member's wiring at most once and later sweeps on the same
revision walk zero, and B2 pins that legality-grammar checks scale linearly
(not quadratically) in sibling-run length.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.visualization import _collapse_runs, _collapse_signatures
from torchlens.visualization.auto_collapse import (
    _run_fold_members_uniform,
    analyze_collapse,
    resolve_repeat_folds,
)
from torchlens.visualization.collapse_optimizer import select_collapse_plan
from torchlens.visualization.collapse_plan import RenderContext


class _Block(nn.Module):
    """One linear+activation block; ``odd=True`` swaps ReLU for Tanh."""

    def __init__(self, width: int = 4, odd: bool = False) -> None:
        super().__init__()
        self.lin = nn.Linear(width, width)
        self.odd = odd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block."""

        y = self.lin(x)
        return torch.tanh(y) if self.odd else torch.relu(y)


class _Stack(nn.Module):
    """A serial chain of same-structure blocks (one optionally odd)."""

    def __init__(self, n: int, odd_at: int | None = None) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_Block(odd=(i == odd_at)) for i in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply every block in sequence."""

        for block in self.blocks:
            x = block(x)
        return x


def _trace(n: int, odd_at: int | None = None) -> tl.Trace:
    """Capture one ``_Stack(n)`` trace."""

    return tl.trace(_Stack(n, odd_at=odd_at), torch.randn(2, 4))


def _count_wiring_walks(monkeypatch, fn) -> int:
    """Run ``fn`` with ``_module_wiring_walk`` instrumented; return call count."""

    calls = {"n": 0}
    real_walk = _collapse_signatures._module_wiring_walk

    def counting_walk(module):  # noqa: ANN001, ANN202
        calls["n"] += 1
        return real_walk(module)

    monkeypatch.setattr(_collapse_signatures, "_module_wiring_walk", counting_walk)
    fn()
    return calls["n"]


@pytest.mark.smoke
def test_fingerprint_memo_walks_each_member_once_then_zero(monkeypatch) -> None:
    """One fold sweep walks each member at most once; a rerun walks zero.

    Collapse memo item 6 (B1): the wiring walk historically ran TWICE per
    member per uniformity check (digest + exterior bindings) and the
    signature was rebuilt for every candidate window. The fingerprint memo
    is keyed on the revision-fresh analysis object, so the second sweep on
    an unchanged trace must be pure cache hits.
    """

    trace = _trace(6)
    module_count = len(list(trace.modules))

    def sweep() -> None:
        resolve_repeat_folds(trace, None, RenderContext(), fold_repeats=True)

    first = _count_wiring_walks(monkeypatch, sweep)
    assert 0 < first <= module_count, (
        f"first sweep walked wiring {first} times for {module_count} modules; "
        "B1 requires at most one walk per member"
    )
    second = _count_wiring_walks(monkeypatch, sweep)
    assert second == 0, f"warm sweep must be pure cache hits, walked {second} times"


@pytest.mark.smoke
def test_cached_fingerprints_match_direct_computation() -> None:
    """Memoized signature/bindings equal the direct uncached functions.

    The B1 prototype's acceptance was byte-identical plans; the in-repo
    equivalence pin is exact value equality between the shared-walk cache
    and the historical two-walk computation, member by member.
    """

    trace = _trace(5, odd_at=2)
    analysis = analyze_collapse(trace)
    cache = _collapse_signatures.fingerprints_for(trace, analysis)
    for module in trace.modules:
        address = module.address
        assert cache.signature(address) == _collapse_signatures._module_structural_signature(module)
        assert cache.bindings(address) == _collapse_signatures._module_exterior_bindings(module)
    assert _collapse_signatures.fingerprints_for(trace, analysis) is cache


def test_memoized_odd_member_still_splits_runs() -> None:
    """The memoized path preserves the odd-member split contract exactly."""

    trace = _trace(7, odd_at=3)
    addresses = tuple(f"blocks.{i}" for i in range(7))
    assert not _run_fold_members_uniform(trace, addresses)
    assert _run_fold_members_uniform(trace, addresses[:3])
    assert _run_fold_members_uniform(trace, addresses[4:])


def test_legality_checks_scale_linearly() -> None:
    """Legality-grammar checks stay ~linear in sibling-run length (B2).

    The historical enumeration paid one full legality check per window
    width per start (quadratic; 24,165 calls measured on densenet201). The
    longest-first scan pays ~one per start. A 4x member growth must not
    grow the check count by more than ~6x (quadratic growth would be 16x).
    """

    small_trace = _trace(12)
    before_small = _collapse_runs.legality_check_count()
    select_collapse_plan(small_trace, RenderContext(), mode="max")
    small_calls = _collapse_runs.legality_check_count() - before_small

    large_trace = _trace(48)
    before_large = _collapse_runs.legality_check_count()
    select_collapse_plan(large_trace, RenderContext(), mode="max")
    large_calls = _collapse_runs.legality_check_count() - before_large

    assert small_calls > 0
    assert large_calls <= 6 * small_calls, (
        f"legality checks grew {small_calls} -> {large_calls} for 12 -> 48 "
        "members; the B2 de-quadratic contract bounds growth at ~linear"
    )


def test_longest_uniform_legal_run_matches_bruteforce() -> None:
    """The longest-first scan equals the historical grow-every-window pick.

    Reference semantics: the longest window ``w`` (representative-first,
    ``RUN_FOLD_MIN_LENGTH <= w``) that is BOTH legal under the v2 grammar
    and member-uniform. Checked on a real captured component, including an
    odd-member trace where the uniform bound truncates the candidate.
    """

    for n, odd_at in ((6, None), (7, 4), (5, 1)):
        trace = _trace(n, odd_at=odd_at)
        analysis = analyze_collapse(trace)
        graph = analysis.child_flow_graphs.get("blocks")
        if graph is None:
            graph = analysis.child_flow_graphs.get("self")
        assert graph is not None
        candidate = tuple(f"blocks.{i}" for i in range(n))
        fingerprints = _collapse_signatures.fingerprints_for(trace, analysis)
        fast = _collapse_runs.longest_uniform_legal_run(candidate, graph, fingerprints)
        best: tuple[str, ...] = ()
        for width in range(_collapse_runs.RUN_FOLD_MIN_LENGTH, len(candidate) + 1):
            run = candidate[:width]
            if _collapse_runs._run_fold_is_legal(run, graph) and _run_fold_members_uniform(
                trace, run
            ):
                best = run
        assert fast == best, f"n={n} odd_at={odd_at}: {fast} != reference {best}"
