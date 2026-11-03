"""Hardening tests for param-free ops at loop boundaries (rounds 22-25).

Locks the round-25 TOPOLOGICAL REDESIGN of bare (unanchored, parameter-free)
op grouping. A bare op's layer membership is no longer guessed from context-set
heuristics (the r22 disjoint-context veto, r23 any-inequality veto plus
entry adoption, r24 global carrier census); it is DERIVED from its parents'
already-solved grouping: every bare op carries a signature -- the multiset of
its parents' pass-free site colors (parameterized call identity, anchored key,
bare-op class, external label) -- and same-key ops partition into the coarsest
classes consistent with that evidence. Lockstep iterations of one source-code
site produce EQUAL signatures by construction; sites whose parameterized
direct parents differ in LAYER split by construction. The one honest same-site
signature difference, the loop ENTRY (pass 1 reads pre-loop state, later
passes read the feedback wire), is re-admitted through a guarded carry-slot
exemption; loop-invariant-fed repeats (in-body factory ops, recomputes of a
pre-loop value) sequence through their consumers instead.

The correctness bar throughout is LOCKSTEP PASS-COHERENCE: sibling bare ops
executing once per iteration of one loop body must partition identically, and
a bare op's per-layer pass counts must mirror its own loop's parameterized
flanks -- never straddle a boundary (a 3-pass and a 2-pass layer cannot share
a 5-pass neighbor).

Two DELIBERATE policy changes from the pre-r25 heuristic ladder, both locked
below with rationale:

* Alternating-site straight-line residual bodies (``h = h + attn(h); h = h +
  mlp(h)`` per iteration) now split PER SITE ([3, 3]), not per loop ([6]): the
  attn-add and the mlp-add are different source-code sites with different
  parameterized direct parents, and [6] beside attn [3] / mlp [3] violated the
  very neighbor-coherence standard the boundary work defends.
* The n1=1 "peeled entry" chain (``for 1: ... ; for 2: ...`` over distinct
  sites) no longer fuses the binary bridge op into the second loop ([3] ->
  [1, 2]): the peeled add's parameterized flank (enc, one pass) positively
  contradicts the second loop's flank (dec, two passes), so [1, 2] is the
  flank-lockstep answer. Flankless unary ops (the tanh head) carry no such
  contradicting evidence and keep the adoption shape ([3]), mirroring the
  accepted singleton-prelude adoption family.
"""

from collections import OrderedDict

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from support.structural_grouper import legacy_structural_grouper  # noqa: F401

from torchlens import trace as trace_fn
from torchlens.validation.invariants import check_metadata_invariants

# This suite drives the structural grouper (grouper_v1), which serves captures
# without call-site evidence; torch captures group by call site
# (tests/test_grouping_call_sites.py).
pytestmark = pytest.mark.usefixtures("legacy_structural_grouper")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _layer_passes(traced) -> "OrderedDict[str, int]":
    """Return ordered mapping of layer_label -> number of grouped passes."""
    layers: OrderedDict[str, list] = OrderedDict()
    for op in traced:
        layers.setdefault(op.layer_label, []).append(op)
    return OrderedDict((label, len(ops)) for label, ops in layers.items())


def _pass_counts(traced, stem: str) -> list[int]:
    """Return the per-layer pass counts of every layer whose label starts with ``stem``.

    Ordered by first appearance in the trace, so a chained 3-iteration loop
    followed by a 2-iteration loop reads ``[3, 2]``.
    """
    return [count for label, count in _layer_passes(traced).items() if label.startswith(f"{stem}_")]


def _assert_coherent_two_loop_partition(traced, param_stem: str, interior_stem: str) -> None:
    """Assert the chained 3+2 loop pair splits into pass-coherent neighborhoods.

    The parameterized flanks must be [3, 2], and the param-free interior must
    mirror its own loop -- two layers of [3, 2] passes, NOT one 5-pass layer
    bridging the boundary.
    """
    assert _pass_counts(traced, param_stem) == [3, 2], _layer_passes(traced)
    assert _pass_counts(traced, interior_stem) == [3, 2], _layer_passes(traced)
    assert check_metadata_invariants(traced)


# ---------------------------------------------------------------------------
# Round 22: chained loops whose parameterized flanks share parameter barcodes.
# The tied/shared-kernel interiors split because their parameterized direct
# parents differ in site identity -- now read directly off the signature.
# ---------------------------------------------------------------------------


class _SharedKernelTwoLoops(nn.Module):
    """ONE kernel applied under different non-tensor args in two chained loops.

    Both calls are shape-preserving on the input (3x3 kernel: padding=1, and
    padding=2 with dilation=2), so the tanh interiors of both loops share one
    structural equivalence key -- the merge is blocked only by the loop
    boundary, exactly the axis under test.
    """

    def __init__(self) -> None:
        super().__init__()
        self.kernel = nn.Parameter(torch.randn(1, 1, 3, 3))

    def forward(self, x):
        for _ in range(3):
            x = torch.tanh(F.conv2d(x, self.kernel, padding=1))
        for _ in range(2):
            x = torch.tanh(F.conv2d(x, self.kernel, padding=2, dilation=2))
        return x


def test_shared_kernel_divergent_args_interiors_respect_loop_boundary() -> None:
    """Same kernel under different args in chained loops: interiors split 3/2."""
    torch.manual_seed(0)
    traced = trace_fn(_SharedKernelTwoLoops(), torch.randn(1, 1, 8, 8))
    _assert_coherent_two_loop_partition(traced, "conv2d", "tanh")


# ---------------------------------------------------------------------------
# Round 23: OVERLAPPING-BUT-UNEQUAL parent structure (a shared module reused
# in both loops next to site-specific halves). Signatures {shared, enc} and
# {shared, dec} differ, so the bridges split with no set-relation ladder.
# ---------------------------------------------------------------------------


def test_multi_output_tied_cells_with_shared_reuse_bridge() -> None:
    """LSTMCell slots stay honest AND the combining bridges split 3/2.

    A reused shared LSTMCell fires in both chained loops (two 5-pass slot
    layers), fully weight-tied DISTINCT enc/dec cells fire 3/2 (honest
    [5, 5, 3, 3, 2, 2] slots), and the ``tanh(h_shared + h_site)`` bridges
    follow the site half: add/tanh 3/2, never 5-pass layers across the
    boundary. The h/c zero inits each feed ONE cell call and stay single-pass
    values (consumer-side sequencing demands disjoint consumers).
    """

    class TiedLSTMOverlap(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.LSTMCell(4, 4)
            self.enc = nn.LSTMCell(4, 4)
            self.dec = nn.LSTMCell(4, 4)
            for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                setattr(self.dec, name, getattr(self.enc, name))

        def forward(self, x):
            hs = torch.zeros(1, 4)
            cs = torch.zeros(1, 4)
            he = torch.zeros(1, 4)
            ce = torch.zeros(1, 4)
            for _ in range(3):
                hs, cs = self.shared(x, (hs, cs))
                he, ce = self.enc(x, (he, ce))
                x = torch.tanh(hs + he)
            hd = torch.zeros(1, 4)
            cd = torch.zeros(1, 4)
            for _ in range(2):
                hs, cs = self.shared(x, (hs, cs))
                hd, cd = self.dec(x, (hd, cd))
                x = torch.tanh(hs + hd)
            return x

    torch.manual_seed(0)
    traced = trace_fn(TiedLSTMOverlap(), torch.randn(1, 4))
    assert _pass_counts(traced, "lstmcell") == [5, 5, 3, 3, 2, 2], _layer_passes(traced)
    assert _pass_counts(traced, "add") == [3, 2], _layer_passes(traced)
    assert _pass_counts(traced, "tanh") == [3, 2], _layer_passes(traced)
    assert all(count == 1 for count in _pass_counts(traced, "zeros")), _layer_passes(traced)
    assert check_metadata_invariants(traced)


# ---------------------------------------------------------------------------
# Loop-entry saturation: pass 1 reads pre-loop state, so its signature differs
# from every later pass in exactly the carry slot. The guarded exemption must
# keep honest entries merged without reopening any boundary.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Round 24: 2-iteration param-free-headed loops feeding a second consumer.
# Signature classes are global per equivalence key, so a saturated interior
# fragmented across iso rounds can never be misread as a dangling entry.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Round 25 seal findings F1-F4: context-REPLACEMENT param-headed loops, the
# post-LN r23 regression, nested inner/outer sites, and same-key preludes.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Loop-invariant-fed repeats: parent signatures carry no sequencing evidence,
# so repetition is certified through consumers -- without conflating sibling
# one-shot values or parallel streams.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Anti-conflation guards: repetition without loop evidence must stay split.
# ---------------------------------------------------------------------------


def test_interleaved_parallel_streams_partition_per_stream() -> None:
    """Interleaved capture order does not conflate parallel streams.

    ``for 2: a = tanh(cell(a)); b = tanh(cell(b))`` interleaves the two
    streams' calls in capture order; connectivity components keep the tanh
    partition per stream ([2, 2]), so raw adjacency in time is never evidence.
    """

    class InterleavedParallelLoops(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cell = nn.Linear(4, 4)

        def forward(self, x):
            a = x
            b = torch.relu(x)
            for _ in range(2):
                a = torch.tanh(self.cell(a))
                b = torch.tanh(self.cell(b))
            return a.sum() + b.sum()

    torch.manual_seed(0)
    traced = trace_fn(InterleavedParallelLoops(), torch.randn(2, 4))
    assert _pass_counts(traced, "linear") == [4], _layer_passes(traced)
    assert sorted(_pass_counts(traced, "tanh")) == [2, 2], _layer_passes(traced)
    assert check_metadata_invariants(traced)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
