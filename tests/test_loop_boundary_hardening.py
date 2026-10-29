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

from tests.support.structural_grouper import legacy_structural_grouper  # noqa: F401
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


class _TiedTwoLoops(nn.Module):
    """Two DISTINCT weight-tied modules driving chained loops with tanh interiors."""

    def __init__(self, tied: bool = True) -> None:
        super().__init__()
        self.enc = nn.Linear(4, 4)
        self.dec = nn.Linear(4, 4)
        if tied:
            self.dec.weight = self.enc.weight
            self.dec.bias = self.enc.bias

    def forward(self, x):
        for _ in range(3):
            x = torch.tanh(self.enc(x))
        for _ in range(2):
            x = torch.tanh(self.dec(x))
        return x


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


class _SameModuleTwoLoops(nn.Module):
    """Genuine reuse control: ONE module drives BOTH chained loops (ALBERT)."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.Linear(4, 4)

    def forward(self, x):
        for _ in range(3):
            x = torch.tanh(self.cell(x))
        for _ in range(2):
            x = torch.tanh(self.cell(x))
        return x


class _DistinctTanhModuleLoops(nn.Module):
    """Anchored-interior control: tied linears with DISTINCT nn.Tanh interiors."""

    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Linear(4, 4)
        self.dec = nn.Linear(4, 4)
        self.dec.weight = self.enc.weight
        self.dec.bias = self.enc.bias
        self.tanh_a = nn.Tanh()
        self.tanh_b = nn.Tanh()

    def forward(self, x):
        for _ in range(3):
            x = self.tanh_a(self.enc(x))
        for _ in range(2):
            x = self.tanh_b(self.dec(x))
        return x


def test_tied_distinct_modules_interiors_respect_loop_boundary() -> None:
    """Tied enc/dec chained 3+2 loops: tanh interiors split 3/2, never 5.

    The tanh signatures are {enc-site} for three calls and {dec-site} for two:
    distinct parameterized parent layers split the interiors by construction,
    with no barcode-context indirection to defeat.
    """
    torch.manual_seed(0)
    traced = trace_fn(_TiedTwoLoops(tied=True), torch.randn(1, 4))
    _assert_coherent_two_loop_partition(traced, "linear", "tanh")


def test_shared_kernel_divergent_args_interiors_respect_loop_boundary() -> None:
    """Same kernel under different args in chained loops: interiors split 3/2."""
    torch.manual_seed(0)
    traced = trace_fn(_SharedKernelTwoLoops(), torch.randn(1, 1, 8, 8))
    _assert_coherent_two_loop_partition(traced, "conv2d", "tanh")


def test_untied_chained_loops_control() -> None:
    """Untied enc/dec chained loops already split 3/2 and must stay split."""
    torch.manual_seed(0)
    traced = trace_fn(_TiedTwoLoops(tied=False), torch.randn(1, 4))
    _assert_coherent_two_loop_partition(traced, "linear", "tanh")


def test_distinct_tanh_modules_control() -> None:
    """Distinct anchored nn.Tanh interiors split 3/2 through module identity."""
    torch.manual_seed(0)
    traced = trace_fn(_DistinctTanhModuleLoops(), torch.randn(1, 4))
    _assert_coherent_two_loop_partition(traced, "linear", "tanh")


def test_same_module_reuse_still_merges_across_chained_loops() -> None:
    """ALBERT policy: ONE module reused in both loops keeps merging 5/5.

    A genuinely reused module keeps ONE site color across both loops, so the
    tanh signatures stay equal and the interiors form one coherent 5-pass
    layer beside the 5-pass linear. The redesign must not over-split
    legitimate recurrence.
    """
    torch.manual_seed(0)
    traced = trace_fn(_SameModuleTwoLoops(), torch.randn(1, 4))
    assert _pass_counts(traced, "linear") == [5], _layer_passes(traced)
    assert _pass_counts(traced, "tanh") == [5], _layer_passes(traced)
    assert check_metadata_invariants(traced)


def test_tied_sites_within_one_loop_body_keep_per_site_interiors() -> None:
    """Tied enc/dec called inside ONE 3-iteration body: two 3-pass tanh layers."""

    class TiedSitesOneBody(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enc = nn.Linear(4, 4)
            self.dec = nn.Linear(4, 4)
            self.dec.weight = self.enc.weight
            self.dec.bias = self.enc.bias

        def forward(self, x):
            for _ in range(3):
                x = torch.tanh(self.enc(x))
                x = torch.tanh(self.dec(x))
            return x

    torch.manual_seed(0)
    traced = trace_fn(TiedSitesOneBody(), torch.randn(1, 4))
    assert _pass_counts(traced, "linear") == [3, 3], _layer_passes(traced)
    assert _pass_counts(traced, "tanh") == [3, 3], _layer_passes(traced)
    assert check_metadata_invariants(traced)


# ---------------------------------------------------------------------------
# Round 23: OVERLAPPING-BUT-UNEQUAL parent structure (a shared module reused
# in both loops next to site-specific halves). Signatures {shared, enc} and
# {shared, dec} differ, so the bridges split with no set-relation ladder.
# ---------------------------------------------------------------------------


class _PartialOverlapLoops(nn.Module):
    """Shared linear reused in BOTH loops plus per-loop site-specific linears."""

    def __init__(self, chained: bool = True) -> None:
        super().__init__()
        self.shared = nn.Linear(4, 4)
        self.enc = nn.Linear(4, 4)
        self.dec = nn.Linear(4, 4)
        self.chained = chained

    def forward(self, x):
        if self.chained:
            for _ in range(3):
                x = torch.tanh(self.shared(x) + self.enc(x))
            for _ in range(2):
                x = torch.tanh(self.shared(x) + self.dec(x))
        else:
            for _ in range(3):
                x = torch.tanh(self.shared(x) + self.enc(x))
                x = torch.tanh(self.shared(x) + self.dec(x))
        return x


def test_overlapping_unequal_contexts_respect_loop_boundary() -> None:
    """Chained 3+2 loops with a shared reused flank: bridges split 3/2.

    The shared linear stays ONE 5-pass layer (locked ALBERT genuine-reuse
    policy) and enc/dec stay 3/2, so the param-free ``add`` and ``tanh``
    bridges must follow their differing parent topology (``shared+enc`` vs
    ``shared+dec``) and split 3/2.
    """
    torch.manual_seed(0)
    traced = trace_fn(_PartialOverlapLoops(chained=True), torch.randn(1, 4))
    assert _pass_counts(traced, "linear") == [5, 3, 2], _layer_passes(traced)
    assert _pass_counts(traced, "add") == [3, 2], _layer_passes(traced)
    assert _pass_counts(traced, "tanh") == [3, 2], _layer_passes(traced)
    assert check_metadata_invariants(traced)


def test_overlapping_unequal_sites_within_one_loop_body() -> None:
    """Both overlapping-unequal sites inside ONE 3-iteration body: 3/3."""
    torch.manual_seed(0)
    traced = trace_fn(_PartialOverlapLoops(chained=False), torch.randn(1, 4))
    assert _pass_counts(traced, "linear") == [6, 3, 3], _layer_passes(traced)
    assert _pass_counts(traced, "add") == [3, 3], _layer_passes(traced)
    assert _pass_counts(traced, "tanh") == [3, 3], _layer_passes(traced)
    assert check_metadata_invariants(traced)


def test_subset_context_across_structurally_unequal_bodies() -> None:
    """Strict-subset parent structure across visibly different bodies splits 3/2.

    ``tanh(shared(x))`` for three iterations then ``tanh(shared(x) + dec(x))``
    for two: the first loop's tanh signature {shared} is carried by THREE
    calls, so its cohort is a realized recurrent site of its own -- never a
    dangling entry -- and the boundary holds (the entry exemption demands a
    globally unique entry signature).
    """

    class SubsetOverlap(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Linear(4, 4)
            self.dec = nn.Linear(4, 4)

        def forward(self, x):
            for _ in range(3):
                x = torch.tanh(self.shared(x))
            for _ in range(2):
                x = torch.tanh(self.shared(x) + self.dec(x))
            return x

    torch.manual_seed(0)
    traced = trace_fn(SubsetOverlap(), torch.randn(1, 4))
    assert _pass_counts(traced, "linear") == [5, 2], _layer_passes(traced)
    assert _pass_counts(traced, "tanh") == [3, 2], _layer_passes(traced)
    assert check_metadata_invariants(traced)


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


def test_three_differing_parent_contexts_split_per_loop() -> None:
    """A param-free op with THREE differing parent structures splits 2/2/2."""

    class ThreeContexts(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Linear(4, 4)
            self.a = nn.Linear(4, 4)
            self.b = nn.Linear(4, 4)
            self.c = nn.Linear(4, 4)

        def forward(self, x):
            for _ in range(2):
                x = torch.tanh(self.shared(x) + self.a(x))
            for _ in range(2):
                x = torch.tanh(self.shared(x) + self.b(x))
            for _ in range(2):
                x = torch.tanh(self.shared(x) + self.c(x))
            return x

    torch.manual_seed(0)
    traced = trace_fn(ThreeContexts(), torch.randn(1, 4))
    assert _pass_counts(traced, "linear") == [6, 2, 2, 2], _layer_passes(traced)
    assert _pass_counts(traced, "add") == [2, 2, 2], _layer_passes(traced)
    assert _pass_counts(traced, "tanh") == [2, 2, 2], _layer_passes(traced)
    assert check_metadata_invariants(traced)


def test_untied_overlapping_control_stays_split() -> None:
    """Untied control: distinct shared_a/shared_b flanks keep 3/2 splits."""

    class DisjointControl(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared_a = nn.Linear(4, 4)
            self.shared_b = nn.Linear(4, 4)
            self.enc = nn.Linear(4, 4)
            self.dec = nn.Linear(4, 4)

        def forward(self, x):
            for _ in range(3):
                x = torch.tanh(self.shared_a(x) + self.enc(x))
            for _ in range(2):
                x = torch.tanh(self.shared_b(x) + self.dec(x))
            return x

    torch.manual_seed(0)
    traced = trace_fn(DisjointControl(), torch.randn(1, 4))
    assert _pass_counts(traced, "linear") == [3, 3, 2, 2], _layer_passes(traced)
    assert _pass_counts(traced, "add") == [3, 2], _layer_passes(traced)
    assert _pass_counts(traced, "tanh") == [3, 2], _layer_passes(traced)
    assert check_metadata_invariants(traced)


# ---------------------------------------------------------------------------
# Loop-entry saturation: pass 1 reads pre-loop state, so its signature differs
# from every later pass in exactly the carry slot. The guarded exemption must
# keep honest entries merged without reopening any boundary.
# ---------------------------------------------------------------------------


def test_residual_loop_entry_pass_stays_merged_per_site() -> None:
    """Two-site residual body: each site's adds group per site, entry included.

    ``h = emb(x); for 3: h = h + attn(h); h = h + mlp(h)``. The attn-adds and
    the mlp-adds are different source-code sites with different parameterized
    direct parents (attn versus mlp), so they form TWO 3-pass layers in
    lockstep with their flanks (attn [3], mlp [3]) -- the entry add (pass 1
    reads emb) is adopted into the attn site through the carry-slot exemption.

    DELIBERATE r25 policy change: the pre-r25 ladder grouped all six adds as
    one 6-pass layer, a 6-pass neighbor between two 3-pass parameterized
    layers -- the same neighbor incoherence this file exists to forbid. The
    independent brute-force site partition is [3, 3], so this corrects a bad
    [6] golden rather than weakening the validation tripwire.
    """

    class ResidualEntry(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Linear(4, 4)
            self.attn = nn.Linear(4, 4)
            self.mlp = nn.Linear(4, 4)

        def forward(self, x):
            h = self.emb(x)
            for _ in range(3):
                h = h + self.attn(h)
                h = h + self.mlp(h)
            return h

    torch.manual_seed(0)
    traced = trace_fn(ResidualEntry(), torch.randn(1, 4))
    assert _pass_counts(traced, "linear") == [1, 3, 3], _layer_passes(traced)
    assert _pass_counts(traced, "add") == [3, 3], _layer_passes(traced)
    assert check_metadata_invariants(traced)


def test_single_site_residual_entry_stays_merged() -> None:
    """One-site residual loop: the unsaturated entry add still merges to [n].

    ``h = emb(x); for 3: h = h + attn(h)``: pass 1's signature {emb, attn}
    differs from the steady {self-class, attn} in exactly the carry slot, the
    odd parent emb is one-shot, the attn flank agrees, and the carry is
    certified -- the flagship honest entry.
    """

    class SingleSiteResidual(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Linear(4, 4)
            self.attn = nn.Linear(4, 4)

        def forward(self, x):
            h = self.emb(x)
            for _ in range(3):
                h = h + self.attn(h)
            return h

    torch.manual_seed(0)
    traced = trace_fn(SingleSiteResidual(), torch.randn(1, 4))
    assert _pass_counts(traced, "add") == [3], _layer_passes(traced)
    assert check_metadata_invariants(traced)


def test_single_site_entry_saturation_stays_merged() -> None:
    """One-site loop with a param-free unary head: pass 1 merges to [3].

    ``h = pre(x); for: t = tanh(h); h = body(t) + t``: the tanh entry is
    flankless (unary), so admission rides the realized-target evidence (two
    later same-signature calls) with the one-shot ``pre`` odd parent.
    """

    class SaturatingTanh(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pre = nn.Linear(4, 4)
            self.body = nn.Linear(4, 4)

        def forward(self, x):
            h = self.pre(x)
            for _ in range(3):
                t = torch.tanh(h)
                h = self.body(t) + t
            return h

    torch.manual_seed(0)
    traced = trace_fn(SaturatingTanh(), torch.randn(1, 4))
    assert _pass_counts(traced, "tanh") == [3], _layer_passes(traced)
    assert _pass_counts(traced, "add") == [3], _layer_passes(traced)
    assert check_metadata_invariants(traced)


def test_different_typed_entries_sharing_one_context_both_adopt() -> None:
    """Same-key census scoping: different-typed honest entries do not collide.

    ``h = emb(x); for 3: h = tanh(h) * sigmoid(h) + body(h)``: on pass 1 BOTH
    the bare ``tanh`` and the bare ``sigmoid`` carry the identical unsaturated
    signature {emb}. Uniqueness is judged within each op's OWN equivalence-key
    universe, so both adoptions are granted.
    """

    class TwoTypedEntries(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Linear(4, 4)
            self.body = nn.Linear(4, 4)

        def forward(self, x):
            h = self.emb(x)
            for _ in range(3):
                h = torch.tanh(h) * torch.sigmoid(h) + self.body(h)
            return h

    torch.manual_seed(0)
    traced = trace_fn(TwoTypedEntries(), torch.randn(1, 4))
    assert _pass_counts(traced, "tanh") == [3], _layer_passes(traced)
    assert _pass_counts(traced, "sigmoid") == [3], _layer_passes(traced)
    assert _pass_counts(traced, "mul") == [3], _layer_passes(traced)
    assert _pass_counts(traced, "add") == [3], _layer_passes(traced)
    assert check_metadata_invariants(traced)


def test_param_free_bridged_chained_loops_stay_split() -> None:
    """A realized 3-member signature class never dissolves into loop 2.

    Loop 2's head reads loop 1's output PARAM-FREE (``tanh(x + dec(x))``), so
    a param-free path crosses the boundary -- only the globally-unique-entry
    census blocks adoption. Expect tanh 3/2, never a 5-pass bridge.
    """

    class PfBridgedChained(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Linear(4, 4)
            self.enc = nn.Linear(4, 4)
            self.dec = nn.Linear(4, 4)

        def forward(self, x):
            for _ in range(3):
                x = torch.tanh(self.shared(x) + self.enc(x))
            for _ in range(2):
                x = torch.tanh(x + self.dec(x))
            return x

    torch.manual_seed(0)
    traced = trace_fn(PfBridgedChained(), torch.randn(1, 4))
    assert _pass_counts(traced, "add") == [3, 2], _layer_passes(traced)
    assert _pass_counts(traced, "tanh") == [3, 2], _layer_passes(traced)
    assert check_metadata_invariants(traced)


def test_recalled_shared_identity_is_not_adopted() -> None:
    """Tapped-entry conservative refusal: a realized odd parent is a boundary.

    ``x = tanh(shared(x))`` once, then ``tanh(shared(x) + dec(x))`` twice: the
    prelude tanh's odd parent is the ``shared`` site, realized THREE times --
    a recurring neighbor, not a one-shot pre-loop producer -- so the entry
    exemption refuses and the prelude stays a separate 1-pass layer.
    """

    class OneShotThenLoop(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Linear(4, 4)
            self.dec = nn.Linear(4, 4)

        def forward(self, x):
            x = torch.tanh(self.shared(x))
            for _ in range(2):
                x = torch.tanh(self.shared(x) + self.dec(x))
            return x

    torch.manual_seed(0)
    traced = trace_fn(OneShotThenLoop(), torch.randn(1, 4))
    assert _pass_counts(traced, "tanh") == [1, 2], _layer_passes(traced)
    assert check_metadata_invariants(traced)


def test_entry_singleton_cannot_bridge_two_vetoed_loops() -> None:
    """At-most-one adoption: one prelude call never fuses two split loops.

    ``x = tanh(shared(x))`` then two param-free-bridged 2-iteration loops: the
    prelude may join at most its EARLIEST admissible target, so the two loops'
    tanh classes never merge into a single 5-pass layer.
    """

    class SingletonBridge(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Linear(4, 4)
            self.enc = nn.Linear(4, 4)
            self.dec = nn.Linear(4, 4)

        def forward(self, x):
            x = torch.tanh(self.shared(x))
            for _ in range(2):
                x = torch.tanh(x + self.enc(x))
            for _ in range(2):
                x = torch.tanh(x + self.dec(x))
            return x

    torch.manual_seed(0)
    traced = trace_fn(SingletonBridge(), torch.randn(1, 4))
    tanh_counts = _pass_counts(traced, "tanh")
    assert tanh_counts in ([3, 2], [1, 2, 2]), _layer_passes(traced)
    assert _pass_counts(traced, "add") == [2, 2], _layer_passes(traced)
    assert check_metadata_invariants(traced)


def test_entry_adoption_never_fuses_parallel_vetoed_branches() -> None:
    """At-most-one adoption, parallel form: branches stay apart.

    One prelude ``tanh(shared(x))`` param-free-feeds TWO PARALLEL loops whose
    tanh classes are mutually split. The prelude joins at most one branch
    ([3, 2]) or stays a singleton ([1, 2, 2]); a cross-branch three-call
    "3-pass layer" ([3, 1, 1]-style fusion) is forbidden.
    """

    class ParallelBranches(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Linear(4, 4)
            self.enc = nn.Linear(4, 4)
            self.dec = nn.Linear(4, 4)

        def forward(self, x):
            t = torch.tanh(self.shared(x))
            a = t
            for _ in range(2):
                a = torch.tanh(a + self.enc(a))
            b = t
            for _ in range(2):
                b = torch.tanh(b + self.dec(b))
            return a + b

    torch.manual_seed(0)
    traced = trace_fn(ParallelBranches(), torch.randn(1, 4))
    tanh_counts = _pass_counts(traced, "tanh")
    assert tanh_counts in ([1, 2, 2], [3, 2]), _layer_passes(traced)
    assert check_metadata_invariants(traced)


# ---------------------------------------------------------------------------
# Round 24: 2-iteration param-free-headed loops feeding a second consumer.
# Signature classes are global per equivalence key, so a saturated interior
# fragmented across iso rounds can never be misread as a dangling entry.
# ---------------------------------------------------------------------------


class _PfHeadedLoopChain(nn.Module):
    """Chained param-free-headed residual loops sharing one prelude.

    ``h = emb(x)`` then, per loop ``i``, ``n_i`` iterations of
    ``h = tanh(h + site_i(h))``. Each iteration runs exactly one bare ``add``
    and one bare ``tanh`` in lockstep, so a correct grouping must partition
    the two op types identically -- per loop, matching the parameterized
    flank pass counts.
    """

    def __init__(self, *iteration_counts: int) -> None:
        super().__init__()
        self.emb = nn.Linear(4, 4)
        self.sites = nn.ModuleList(nn.Linear(4, 4) for _ in iteration_counts)
        self.iteration_counts = iteration_counts

    def forward(self, x):
        h = self.emb(x)
        for site, iterations in zip(self.sites, self.iteration_counts):
            for _ in range(iterations):
                h = torch.tanh(h + site(h))
        return h


@pytest.mark.parametrize(
    "iteration_counts",
    [(2, 2), (2, 3), (3, 2), (2, 4), (4, 2), (3, 3)],
    ids=lambda counts: "x".join(str(count) for count in counts),
)
def test_two_loop_chain_partitions_add_and_tanh_in_lockstep(iteration_counts) -> None:
    """Chained pf-headed loops split add AND tanh per loop for EVERY length mix.

    The historical r24 trigger was a 2-ITERATION loop feeding a second loop:
    the saturated 2nd ``add`` was misread as a loop entry and adopted across
    the boundary (add [1, 3] beside tanh [2, 2]). Under the topological rule
    the straddle pair's odd parents are the REALIZED enc/dec sites, so the
    one-shot-odd-parent condition refuses regardless of fragmentation.
    """
    torch.manual_seed(0)
    traced = trace_fn(_PfHeadedLoopChain(*iteration_counts), torch.randn(1, 4))
    expected = list(iteration_counts)
    assert _pass_counts(traced, "add") == expected, _layer_passes(traced)
    assert _pass_counts(traced, "tanh") == expected, _layer_passes(traced)
    assert _pass_counts(traced, "linear") == [1, *expected], _layer_passes(traced)
    assert check_metadata_invariants(traced)


@pytest.mark.parametrize(
    "iteration_counts",
    [(2, 2, 2), (2, 3, 2), (3, 2, 2)],
    ids=lambda counts: "x".join(str(count) for count in counts),
)
def test_three_chained_pf_headed_loops_partition_per_loop(iteration_counts) -> None:
    """Three chained pf-headed loops: every boundary holds, add mirrors tanh."""
    torch.manual_seed(0)
    traced = trace_fn(_PfHeadedLoopChain(*iteration_counts), torch.randn(1, 4))
    expected = list(iteration_counts)
    assert _pass_counts(traced, "add") == expected, _layer_passes(traced)
    assert _pass_counts(traced, "tanh") == expected, _layer_passes(traced)
    assert check_metadata_invariants(traced)


def test_one_iteration_loop_keeps_flank_lockstep_partition() -> None:
    """n1=1 peel: the single-iteration loop's add stays with its own flank.

    ``for 1: tanh(h + enc(h)); for 2: tanh(h + dec(h))``: the peeled add's
    signature {emb, enc} differs from loop 2's {tanh-class, dec} in BOTH
    slots -- its enc flank positively contradicts loop 2's dec flank -- so it
    stays a 1-pass layer in lockstep with enc [1] (DELIBERATE r25 change from
    the ladder's fused [3], which straddled a 1-pass and a 2-pass flank). The
    flankless unary tanh carries no contradicting evidence and keeps the
    accepted adoption shape [3]. The independent brute-force add partition is
    [1, 2], so this corrects a bad [3] golden rather than weakening the
    validation tripwire.
    """
    torch.manual_seed(0)
    traced = trace_fn(_PfHeadedLoopChain(1, 2), torch.randn(1, 4))
    assert _pass_counts(traced, "linear") == [1, 1, 2], _layer_passes(traced)
    assert _pass_counts(traced, "add") == [1, 2], _layer_passes(traced)
    assert _pass_counts(traced, "tanh") == [3], _layer_passes(traced)
    assert check_metadata_invariants(traced)


def test_two_iteration_loop_bridging_into_non_loop_site() -> None:
    """A 2-iter pf-headed loop feeding a single NON-loop site: no false 2-pass."""

    class LoopThenSite(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Linear(4, 4)
            self.enc = nn.Linear(4, 4)
            self.post = nn.Linear(4, 4)

        def forward(self, x):
            h = self.emb(x)
            for _ in range(2):
                h = torch.tanh(h + self.enc(h))
            h = torch.tanh(h + self.post(h))
            return h

    torch.manual_seed(0)
    traced = trace_fn(LoopThenSite(), torch.randn(1, 4))
    assert _pass_counts(traced, "add") == [2, 1], _layer_passes(traced)
    assert _pass_counts(traced, "tanh") == [2, 1], _layer_passes(traced)
    assert check_metadata_invariants(traced)


def test_residual_prelude_loop_then_two_iteration_chain() -> None:
    """A saturating residual loop ahead of the 2+2 chain keeps every class.

    ``h = emb(x); for 3: h = h + attn(h); h = h + mlp(h)`` (per-site adds
    [3, 3] with honest entry adoption) followed by two 2-iteration pf-headed
    loops (adds [2, 2] with tanh [2, 2]): per-site residual splitting and the
    boundary refusals coexist in ONE model.
    """

    class ResidualThenChain(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Linear(4, 4)
            self.attn = nn.Linear(4, 4)
            self.mlp = nn.Linear(4, 4)
            self.enc = nn.Linear(4, 4)
            self.dec = nn.Linear(4, 4)

        def forward(self, x):
            h = self.emb(x)
            for _ in range(3):
                h = h + self.attn(h)
                h = h + self.mlp(h)
            for _ in range(2):
                h = torch.tanh(h + self.enc(h))
            for _ in range(2):
                h = torch.tanh(h + self.dec(h))
            return h

    torch.manual_seed(0)
    traced = trace_fn(ResidualThenChain(), torch.randn(1, 4))
    assert _pass_counts(traced, "add") == [3, 3, 2, 2], _layer_passes(traced)
    assert _pass_counts(traced, "tanh") == [2, 2], _layer_passes(traced)
    assert check_metadata_invariants(traced)


def test_nested_loops_shared_and_site_interiors_stay_coherent() -> None:
    """Nested-loop control: 2x3 nested iterations keep 6-pass layers throughout."""

    class NestedSharedSite(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Linear(4, 4)
            self.inner = nn.Linear(4, 4)

        def forward(self, x):
            for _ in range(2):
                for _ in range(3):
                    x = torch.tanh(self.shared(x) + self.inner(x))
            return x

    torch.manual_seed(0)
    traced = trace_fn(NestedSharedSite(), torch.randn(1, 4))
    assert _pass_counts(traced, "linear") == [6, 6], _layer_passes(traced)
    assert _pass_counts(traced, "add") == [6], _layer_passes(traced)
    assert _pass_counts(traced, "tanh") == [6], _layer_passes(traced)
    assert check_metadata_invariants(traced)


def test_conditional_in_loop_alternating_arms_stay_per_arm() -> None:
    """Conditional-in-loop control: per-arm interiors keep per-arm pass counts."""

    class ConditionalInLoop(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Linear(4, 4)
            self.b = nn.Linear(4, 4)

        def forward(self, x):
            for i in range(4):
                if i % 2 == 0:
                    x = torch.tanh(self.a(x))
                else:
                    x = torch.tanh(self.b(x))
            return x

    torch.manual_seed(0)
    traced = trace_fn(ConditionalInLoop(), torch.randn(1, 4))
    assert _pass_counts(traced, "linear") == [2, 2], _layer_passes(traced)
    assert _pass_counts(traced, "tanh") == [2, 2], _layer_passes(traced)
    assert check_metadata_invariants(traced)


# ---------------------------------------------------------------------------
# Round 25 seal findings F1-F4: context-REPLACEMENT param-headed loops, the
# post-LN r23 regression, nested inner/outer sites, and same-key preludes.
# ---------------------------------------------------------------------------


class _PostLN(nn.Module):
    """Hand-rolled post-norm residual loop: ``h = ln(h + attn(h))``."""

    def __init__(self, iterations: int) -> None:
        super().__init__()
        self.emb = nn.Linear(4, 4)
        self.attn = nn.Linear(4, 4)
        self.ln = nn.LayerNorm(4)
        self.iterations = iterations

    def forward(self, x):
        h = self.emb(x)
        for _ in range(self.iterations):
            h = self.ln(h + self.attn(h))
        return h


@pytest.mark.parametrize("iterations", [2, 3, 4])
def test_post_ln_param_headed_loop_interior_stays_whole(iterations) -> None:
    """F2 (r23 regression): a param-headed loop's bare interior groups to [n].

    The feedback wire crosses the parameterized ``ln``, so pass 1's signature
    {emb, attn} is REPLACED (not grown) at pass 2 by {ln, attn}. The subset
    test of the heuristic ladder could never re-admit it (add split [1, n-1]
    from r23 through r24); the carry-slot exemption admits it because emb is
    one-shot, the attn flank agrees, and ln pass 1 is computed from the entry
    add. Pre-r23 behavior ([n]) is restored.
    """
    torch.manual_seed(0)
    traced = trace_fn(_PostLN(iterations), torch.randn(1, 4))
    assert _pass_counts(traced, "add") == [iterations], _layer_passes(traced)
    assert _pass_counts(traced, "layernorm") == [iterations], _layer_passes(traced)
    assert check_metadata_invariants(traced)


def test_post_ln_sibling_lockstep() -> None:
    """F2/C6 lockstep witness: two bare siblings in one post-LN body match.

    ``h = ln(h + tanh(attn(h)))``: the tanh (saturating immediately) and the
    add (context-replaced by ln) execute once per iteration in lockstep and
    must partition identically -- [3] and [3], never tanh [3] beside add
    [1, 2] (the definitive r25 lockstep violation).
    """

    class PostLNSiblingTanh(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Linear(4, 4)
            self.attn = nn.Linear(4, 4)
            self.ln = nn.LayerNorm(4)

        def forward(self, x):
            h = self.emb(x)
            for _ in range(3):
                h = self.ln(h + torch.tanh(self.attn(h)))
            return h

    torch.manual_seed(0)
    traced = trace_fn(PostLNSiblingTanh(), torch.randn(1, 4))
    assert _pass_counts(traced, "add") == [3], _layer_passes(traced)
    assert _pass_counts(traced, "tanh") == [3], _layer_passes(traced)
    assert _pass_counts(traced, "layernorm") == [3], _layer_passes(traced)
    assert check_metadata_invariants(traced)


@pytest.mark.parametrize("iterations", [2, 3])
def test_post_ln_unary_interior_stays_whole(iterations) -> None:
    """F2 unary flavor: ``h = ln(tanh(h))`` groups its tanh to [n].

    The flankless unary entry rides the realized parameterized odd parent
    (``ln``, n passes): a 2-iteration loop has only one steady tanh call, so
    target-cohort realization alone cannot admit it, but the ln beacon can --
    while a bare ``tanh(tanh(x))`` chain (no parameterized site anywhere)
    still refuses.
    """

    class PostLNUnary(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Linear(4, 4)
            self.ln = nn.LayerNorm(4)

        def forward(self, x):
            h = self.emb(x)
            for _ in range(iterations):
                h = self.ln(torch.tanh(h))
            return h

    torch.manual_seed(0)
    traced = trace_fn(PostLNUnary(), torch.randn(1, 4))
    assert _pass_counts(traced, "tanh") == [iterations], _layer_passes(traced)
    assert _pass_counts(traced, "layernorm") == [iterations], _layer_passes(traced)
    assert check_metadata_invariants(traced)


def test_post_ln_tap_chain_partitions_in_lockstep() -> None:
    """F1: a 2-iter param-headed loop's pre-norm tap feeding a 2-iter pf loop.

    ``for 2: u = h + attn(h); h = ln(u)`` then ``for 2: u = u + dec(u)``: the
    r24 census misread loop 1's context-REPLACED pass-2 add as a dangling
    entry and fused it across the boundary (add [1, 3] beside layernorm [2]).
    Under the topological rule the straddle pair differs in the enc/dec-side
    slot with a REALIZED odd parent, so it splits; each loop's adds stay
    together through the flanked carry exemption: add [2, 2].
    """

    class PostLNTap(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Linear(4, 4)
            self.attn = nn.Linear(4, 4)
            self.dec = nn.Linear(4, 4)
            self.ln = nn.LayerNorm(4)

        def forward(self, x):
            h = self.emb(x)
            u = h
            for _ in range(2):
                u = h + self.attn(h)
                h = self.ln(u)
            for _ in range(2):
                u = u + self.dec(u)
            return u

    torch.manual_seed(0)
    traced = trace_fn(PostLNTap(), torch.randn(1, 4))
    assert _pass_counts(traced, "add") == [2, 2], _layer_passes(traced)
    assert _pass_counts(traced, "layernorm") == [2], _layer_passes(traced)
    assert _pass_counts(traced, "linear") == [1, 2, 2], _layer_passes(traced)
    assert check_metadata_invariants(traced)


def test_nested_inner_and_outer_sites_partition_per_site() -> None:
    """F3: nested 2-outer x (2-inner + outer site) partitions per SITE.

    ``for 2: (for 2: h = tanh(h + enc(h))); h = tanh(h + dec(h))``: the inner
    adds' parameterized direct parent is enc (4 passes) and the outer adds' is
    dec (2 passes) -- graph-visible site evidence the equal-context quadrant
    of the ladder washed out (it merged across the site boundary as [2, 4]).
    Expected: add [4, 2] and tanh [4, 2] in lockstep with linear [1, 4, 2].
    """

    class NestedOuterSite(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Linear(4, 4)
            self.enc = nn.Linear(4, 4)
            self.dec = nn.Linear(4, 4)

        def forward(self, x):
            h = self.emb(x)
            for _ in range(2):
                for _ in range(2):
                    h = torch.tanh(h + self.enc(h))
                h = torch.tanh(h + self.dec(h))
            return h

    torch.manual_seed(0)
    traced = trace_fn(NestedOuterSite(), torch.randn(1, 4))
    assert _pass_counts(traced, "linear") == [1, 4, 2], _layer_passes(traced)
    assert _pass_counts(traced, "add") == [4, 2], _layer_passes(traced)
    assert _pass_counts(traced, "tanh") == [4, 2], _layer_passes(traced)
    assert check_metadata_invariants(traced)


class _SameKeyPrelude(nn.Module):
    """A bare same-key prelude fanning into two parallel pf-headed loops."""

    def __init__(self, iterations: int, prelude: str = "add") -> None:
        super().__init__()
        self.emb = nn.Linear(4, 4)
        self.attn = nn.Linear(4, 4)
        self.xlin = nn.Linear(4, 4)
        self.ylin = nn.Linear(4, 4)
        self.iterations = iterations
        self.prelude = prelude

    def forward(self, x):
        h = self.emb(x)
        if self.prelude == "add":
            g = h + self.attn(h)
        elif self.prelude == "mul":
            g = h * self.attn(h)
        else:
            g = self.attn(h)
        p, q = g, g
        for _ in range(self.iterations):
            p = p + self.xlin(p)
        for _ in range(self.iterations):
            q = q + self.ylin(q)
        return p * q


@pytest.mark.parametrize("iterations", [2, 3])
def test_same_key_prelude_into_parallel_loops_keeps_both_loops(iterations) -> None:
    """F4: a same-key prelude no longer fragments BOTH parallel loops.

    The prelude add's signature {emb, attn} differs from each loop's
    {self-class, site} in BOTH slots, so it stays a separate 1-pass layer and
    each loop's interior groups whole ([1, n, n]) -- the ladder's subgraph
    machinery let the prelude steal each loop's first parameterized call and
    orphaned both last passes (add [2, 1, 1, 1] / [3, 1, 2, 1]).
    """
    torch.manual_seed(0)
    traced = trace_fn(_SameKeyPrelude(iterations, "add"), torch.randn(1, 4))
    expected = [1, iterations, iterations]
    assert _pass_counts(traced, "add") == expected, _layer_passes(traced)
    assert check_metadata_invariants(traced)


@pytest.mark.parametrize("prelude", ["mul", "none"])
def test_different_key_prelude_controls_stay_clean(prelude) -> None:
    """F4 controls: different-key and absent preludes keep clean [2, 2] loops."""
    torch.manual_seed(0)
    traced = trace_fn(_SameKeyPrelude(2, prelude), torch.randn(1, 4))
    assert _pass_counts(traced, "add") == [2, 2], _layer_passes(traced)
    assert check_metadata_invariants(traced)


# ---------------------------------------------------------------------------
# Loop-invariant-fed repeats: parent signatures carry no sequencing evidence,
# so repetition is certified through consumers -- without conflating sibling
# one-shot values or parallel streams.
# ---------------------------------------------------------------------------


def test_in_loop_factory_op_groups_with_its_body() -> None:
    """A per-iteration factory op groups [n] through consumer-side sequencing.

    ``for 3: h = body(h) + torch.ones(4)``: the ones calls are parentless, so
    their repetition is certified by disjoint same-class consumers connected
    in sequence -- and the adds stay [3] because the ones contribute ONE class
    color, not three singleton colors.
    """

    class InLoopFactory(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Linear(4, 4)
            self.body = nn.Linear(4, 4)

        def forward(self, x):
            h = self.emb(x)
            for _ in range(3):
                h = self.body(h) + torch.ones(4)
            return h

    torch.manual_seed(0)
    traced = trace_fn(InLoopFactory(), torch.randn(1, 4))
    assert _pass_counts(traced, "ones") == [3], _layer_passes(traced)
    assert _pass_counts(traced, "add") == [3], _layer_passes(traced)
    assert check_metadata_invariants(traced)


def test_loop_invariant_recompute_groups_with_its_body() -> None:
    """A per-iteration recompute of a pre-loop value groups [n].

    ``for 3: h = body(h) + tanh(x)``: every ``tanh(x)`` reads only the
    unchanging input (equal all-external signatures, no connecting data path),
    so consumer-side sequencing carries the grouping.
    """

    class LoopInvariantRecompute(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Linear(4, 4)
            self.body = nn.Linear(4, 4)

        def forward(self, x):
            h = self.emb(x)
            for _ in range(3):
                h = self.body(h) + torch.tanh(x)
            return h

    torch.manual_seed(0)
    traced = trace_fn(LoopInvariantRecompute(), torch.randn(1, 4))
    assert _pass_counts(traced, "tanh") == [3], _layer_passes(traced)
    assert _pass_counts(traced, "add") == [3], _layer_passes(traced)
    assert check_metadata_invariants(traced)


def test_shared_mask_scan_entry_merges() -> None:
    """A pre-loop mask flank plus carry rotation still admits the entry.

    ``mask = sigmoid(x); for 3: h = body(h * mask) + h``: the mul entry's odd
    parent is the one-shot emb, the agreeing remainder is the (param-free)
    mask class, and the target cohort is realized -- mul [3] in lockstep with
    add [3], never a sheared [1, 2].
    """

    class MaskedScan(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Linear(4, 4)
            self.body = nn.Linear(4, 4)

        def forward(self, x):
            h = self.emb(x)
            mask = torch.sigmoid(x)
            for _ in range(3):
                h = self.body(h * mask) + h
            return h

    torch.manual_seed(0)
    traced = trace_fn(MaskedScan(), torch.randn(1, 4))
    assert _pass_counts(traced, "mul") == [3], _layer_passes(traced)
    assert _pass_counts(traced, "add") == [3], _layer_passes(traced)
    assert check_metadata_invariants(traced)


# ---------------------------------------------------------------------------
# Anti-conflation guards: repetition without loop evidence must stay split.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("length", [2, 3])
def test_bare_functional_chain_is_never_a_loop(length) -> None:
    """A bare single-op self-chain stays single-pass at every length.

    ``tanh(tanh(x))`` and ``tanh(tanh(tanh(x)))`` carry no recurrence evidence
    beyond their own repetition (the steady calls are fed exclusively by the
    class itself), so the self-evidence gate dissolves them -- the historical
    minimum-body-size policy, restated topologically.
    """

    class BareChain(nn.Module):
        def forward(self, x):
            for _ in range(length):
                x = torch.tanh(x)
            return x

    torch.manual_seed(0)
    traced = trace_fn(BareChain(), torch.randn(1, 4))
    assert _pass_counts(traced, "tanh") == [1] * length, _layer_passes(traced)
    assert check_metadata_invariants(traced)


def test_two_op_param_free_motif_still_groups() -> None:
    """A straight-line >=2-op motif keeps grouping (locked owner policy).

    ``for 3: x = tanh(-x)``: the neg and tanh classes support each other
    (each is the other's steady evidence), so both group [3] -- the
    topological restatement of the >=2-op body policy.
    """

    class TwoOpMotif(nn.Module):
        def forward(self, x):
            for _ in range(3):
                x = torch.tanh(-x)
            return x

    torch.manual_seed(0)
    traced = trace_fn(TwoOpMotif(), torch.randn(1, 4))
    assert _pass_counts(traced, "tanh") == [3], _layer_passes(traced)
    assert _pass_counts(traced, "neg") == [3], _layer_passes(traced)
    assert check_metadata_invariants(traced)


def test_disconnected_singleton_is_never_adopted() -> None:
    """Carry certificate: a parallel one-shot call never joins a loop it feeds nothing.

    ``t = tanh(pre(x))`` on a PARALLEL branch beside ``for 3: h = tanh(h +
    attn(h))``: the prelude tanh's signature {pre} is globally unique, its odd
    parent is one-shot, and the loop's tanh cohort is realized -- every entry
    condition holds EXCEPT the carry certificate: no data path from the
    prelude computes any loop call's carry parent. It must stay a separate
    1-pass layer, never pass 1 of a 4-pass layer.
    """

    class DisconnectedSingleton(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pre = nn.Linear(4, 4)
            self.emb = nn.Linear(4, 4)
            self.attn = nn.Linear(4, 4)

        def forward(self, x):
            t = torch.tanh(self.pre(x))
            h = self.emb(x)
            for _ in range(3):
                h = torch.tanh(h + self.attn(h))
            return h + t

    torch.manual_seed(0)
    traced = trace_fn(DisconnectedSingleton(), torch.randn(1, 4))
    assert _pass_counts(traced, "tanh") == [1, 3], _layer_passes(traced)
    assert _pass_counts(traced, "add") == [3, 1], _layer_passes(traced)
    assert check_metadata_invariants(traced)


def test_parallel_stream_interiors_never_merge() -> None:
    """Equal signatures without a connecting data path stay per-stream.

    One tied cell drives loops on two PARALLEL streams: the tanh interiors of
    the two streams share the {cell} signature but no data path connects them,
    so they stay [2, 3] beside the one 5-pass tied cell, and the terminal
    sums stay single-pass.
    """

    class SharedCellTwoLoops(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cell = nn.Linear(4, 4)

        def forward(self, x):
            a = x
            for _ in range(2):
                a = torch.tanh(self.cell(a))
            b = torch.relu(x)
            for _ in range(3):
                b = torch.tanh(self.cell(b))
            return a.sum() + b.sum()

    torch.manual_seed(0)
    traced = trace_fn(SharedCellTwoLoops(), torch.randn(2, 4))
    assert _pass_counts(traced, "linear") == [5], _layer_passes(traced)
    assert sorted(_pass_counts(traced, "tanh")) == [2, 3], _layer_passes(traced)
    assert _pass_counts(traced, "sum") == [1, 1], _layer_passes(traced)
    assert check_metadata_invariants(traced)


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
