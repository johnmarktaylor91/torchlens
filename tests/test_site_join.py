"""Site-key join (three-layer rule, verdict tiers) and the tier-(a) fold
closure: the memo's adversarial corpus, each case probe-validated in the
canonical census drivers before becoming a test obligation here.

The join corpus (memo 3.6):
(i)   equal-cardinality alternative branches -- guard passes, witness
      REFUSES -- in both the flat and the NESTED form (the source-line
      oracle asserts the SELECTED witness equals the two distinct inner
      lines before asserting refusal, so a regressed frame selector fails
      the oracle, not just the verdict);
(ii)  permuted per-call-instance cardinality -- weak/multiset guard passes,
      strong guard REFUSES;
(iii) same-line reorder -- joins as corroborated, PINNED AS THE DISCLOSED
      R4 RESIDUAL (if a future key version closes R4, this pin flips and
      the glossary entry must change with it);
(iv)  verdict-tier totality -- every joined key carries exactly one tier.
"""

from __future__ import annotations

import inspect
from collections import Counter

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.postprocess._site_join import (
    FoldRow,
    _SiteJoinVerdict,
    fold_rows_from_trace,
    fold_site_groups,
    join_site_profiles,
    site_profile,
)


class _FlatBranch(nn.Module):
    """Equal-cardinality alternative branches at ROOT depth: two same-type
    functional calls in alternative arms collide on the full key with
    cardinality 1 = 1 -- only the source-location witness can refuse."""

    def __init__(self, flag: bool) -> None:
        super().__init__()
        self.flag = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.flag:
            return torch.tanh(x)
        return torch.tanh(x)


class _NestedInner(nn.Module):
    """The nested form (r4, sol M3-1): alternative sites one module deep
    behind ONE outer call line, where the shallowest frame is
    capture-identical -- a cc[0] selector would silently corroborate."""

    def __init__(self, flag: bool) -> None:
        super().__init__()
        self.flag = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.flag:
            return torch.tanh(x)
        return torch.tanh(x)


class _NestedOuter(nn.Module):
    def __init__(self, flag: bool) -> None:
        super().__init__()
        self.inner = _NestedInner(flag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)


class _PermBody(nn.Module):
    def forward(self, x: torch.Tensor, n: int) -> torch.Tensor:
        for _ in range(n):
            x = torch.tanh(x)
        return x


class _Perm(nn.Module):
    """Permuted per-call-instance cardinality: counts (1,3) vs (3,1) across
    two call instances share the sorted multiset but mis-pair ordinals."""

    def __init__(self, counts: tuple[int, ...]) -> None:
        super().__init__()
        self.counts = counts
        self.body = _PermBody()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for n in self.counts:
            x = self.body(x, n)
        return x


class _SameLineLoop(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = torch.tanh(x)
        return x


class _Tied(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(8, 8)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = self.act(self.lin(x))
        return x


class _Shrink(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for k in (8, 4, 2):
            x = self.act(x[:, :k])
        return x


def _tanh_key(profile) -> str:
    return next(key for key in profile.keys.values() if "tanh" in key)


# ---------------------------------------------------------------------------
# (i) equal-cardinality alternative branches: witness refusal
# ---------------------------------------------------------------------------


def test_flat_branch_refused_by_witness_not_guard() -> None:
    left = tl.trace(_FlatBranch(True), torch.randn(2, 4))
    right = tl.trace(_FlatBranch(False), torch.randn(2, 4))
    rows = join_site_profiles(site_profile(left), site_profile(right))
    row = rows[_tanh_key(site_profile(left))]
    # NO cardinality guard of any strength can catch this class.
    assert row.strong_guard_ok and row.weak_guard_ok
    assert row.verdict is _SiteJoinVerdict.REFUSED_WITNESS
    assert not row.joined


def test_nested_branch_witness_oracle_then_refusal() -> None:
    left = tl.trace(_NestedOuter(True), torch.randn(2, 4))
    right = tl.trace(_NestedOuter(False), torch.randn(2, 4))
    left_profile, right_profile = site_profile(left), site_profile(right)
    key = _tanh_key(left_profile)
    # SOURCE-LINE ORACLE first (r4, sol M3-1): the selected witness must be
    # the two DISTINCT inner branch lines, derived independently via
    # inspect -- a regressed selector (cc[0]: the outer call line, identical
    # across captures) fails HERE, not merely at the verdict.
    source_lines, start = inspect.getsourcelines(_NestedInner.forward)
    branch_lines = {
        start + offset for offset, line in enumerate(source_lines) if "torch.tanh" in line
    }
    assert len(branch_lines) == 2
    (left_witness,) = left_profile.witnesses[key]
    (right_witness,) = right_profile.witnesses[key]
    assert left_witness is not None and right_witness is not None
    assert {left_witness[1], right_witness[1]} == branch_lines
    assert left_witness != right_witness
    rows = join_site_profiles(left_profile, right_profile)
    assert rows[key].verdict is _SiteJoinVerdict.REFUSED_WITNESS


# ---------------------------------------------------------------------------
# (ii) permuted per-call-instance cardinality: strong guard refusal
# ---------------------------------------------------------------------------


def test_permuted_cardinality_weak_passes_strong_refuses() -> None:
    left = tl.trace(_Perm((1, 3)), torch.randn(2, 4))
    right = tl.trace(_Perm((3, 1)), torch.randn(2, 4))
    rows = join_site_profiles(site_profile(left), site_profile(right))
    tanh_rows = [row for key, row in rows.items() if "tanh" in key]
    assert len(tanh_rows) == 3
    for row in tanh_rows:
        # The exact case the r2 sorted-multiset guard wrongly admitted.
        assert row.weak_guard_ok and not row.strong_guard_ok
        assert row.verdict is _SiteJoinVerdict.REFUSED_CARDINALITY


# ---------------------------------------------------------------------------
# (iii) same-line re-execution: the DISCLOSED R4 residual
# ---------------------------------------------------------------------------


def test_same_line_reorder_residual_stays_disclosed() -> None:
    left = tl.trace(_SameLineLoop(), torch.randn(2, 4))
    right = tl.trace(_SameLineLoop(), torch.randn(2, 4))
    rows = join_site_profiles(site_profile(left), site_profile(right))
    tanh_verdicts = {key: row.verdict for key, row in rows.items() if "tanh" in key}
    assert len(tanh_verdicts) == 3
    # Guard and witness both pass on one re-executed source line: the
    # ordinal pairing is positional only, yet the tier reads corroborated
    # BY DEFINITION (position + call-site provenance, never semantics).
    # This pin asserts the disclosure stays true; a future key version that
    # closes R4 must flip this test and the glossary entry together.
    assert set(tanh_verdicts.values()) == {_SiteJoinVerdict.CORROBORATED}


# ---------------------------------------------------------------------------
# (iv) verdict totality + boundary behaviors
# ---------------------------------------------------------------------------


def test_every_joined_key_carries_exactly_one_tier() -> None:
    left = tl.trace(_Perm((1, 3)), torch.randn(2, 4))
    right = tl.trace(_Perm((3, 1)), torch.randn(2, 4))
    rows = join_site_profiles(site_profile(left), site_profile(right))
    assert rows  # non-vacuous
    for row in rows.values():
        assert row.verdict in tuple(_SiteJoinVerdict)
        assert row.joined == (
            row.verdict in (_SiteJoinVerdict.CORROBORATED, _SiteJoinVerdict.POSITIONAL)
        )


def test_io_boundary_ops_join_positional() -> None:
    # Input/output ops carry no code_context: witness-absent, never refused.
    left = tl.trace(_SameLineLoop(), torch.randn(2, 4))
    right = tl.trace(_SameLineLoop(), torch.randn(2, 4))
    rows = join_site_profiles(site_profile(left), site_profile(right))
    assert rows["s1||input||1"].verdict is _SiteJoinVerdict.POSITIONAL
    assert rows["s1||output||1"].verdict is _SiteJoinVerdict.POSITIONAL


@pytest.mark.smoke
def test_site_profile_refuses_typed_on_keyless_trace() -> None:
    log = tl.trace(_SameLineLoop(), torch.randn(2, 4))
    for label in log.op_labels:
        log.ops[label].site_key = None
    with pytest.raises(InvalidArgumentError) as excinfo:
        site_profile(log)
    assert excinfo.value.fields["code"] == "site_key_unavailable"


@pytest.mark.heavy
def test_gpt2_cross_length_alignment_pins() -> None:
    # Cross-capture alignment machinery (the cross-stamp exit-gate shape):
    # two captures of one program at DIFFERENT sequence lengths align 100%
    # on sites with exact pinned tier counts -- while their label spaces
    # would disagree wherever step counts differ. Exact counts, not
    # thresholds; drift = investigation.
    transformers = pytest.importorskip("transformers")
    config = transformers.GPT2Config(n_layer=4, n_head=4, n_embd=128, vocab_size=512)
    model = transformers.GPT2LMHeadModel(config)
    model.eval()
    with torch.no_grad():
        long_log = tl.trace(model, torch.randint(0, 512, (1, 16)))
        short_log = tl.trace(model, torch.randint(0, 512, (1, 8)))
    rows = join_site_profiles(site_profile(long_log), site_profile(short_log))
    assert len(rows) == 199
    tiers = Counter(row.verdict for row in rows.values())
    assert tiers == Counter(
        {
            _SiteJoinVerdict.CORROBORATED: 189,
            _SiteJoinVerdict.POSITIONAL: 10,
        }
    )


# ---------------------------------------------------------------------------
# Tier-(a) fold closure (entry-dark; normative equivalence guard)
# ---------------------------------------------------------------------------


def test_closure_is_identity_when_recurrence_already_grouped() -> None:
    log = tl.trace(_Tied(), torch.randn(2, 8))
    groups = fold_site_groups(fold_rows_from_trace(log))
    ops = {label: log.ops[label] for label in log.op_labels}
    for label, group in groups.items():
        recurrence = set(ops[label].recurrent_ops or ()) or {label}
        assert group == frozenset(recurrence)


@pytest.mark.smoke
def test_closure_reunites_degraded_same_site_cohorts() -> None:
    # On the recurrence_detection=False path the param-free relus stay
    # single layers; the closure folds them back into one group via
    # (same site_key AND same equivalence_class).
    log = tl.trace(
        _Tied(), torch.randn(2, 8), capture=tl.options.CaptureOptions(recurrence_detection=False)
    )
    groups = fold_site_groups(fold_rows_from_trace(log))
    relu_groups = {groups[label] for label in log.op_labels if "relu" in label}
    assert len(relu_groups) == 1
    assert len(next(iter(relu_groups))) == 3


def test_closure_equivalence_guard_refuses_cross_shape_folds() -> None:
    # The reused relu spans three shapes -> three distinct (shape-bearing)
    # equivalence classes: the NORMATIVE guard keeps them separate. An
    # unguarded closure would mint a cross-shape group violating the live
    # shared-equivalence invariant.
    log = tl.trace(_Shrink(), torch.randn(2, 16))
    groups = fold_site_groups(fold_rows_from_trace(log))
    relu_groups = {groups[label] for label in log.op_labels if "relu" in label}
    assert len(relu_groups) == 3
    assert all(len(group) == 1 for group in relu_groups)


def test_closure_never_splits_and_stays_equivalence_uniform() -> None:
    for build in (
        lambda: tl.trace(_Tied(), torch.randn(2, 8)),
        lambda: tl.trace(_Perm((1, 3)), torch.randn(2, 4)),
    ):
        log = build()
        ops = {label: log.ops[label] for label in log.op_labels}
        groups = fold_site_groups(fold_rows_from_trace(log))
        for label, group in groups.items():
            # Coarsen-only: the recurrence group is always contained.
            assert set(ops[label].recurrent_ops or ()) <= set(group) | {label}
            # Every fold group shares ONE equivalence class (invariant-safe
            # by construction).
            assert len({ops[member].equivalence_class for member in group}) == 1


@pytest.mark.smoke
def test_closure_keyless_rows_never_fold() -> None:
    rows = [
        FoldRow("a_1_1:1", None, "tanh_x", ()),
        FoldRow("a_2_2:1", None, "tanh_x", ()),
    ]
    groups = fold_site_groups(rows)
    assert groups["a_1_1:1"] == frozenset({"a_1_1:1"})
    assert groups["a_2_2:1"] == frozenset({"a_2_2:1"})


@pytest.mark.heavy
def test_resnet18_fold_group_census_pin() -> None:
    # Pinned census fact (I-S3' family): the guarded closure mints exactly
    # 8 beyond-recurrence fold groups on resnet18, all equivalence-uniform.
    torchvision = pytest.importorskip("torchvision")
    model = torchvision.models.resnet18()
    model.eval()
    with torch.no_grad():
        log = tl.trace(model, torch.randn(1, 3, 64, 64))
    ops = {label: log.ops[label] for label in log.op_labels}
    groups = fold_site_groups(fold_rows_from_trace(log))
    beyond_recurrence = [
        group
        for group in set(groups.values())
        if len(group) > 1 and len(ops[next(iter(group))].recurrent_ops) != len(group)
    ]
    assert len(beyond_recurrence) == 8
    for group in beyond_recurrence:
        assert len({ops[member].equivalence_class for member in group}) == 1
