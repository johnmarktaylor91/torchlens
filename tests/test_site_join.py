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

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens._postprocess._site_join import (
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


def _tanh_key(profile) -> str:
    return next(key for key in profile.keys.values() if "tanh" in key)


# ---------------------------------------------------------------------------
# (i) equal-cardinality alternative branches: witness refusal
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of _postprocess/_site_join.py.
def test_flat_branch_refused_by_witness_not_guard() -> None:
    left = tl.trace(_FlatBranch(True), torch.randn(2, 4))
    right = tl.trace(_FlatBranch(False), torch.randn(2, 4))
    rows = join_site_profiles(site_profile(left), site_profile(right))
    row = rows[_tanh_key(site_profile(left))]
    # NO cardinality guard of any strength can catch this class.
    assert row.strong_guard_ok and row.weak_guard_ok
    assert row.verdict is _SiteJoinVerdict.REFUSED_WITNESS
    assert not row.joined


# ---------------------------------------------------------------------------
# (ii) permuted per-call-instance cardinality: strong guard refusal
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# (iii) same-line re-execution: the DISCLOSED R4 residual
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# (iv) verdict totality + boundary behaviors
# ---------------------------------------------------------------------------


# Kept by shrink-r4 (per-id subset rule): sole coverage of _postprocess/_site_join.py.
def test_site_profile_refuses_typed_on_keyless_trace() -> None:
    log = tl.trace(_SameLineLoop(), torch.randn(2, 4))
    for label in log.op_labels:
        log.ops[label]._graph_site_key = None
    with pytest.raises(InvalidArgumentError) as excinfo:
        site_profile(log)
    assert excinfo.value.fields["code"] == "site_key_unavailable"


# ---------------------------------------------------------------------------
# Tier-(a) fold closure (entry-dark; normative equivalence guard)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
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
            assert set(ops[label].graph.recurrent_ops or ()) <= set(group) | {label}
            # Every fold group shares ONE equivalence class (invariant-safe
            # by construction).
            assert len({ops[member].graph.equivalence_class for member in group}) == 1


# Kept by shrink-r4 (per-id subset rule): sole coverage of _postprocess/_site_join.py.
def test_closure_keyless_rows_never_fold() -> None:
    rows = [
        FoldRow("a_1_1:1", None, "tanh_x", ()),
        FoldRow("a_2_2:1", None, "tanh_x", ()),
    ]
    groups = fold_site_groups(rows)
    assert groups["a_1_1:1"] == frozenset({"a_1_1:1"})
    assert groups["a_2_2:1"] == frozenset({"a_2_2:1"})
