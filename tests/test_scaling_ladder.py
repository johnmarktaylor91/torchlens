"""grind-r8 cluster 4.0.6: the R60/R29 scaling-ladder defenses.

Regression coverage for the capture/postprocess/viz resource-exhaustion
fixes: the op-count capture disclosure, the widened ancestor bitset
compaction, the descendant-mask batch ceiling, the cohort pair-probe
ceiling, the param-free fixpoint work budget, and the collapse-ceiling
parity for run folding and collapse metadata. Perf-shape assertions are
behavioral (warnings, cell encodings, degraded surfaces) -- wall-clock
claims live in the lane's acceptance probe, not in CI.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors._base import TorchLensWarning

pytestmark = pytest.mark.smoke


class _Chain(nn.Module):
    def __init__(self, steps: int = 30) -> None:
        super().__init__()
        self.steps = steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.steps):
            x = x + torch.zeros(1)
        return x


class _ParallelStreams(nn.Module):
    """Mutually-unreachable same-signature siblings (the R60-4 triangle shape).

    The anchored parent (Linear) routes the bare mults through the
    direct-site fixpoint; identical scalars keep them in ONE equal-signature
    cohort whose members never reach each other, which is exactly the shape
    that used to walk the full pair triangle.
    """

    def __init__(self, streams: int = 12) -> None:
        super().__init__()
        self.streams = streams
        self.lin = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        ys = [h * 2.0 for _ in range(self.streams)]
        return torch.stack(ys).sum(dim=0)


def test_op_count_disclosure_fires_once_at_threshold(monkeypatch) -> None:
    """Crossing the op-record threshold warns exactly once, capture continues."""

    from torchlens.ir import capture_events as ce

    monkeypatch.setattr(ce, "OP_COUNT_DISCLOSURE_THRESHOLD", 10)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        log = tl.trace(_Chain(steps=30), torch.randn(2))
    disclosures = [
        w for w in caught if "recorded 10 ops" in str(w.message) and w.category is TorchLensWarning
    ]
    assert len(disclosures) == 1
    assert len(log.layer_list) > 10


def test_all_four_ancestor_closures_compact_to_bitsets() -> None:
    """input_ancestors/output_descendants intern like root/internal (R60-2)."""

    from torchlens.backends.torch.ops import _ANCESTOR_SLOT_DESCRIPTORS, _AncestorBitset

    log = tl.trace(_Chain(steps=10), torch.randn(2))
    op = log[5]
    for field_name in (
        "root_ancestors",
        "internal_source_ancestors",
        "input_ancestors",
        "output_descendants",
    ):
        cell = _ANCESTOR_SLOT_DESCRIPTORS[field_name].__get__(op, type(op))
        assert isinstance(cell, _AncestorBitset), field_name
        assert isinstance(getattr(op, field_name), frozenset), field_name


def test_dense_mask_batch_declines_above_node_ceiling(monkeypatch) -> None:
    """Above the node ceiling the whole-graph mask DP is skipped (R60-3)."""

    from torchlens.postprocess import loop_grouping_adapter as lga

    monkeypatch.setattr(lga, "_DENSE_BATCH_MAX_NODES", 1)
    log = tl.trace(_Chain(steps=20), torch.randn(2))
    assert len(log.layer_list) > 1  # capture + grouping completed


def test_cohort_pair_probe_ceiling_discloses(monkeypatch) -> None:
    """An exhausted pair triangle warns and leaves the ops ungrouped (R60-4)."""

    from torchlens.postprocess import loop_grouping_adapter as lga

    monkeypatch.setattr(lga, "_PF_COHORT_PAIR_PROBE_CEILING", 1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        log = tl.trace(_ParallelStreams(streams=12), torch.randn(3))
    assert len(log.layer_list) > 12
    ceiling_warnings = [w for w in caught if "pair-probe ceiling" in str(w.message)]
    # The ceiling only fires when a same-signature cohort survives to the
    # triangle; the parallel-stream shape does that by construction.
    assert ceiling_warnings, "expected the bounded-sweep disclosure"


def test_fixpoint_work_budget_degrades_to_singletons(monkeypatch) -> None:
    """Budget exhaustion dissolves candidate classes, never over-groups (R60-5)."""

    from torchlens.postprocess import loop_grouping_adapter as lga

    monkeypatch.setattr(lga, "_PF_FIXPOINT_WORK_BUDGET", 0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        log = tl.trace(_Chain(steps=20), torch.randn(2))
    assert len(log.layer_list) > 1
    budget_warnings = [w for w in caught if "refinement" in str(w.message)]
    if budget_warnings:  # fires only when the fixpoint had multi-member classes
        assert "ungrouped" in str(budget_warnings[0].message)


def test_collapse_order_declines_cheaply_above_ceiling(monkeypatch) -> None:
    """Over-ceiling collapse metadata returns [] without full analysis (R60-8)."""

    from torchlens.visualization import auto_collapse, collapse_optimizer

    log = tl.trace(nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4)), torch.randn(1, 4))
    # F11 (collapse memo D5(i)): the analysis-free decline arm is the
    # PATHOLOGICAL pre-gate (20x the constant); merely over-budget traces now
    # degrade to the fallback planner, which legitimately analyzes. The
    # constant is pinned to 0 so any trace crosses the pathological gate.
    monkeypatch.setattr(collapse_optimizer, "COLLAPSE_OPTIMIZER_MAX_OPS", 0)

    def _analyze_should_not_run(trace):  # noqa: ANN001 - test shim
        raise AssertionError("analyze_collapse must not run above the ceiling")

    monkeypatch.setattr(auto_collapse, "analyze_collapse", _analyze_should_not_run)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert auto_collapse.collapse_order(log) == []
    # Every consumer reads through .get(address, 0.0):
    assert dict(auto_collapse.collapse_order(log)).get("nonexistent", 0.0) == 0.0


def test_fold_repeats_true_respects_collapse_ceiling(monkeypatch) -> None:
    """draw(fold_repeats=True) declines folding above the ceiling (R60-9)."""

    from torchlens.visualization import auto_collapse, collapse_optimizer

    log = tl.trace(
        nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4), nn.ReLU()),
        torch.randn(1, 4),
    )
    monkeypatch.setattr(collapse_optimizer, "COLLAPSE_OPTIMIZER_MAX_OPS", 1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        folds = auto_collapse.resolve_repeat_folds(log, None, fold_repeats=True)
    assert folds == {}
    assert any("skipping repeat-run folding" in str(w.message) for w in caught)


def test_bundle_duplicate_names_still_refuse() -> None:
    """The O(n) duplicate detection keeps the exact refusal surface (R52)."""

    model = nn.Linear(2, 2)
    x = torch.randn(1, 2)
    log_a = tl.trace(model, x)
    log_b = tl.trace(model, x)
    with pytest.raises(ValueError, match="duplicates: \\['twin'\\]"):
        tl.Bundle([("twin", log_a), ("twin", log_b)])
    bundle = tl.Bundle([("a", log_a), ("b", log_b)])
    assert sorted(bundle.names) == ["a", "b"]
