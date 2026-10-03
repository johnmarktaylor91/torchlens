"""F16 B6: the diagnostic frontier query (treescope memo F-E).

One injected NaN flags the whole downstream cone; the frontier is the
clean-to-dirty transition -- exactly one first-dirty site with its last
clean parent and first dirty child, in deterministic order.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from torchlens.notebook.frontier import nonfinite_frontier


class _Poisoned(nn.Module):
    """Mid-graph 0/0 injection."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)
        self.c = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """NaN enters between relu(a(x)) and b."""
        h = torch.relu(self.a(x))
        h = h / 0.0 * 0.0
        h = torch.relu(self.b(h))
        return self.c(h)


def test_frontier_is_the_clean_to_dirty_transition() -> None:
    """Exactly one first-dirty site; parent clean; child flagged."""

    log = tl.trace(_Poisoned(), torch.randn(2, 4))
    frontier = nonfinite_frontier(log)
    assert frontier.flagged_total >= 4  # the propagated cone
    first_dirty = [site for site in frontier.sites if site.role == "first_dirty"]
    assert [site.label for site in first_dirty] == ["truediv_1_3:1"]
    parents = [site.label for site in frontier.sites if site.role == "parent"]
    assert parents == ["relu_1_2:1"]  # the last clean value
    children = [site.label for site in frontier.sites if site.role == "child"]
    assert children == ["mul_1_4:1"]
    # Deterministic priority: first-dirty, then parents, then children.
    assert [site.role for site in frontier.sites] == ["first_dirty", "parent", "child"]
    # The frontier is a tiny, bounded slice of the flagged cone.
    assert len(frontier.sites) < frontier.flagged_total


def test_clean_capture_has_empty_frontier_with_coverage() -> None:
    """No flags -> empty frontier, and the coverage basis is disclosed."""

    log = tl.trace(nn.Sequential(nn.Linear(4, 4), nn.ReLU()), torch.randn(2, 4))
    frontier = nonfinite_frontier(log)
    assert frontier.flagged_total == 0 and frontier.sites == ()
    assert frontier.coverage  # never an undisclosed empty answer


def test_frontier_query_is_side_effect_free() -> None:
    """A pure graph query: no payload reads, no trace mutation."""

    log = tl.trace(_Poisoned(), torch.randn(2, 4))
    before = (log.num_ops, tuple(log.op_labels))
    nonfinite_frontier(log)
    nonfinite_frontier(log)
    assert (log.num_ops, tuple(log.op_labels)) == before


def test_distribution_sketch_slot_is_the_c02_histogram() -> None:
    """B6's typed sketch slot: TensorStats histogram fields, absent-safe."""

    from torchlens.stats import tensor_stats

    stats = tensor_stats(torch.randn(1024))
    assert stats.histogram_counts is not None
    assert stats.histogram_edges is not None
    assert stats.histogram_evidence.policy in ("exact", "sampled")
    tiny = tensor_stats(torch.randn(3))
    # Below the sketch floor the slot is ABSENT (None), never fabricated.
    assert tiny.histogram_counts is None
