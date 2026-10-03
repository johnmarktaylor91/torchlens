"""W051-STOCH / AUD-CODE 2.2: batch ``do()`` draws never depend on clause order.

The batch normalizer used the clause ORDINAL as the disambiguation suffix, so
the second clause's draw changed whenever an equal-content plan preceded it and
batch ``do()`` differed from sequential ``do()``. Suffixes now derive from each
plan object's clause selections; ``per_firing`` plans are never suffixed.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from torchlens.intervention import reference, sample_from, sampling_records
from torchlens.intervention.stochastic import assign_batch_donor_groups


class _TwoBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.b(torch.relu(self.a(x))))


def _trace() -> tl.Trace:
    torch.manual_seed(0)
    return tl.trace(
        _TwoBlock(),
        torch.randn(5, 4),
        capture=tl.options.CaptureOptions(intervention_ready=True, random_seed=0),
    )


def _draws(fork: tl.Trace) -> dict[str, list[int]]:
    return {
        row["logical_firing_coordinate"][0]: row["donor_ids"]
        for row in sampling_records(fork)
        if row["share_draw"] == "per_firing"
    }


def test_per_firing_batch_equals_sequential_in_any_order() -> None:
    """The most common spelling: two equal-content per_firing plans at two sites."""

    log = _trace()
    baseline = log["relu_1_2"].out
    donors = reference(torch.stack([baseline + k for k in range(1, 6)]), origin="d")
    site_a = log["relu_1_2"].__selection__()
    site_b = log["relu_2_4"].__selection__()
    plan_a = sample_from(donors, seed=3)
    plan_b = sample_from(donors, seed=3)

    sequential = log.fork()
    sequential.do(site_a, tl.patch_from(plan_a))
    sequential.do(site_b, tl.patch_from(plan_b))
    batch = log.fork()
    batch.do([(site_a, tl.patch_from(plan_a)), (site_b, tl.patch_from(plan_b))])
    reversed_batch = log.fork()
    reversed_batch.do([(site_b, tl.patch_from(plan_b)), (site_a, tl.patch_from(plan_a))])

    expected = _draws(sequential)
    assert len(expected) == 2
    assert _draws(batch) == expected, "batch do() must reproduce sequential do()"
    assert _draws(reversed_batch) == expected, "clause order never moves a draw"
    for fork in (batch, reversed_batch):
        assert {row["donor_group_id"] for row in sampling_records(fork)} == {
            plan_a.donor_group_id
        }, "per_firing plans are never suffixed: the coordinate already carries the site"
    assert torch.equal(batch["relu_2_4"].out, reversed_batch["relu_2_4"].out)


def test_per_group_disambiguation_is_order_independent() -> None:
    """Distinct equal-content per_group objects never share, in either clause order."""

    log = _trace()
    baseline = log["relu_1_2"].out
    donors = reference(torch.stack([baseline + k for k in range(1, 6)]), origin="d")
    site_a = log["relu_1_2"].__selection__()
    site_b = log["relu_2_4"].__selection__()
    plan_a = sample_from(donors, seed=3, share_draw="per_group")
    plan_b = sample_from(donors, seed=3, share_draw="per_group")
    assert plan_a.donor_group_id == plan_b.donor_group_id

    forward = log.fork()
    forward.do([(site_a, tl.patch_from(plan_a)), (site_b, tl.patch_from(plan_b))])
    backward = log.fork()
    backward.do([(site_b, tl.patch_from(plan_b)), (site_a, tl.patch_from(plan_a))])

    def by_group(fork: tl.Trace) -> dict[str, list[int]]:
        return {row["donor_group_id"]: row["donor_ids"] for row in sampling_records(fork)}

    forward_groups = by_group(forward)
    assert len(forward_groups) == 2, "kwargs coincidence never shares a donor group"
    assert all(gid.startswith(f"{plan_a.donor_group_id}#") for gid in forward_groups)
    assert by_group(backward) == forward_groups
    assert torch.equal(forward["relu_1_2"].out, backward["relu_1_2"].out)
    assert torch.equal(forward["relu_2_4"].out, backward["relu_2_4"].out)


def test_normalizer_keys_on_selection_content_not_position() -> None:
    """Pure-function check on the normalizer: same clauses, any order, same groups."""

    donors = reference(torch.randn(4, 3), origin="d")
    plan_a = sample_from(donors, seed=1, share_draw="per_rule")
    plan_b = sample_from(donors, seed=1, share_draw="per_rule")
    site_a = tl.units("relu_1_2", [(0, 0)])
    site_b = tl.units("relu_2_4", [(0, 0)])
    pairs = [(site_a, tl.patch_from(plan_a)), (site_b, tl.patch_from(plan_b))]
    forward = assign_batch_donor_groups(pairs)
    backward = assign_batch_donor_groups(list(reversed(pairs)))
    groups_forward = {
        repr(selection): edit._tl_sampling_plan.donor_group_id for selection, edit in forward
    }
    groups_backward = {
        repr(selection): edit._tl_sampling_plan.donor_group_id for selection, edit in backward
    }
    assert groups_forward == groups_backward
    assert len(set(groups_forward.values())) == 2
    # A reused OBJECT keeps ONE group across its clauses (D8 sharing is object identity).
    shared = assign_batch_donor_groups(
        [(site_a, tl.patch_from(plan_a)), (site_b, tl.patch_from(plan_a))]
    )
    assert len({edit._tl_sampling_plan.donor_group_id for _s, edit in shared}) == 1
