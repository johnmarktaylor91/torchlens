"""F02 selection-batch ``do()`` v1 (edits memo D33; row B6 analog).

One atomic ACT-only transaction: all-or-nothing attachment, ONE envelope with
the batch disclosure and kind-specific staged-store slots, per-pair ACT audit
rows, donor-group sharing by plan object identity (D8), cross-kind capability
refusal, and the abort row -- one valid + one invalid clause leaves the trace
value-equivalent to its pre-call state with one aborted-transaction record.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention import reference, sample_from, sampling_records


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
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )


def _envelopes(log: tl.Trace) -> list[dict]:
    return [
        row
        for row in log.state_history
        if isinstance(row, dict) and row.get("op") == "intervention_event"
    ]


@pytest.mark.smoke
def test_batch_applies_every_pair_in_one_transaction() -> None:
    """Two ACT pairs, one push, one envelope, per-pair ACT rows."""

    log = _trace()
    first = log["relu_1_2"].out
    second = log["relu_2_4"].out
    fork = log.fork()
    fork.do(
        [
            (log["relu_1_2"].__selection__(), tl.scale(2.0)),
            (log["relu_2_4"].__selection__(), tl.zero_ablate()),
        ]
    )
    assert torch.allclose(fork["relu_1_2"].out, first * 2)
    assert torch.equal(fork["relu_2_4"].out, torch.zeros_like(second))
    envelopes = _envelopes(fork)
    assert len(envelopes) == 1, "ONE transactional envelope for the whole batch"
    assert envelopes[0]["edit_names"] == ["scale", "zero_ablate"]
    assert envelopes[0]["selection_batch"]["pairs"] == 2
    assert envelopes[0]["selection_batch"]["staged_stores"] == {
        "act": 2,
        "param": 0,
        "edge": 0,
    }
    act_rows = [row for row in fork.intervention_audit if row.get("kind") == "ACT"]
    assert len(act_rows) == 2, "per-pair ACT disclosure rows"


@pytest.mark.smoke
def test_shared_plan_identity_shares_one_donor_group() -> None:
    """D8: ONE plan reused across clauses = one donor_group_id; per_group shares the draw."""

    log = _trace()
    baseline = log["relu_1_2"].out
    donors = reference(torch.stack([baseline + 1, baseline + 2, baseline + 3]), origin="d")
    plan = sample_from(donors, seed=3, share_draw="per_group")
    fork = log.fork()
    fork.do(
        [
            (log["relu_1_2"].__selection__(), tl.patch_from(plan)),
            (log["relu_2_4"].__selection__(), tl.patch_from(plan)),
        ]
    )
    records = sampling_records(fork)
    assert len(records) == 2
    assert records[0]["donor_group_id"] == records[1]["donor_group_id"]
    assert records[0]["donor_ids"] == records[1]["donor_ids"], "per_group shares the draw"
    # Content-digest minting: the same declared inputs reconstruct the same
    # group key (the derived-seed law's rerun-reproducibility half) ...
    plan_a = sample_from(donors, seed=3, share_draw="per_group")
    plan_b = sample_from(donors, seed=3, share_draw="per_group")
    assert plan_a.donor_group_id == plan_b.donor_group_id
    # ... while the batch normalizer keeps D8's other half: distinct plan
    # OBJECTS inside ONE transaction never share a donor group, even with
    # identical visible arguments -- the second object gets a deterministic
    # clause-order suffix, so a rerun of this batch reproduces it too.
    fork_two = log.fork()
    fork_two.do(
        [
            (log["relu_1_2"].__selection__(), tl.patch_from(plan_a)),
            (log["relu_2_4"].__selection__(), tl.patch_from(plan_b)),
        ]
    )
    two_plan_records = sampling_records(fork_two)[-2:]
    group_ids = {row["donor_group_id"] for row in two_plan_records}
    assert len(group_ids) == 2, "kwargs coincidence never shares a donor group"
    assert plan_a.donor_group_id in group_ids
    assert f"{plan_b.donor_group_id}#1" in group_ids


@pytest.mark.smoke
def test_leaf_site_pair_refuses_typed() -> None:
    """D33 v1: leaf sites (inputs/buffers) cannot join one hook transaction."""

    log = _trace()
    with pytest.raises(Exception) as excinfo:
        log.fork().do(
            [
                (log.input_ops[0].__selection__(), tl.scale(2.0)),
                (log["relu_1_2"].__selection__(), tl.scale(2.0)),
            ]
        )
    assert excinfo.value.fields["code"] == "selection_batch_leaf_unsupported"


@pytest.mark.smoke
def test_cross_kind_batch_refuses_with_capability_report() -> None:
    """D33: PARAM/EDGE selections in a batch refuse typed, naming the v1 scope."""

    log = _trace()
    param_name = next(iter(dict(log.params.items())))
    with pytest.raises(Exception) as excinfo:
        log.fork().do(
            [
                (log["relu_1_2"].__selection__(), tl.scale(2.0)),
                (tl.params(param_name), tl.scale(0.5)),
            ]
        )
    assert excinfo.value.fields["code"] == "selection_batch_cross_kind"
    assert "ACT-only" in str(excinfo.value)


@pytest.mark.smoke
def test_mixed_selection_and_legacy_pairs_refuse() -> None:
    """A batch mixing Selections with legacy selector sites refuses typed."""

    log = _trace()
    with pytest.raises(Exception) as excinfo:
        log.fork().do(
            [
                (log["relu_1_2"].__selection__(), tl.scale(2.0)),
                (tl.label("relu_2_4"), tl.zero_ablate()),
            ]
        )
    assert excinfo.value.fields["code"] == "selection_batch_pair_invalid"
    with pytest.raises(Exception) as excinfo:
        log.fork().do([(log["relu_1_2"].__selection__(), None)], tl.scale(2.0))
    assert excinfo.value.fields["code"] == "selection_batch_pair_invalid"


@pytest.mark.smoke
def test_abort_leaves_pre_call_state_with_one_error_record() -> None:
    """B6 analog: one valid + one invalid clause rolls back everything."""

    log = _trace()
    fork = log.fork()
    baseline_first = fork["relu_1_2"].out.clone()
    baseline_second = fork["relu_2_4"].out.clone()
    hooks_before = len(fork._ensure_intervention_spec().hook_specs)
    with pytest.raises(ValueError):
        fork.do(
            [
                (log["relu_1_2"].__selection__(), tl.scale(2.0)),
                (log["relu_2_4"].__selection__(), "not-an-edit"),
            ]
        )
    assert torch.equal(fork["relu_1_2"].out, baseline_first), "values untouched"
    assert torch.equal(fork["relu_2_4"].out, baseline_second)
    assert len(fork._ensure_intervention_spec().hook_specs) == hooks_before, "no hooks leak"
    envelopes = _envelopes(fork)
    assert len(envelopes) == 1 and envelopes[0]["status"] == "error", (
        "the fire-evidence rule: exactly one aborted-transaction record"
    )


@pytest.mark.smoke
def test_batch_failure_mid_attach_detaches_earlier_pairs() -> None:
    """Atomicity at the attach seam: a refused later pair unhooks earlier ones."""

    log = _trace()
    fork = log.fork()
    hooks_before = len(fork._ensure_intervention_spec().hook_specs)
    ragged = tl.units("relu_2_4", [(0, 1), (1, 2)])
    from torchlens.intervention import permute_batch

    with pytest.raises(Exception) as excinfo:
        fork.do(
            [
                (log["relu_1_2"].__selection__(), tl.scale(2.0)),
                (ragged.resolve(fork), permute_batch(seed=1, axis=0)),
            ]
        )
    assert excinfo.value.fields["code"] == "mask_not_row_equivariant"
    assert len(fork._ensure_intervention_spec().hook_specs) == hooks_before
