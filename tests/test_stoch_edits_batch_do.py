"""F02 selection-batch ``do()`` v1 (edits memo D33; row B6 analog).

One atomic transaction: all-or-nothing attachment, ONE envelope with
the batch disclosure and kind-specific staged-store slots, per-pair ACT audit
rows, donor-group sharing by plan object identity (D8), the cross-kind batch
(ACT with PARAM and EDGE: equal to one door at a time, atomic on refusal), and
the abort row -- one valid + one invalid clause leaves the trace
value-equivalent to its pre-call state with one aborted-transaction record.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.intervention import population, sample_from
from torchlens.intervention.stochastic import sampling_records


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


def test_shared_plan_identity_shares_one_donor_group() -> None:
    """D8: ONE plan reused across clauses = one donor_group_id; per_group shares the draw."""

    log = _trace()
    baseline = log["relu_1_2"].out
    donors = population(torch.stack([baseline + 1, baseline + 2, baseline + 3]), origin="d")
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
    # identical visible arguments -- each colliding per_group object gets a
    # suffix derived from its OWN clause selections (never the clause
    # ordinal, which made the draw depend on clause order -- AUD-CODE 2.2),
    # so a rerun of this batch in any clause order reproduces it.
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
    assert all(gid.startswith(f"{plan_a.donor_group_id}#") for gid in group_ids)
    fork_three = log.fork()
    fork_three.do(
        [
            (log["relu_2_4"].__selection__(), tl.patch_from(plan_b)),
            (log["relu_1_2"].__selection__(), tl.patch_from(plan_a)),
        ]
    )
    reversed_ids = {row["donor_group_id"] for row in sampling_records(fork_three)[-2:]}
    assert reversed_ids == group_ids, "clause order never moves a donor group"


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


def _trace64() -> tl.Trace:
    torch.manual_seed(0)
    return tl.trace(
        _TwoBlock().double(),
        torch.randn(5, 4, dtype=torch.float64),
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )


def _cross_kind_pairs(fork: tl.Trace, order: tuple[str, ...], bad: str = "") -> list:
    """ACT, PARAM and EDGE pairs on ``fork`` in ``order``; ``bad`` names a refusing pair."""

    def _wrong_shape(tensor, *, ctx):  # noqa: ANN001, ANN202 - edit callable
        return tensor[:1]

    def _refuse(tensor, *, ctx):  # noqa: ANN001, ANN202 - edit callable
        raise InvalidArgumentError(
            "refused by the test edit", code="test_edit_refused", remedy="none"
        )

    edge = next(e for e in fork.graph.edges if e.parent_label == "relu_1_2")
    pairs = {
        "ACT": (fork["relu_1_2"].__selection__(), _refuse if bad == "ACT" else tl.scale(2.0)),
        "PARAM": (tl.param("b.weight"), _wrong_shape if bad == "PARAM" else tl.scale(0.25)),
        "EDGE": (edge.__selection__(), _wrong_shape if bad == "EDGE" else tl.add(0.25)),
    }
    return [pairs[kind] for kind in order]


def _fingerprint(log: tl.Trace) -> dict:
    """Everything a failed batch must leave as it was (values, spec, ledger rows)."""

    ops = {}
    for op in log.layer_list:
        store = op.edge_substitutions or {}
        ops[op.label] = (
            None if op.out is None else op.out.clone(),
            tuple(repr(record) for record in op.interventions or ()),
            bool(op.intervention_replaced),
            tuple(sorted(repr(key) for key in store)),
        )
    return {
        "ops": ops,
        "hooks": tuple(repr(spec) for spec in log._ensure_intervention_spec().hook_specs),
        "audit": tuple(repr(row) for row in log.intervention_audit),
        "history": tuple(repr(row) for row in log.state_history),
        "state": log.state,
    }


def _assert_same_values(left: dict, right: dict) -> None:
    assert left.keys() == right.keys()
    for label, (left_out, *left_rest) in left.items():
        right_out, *right_rest = right[label]
        assert (left_out is None) == (right_out is None), label
        if left_out is not None:
            assert torch.equal(left_out, right_out), label
        assert left_rest == right_rest, label


@pytest.mark.parametrize(
    "order", [("ACT", "PARAM", "EDGE"), ("EDGE", "ACT", "PARAM"), ("PARAM", "EDGE", "ACT")]
)
def test_cross_kind_batch_equals_the_doors_one_at_a_time(order: tuple[str, ...]) -> None:
    """A batch mixing ACT, PARAM and EDGE equals one do() per pair, in order, exactly."""

    log = _trace64()
    batch = log.fork()
    batch.do(_cross_kind_pairs(batch, order))
    sequential = log.fork()
    for selection, edit in _cross_kind_pairs(sequential, order):
        sequential.do(selection, edit)
    for batch_op, seq_op in zip(batch.layer_list, sequential.layer_list, strict=True):
        assert batch_op.label == seq_op.label
        assert batch_op.out.dtype == torch.float64
        assert torch.equal(batch_op.out, seq_op.out), batch_op.label
        assert sorted(map(repr, batch_op.edge_substitutions or {})) == sorted(
            map(repr, seq_op.edge_substitutions or {})
        )
    assert not torch.equal(batch[batch.output_layers[0]].out, log[log.output_layers[0]].out), (
        "the batch changed the output"
    )
    envelopes = _envelopes(batch)
    assert len(envelopes) == 1, "ONE transactional envelope for the whole batch"
    assert envelopes[0]["selection_batch"]["kinds"] == list(order)
    assert envelopes[0]["selection_batch"]["staged_stores"] == {"act": 1, "param": 1, "edge": 1}
    kinds = [row.get("kind") for row in batch.intervention_audit if row.get("kind") != "EVENT"]
    assert kinds == list(order), "one audit row per pair, in pair order"


@pytest.mark.parametrize("bad", ["ACT", "PARAM", "EDGE"])
def test_cross_kind_batch_with_a_refusing_pair_leaves_the_fork_as_before(bad: str) -> None:
    """Atomicity: the refusing pair runs last, after the other doors changed values."""

    order = tuple(kind for kind in ("ACT", "PARAM", "EDGE") if kind != bad) + (bad,)
    log = _trace64()
    fork = log.fork()
    before = _fingerprint(fork)
    probe = log.fork()
    probe.do(_cross_kind_pairs(probe, order[:2]))
    assert not torch.equal(probe[probe.output_layers[0]].out, log[log.output_layers[0]].out), (
        "the pairs before the refusal do change values, so the rollback is exercised"
    )
    with pytest.raises(Exception) as excinfo:
        fork.do(_cross_kind_pairs(fork, order, bad=bad))
    assert "code" in getattr(excinfo.value, "fields", {}), "a typed refusal"
    after = _fingerprint(fork)
    _assert_same_values(after["ops"], before["ops"])
    assert after["hooks"] == before["hooks"], "no hook leaks"
    assert after["state"] == before["state"]
    assert after["audit"][: len(before["audit"])] == before["audit"]
    assert len(after["audit"]) - len(before["audit"]) <= 1, "at most the error row"
    assert after["history"][: len(before["history"])] == before["history"]
    added = list(fork.state_history[len(before["history"]) :])
    assert [row.get("op") for row in added] == ["intervention_event"]
    assert added[0]["status"] == "error", "the fire-evidence rule: one aborted record"
    assert added[0]["selection_batch"]["kinds"] == list(order)
    fork.do(_cross_kind_pairs(fork, order[:2]))
    for fork_op, probe_op in zip(fork.layer_list, probe.layer_list, strict=True):
        assert torch.equal(fork_op.out, probe_op.out), "the fork is usable after the abort"


def test_cross_kind_batch_off_replay_refuses_before_anything_runs() -> None:
    """PARAM/EDGE edits ship on replay only: other engines refuse typed, nothing applied."""

    log = _trace64()
    fork = log.fork()
    before = _fingerprint(fork)
    with pytest.raises(InvalidArgumentError) as excinfo:
        fork.do(
            _cross_kind_pairs(fork, ("ACT", "PARAM")),
            intervention=tl.options.InterventionOptions(engine="set_only"),
        )
    assert excinfo.value.fields["code"] == "selection_batch_cross_kind"
    assert "engine='replay'" in excinfo.value.fields["remedy"]
    _assert_same_values(_fingerprint(fork)["ops"], before["ops"])


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
                (fork[ragged], permute_batch(seed=1, axis=0)),
            ]
        )
    assert excinfo.value.fields["code"] == "mask_not_row_equivariant"
    assert len(fork._ensure_intervention_spec().hook_specs) == hooks_before
