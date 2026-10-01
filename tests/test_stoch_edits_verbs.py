"""F02 stochastic verbs: draw law, axis law, oracles (edits memo D4-D9, D16-D18, D23-D25).

Tier-A rows exercised here (offline analogs of the memo's real-model rows,
oracle-backed, deterministic):

- A2 core: ``permute_batch`` against an independent index oracle from the
  recorded permutation; byte-identical repeat under one seed; batch-of-one
  refusal; the gradient leg (gradients route to permuted source rows,
  graph-connected).
- A2b axis law: no axis refuses; a wrong-rank axis refuses at fire.
- A3: mask-equivariance -- an all-rows-equal span mask under ``permute_batch``
  succeeds with the mask facts recorded; per-row-differing masks refuse,
  stably across seeds.
- A4b analog: the draw-independence matrix on a REAL weight-tied two-pass
  module -- ``per_firing`` pass draws differ, ``per_rule`` draws are equal,
  both byte-stable across a re-push.
- A1/A1b analog: ``mean_fill`` axis semantics vs the legacy global scalar;
  typo'd and named ``over=`` tokens refuse; ``batch_independent`` is derived.
- A8: eager derivation equality -- the recorded derived seed re-derives from
  the recorded coordinate.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention import (
    OneDatum,
    PerRowDatums,
    make_hook_context,
    mean_fill,
    mean_from,
    permute_batch,
    reference,
    resample_rows_from,
    sample_from,
    sampling_records,
    set_direction_mean,
)
from torchlens.intervention.stochastic import _derived_seed


class _Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))


class _Tied(nn.Module):
    """Weight-tied module called twice: a REAL two-pass recurrence."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(torch.relu(self.lin(x)))


def _toy_trace() -> tl.Trace:
    torch.manual_seed(0)
    model = _Toy()
    x = torch.randn(5, 4)
    return tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))


@pytest.mark.smoke
def test_permute_batch_oracle_reproducibility_and_extent() -> None:
    """A2 core: recorded-permutation oracle + seed law + extent refusal."""

    log = _toy_trace()
    baseline = log["relu_1_2"].out
    fork = log.fork()
    fork.do(tl.label("relu_1_2"), permute_batch(seed=11, axis=0))
    record = sampling_records(fork)[-1]
    permutation = record["permutation"]
    assert sorted(permutation) == list(range(5)), "every output row is some input row"
    oracle = baseline[torch.tensor(permutation)]
    assert torch.equal(fork["relu_1_2"].out, oracle)
    # Byte-identical repeat under the same seed; a different seed differs.
    again = log.fork()
    again.do(tl.label("relu_1_2"), permute_batch(seed=11, axis=0))
    assert sampling_records(again)[-1]["permutation"] == permutation
    other = log.fork()
    other.do(tl.label("relu_1_2"), permute_batch(seed=12, axis=0))
    assert sampling_records(other)[-1]["permutation"] != permutation
    # Batch extent 1 refuses at fire.
    torch.manual_seed(0)
    single = tl.trace(
        _Toy(), torch.randn(1, 4), capture=tl.options.CaptureOptions(intervention_ready=True)
    )
    with pytest.raises(Exception) as excinfo:
        single.fork().do(tl.label("relu_1_2"), permute_batch(seed=1, axis=0))
    assert excinfo.value.fields["code"] == "permute_batch_extent_invalid"


@pytest.mark.smoke
def test_permute_batch_axis_law() -> None:
    """A2b: no axis refuses at construction; wrong-rank axis refuses at fire."""

    with pytest.raises(Exception) as excinfo:
        permute_batch(seed=1)
    assert excinfo.value.fields["code"] == "axis_semantics_unknown"
    log = _toy_trace()
    with pytest.raises(Exception) as excinfo:
        log.fork().do(tl.label("relu_1_2"), permute_batch(seed=1, axis=5))
    assert excinfo.value.fields["code"] == "sampling_geometry_mismatch"


@pytest.mark.smoke
def test_permute_batch_gradients_route_to_permuted_sources() -> None:
    """A2 gradient leg: graph-connected index_select, never a detach."""

    edit = permute_batch(seed=3, axis=0)
    assert edit.factory is not None
    hook_fn = edit.factory()
    x = torch.randn(4, 3, requires_grad=True)
    run_ctx: dict[str, object] = {}
    context = make_hook_context(
        name="permute_batch",
        layer_log={"label": "unit", "pass_index": 1},
        run_ctx=run_ctx,
    )
    out = hook_fn(x, hook=context)
    permutation = run_ctx["sampling_records"][-1]["permutation"]  # type: ignore[index]
    weights = torch.tensor([1.0, 10.0, 100.0, 1000.0]).reshape(4, 1)
    (out * weights).sum().backward()
    # Row j of the output is row permutation[j] of the input, so input row
    # permutation[j] receives weight j's gradient.
    expected = torch.zeros(4, 3)
    for out_row, in_row in enumerate(permutation):  # type: ignore[arg-type]
        expected[in_row] = weights[out_row]
    assert x.grad is not None and torch.equal(x.grad, expected)


@pytest.mark.smoke
def test_draw_independence_matrix_on_real_two_pass_recurrence() -> None:
    """A4b analog: per_firing pass draws DIFFER, per_rule draws are EQUAL.

    The one row the shipped library failed both ways: unseeded edits were
    re-rolled by unrelated pushes, seeded ones drew identically at every pass.
    """

    torch.manual_seed(0)
    model = _Tied()
    x = torch.randn(6, 4)
    log = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    layer = next(lab for lab in log.layer_labels if lab.startswith("linear"))
    assert log[layer].num_passes == 2, "the fixture must be a real two-pass layer"

    donors = reference(torch.randn(8, *log[f"{layer}:1"].out.shape), origin="donor stack")
    per_firing = sample_from(donors, seed=13, share_draw="per_firing")
    fork = log.fork()
    # The explicit Layer selection is the all-passes spelling (one rule, two
    # pass firings) -- exactly the treeified-scrub shape A4b pins.
    fork.do(fork[layer].__selection__(), tl.patch_from(per_firing))
    records = [r for r in sampling_records(fork) if r["share_draw"] == "per_firing"]
    assert len(records) == 2, "one draw per pass firing"
    assert records[0]["derived_seed"] != records[1]["derived_seed"]
    assert records[0]["logical_firing_coordinate"] != records[1]["logical_firing_coordinate"]

    per_rule = sample_from(donors, seed=13, share_draw="per_rule")
    fork2 = log.fork()
    fork2.do(fork2[layer].__selection__(), tl.patch_from(per_rule))
    records2 = [r for r in sampling_records(fork2) if r["share_draw"] == "per_rule"]
    assert len(records2) == 2
    assert records2[0]["derived_seed"] == records2[1]["derived_seed"]
    assert records2[0]["donor_ids"] == records2[1]["donor_ids"]

    # Byte-stability across a re-push (idempotent logical coordinates).
    before = [r["derived_seed"] for r in sampling_records(fork)]
    fork.push()
    after = [r["derived_seed"] for r in sampling_records(fork) if r["share_draw"] == "per_firing"]
    assert after[-2:] == before[-2:]


@pytest.mark.smoke
def test_derived_seed_is_rederivable_from_recorded_facts() -> None:
    """A8: the recorded derived seed equals the eager re-derivation."""

    log = _toy_trace()
    fork = log.fork()
    fork.do(tl.label("relu_1_2"), permute_batch(seed=21, axis=0))
    record = sampling_records(fork)[-1]
    rederived = _derived_seed(
        record["base_seed"],
        population_identity=record["population_identity"],
        donor_group_id=record["donor_group_id"],
        coordinate=tuple(record["logical_firing_coordinate"]),
        leaf_path=(),
    )
    assert rederived == record["derived_seed"]


@pytest.mark.smoke
def test_seed_is_required_and_auto_derives_from_trace_seed() -> None:
    """D4/D5: seed=None refuses; 'auto' canonicalizes against trace.random_seed."""

    with pytest.raises(Exception) as excinfo:
        permute_batch(seed=None, axis=0)  # type: ignore[arg-type]
    assert excinfo.value.fields["code"] == "sampling_seed_required"
    log = _toy_trace()
    fork_a = log.fork()
    fork_a.do(tl.label("relu_1_2"), permute_batch(seed="auto", axis=0))
    fork_b = log.fork()
    fork_b.do(tl.label("relu_1_2"), permute_batch(seed="auto", axis=0))
    rec_a = sampling_records(fork_a)[-1]
    rec_b = sampling_records(fork_b)[-1]
    assert rec_a["base_seed"] == rec_b["base_seed"], "auto base is capture-derived, not ambient"
    assert rec_a["permutation"] == rec_b["permutation"]
    # No run context -> no trace seed -> typed refusal, never ambient RNG.
    edit = permute_batch(seed="auto", axis=0)
    assert edit.factory is not None
    context = make_hook_context(name="p", layer_log={"label": "u", "pass_index": 1})
    with pytest.raises(Exception) as excinfo:
        edit.factory()(torch.randn(3, 2), hook=context)
    assert excinfo.value.fields["code"] == "sampling_trace_seed_unavailable"


@pytest.mark.smoke
def test_mask_equivariance_law() -> None:
    """A3: uniform span masks pass with recorded facts; ragged masks refuse."""

    log = _toy_trace()
    baseline = log["relu_1_2"].out
    span = tl.units("relu_1_2", [(row, col) for row in range(5) for col in (1, 2)])
    fork = log.fork()
    fork.do(span.resolve(fork), permute_batch(seed=4, axis=0))
    record = sampling_records(fork)[-1]
    permutation = torch.tensor(record["permutation"])
    expected = baseline.clone()
    expected[:, 1:3] = baseline[permutation][:, 1:3]
    assert torch.allclose(fork["relu_1_2"].out, expected)
    fire = list(fork.ops["relu_1_2"].interventions or ())[-1]
    assert fire.helper is not None
    metadata = dict(fire.helper.metadata)
    assert "selection_mask_digest" in metadata
    assert metadata["selection_mask_rows_equal"] is True
    # Per-row-differing masks refuse typed, stably across seeds.
    ragged = tl.units("relu_1_2", [(0, 1), (1, 2), (2, 3)])
    for seed in (1, 99):
        fork_bad = log.fork()
        with pytest.raises(Exception) as excinfo:
            fork_bad.do(ragged.resolve(fork_bad), permute_batch(seed=seed, axis=0))
        assert excinfo.value.fields["code"] == "mask_not_row_equivariant"


@pytest.mark.smoke
def test_resample_rows_from_per_row_coherent_draws() -> None:
    """Whole-row coherent donor sampling with replacement, agreement-aware."""

    log = _toy_trace()
    rows = reference(torch.randn(3, 4), origin="row donors", data=["a", "a", "b"])
    fork = log.fork()
    # Class 'b' deliberately has ONE member: the D15 disclosure must fire.
    with pytest.warns(UserWarning, match="agreement class of size 1"):
        fork.do(
            tl.label("relu_1_2"),
            resample_rows_from(
                rows,
                seed=2,
                axis=0,
                group_by=lambda d: d,
                matching=PerRowDatums(["a", "a", "b", "b", "a"]),
            ),
        )
    record = sampling_records(fork)[-1]
    donor_ids = record["donor_ids"]
    assert len(donor_ids) == 5
    # Agreement law: rows matched to class 'a' draw members {0,1}; 'b' draws {2}.
    for row, key in enumerate(["a", "a", "b", "b", "a"]):
        assert donor_ids[row] in ((0, 1) if key == "a" else (2,))
    expected = torch.stack([rows.members[index] for index in donor_ids], dim=0)
    assert torch.equal(fork["relu_1_2"].out, expected)
    # Geometry proof: row donors of the wrong shape refuse printing both shapes.
    bad = reference(torch.randn(3, 7), origin="bad rows")
    with pytest.raises(Exception) as excinfo:
        log.fork().do(tl.label("relu_1_2"), resample_rows_from(bad, seed=2, axis=0))
    assert excinfo.value.fields["code"] == "sampling_geometry_mismatch"


@pytest.mark.smoke
def test_singleton_agreement_class_discloses_and_strict_refuses() -> None:
    """D15: class of one -> coded warning by default, refusal under strict."""

    log = _toy_trace()
    rows = reference(torch.randn(2, 4), origin="rows", data=["a", "b"])
    fork = log.fork()
    with pytest.warns(UserWarning, match="agreement class of size 1") as disclosed:
        fork.do(
            tl.label("relu_1_2"),
            resample_rows_from(rows, seed=3, axis=0, group_by=lambda d: d, matching=OneDatum("a")),
        )
    assert any(
        getattr(record.message, "fields", {}).get("code") == "sampling_agreement_class_singleton"
        for record in disclosed
    ), "the S-18 disclosure carries its contract code"
    with pytest.raises(Exception) as excinfo:
        log.fork().do(
            tl.label("relu_1_2"),
            resample_rows_from(
                rows, seed=3, axis=0, group_by=lambda d: d, matching=OneDatum("a"), strict=True
            ),
        )
    assert excinfo.value.fields["code"] == "sampling_agreement_class_too_small"


@pytest.mark.smoke
def test_mean_fill_axis_semantics_and_derived_flag() -> None:
    """A1/A1b analog: real over= vocabulary; batch_independent DERIVED."""

    log = _toy_trace()
    baseline = log["relu_1_2"].out
    fork = log.fork()
    fork.do(tl.label("relu_1_2"), mean_fill(over="all"))
    filled = fork["relu_1_2"].out
    assert torch.allclose(filled, torch.zeros_like(baseline) + baseline.mean())
    assert filled.unique().numel() == 1, "over='all' is the explicit global scalar"
    fork2 = log.fork()
    fork2.do(tl.label("relu_1_2"), mean_fill(over=0))
    axis_filled = fork2["relu_1_2"].out
    assert torch.allclose(axis_filled, torch.zeros_like(baseline) + baseline.mean(0, keepdim=True))
    assert axis_filled[0].unique().numel() > 1, "axis-aware fill varies per feature"
    # The one line pinned in both failure directions (A1b):
    with pytest.raises(Exception) as excinfo:
        mean_fill(over="btach")
    assert excinfo.value.fields["code"] == "intervention_over_invalid"
    with pytest.raises(Exception) as excinfo:
        mean_fill(over="batch")
    assert excinfo.value.fields["code"] == "axis_semantics_unknown"
    # D18: derived, never tabled. Self-sourced reads the traced batch.
    assert mean_fill(over="all").batch_independent is False
    donors = reference(torch.randn(4, *baseline.shape), origin="pop")
    assert mean_fill(donors, over="all").batch_independent is True
    # 'self' warns and maps to 'all' (one migration window).
    with pytest.warns(UserWarning, match="over='all'") as aliased_warnings:
        aliased = mean_fill(over="self")
    assert dict(aliased.kwargs)["over"] == "all"
    assert any(
        getattr(record.message, "fields", {}).get("code") == "mean_over_self_alias"
        for record in aliased_warnings
    ), "the S-18 alias disclosure carries its contract code"


@pytest.mark.smoke
def test_trace_backed_population_reduce_refuses() -> None:
    """D3: reductions need a tensor stack; trace members have none to reduce."""

    log = _toy_trace()
    donors = reference([log], origin="one captured run")
    with pytest.raises(Exception) as excinfo:
        mean_from(donors)
    assert excinfo.value.fields["code"] == "population_reduce_unsupported"


@pytest.mark.smoke
def test_mean_from_evidence_set_fill() -> None:
    """Deterministic elementwise evidence mean; stochastic=False recorded."""

    log = _toy_trace()
    baseline = log["relu_1_2"].out
    donors = reference(torch.stack([baseline, baseline + 2.0]), origin="evidence")
    fork = log.fork()
    fork.do(tl.label("relu_1_2"), mean_from(donors))
    assert torch.allclose(fork["relu_1_2"].out, baseline + 1.0)
    record = sampling_records(fork)[-1]
    assert record["stochastic"] is False
    assert record["base_seed"] is None, "a mean never invents a seed"


@pytest.mark.smoke
def test_set_direction_mean_projects_and_recenters() -> None:
    """out - proj_v(out) + mean(coef(ref)) * v_hat, gradients orthogonal-only."""

    log = _toy_trace()
    baseline = log["relu_1_2"].out
    donors = reference(torch.stack([baseline, baseline + 1.0]), origin="coef evidence")
    direction = torch.randn(4)
    edit = set_direction_mean(direction, donors, feature_axis=1)
    fork = log.fork()
    fork.do(tl.label("relu_1_2"), edit)
    unit = direction / direction.norm()
    projection = (baseline * unit).sum(1, keepdim=True) * unit
    mean_coef = dict(edit.kwargs)["mean_coefficient"]
    assert torch.allclose(fork["relu_1_2"].out, baseline - projection + mean_coef * unit, atol=1e-5)
    # Geometry refusals: direction rank and extent.
    with pytest.raises(Exception) as excinfo:
        set_direction_mean(torch.randn(2, 2), donors, feature_axis=1)
    assert excinfo.value.fields["code"] == "direction_vector_invalid"
    with pytest.raises(Exception) as excinfo:
        set_direction_mean(torch.randn(9), donors, feature_axis=1)
    assert excinfo.value.fields["code"] == "sampling_geometry_mismatch"


@pytest.mark.smoke
def test_patch_from_plan_geometry_and_matching_laws() -> None:
    """D9 whole-event geometry proof + whole-event matching carrier law."""

    log = _toy_trace()
    baseline = log["relu_1_2"].out
    wrong = reference(torch.randn(3, 9), origin="wrong shape")
    plan = sample_from(wrong, seed=5)
    with pytest.raises(Exception) as excinfo:
        log.fork().do(tl.label("relu_1_2"), tl.patch_from(plan))
    assert excinfo.value.fields["code"] == "sampling_geometry_mismatch"
    donors = reference(torch.stack([baseline + 1, baseline + 2]), origin="ok")
    with pytest.raises(Exception) as excinfo:
        sample_from(donors, seed=5, agree_on=lambda d: d, matching=PerRowDatums(["a"] * 5))
    assert excinfo.value.fields["code"] == "population_matching_invalid"
    with pytest.raises(Exception) as excinfo:
        sample_from(donors, seed=5, matching=OneDatum("a"))
    assert excinfo.value.fields["code"] == "population_matching_invalid"
    with pytest.raises(Exception) as excinfo:
        sample_from(donors, seed=5, share_draw="sometimes")
    assert excinfo.value.fields["code"] == "sampling_share_draw_invalid"


@pytest.mark.smoke
def test_chunked_capture_refuses_batch_coherent_edits() -> None:
    """D34: a chunked-forward capture refuses whole-batch edits at attach."""

    log = _toy_trace()
    fork = log.fork()
    fork.chunked_forward = True
    with pytest.raises(Exception) as excinfo:
        fork.do(tl.label("relu_1_2"), permute_batch(seed=1, axis=0))
    assert excinfo.value.fields["code"] == "chunked_replay_batch_coupling"
    with pytest.raises(Exception) as excinfo:
        fork.do(tl.label("relu_1_2"), mean_fill(over="all"))
    assert excinfo.value.fields["code"] == "chunked_replay_batch_coupling"
    # Per-row edits never couple rows and stay legal on chunked captures.
    rows = reference(torch.randn(2, 4), origin="rows")
    fork.do(tl.label("relu_1_2"), resample_rows_from(rows, seed=1, axis=0))


@pytest.mark.smoke
def test_fire_record_carries_the_sampling_note_via_one_builder() -> None:
    """D12: realized draws ride determinism_note through the ONE builder."""

    log = _toy_trace()
    fork = log.fork()
    fork.do(tl.label("relu_1_2"), permute_batch(seed=11, axis=0))
    fire = list(fork.ops["relu_1_2"].interventions or ())[-1]
    note = fire.determinism_note or ""
    assert "sampling[permute_batch]" in note
    assert "base_seed=11" in note
    assert "draw_digest=" in note
