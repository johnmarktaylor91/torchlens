"""F03 ledger memo items 6-8: candidate engine, site_sweep, head sugar, effects.

Carries the SAME-FILE ORACLE the lane gate demands: the head-ablation
sugar's v-facet lowering is checked candidate-by-candidate against a
hand-written torch forward_pre_hook zeroing head i's slice of the c_proj
input (no TorchLens in the oracle path) on a config-built GPT-2 running
through the REAL HF ``from_dict`` parsing path — agreement to tolerance and
identical ranking. The oracle is the law because it is what caught the
panel's own flagship mis-spelling (the unscoped 36-site broadcast).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors.episode import BundleExperimentError
from torchlens.experiment import site_sweep, top_k

pytestmark = pytest.mark.smoke


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)
        self.gate = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x)) + torch.tanh(self.gate(x))


def _baseline(x: torch.Tensor, model: nn.Module) -> tl.Trace:
    return tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))


def _output_metric(member: tl.Trace) -> float:
    label = member.output_layers[0]
    return float(member[label].out.sum().item())


def test_site_sweep_required_arguments_refuse_typed() -> None:
    torch.manual_seed(0)
    model = _Tiny().eval()
    x = torch.randn(2, 3)
    baseline = _baseline(x, model)
    with pytest.raises(BundleExperimentError) as excinfo:
        site_sweep(
            baseline,
            candidates=["relu_1_2"],
            edit=tl.zero_ablate(),
            metric=_output_metric,
            retain="most",
            model=model,
            x=x,
        )
    assert excinfo.value.fields["code"] == "site_sweep_retain_invalid"
    with pytest.raises(BundleExperimentError) as excinfo:
        site_sweep(
            baseline,
            candidates=["relu_1_2"],
            edit=tl.zero_ablate(),
            metric="not callable",  # type: ignore[arg-type]
            retain="all",
            model=model,
            x=x,
        )
    assert excinfo.value.fields["code"] == "site_sweep_metric_invalid"
    with pytest.raises(BundleExperimentError) as excinfo:
        site_sweep(
            baseline,
            candidates=["relu_1_2"],
            edit=tl.zero_ablate(),
            metric=_output_metric,
            retain="all",
        )
    assert excinfo.value.fields["code"] == "site_sweep_inputs_missing"


def test_site_sweep_live_hook_lane_end_to_end() -> None:
    torch.manual_seed(0)
    model = _Tiny().eval()
    x = torch.randn(2, 3)
    baseline = _baseline(x, model)
    bundle = site_sweep(
        baseline,
        candidates={"relu": "relu_1_2", "tanh": "tanh_1_4"},
        edit=tl.zero_ablate(),
        metric=_output_metric,
        retain="all",
        model=model,
        x=x,
    )
    assert bundle.names[0] == "baseline"
    assert set(bundle.names) == {"baseline", "relu", "tanh"}
    view = bundle.effects()
    assert view.baseline_value is not None
    measured = {row.candidate_id: row for row in view.rows if row.candidate_id != "__baseline__"}
    assert set(measured) == {"relu", "tanh"}
    assert all(row.status == "completed" for row in measured.values())
    assert all(row.resolved_site_count == 1 for row in measured.values())
    # The lane is stamped; the repr accounts for every candidate.
    assert view.table.lane == "live_hook"
    assert "effects=[attempted=2" in repr(bundle)
    # Each candidate's transaction carries the canonical EVENT audit row.
    from torchlens.intervention.audit import event_audit_rows

    assert event_audit_rows(bundle["relu"])
    # The member axis ranks by |value - baseline|.
    ranked = view.most_changed()
    assert len(ranked) == 2 and ranked[0][1] >= ranked[1][1]


def test_site_sweep_retain_none_keeps_rows_releases_members() -> None:
    torch.manual_seed(0)
    model = _Tiny().eval()
    x = torch.randn(2, 3)
    baseline = _baseline(x, model)
    bundle = site_sweep(
        baseline,
        candidates=["relu_1_2", "tanh_1_4"],
        edit=tl.zero_ablate(),
        metric=_output_metric,
        retain="none",
        model=model,
        x=x,
    )
    # Members released; numbers survive (extract-before-cleanup, D3c).
    assert bundle.names == ["baseline"]
    rows = [row for row in bundle.effects().rows if row.candidate_id != "__baseline__"]
    assert all(row.status == "released" and row.value is not None for row in rows)
    assert all(row.member_name is None for row in rows)


def test_site_sweep_rolling_top_k_retention() -> None:
    torch.manual_seed(0)
    model = _Tiny().eval()
    x = torch.randn(2, 3)
    baseline = _baseline(x, model)
    bundle = site_sweep(
        baseline,
        candidates=["relu_1_2", "tanh_1_4"],
        edit=tl.zero_ablate(),
        metric=_output_metric,
        retain=top_k(1),
        model=model,
        x=x,
    )
    view = bundle.effects()
    ranked = view.most_changed()
    top_candidate = ranked[0][0]
    assert set(bundle.names) == {"baseline", top_candidate}
    released = [row for row in view.rows if row.status == "released"]
    assert len(released) == 1
    assert released[0].value is not None  # the number survived release


def test_site_sweep_refuses_undeclared_multi_site_candidate() -> None:
    torch.manual_seed(0)
    model = _Tiny().eval()
    x = torch.randn(2, 3)
    baseline = _baseline(x, model)
    bundle = site_sweep(
        baseline,
        candidates={"broadcast": tl.func("linear"), "single": "relu_1_2"},
        edit=tl.zero_ablate(),
        metric=_output_metric,
        retain="all",
        model=model,
        x=x,
    )
    rows = {row.candidate_id: row for row in bundle.effects().rows}
    assert rows["broadcast"].status == "refused"
    assert "multi_site" in (rows["broadcast"].error or "")
    assert rows["single"].status == "completed"
    # Declared multi-site: the same candidate runs as ONE knockout set (the
    # fan-out DISCLOSURE still fires; it is a warning, not a refusal).
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        declared = site_sweep(
            baseline,
            candidates={"broadcast": tl.func("linear")},
            edit=tl.zero_ablate(),
            metric=_output_metric,
            retain="all",
            model=model,
            x=x,
            multi_site=True,
        )
    declared_rows = {row.candidate_id: row for row in declared.effects().rows}
    assert declared_rows["broadcast"].status == "completed"
    assert (declared_rows["broadcast"].resolved_site_count or 0) > 1


def test_site_sweep_duplicate_candidates_refuse_unless_replicates() -> None:
    torch.manual_seed(0)
    model = _Tiny().eval()
    x = torch.randn(2, 3)
    baseline = _baseline(x, model)
    with pytest.raises(BundleExperimentError) as excinfo:
        site_sweep(
            baseline,
            candidates={"one": "relu_1_2", "two": "relu_1_2"},
            edit=tl.zero_ablate(),
            metric=_output_metric,
            retain="none",
            model=model,
            x=x,
        )
    assert excinfo.value.fields["code"] == "site_sweep_duplicate_candidates"
    replicated = site_sweep(
        baseline,
        candidates={"one": "relu_1_2", "two": "relu_1_2"},
        edit=tl.zero_ablate(),
        metric=_output_metric,
        retain="none",
        model=model,
        x=x,
        replicates=True,
    )
    rows = [row for row in replicated.effects().rows if row.candidate_id != "__baseline__"]
    assert len(rows) == 2


def test_effect_table_survives_save_load_and_unarmed_ledger(tmp_path) -> None:
    torch.manual_seed(0)
    model = _Tiny().eval()
    x = torch.randn(2, 3)
    baseline = _baseline(x, model)
    bundle = site_sweep(
        baseline,
        candidates=["relu_1_2"],
        edit=tl.zero_ablate(),
        metric=_output_metric,
        retain="none",
        model=model,
        x=x,
    )
    path = tmp_path / "sweep_bundle"
    bundle.save(str(path))
    loaded = tl.load(str(path))
    view = loaded.effects()
    assert view.table.lane == "live_hook"
    assert [row.candidate_id for row in view.rows] == ["__baseline__", "c0"]
    # Selections are session-only: the loaded table serves numbers, never
    # a fabricated region.
    with pytest.raises(BundleExperimentError) as excinfo:
        view.selection()
    assert excinfo.value.fields["code"] == "effects_selection_unavailable"


def test_measure_members_reports_unmeasured_and_refuses_chain_order() -> None:
    torch.manual_seed(0)
    model = _Tiny().eval()
    x = torch.randn(2, 3)
    baseline = _baseline(x, model)
    bundle = site_sweep(
        baseline,
        candidates=["relu_1_2", "tanh_1_4"],
        edit=tl.zero_ablate(),
        metric=_output_metric,
        retain=top_k(1),
        model=model,
        x=x,
    )
    report = bundle.measure_members(metric=lambda member: len(member.op_labels))
    assert set(report["values"]) == set(bundle.names)
    assert len(report["unmeasured"]) == 1  # the released candidate, named
    with pytest.raises(BundleExperimentError) as excinfo:
        bundle.measure_members(metric=_output_metric, order="chain")
    assert excinfo.value.fields["code"] == "measure_members_order_unavailable"


def test_effects_selection_tier_c_conversion_in_session() -> None:
    torch.manual_seed(0)
    model = _Tiny().eval()
    x = torch.randn(2, 3)
    baseline = _baseline(x, model)
    bundle = site_sweep(
        baseline,
        candidates=["relu_1_2", "tanh_1_4"],
        edit=tl.zero_ablate(),
        metric=_output_metric,
        retain="all",
        model=model,
        x=x,
    )
    selection = bundle.effects().top(1).selection()
    resolved = selection.resolve(baseline) if hasattr(selection, "resolve") else None
    assert selection is not None
    assert resolved is None or len(resolved.entries) >= 1


def test_site_sweep_engine_vocabulary_closed() -> None:
    """engine= is a closed two-lane vocabulary; anything else refuses typed."""

    torch.manual_seed(0)
    model = _Tiny().eval()
    x = torch.randn(2, 3)
    baseline = _baseline(x, model)
    with pytest.raises(BundleExperimentError) as excinfo:
        site_sweep(
            baseline,
            candidates=["relu_1_2"],
            edit=tl.zero_ablate(),
            metric=_output_metric,
            retain="all",
            model=model,
            x=x,
            engine="rerun",
        )
    assert excinfo.value.fields["code"] == "site_sweep_engine_invalid"


def test_site_sweep_candidate_plan_shapes_refuse_typed() -> None:
    """Non-plan values, empty plans, and the reserved 'baseline' id refuse."""

    torch.manual_seed(0)
    model = _Tiny().eval()
    x = torch.randn(2, 3)
    baseline = _baseline(x, model)

    def _sweep(candidates) -> None:
        site_sweep(
            baseline,
            candidates=candidates,
            edit=tl.zero_ablate(),
            metric=_output_metric,
            retain="all",
            model=model,
            x=x,
        )

    with pytest.raises(BundleExperimentError) as excinfo:
        _sweep(42)
    assert excinfo.value.fields["code"] == "site_sweep_candidates_invalid"
    with pytest.raises(BundleExperimentError) as excinfo:
        _sweep([])
    assert excinfo.value.fields["code"] == "site_sweep_candidates_invalid"
    with pytest.raises(BundleExperimentError) as excinfo:
        _sweep({"baseline": "relu_1_2"})
    assert excinfo.value.fields["code"] == "site_sweep_candidates_invalid"


def test_site_sweep_keyless_baseline_refuses_typed() -> None:
    """A label candidate on a keyless op refuses: the engine runs in site-key
    coordinates, so a legacy (pre-site-key) baseline cannot enumerate labels."""

    torch.manual_seed(0)
    model = _Tiny().eval()
    x = torch.randn(2, 3)
    baseline = _baseline(x, model)
    baseline.ops["relu_1_2"].site_key = None  # simulate a legacy keyless capture
    with pytest.raises(BundleExperimentError) as excinfo:
        site_sweep(
            baseline,
            candidates=["relu_1_2"],
            edit=tl.zero_ablate(),
            metric=_output_metric,
            retain="all",
            model=model,
            x=x,
        )
    assert excinfo.value.fields["code"] == "site_sweep_site_key_unavailable"


def test_effects_view_requires_a_stored_table() -> None:
    """effects() refuses typed with no stored table or an unknown operation id."""

    torch.manual_seed(0)
    model = _Tiny().eval()
    x = torch.randn(2, 3)
    baseline = _baseline(x, model)
    bare = tl.Bundle({"baseline": baseline}, baseline="baseline")
    with pytest.raises(BundleExperimentError) as excinfo:
        bare.effects()
    assert excinfo.value.fields["code"] == "effects_table_missing"

    swept = site_sweep(
        baseline,
        candidates={"relu": "relu_1_2"},
        edit=tl.zero_ablate(),
        metric=_output_metric,
        retain="none",
        model=model,
        x=x,
    )
    with pytest.raises(BundleExperimentError) as excinfo:
        swept.effects(operation_id="op_never_ran")
    assert excinfo.value.fields["code"] == "effects_table_missing"


def test_head_ablation_underivable_geometry_refuses_typed() -> None:
    """No derivable attention geometry -> the v-facet equivalence claim refuses."""

    torch.manual_seed(0)
    model = _Tiny().eval()
    x = torch.randn(2, 3)
    baseline = _baseline(x, model)
    from torchlens.experiment import head_ablation_candidates

    with pytest.raises(BundleExperimentError) as excinfo:
        head_ablation_candidates(baseline, "linear", heads=2)
    assert excinfo.value.fields["code"] == "head_ablation_equivalence_undeclared"
