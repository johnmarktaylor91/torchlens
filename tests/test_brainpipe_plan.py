"""Extraction planner, plan table, run door, and npz oracles (F20).

The brainpipe centerpiece (memo section 3): two real probes, per-site
linear byte fits with batch-invariant detection, budget arithmetic that
refuses instead of shrinking (D-10), full-signature plan keying (D-15),
the trace-only engine gate (D-11), the run door onto the shipped
extraction artifact machinery (D-8 single-pass-primary), and D-21's npz
interchange with the Net2Brain lexicographic file-naming contract.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from torchlens import brainpipe as bp

pytestmark = pytest.mark.smoke  # measured <0.5s per test (W051-GATE, AUD-CODE 0.1)


class _BatchInvariantModel(nn.Module):
    """Model with one batch-invariant site (a broadcast constant)."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(16, 8)
        self.register_buffer("scale_row", torch.ones(8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.scale_row, dim=0)  # batch-invariant site
        return self.linear(x) * weights


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8)).eval()


def _plan(**overrides):
    kwargs = {"n_stimuli": 20, "batch_size": 5}
    kwargs.update(overrides)
    return bp.extraction_plan(_model(), torch.randn(4, 16), **kwargs)


def test_parse_bytes_units_and_refusals() -> None:
    """D-26 human-unit budgets: decimal and binary units, typed refusals."""

    assert bp.parse_bytes("2GB") == 2 * 10**9
    assert bp.parse_bytes("1 GiB") == 2**30
    assert bp.parse_bytes("512 MB") == 512 * 10**6
    assert bp.parse_bytes(4096) == 4096
    for bad in ("fast", "-2GB", 0, True):
        with pytest.raises(Exception) as excinfo:
            bp.parse_bytes(bad)
        assert excinfo.value.fields["code"] == "byte_budget_invalid"


def test_two_probe_plan_measures_sites_and_prices_passes() -> None:
    """The plan holds every observed site with fitted bytes + exact passes."""

    plan = _plan()
    labels = {site.label for site in plan.sites}
    assert {"linear_1_1", "relu_1_2", "linear_2_3"} <= labels
    assert plan.planned_passes == 3 + 4  # warm-up + two probes + ceil(20/5) batches
    linear_site = next(s for s in plan.sites if s.label == "linear_1_1")
    # 5 stimuli x 32 features x 4 bytes at the run batch.
    assert linear_site.bytes_at_batch(5) == 5 * 32 * 4
    assert not linear_site.batch_invariant


def test_batch_invariant_site_is_detected_not_total_scaled() -> None:
    """D-15: a site whose bytes ignore batch is flagged and priced flat."""

    model = _BatchInvariantModel().eval()
    plan = bp.extraction_plan(model, torch.randn(4, 16), n_stimuli=64, batch_size=16)
    invariant = [site for site in plan.sites if site.batch_invariant]
    assert invariant, "the broadcast-constant site must be detected"
    site = invariant[0]
    assert site.bytes_at_batch(16) == site.bytes_at_batch(4)


def test_plan_table_prints_measured_footer() -> None:
    """The table carries passes, the peak pair, and budget lines."""

    plan = _plan(memory_budget="4 GiB")
    table = plan.table()
    assert "planned passes: 7" in table
    assert "probe peak pairs" in table
    assert "memory budget: 4.0 GiB" in table
    assert "engine: trace" in table
    # Object-first surfaces exist.
    as_dict = plan.to_dict()
    assert as_dict["planned_passes"] == 7
    assert as_dict["sites"]


def test_engine_gate_is_typed(recwarn) -> None:
    """D-11: fastlog refuses typed until the parity suite exists."""

    with pytest.raises(Exception) as excinfo:
        _plan(engine="fastlog")
    err = excinfo.value
    assert err.fields["code"] == "extraction_engine_unsupported"
    assert "parity" in str(err)


def test_signature_mismatch_refuses_typed() -> None:
    """D-15: a drifted run request refuses; it never silently over-runs."""

    model = _model()  # held alive: the signature door is under test here
    plan = bp.extraction_plan(model, torch.randn(4, 16), n_stimuli=20, batch_size=5)
    with pytest.raises(Exception) as excinfo:
        plan.run(torch.randn(20, 17), progress=False)
    assert excinfo.value.fields["code"] == "extraction_plan_signature_mismatch"


def test_over_budget_plan_refuses_with_arithmetic() -> None:
    """D-10: the refusal names the numbers and the batch-size lever."""

    with pytest.raises(Exception) as excinfo:
        _plan(memory_budget=1024)  # 1 KiB: absurdly small on purpose
    err = excinfo.value
    assert err.fields["code"] == "extraction_plan_over_budget"
    assert "batch" in str(err)


def test_plan_run_executes_through_the_artifact_runner() -> None:
    """D-8 single-pass-primary: run() extracts every included site once."""

    model = _model()
    plan = bp.extraction_plan(
        model,
        torch.randn(4, 16),
        n_stimuli=20,
        batch_size=5,
        transform=lambda t: t.mean(dim=-1, keepdim=True),
    )
    result = plan.run(torch.randn(20, 16), progress=False)
    assert set(result) == {site.label for site in plan.included_sites}
    for value in result.values():
        assert value.shape == (20, 1)


def test_plan_holds_model_weakly() -> None:
    """The plan itself never adds a strong model reference.

    (TorchLens session registries may keep prepared models alive
    independently, so outright death is not asserted here; the
    ``extraction_plan_model_gone`` refusal stays as the defensive door.)
    """

    import weakref

    model = _model()
    plan = bp.extraction_plan(model, torch.randn(4, 16), n_stimuli=20, batch_size=5)
    assert isinstance(plan._model_ref, weakref.ref)
    assert plan._model_ref() is model


def test_unknown_requested_site_is_excluded_actionably() -> None:
    """Never silently dropped: unknown sites carry an exclusion reason."""

    plan = _plan(sites=["linear_1_1", "nonexistent_9"])
    by_label = {site.label: site for site in plan.sites}
    assert by_label["linear_1_1"].included
    assert not by_label["nonexistent_9"].included
    assert "not observed" in by_label["nonexistent_9"].exclusion_reason


def test_npz_per_stimulus_net2brain_naming_contract(tmp_path) -> None:
    """D-21: filenames sort lexicographically in stimulus order + sidecar."""

    features = {
        "siteA": torch.arange(24, dtype=torch.float32).reshape(12, 2),
        "siteB": torch.arange(36, dtype=torch.float32).reshape(12, 3),
    }
    ids = [f"img_{i}" for i in range(12)]
    paths = bp.export_npz(
        features,
        tmp_path / "n2b",
        layout="per_stimulus",
        compatibility="net2brain",
        stimulus_ids=ids,
    )
    names = [p.name for p in paths]
    assert names == sorted(names), "lexicographic order must equal stimulus order"
    # Round-trip: glob+sort (the Net2Brain loader behavior) realigns rows.
    reread = []
    for file_path in sorted((tmp_path / "n2b").glob("stimulus_*.npz")):
        with np.load(file_path, allow_pickle=False) as data:
            reread.append(float(data["siteA"][0]))
    assert reread == [float(features["siteA"][i, 0]) for i in range(12)]
    sidecar = json.loads((tmp_path / "n2b" / "manifest.json").read_text())
    assert sidecar["name_to_stimulus_id"]["stimulus_00000.npz"] == "img_0"
    assert sidecar["compatibility"] == "net2brain"


def test_npz_consolidated_roundtrip(tmp_path) -> None:
    """Consolidated layout: one npz per run plus a provenance sidecar."""

    features = {"siteA": torch.ones(4, 3)}
    paths = bp.export_npz(features, tmp_path / "all.npz")
    with np.load(paths[0], allow_pickle=False) as data:
        assert data["siteA"].shape == (4, 3)
    sidecar = json.loads((tmp_path / "all.manifest.json").read_text())
    assert sidecar["sites"] == ["siteA"]


def test_npz_refusals_are_typed(tmp_path) -> None:
    """Object arrays, bad layouts, and cardinality mismatches refuse."""

    features = {"siteA": torch.ones(4, 3)}
    cases = [
        {"layout": "zip"},
        {"compatibility": "deepjuice"},
        {"layout": "consolidated", "compatibility": "net2brain"},
        {"layout": "per_stimulus", "stimulus_ids": ["only_one"]},
    ]
    for kwargs in cases:
        with pytest.raises(Exception) as excinfo:
            bp.export_npz(features, tmp_path / "x", **kwargs)
        assert excinfo.value.fields["code"] == "npz_export_invalid"
    with pytest.raises(Exception) as excinfo:
        bp.export_npz({"bad": np.array([object()], dtype=object)}, tmp_path / "y.npz")
    assert excinfo.value.fields["code"] == "npz_export_invalid"


def test_lazy_facade_reaches_brainpipe() -> None:
    """tl.brainpipe resolves through the lazy facade (no eager import)."""

    import torchlens as tl

    assert tl.brainpipe.extraction_plan is bp.extraction_plan


def test_probe_invalid_refusals_are_typed() -> None:
    """Non-tensor probes and non-positive sizes refuse extraction_probe_invalid."""

    model = _model()
    with pytest.raises(Exception) as excinfo:
        bp.extraction_plan(model, "not a tensor", n_stimuli=8, batch_size=4)
    assert excinfo.value.fields["code"] == "extraction_probe_invalid"
    with pytest.raises(Exception) as excinfo:
        bp.extraction_plan(model, torch.randn(4, 16), n_stimuli=8, batch_size=0)
    assert excinfo.value.fields["code"] == "extraction_probe_invalid"


def test_dead_model_run_refuses_extraction_plan_model_gone() -> None:
    """run() on a plan whose model died refuses extraction_plan_model_gone."""

    import weakref

    class _Ephemeral(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    holder = _Ephemeral()
    dead_ref = weakref.ref(holder)
    del holder
    plan = bp.ExtractionPlan(
        sites=(),
        batch_size=4,
        n_stimuli=8,
        input_signature=((16,), "torch.float32", 4, None, "_Ephemeral"),
        transform_repr=None,
        engine="trace",
        memory_budget_bytes=None,
        safety_margin=0.15,
        probe_batch_sizes=(4, 8),
        probe_peak_pairs=(None, None),
        predicted_peak_bytes=None,
        peak_basis="unmeasured",
        model_class="_Ephemeral",
        _model_ref=dead_ref,
        _transform=None,
    )
    with pytest.raises(Exception) as excinfo:
        plan.run(torch.randn(8, 16), progress=False)
    assert excinfo.value.fields["code"] == "extraction_plan_model_gone"
