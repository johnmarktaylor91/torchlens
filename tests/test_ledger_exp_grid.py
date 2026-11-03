"""``tl.experiment.sweep_grid``: named axes over the candidate engine.

Spellings and the per-cell refusal behaviour follow JMT's rulings of
1 Nov 2026 (``sweep_grid`` returning ``SweepGrid``; one edit axis; a hole is a
refused cell and the rest run; float64 values with NaN; plain ordered axes;
comma-joined cell names; axes recorded in the operation's params). Every test
runs on the tiny fixtures of ``tests/test_ledger_exp_engine.py`` except the
attention facet row, which uses the GPT-2 fixture of the head oracle test.
"""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors._episode import BundleExperimentError
from torchlens.experiment import GridCell, SweepGrid, sweep_grid, sweep_sites
from torchlens.experiment._engine import top_k

_REPO_ROOT = Path(__file__).resolve().parents[1]


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)
        self.gate = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x)) + torch.tanh(self.gate(x))


class _TwoRelu(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)
        self.gate = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x)) + torch.relu(self.gate(x))


class _TinyConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.conv2(torch.relu(self.conv1(x))))


def _baseline(x: torch.Tensor, model: nn.Module) -> tl.Trace:
    return tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))


def _output_metric(member: tl.Trace) -> float:
    label = member.output_layers[0]
    return float(member[label].out.sum().item())


def _tiny(shape: tuple[int, ...] = (2, 3)) -> tuple[nn.Module, torch.Tensor, tl.Trace]:
    torch.manual_seed(0)
    model = _Tiny().eval()
    x = torch.randn(*shape)
    return model, x, _baseline(x, model)


def _grid(baseline: tl.Trace, model: nn.Module, x: torch.Tensor, **kwargs: object) -> SweepGrid:
    kwargs.setdefault("metric", _output_metric)
    kwargs.setdefault("retain", "none")
    return sweep_grid(baseline, model=model, inputs=x, **kwargs)  # type: ignore[arg-type]


_SITES = {"a": "relu_1_2", "b": "tanh_1_4"}


def _code(excinfo: pytest.ExceptionInfo[BundleExperimentError]) -> str:
    return str(excinfo.value.fields["code"])


# ---- shape, equality with the flat sweep, connectedness ----------------------


def test_grid_shape_axes_cells_follow_the_declared_axes() -> None:
    model, x, baseline = _tiny()
    grid = _grid(
        baseline,
        model,
        x,
        axes={"site": _SITES, "scale": [tl.scale(0.0), tl.scale(0.5), tl.scale(2.0)]},
    )
    assert isinstance(grid, SweepGrid)
    assert grid.values.shape == (2, 3)
    assert grid.values.dtype is torch.float64
    assert grid.axes == {"site": ("a", "b"), "scale": ("0", "1", "2")}
    assert list(grid.axes) == ["site", "scale"]
    assert len(grid.cells) == 6
    assert all(isinstance(cell, GridCell) for cell in grid.cells)
    assert bool(torch.isfinite(grid.values).all())
    assert grid.edit_axis == "scale"
    assert grid.refused == () and grid.failed == ()
    assert "SweepGrid(site=2, scale=3, measured=6, failed=0" in repr(grid)


def test_grid_values_equal_the_flat_sweep_values() -> None:
    model, x, baseline = _tiny()
    grid = _grid(baseline, model, x, axes={"site": _SITES}, edit=tl.scale(0.5))
    flat = sweep_sites(
        baseline,
        candidates=dict(_SITES),
        edit=tl.scale(0.5),
        metric=_output_metric,
        retain="none",
        model=model,
        inputs=x,
    )
    flat_rows = {row.candidate_id: row for row in flat.effects().rows}
    for name in _SITES:
        assert grid.cell(site=name).value == flat_rows[name].value
        assert float(grid.values[grid.index(site=name)]) == flat_rows[name].value
    assert grid.baseline_value == flat_rows["__baseline__"].value
    assert torch.equal(grid.deltas, grid.values - grid.baseline_value)


def test_changing_an_edit_axis_value_changes_only_that_cell() -> None:
    model, x, baseline = _tiny()
    axes = {"site": {"a": "relu_1_2"}, "scale": {"half": tl.scale(0.5), "zero": tl.scale(0.0)}}
    grid = _grid(baseline, model, x, axes=axes)
    half = grid.cell(site="a", scale="half").value
    zero = grid.cell(site="a", scale="zero").value
    assert half is not None and zero is not None
    assert half != zero
    assert half != grid.baseline_value and zero != grid.baseline_value
    axes["scale"]["zero"] = tl.scale(0.25)
    again = _grid(baseline, model, x, axes=axes)
    assert again.cell(site="a", scale="zero").value != zero
    assert again.cell(site="a", scale="half").value == half


# ---- holes and refusals --------------------------------------------------------


def test_unresolved_cell_is_refused_and_the_rest_run() -> None:
    model, x, baseline = _tiny()
    calls = {"n": 0}

    def counting_metric(member: tl.Trace) -> float:
        calls["n"] += 1
        return _output_metric(member)

    axes = {"site": {"bad": tl.label("no_such_op"), "good": tl.label("relu_1_2")}}
    grid = _grid(
        baseline, model, x, axes=axes, edit=tl.zero_ablate(), metric=counting_metric, retain="all"
    )
    bad = grid.cell(site="bad")
    assert bad.status == "refused" and bad.value is None and bad.member_name is None
    assert bad.error is not None and bad.error.startswith("grid_cell_unresolved")
    assert "'bad'" in bad.error and "no_such_op" in bad.error
    assert math.isnan(float(grid.values[grid.index(site="bad")]))
    assert grid.refused == ("bad",)
    assert grid.cell(site="good").status == "completed"
    assert calls["n"] == 2  # the baseline, then the one measured member
    assert set(grid.bundle.members) == {"baseline", "good"}
    assert grid.coords("bad") == {"site": "bad"}
    with pytest.raises(BundleExperimentError) as excinfo:
        _grid(baseline, model, x, axes={"site": []}, edit=tl.zero_ablate(), metric=counting_metric)
    assert _code(excinfo) == "grid_axes_invalid"
    assert calls["n"] == 2


def test_edit_axis_and_edit_keyword_are_exclusive_and_one_is_required() -> None:
    model, x, baseline = _tiny()
    with pytest.raises(BundleExperimentError) as excinfo:
        _grid(
            baseline,
            model,
            x,
            axes={"site": _SITES, "scale": [tl.scale(0.5)]},
            edit=tl.zero_ablate(),
        )
    assert _code(excinfo) == "grid_edit_conflict"
    with pytest.raises(BundleExperimentError) as excinfo:
        _grid(baseline, model, x, axes={"site": _SITES})
    assert _code(excinfo) == "grid_edit_missing"
    with pytest.raises(BundleExperimentError) as excinfo:
        _grid(
            baseline, model, x, axes={"site": _SITES, "s1": [tl.scale(0.5)], "s2": [tl.scale(0.0)]}
        )
    assert _code(excinfo) == "grid_edit_axis_multiple"
    with pytest.raises(BundleExperimentError) as excinfo:
        _grid(baseline, model, x, axes={"scale": [tl.scale(0.5)]})
    assert _code(excinfo) == "grid_site_axis_missing"


# Kept by the shrink-r4 coverage proof: sole branch coverage of experiment/_grid.py.
@pytest.mark.parametrize(
    "axes",
    [
        {},
        {"site": {"baseline": "relu_1_2"}},
        {"site": {"a": "relu_1_2", "a2": "relu_1_2"}, "": [tl.scale(0.5)]},
        {"site": "relu_1_2"},
    ],
)
def test_axis_shapes_refuse_grid_axes_invalid(axes: dict[str, object]) -> None:
    model, x, baseline = _tiny()
    with pytest.raises(BundleExperimentError) as excinfo:
        _grid(baseline, model, x, axes=axes, edit=tl.zero_ablate())
    assert _code(excinfo) == "grid_axes_invalid"


def test_mixed_axis_refuses_grid_axis_mixed() -> None:
    model, x, baseline = _tiny()
    with pytest.raises(BundleExperimentError) as excinfo:
        _grid(baseline, model, x, axes={"site": {"a": "relu_1_2", "e": tl.scale(0.5)}})
    assert _code(excinfo) == "grid_axis_mixed"
    with pytest.raises(BundleExperimentError) as excinfo:
        _grid(baseline, model, x, axes={"site": {"a": "relu_1_2", "n": 3}}, edit=tl.zero_ablate())
    assert _code(excinfo) == "grid_axis_mixed"


def test_cell_ids_are_comma_joined_value_names_in_axis_order() -> None:
    model, x, baseline = _tiny((2, 4, 3))
    axes = {"layer": {"L3": "relu_1_2"}, "position": {"t7": tl.positions(tl.zero_ablate(), 1)}}
    grid = _grid(baseline, model, x, axes=axes, retain="all")
    assert grid.candidate_id(layer="L3", position="t7") == "L3,t7"
    assert grid.coords("L3,t7") == {"layer": "L3", "position": "t7"}
    assert grid.cell(layer="L3", position="t7").candidate_id == "L3,t7"
    assert "L3,t7" in {row.candidate_id for row in grid.effects.rows}
    assert "L3,t7" in grid.bundle.members
    listed = _grid(
        baseline, model, x, axes={"site": ["relu_1_2", "tanh_1_4"]}, edit=tl.zero_ablate()
    )
    assert listed.axes == {"site": ("0", "1")}
    with pytest.raises(BundleExperimentError) as excinfo:
        grid.coords("L9,t7")
    assert _code(excinfo) == "grid_cell_unknown"
    with pytest.raises(BundleExperimentError) as excinfo:
        grid.index(layer="L3")
    assert _code(excinfo) == "grid_coords_invalid"


def test_facet_with_a_non_module_partner_refuses_grid_cell_uncombinable() -> None:
    model, x, baseline = _tiny()
    with pytest.raises(BundleExperimentError) as excinfo:
        _grid(
            baseline,
            model,
            x,
            axes={"head": {"h1": tl.head(1, "v")}, "other": {"f": tl.func("relu")}},
            edit=tl.zero_ablate(),
        )
    assert _code(excinfo) == "grid_cell_uncombinable"
    assert excinfo.value.fields["cell"] == "h1,f"


@pytest.mark.heavy
def test_facet_plus_module_combines_to_the_scoped_facet_selector() -> None:
    pytest.importorskip("transformers")
    from tests.real_model.r0.families import build_gpt2

    model = build_gpt2("eager").eval()
    torch.manual_seed(0)
    ids = torch.randint(0, 100, (1, 8))
    address = "transformer.h.1.attn"

    def metric(member: tl.Trace) -> float:
        label = member.output_layers[0]
        return float(member[label].out[0, -1, 17].item())

    baseline = tl.trace(model, ids, capture=tl.options.CaptureOptions(replayable=True))
    grid = sweep_grid(
        baseline,
        axes={"block": {"b": tl.module(address)}, "head": {"h1": tl.head(1, "v")}},
        edit=tl.zero_ablate(),
        metric=metric,
        retain="none",
        model=model,
        inputs=ids,
    )
    flat = sweep_sites(
        baseline,
        candidates={"b,h1": tl.head(1, "v").in_module(address)},
        edit=tl.zero_ablate(),
        metric=metric,
        retain="none",
        model=model,
        inputs=ids,
    )
    (flat_row,) = [row for row in flat.effects().rows if row.candidate_id == "b,h1"]
    assert grid.cell(block="b", head="h1").value == flat_row.value
    assert grid.cell(block="b", head="h1").status == "released"


def test_multi_site_cell_is_refused_unless_declared() -> None:
    torch.manual_seed(0)
    model = _TwoRelu().eval()
    x = torch.randn(2, 3)
    baseline = _baseline(x, model)
    axes = {"site": {"both": tl.func("relu"), "one": tl.label("relu_1_2")}}
    grid = _grid(baseline, model, x, axes=axes, edit=tl.zero_ablate())
    both = grid.cell(site="both")
    assert both.status == "refused"
    assert both.error is not None and both.error.startswith("grid_cell_multi_site")
    assert "'both'" in both.error and "resolved_site_count=2" in both.error
    assert math.isnan(float(grid.values[grid.index(site="both")]))
    assert grid.cell(site="one").status == "released"
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        declared = _grid(baseline, model, x, axes=axes, edit=tl.zero_ablate(), multi_site=True)
    assert declared.cell(site="both").status == "released"
    (row,) = [row for row in declared.effects.rows if row.candidate_id == "both"]
    assert row.resolved_site_count == 2
    assert bool(torch.isfinite(declared.values).all())


def test_duplicate_identity_includes_the_edit() -> None:
    model, x, baseline = _tiny((2, 4, 3))
    zero = tl.zero_ablate()
    grid = _grid(
        baseline,
        model,
        x,
        axes={
            "site": {"a": "relu_1_2"},
            "pos": {"t0": tl.positions(zero, 0), "t1": tl.positions(zero, 1)},
        },
    )
    t0, t1 = grid.cell(site="a", pos="t0").value, grid.cell(site="a", pos="t1").value
    assert t0 is not None and t1 is not None and t0 != t1
    twice = {
        "site": {"a": "relu_1_2"},
        "pos": {"t0": tl.positions(zero, 0), "again": tl.positions(zero, 0)},
    }
    with pytest.raises(BundleExperimentError) as excinfo:
        _grid(baseline, model, x, axes=twice)
    assert _code(excinfo) == "site_sweep_duplicate_candidates"
    replicated = _grid(baseline, model, x, axes=twice, replicates=True)
    assert replicated.cell(site="a", pos="t0").value == replicated.cell(site="a", pos="again").value


@pytest.mark.smoke
def test_position_preflight_against_recorded_shapes() -> None:
    model, x, baseline = _tiny((2, 4, 3))
    calls = {"n": 0}

    def counting_metric(member: tl.Trace) -> float:
        calls["n"] += 1
        return _output_metric(member)

    with pytest.raises(BundleExperimentError) as excinfo:
        _grid(
            baseline,
            model,
            x,
            axes={"site": {"a": "relu_1_2"}, "pos": {"t9": tl.positions(tl.zero_ablate(), 9)}},
            metric=counting_metric,
        )
    assert _code(excinfo) == "grid_position_out_of_range"
    assert excinfo.value.fields["cell"] == "a,t9"
    assert excinfo.value.fields["positions"] == (9,) and excinfo.value.fields["extent"] == 4
    assert (
        "'a,t9'" in str(excinfo.value)
        and "[9]" in str(excinfo.value)
        and "extent 4" in str(excinfo.value)
    )
    assert calls["n"] == 0
    sliced = _grid(
        baseline,
        model,
        x,
        axes={"site": {"a": "relu_1_2"}, "pos": {"s": tl.positions(tl.zero_ablate(), slice(0, 2))}},
        metric=counting_metric,
    )
    assert sliced.cell(site="a", pos="s").status == "released"


def test_zero_site_candidate_refuses_the_flat_sweep_typed() -> None:
    """The engine-level tripwire the grid's refused cell rides on."""

    model, x, baseline = _tiny()
    with pytest.raises(BundleExperimentError) as excinfo:
        sweep_sites(
            baseline,
            candidates={"ghost": tl.label("no_such_op")},
            edit=tl.zero_ablate(),
            metric=_output_metric,
            retain="none",
            model=model,
            inputs=x,
        )
    assert _code(excinfo) == "site_sweep_candidate_unresolved"
    assert excinfo.value.fields["candidate_id"] == "ghost"


# ---- retention, failure, record, round trip -----------------------------------


def test_retention_policy_composes_with_the_grid() -> None:
    model, x, baseline = _tiny()
    grid = _grid(
        baseline,
        model,
        x,
        axes={"site": _SITES, "scale": [tl.scale(0.0), tl.scale(0.5), tl.scale(3.0)]},
        retain=top_k(2),
    )
    assert bool(torch.isfinite(grid.values).all())
    members = set(grid.bundle.members)
    assert "baseline" in members and len(members) == 3
    by_delta = sorted(
        grid.cells, key=lambda cell: -abs(float(cell.value or 0.0) - grid.baseline_value)
    )
    assert {cell.candidate_id for cell in by_delta[:2]} == members - {"baseline"}
    rows = grid.effects.rows
    assert len(rows) == 7
    assert sum(1 for row in rows if row.status == "released") == 4


def test_execution_failure_is_a_failed_cell_not_a_crash() -> None:
    model, x, baseline = _tiny()
    axes = {
        "site": {"a": "relu_1_2"},
        "edit": {"ok": tl.scale(0.5), "boom": tl.steer(torch.ones(7))},
    }
    grid = _grid(baseline, model, x, axes=axes)
    boom = grid.cell(site="a", edit="boom")
    assert boom.status == "failed" and boom.value is None
    assert boom.error
    assert math.isnan(float(grid.values[grid.index(site="a", edit="boom")]))
    assert grid.failed == ("a,boom",)
    assert math.isfinite(float(grid.values[grid.index(site="a", edit="ok")]))
    assert "failed=1" in repr(grid)


def test_axes_are_recorded_in_the_operation_params() -> None:
    model, x, baseline = _tiny()
    grid = _grid(baseline, model, x, axes={"site": _SITES, "scale": [tl.scale(0.0), tl.scale(0.5)]})
    sweeps = [op for op in grid.bundle.history if op.kind == "site_sweep"]
    assert len(sweeps) == 1
    assert sweeps[0].params["axes"] == {"site": ["a", "b"], "scale": ["0", "1"]}
    assert sweeps[0].params["edit_axis"] == "scale"
    assert sweeps[0].params["cells"] == 4
    assert sweeps[0].params["attempted"] == 4
    assert grid.effects.table.edit_repr == "axis:scale"
    assert grid.history_id == sweeps[0].operation_id


@pytest.mark.smoke
def test_grid_round_trips_through_save_and_load(tmp_path: Path) -> None:
    model, x, baseline = _tiny()
    grid = _grid(baseline, model, x, axes={"site": _SITES, "scale": [tl.scale(0.0), tl.scale(0.5)]})
    path = tmp_path / "grid_bundle"
    grid.bundle.save(str(path))
    loaded = tl.load(str(path))
    again = loaded.effects().to_grid()
    assert again.axes == grid.axes
    assert torch.equal(again.values, grid.values)
    assert again.baseline_value == grid.baseline_value
    assert [cell.candidate_id for cell in again.cells] == [cell.candidate_id for cell in grid.cells]
    flat = sweep_sites(
        baseline,
        candidates=dict(_SITES),
        edit=tl.zero_ablate(),
        metric=_output_metric,
        retain="none",
        model=model,
        inputs=x,
    )
    with pytest.raises(BundleExperimentError) as excinfo:
        flat.effects().to_grid()
    assert _code(excinfo) == "grid_axes_unavailable"


def test_tampered_effect_table_refuses_grid_result_shape_mismatch() -> None:
    import dataclasses

    model, x, baseline = _tiny()
    grid = _grid(baseline, model, x, axes={"site": _SITES}, edit=tl.zero_ablate())
    table = grid.bundle._effect_tables[grid.history_id]
    grid.bundle._effect_tables[grid.history_id] = dataclasses.replace(table, rows=table.rows[:-1])
    with pytest.raises(BundleExperimentError) as excinfo:
        grid.bundle.effects().to_grid()
    assert _code(excinfo) == "grid_result_shape_mismatch"
    assert excinfo.value.fields["rows"] == 1 and excinfo.value.fields["cells"] == 2


def test_to_pandas_is_long_format_one_row_per_cell() -> None:
    pytest.importorskip("pandas")
    model, x, baseline = _tiny()
    grid = _grid(baseline, model, x, axes={"site": _SITES, "scale": [tl.scale(0.0), tl.scale(0.5)]})
    frame = grid.to_pandas()
    assert len(frame) == 4
    for column in ("site", "scale", "candidate_id", "status", "value", "delta", "member_name"):
        assert column in frame.columns
    (row,) = frame[frame["candidate_id"] == "a,0"].to_dict("records")
    assert row["value"] == grid.cell(site="a", scale="0").value
    assert row["site"] == "a" and row["scale"] == "0"


def test_max_cells_guards_the_product_of_the_axes() -> None:
    model, x, baseline = _tiny()
    axes = {"site": _SITES, "scale": [tl.scale(0.0), tl.scale(0.5), tl.scale(2.0)]}
    with pytest.raises(BundleExperimentError) as excinfo:
        _grid(baseline, model, x, axes=axes, max_cells=5, metric=lambda member: 1 / 0)
    assert _code(excinfo) == "grid_too_large"
    assert excinfo.value.fields["cells"] == 6
    assert _grid(baseline, model, x, axes=axes, max_cells=6).values.shape == (2, 3)


def test_non_transformer_grid_runs_on_a_conv_net() -> None:
    torch.manual_seed(0)
    model = _TinyConv().eval()
    x = torch.randn(1, 1, 6, 6)
    baseline = _baseline(x, model)
    grid = _grid(
        baseline,
        model,
        x,
        axes={
            "stage": {"s1": tl.module("conv1"), "s2": tl.module("conv2")},
            "channel": {f"c{c}": tl.positions(tl.zero_ablate(), c, axis=1) for c in range(4)},
        },
    )
    assert grid.values.shape == (2, 4)
    assert bool(torch.isfinite(grid.values).all())
    assert bool((grid.deltas != 0).any())


# ---- surface and docs lockstep ----------------------------------------------------


def test_public_surface_and_docs_lockstep() -> None:
    import torchlens.experiment as experiment

    assert {"sweep_grid", "SweepGrid", "GridCell"} <= set(experiment.__all__)
    assert not hasattr(tl, "sweep_grid") and "sweep_grid" not in tl.__all__
    glossary = (_REPO_ROOT / "docs" / "reference" / "glossary.md").read_text()
    for name in (
        "`tl.experiment.sweep_grid`",
        "`tl.experiment.SweepGrid`",
        "`EffectsView.to_grid`",
    ):
        assert name in glossary, name
    contract = (_REPO_ROOT / "docs" / "reference" / "error_refusal_contract.md").read_text()
    documented = set(re.findall(r"^\| `(grid_[a-z_]+)` \|", contract, flags=re.MULTILINE))
    source = (_REPO_ROOT / "torchlens" / "experiment" / "_grid.py").read_text()
    raised = set(re.findall(r'code="(grid_[a-z_]+)"', source)) | set(
        re.findall(r'^_[A-Z_]+ = "(grid_cell_[a-z_]+)"', source, flags=re.MULTILINE)
    )
    assert raised and raised <= documented, sorted(raised - documented)
    for code in documented:
        row = next(line for line in contract.splitlines() if line.startswith(f"| `{code}` |"))
        assert row.count("|") == 4 and row.rsplit("|", 2)[-2].strip(), code
    for removed in ("read_ledger", "ExperimentLedger", "LedgerEntry", "active_ledger"):
        assert removed not in glossary, removed
        for page in (_REPO_ROOT / "docs" / "internal" / "agent-reference").glob("*.md"):
            assert removed not in page.read_text(), (page.name, removed)
