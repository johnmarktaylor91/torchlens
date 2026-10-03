"""Op/facet tier, Route B via fastlog (explorer memo item 8; lane F25).

The differentiator: op-grain sites via one fastlog pass per logged step
under the P1 summary role and the P2 reduce-only path, folded into the
SAME artifact schema (site kinds ``op`` / ``facet``), with disclosed cost
(the plan table), selector-algebra scoping, training-graph neutrality
(the wrapped step returns the LIVE output), and per-head facet identity.
"""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from torchlens.observability import (
    HistoryView,
    StepTruth,
    WatchLifecycleError,
    WatchPlanError,
    render_contact_sheet,
    render_fan,
)
from torchlens.observability._op_tier import OpTierCollector, split_heads
from torchlens.observability._quantiles import WatchRenderError


def _model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 4))


def _run_steps(collector: OpTierCollector, model: torch.nn.Module, n: int = 3) -> None:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    for step in range(n):
        with collector.step(step, truth=StepTruth()):
            output = collector.observe_step(torch.randn(4, 8))
            loss = output.square().mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


class TestOpTierLifecycle:
    @pytest.mark.smoke
    def test_discover_prices_plan_and_catalogs_op_sites(self) -> None:
        model = _model()
        collector = OpTierCollector(model)
        collector.discover(torch.randn(4, 8))
        plan = collector.plan
        assert plan.total_elements_per_step > 0
        table = plan.format_table()
        assert "op tier (fastlog)" in table
        kinds = {site.kind for site in collector.site_catalog.values()}
        assert kinds == {"op"}
        # gpt2-claim shape: op sites outnumber the 3 modules (inputs/ops).
        assert len(collector.site_catalog) >= 3

    def test_zero_match_selection_refuses_at_plan_time(self) -> None:
        collector = OpTierCollector(_model(), save=tl.func("conv2d"))
        with pytest.raises(WatchPlanError) as excinfo:
            collector.discover(torch.randn(4, 8))
        assert excinfo.value.fields["code"] == "watch_plan_empty"

    def test_step_before_discover_refuses(self) -> None:
        collector = OpTierCollector(_model())
        with pytest.raises(WatchLifecycleError) as excinfo, collector.step(0):
            pass
        assert excinfo.value.fields["code"] == "watch_lifecycle_invalid"

    def test_observe_outside_step_refuses(self) -> None:
        model = _model()
        collector = OpTierCollector(model)
        collector.discover(torch.randn(4, 8))
        with pytest.raises(WatchLifecycleError) as excinfo:
            collector.observe_step(torch.randn(4, 8))
        assert excinfo.value.fields["code"] == "watch_lifecycle_invalid"


class TestOpTierObservations:
    def test_op_grain_series_land_and_render(self) -> None:
        model = _model()
        collector = OpTierCollector(model)
        collector.discover(torch.randn(4, 8))
        _run_steps(collector, model, n=3)
        view = HistoryView.from_collector(collector)
        assert len(view.blocks) == 3
        linear_sites = [s for s in view.sites if "linear" in s]
        assert linear_sites, f"expected linear op sites, got {sorted(view.sites)[:5]}"
        series = view.series(linear_sites[0])
        assert len(series) == 3
        for point in series:
            spine = point.observation.spine
            assert spine is not None and spine.count_total > 0
            assert point.observation.sketch is not None
        # The renderers accept op-tier views unchanged (grain-agnostic).
        assert render_fan(view, linear_sites[0]).startswith("<svg")
        assert "of" in render_contact_sheet(view)

    def test_training_graph_survives_wrapped_step(self) -> None:
        model = _model()
        collector = OpTierCollector(model)
        collector.discover(torch.randn(4, 8))
        before = [p.detach().clone() for p in model.parameters()]
        _run_steps(collector, model, n=2)
        after = list(model.parameters())
        changed = any(not torch.equal(b, a.detach()) for b, a in zip(before, after, strict=True))
        assert changed, "optimizer steps through observe_step outputs must train"

    def test_selector_algebra_scopes_sites(self) -> None:
        model = _model()
        collector = OpTierCollector(model, save=tl.func("relu"))
        collector.discover(torch.randn(4, 8))
        labels = {site.display_label for site in collector.site_catalog.values()}
        assert labels and all("relu" in label for label in labels)

    @pytest.mark.smoke
    def test_exact_totals_per_site(self) -> None:
        model = _model()
        collector = OpTierCollector(model, save=tl.func("relu"))
        collector.discover(torch.randn(4, 8))
        with collector.step(0):
            collector.observe_step(torch.randn(4, 8))
        view = HistoryView.from_collector(collector)
        site_id = next(iter(view.sites))
        spine = view.series(site_id)[0].observation.spine
        assert spine is not None
        assert spine.count_total == 4 * 8  # relu output numel, exactly


class TestFacetTier:
    @pytest.mark.smoke
    def test_per_head_facets_have_stable_identity(self) -> None:
        torch.manual_seed(1)
        model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU())

        class Reshaper(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                batch = x.shape[0]
                return x.view(batch, 4, 4)  # [B, H=4, D]

        model.append(Reshaper())
        probe = OpTierCollector(model, save=tl.func("view"))
        probe.discover(torch.randn(2, 8))
        view_label = next(iter(probe.site_catalog.values())).display_label
        collector = OpTierCollector(
            model,
            save=tl.func("view"),
            facets={view_label: split_heads(4, dim=1)},
        )
        collector.discover(torch.randn(2, 8))
        for step in range(2):
            with collector.step(step):
                collector.observe_step(torch.randn(2, 8))
        view = HistoryView.from_collector(collector)
        facet_sites = [s for s, rec in view.sites.items() if rec.kind == "facet"]
        assert len(facet_sites) == 4, sorted(view.sites)
        for facet_id in facet_sites:
            series = view.series(facet_id)
            assert len(series) == 2, "facet identity stable across steps"
            spine = series[0].observation.spine
            assert spine is not None and spine.count_total == 2 * 4  # one head slice

    @pytest.mark.smoke
    def test_missplit_head_axis_refuses(self) -> None:
        splitter = split_heads(4, dim=1)
        with pytest.raises(WatchRenderError) as excinfo:
            splitter(torch.randn(2, 5, 4))
        assert excinfo.value.fields["code"] == "watch_facet_shape_invalid"
