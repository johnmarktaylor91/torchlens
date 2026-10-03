"""Watch v1 renderers, derived quantiles, and payload conversion (F25 item 7).

Byte-determinism, honesty marks (presence gaps, nonfinite badges, N-of-M
footers, coarsening ticks), typed refusals, and the cairosvg round-trip
(the SVG root must declare xmlns:xlink -- the round-3 gate finding) over
BOTH a real collector session and synthetic edge-case views.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import pytest
import torch

from torchlens.observability import (
    CommittedBlock,
    Histogram,
    HistoryCollector,
    HistoryView,
    ObservationRecord,
    RunRecord,
    SiteRecord,
    Spine,
    StepBlockRecord,
    StepTruth,
    WatchRenderError,
    color_by_watch,
    contact_sheet_pages,
    derived_quantile,
    histogram_payload,
    observation_row,
    rank_sites,
    render_contact_sheet,
    render_detail,
    render_fan,
    render_waterfall,
)


def _observed(
    step: int,
    site: str,
    tensor: torch.Tensor,
    *,
    stream: str = "activation",
    phase: str = "forward",
    with_sketch: bool = True,
    grad_scale: str | None = None,
) -> ObservationRecord:
    spine = Spine()
    spine.update(tensor)
    sketch = None
    if with_sketch:
        histogram = Histogram()
        histogram.update(tensor)
        sketch = histogram.result()
    return ObservationRecord(
        global_step=step,
        site_id=site,
        stream=stream,
        phase=phase,
        presence="observed",
        spine=spine.result(),
        sketch=sketch,
        grad_scale=grad_scale,
    )


def _block(
    step: int,
    observations: tuple[ObservationRecord, ...],
    *,
    skipped: bool = False,
    segment: str = "seg-1",
) -> CommittedBlock:
    return CommittedBlock(
        block=StepBlockRecord(
            segment_id=segment,
            global_step=step,
            provenance="explicit",
            optimizer_status="skipped" if skipped else "applied",
        ),
        observations=observations,
        step_lo=step,
        step_hi=step,
    )


def _synthetic_view() -> HistoryView:
    torch.manual_seed(3)
    run = RunRecord(run_id="run-synth", segment_id="seg-1")
    sites = {
        "m:a": SiteRecord(site_id="m:a", kind="module", display_label="a", module_path="a"),
        "m:b": SiteRecord(site_id="m:b", kind="module", display_label="b", module_path="b"),
    }
    blocks = []
    for step in range(5):
        tensor_a = torch.randn(200) * (1.0 + step)
        tensor_b = torch.randn(200)
        obs_a = _observed(step, "m:a", tensor_a)
        if step == 2:
            gap = ObservationRecord(
                global_step=step,
                site_id="m:b",
                stream="activation",
                phase="forward",
                presence="capture_failed",
                reason="boom",
            )
            blocks.append(_block(step, (obs_a, gap), skipped=True))
            continue
        if step == 3:
            tensor_b = tensor_b.clone()
            tensor_b[0] = float("nan")
        blocks.append(_block(step, (obs_a, _observed(step, "m:b", tensor_b))))
    return HistoryView.from_blocks(run, sites, tuple(blocks))


@pytest.fixture(scope="module")
def collector_view() -> HistoryView:
    """A real Route-A collector session: 6 steps, one skipped."""

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 4))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    collector = HistoryCollector(model, streams=("activation", "param", "param_delta"))
    collector.discover(torch.randn(4, 8))
    collector.attach(optimizer=optimizer)
    try:
        for step in range(6):
            skipped = step == 3
            with collector.step(step, truth=StepTruth(applied=not skipped)):
                loss = model(torch.randn(4, 8)).square().mean()
                loss.backward()
                if not skipped:
                    optimizer.step()
                optimizer.zero_grad()
    finally:
        collector.detach()
    return HistoryView.from_collector(collector)


class TestDerivedQuantiles:
    def test_matches_torch_quantile_within_grid_tolerance(self) -> None:
        torch.manual_seed(1)
        data = torch.randn(20000)
        histogram = Histogram()
        histogram.update(data)
        sketch = histogram.result()
        for q in (0.05, 0.25, 0.5, 0.75, 0.95):
            estimate = derived_quantile(sketch, q)
            truth = float(torch.quantile(data, q))
            assert estimate.approximate
            # bpo=4 grid: one bin spans a 2^(1/4) magnitude ratio (~19%).
            assert abs(estimate.value - truth) <= max(abs(truth) * 0.2, 0.02), (
                q,
                estimate.value,
                truth,
            )

    def test_monotone_in_q(self) -> None:
        torch.manual_seed(2)
        histogram = Histogram()
        histogram.update(torch.randn(5000))
        sketch = histogram.result()
        values = [derived_quantile(sketch, q).value for q in (0.1, 0.3, 0.5, 0.7, 0.9)]
        assert values == sorted(values)

    @pytest.mark.smoke
    def test_zero_heavy_population_median_is_zero(self) -> None:
        histogram = Histogram()
        histogram.update(torch.zeros(100))
        assert derived_quantile(histogram.result(), 0.5).value == 0.0

    def test_overflow_clamp_is_disclosed(self) -> None:
        histogram = Histogram()
        histogram.update(torch.full((50,), 2.0**40))  # beyond hi_exp=16 window
        estimate = derived_quantile(histogram.result(), 0.5)
        assert estimate.clamped_to_window
        assert estimate.value == 2.0**16

    def test_bad_q_refuses(self) -> None:
        histogram = Histogram()
        histogram.update(torch.randn(10))
        with pytest.raises(WatchRenderError) as excinfo:
            derived_quantile(histogram.result(), 1.5)
        assert excinfo.value.fields["code"] == "watch_quantile_invalid"

    def test_empty_population_refuses(self) -> None:
        histogram = Histogram()
        histogram.update(torch.full((5,), float("nan")))
        with pytest.raises(WatchRenderError) as excinfo:
            derived_quantile(histogram.result(), 0.5)
        assert excinfo.value.fields["code"] == "watch_population_empty"


class TestHistogramPayload:
    @pytest.mark.smoke
    def test_bucket_counts_cover_finite_population(self) -> None:
        torch.manual_seed(4)
        data = torch.randn(3000)
        data[5] = float("nan")
        data[6] = float("inf")
        obs = _observed(0, "m:a", data)
        payload = histogram_payload(obs)
        assert sum(payload["bucket_counts"]) == payload["num"] == 2998
        limits = payload["bucket_limits"]
        assert limits == sorted(limits)
        assert payload["excluded_nonfinite"] == {"nan": 1, "posinf": 1, "neginf": 0}
        assert payload["min"] is not None and payload["max"] is not None

    def test_not_observed_refuses(self) -> None:
        gap = ObservationRecord(
            global_step=0,
            site_id="m:a",
            stream="activation",
            phase="forward",
            presence="not_scheduled",
        )
        with pytest.raises(WatchRenderError) as excinfo:
            histogram_payload(gap)
        assert excinfo.value.fields["code"] == "watch_payload_not_observed"

    def test_spine_only_refuses_with_remedy(self) -> None:
        obs = _observed(0, "m:a", torch.randn(10), with_sketch=False)
        with pytest.raises(WatchRenderError) as excinfo:
            histogram_payload(obs)
        assert excinfo.value.fields["code"] == "watch_sketch_missing"

    @pytest.mark.smoke
    def test_observation_row_none_payload_for_gaps(self) -> None:
        gap = ObservationRecord(
            global_step=1,
            site_id="m:a",
            stream="activation",
            phase="forward",
            presence="site_absent",
        )
        row = observation_row(gap)
        assert row["presence"] == "site_absent"
        assert row["mean"] is None and row["count_total"] is None


class TestHistoryView:
    @pytest.mark.smoke
    def test_series_and_refusals(self) -> None:
        view = _synthetic_view()
        series = view.series("m:a")
        assert [p.step_lo for p in series] == [0, 1, 2, 3, 4]
        with pytest.raises(WatchRenderError) as unknown:
            view.series("m:zzz")
        assert unknown.value.fields["code"] == "watch_site_unknown"
        with pytest.raises(WatchRenderError) as empty:
            view.series("m:a", stream="param")
        assert empty.value.fields["code"] == "watch_series_empty"

    def test_phase_ambiguity_refuses(self) -> None:
        run = RunRecord(run_id="r", segment_id="s")
        sites = {"p:w": SiteRecord(site_id="p:w", kind="param", display_label="w")}
        observations = (
            _observed(
                0,
                "p:w",
                torch.randn(8),
                stream="param_grad",
                phase="pre_clip",
                grad_scale="unscaled",
            ),
            _observed(
                0,
                "p:w",
                torch.randn(8),
                stream="param_grad",
                phase="post_clip",
                grad_scale="unscaled",
            ),
        )
        view = HistoryView.from_blocks(run, sites, (_block(0, observations),))
        with pytest.raises(WatchRenderError) as excinfo:
            view.series("p:w", stream="param_grad")
        assert excinfo.value.fields["code"] == "watch_phase_ambiguous"
        assert len(view.series("p:w", "param_grad", "pre_clip")) == 1

    def test_rows_long_form(self) -> None:
        view = _synthetic_view()
        rows = view.rows()
        assert len(rows) == 10
        gap_rows = [r for r in rows if r["presence"] != "observed"]
        assert len(gap_rows) == 1 and gap_rows[0]["mean"] is None

    def test_to_pandas_when_available(self) -> None:
        pandas = pytest.importorskip("pandas")
        frame = _synthetic_view().to_pandas()
        assert isinstance(frame, pandas.DataFrame)
        assert len(frame) == 10

    @pytest.mark.smoke
    def test_to_pandas_without_pandas_refuses_typed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Absent pandas refuses ``watch_tabular_extra_missing`` with the remedy."""

        import sys

        monkeypatch.setitem(sys.modules, "pandas", None)
        with pytest.raises(WatchRenderError) as excinfo:
            _synthetic_view().to_pandas()
        assert excinfo.value.fields["code"] == "watch_tabular_extra_missing"
        assert "pip install pandas" in excinfo.value.fields["remedy"]


def _parse_svg(document: str) -> ET.Element:
    return ET.fromstring(document)


class TestRenderers:
    def test_fan_deterministic_and_honest(self) -> None:
        view = _synthetic_view()
        svg_1 = render_fan(view, "m:b")
        svg_2 = render_fan(view, "m:b")
        assert svg_1 == svg_2, "fan must be byte-deterministic"
        assert 'xmlns:xlink="http://www.w3.org/1999/xlink"' in svg_1
        assert "min/max exact" in svg_1
        assert "presence gap" in svg_1  # the capture_failed step
        assert "nonfinite event" in svg_1  # the injected NaN
        _parse_svg(svg_1)

    @pytest.mark.smoke
    def test_waterfall_deterministic_with_raster_and_caption(self) -> None:
        view = _synthetic_view()
        svg_1 = render_waterfall(view, "m:a")
        assert svg_1 == render_waterfall(view, "m:a")
        assert "data:image/png;base64," in svg_1
        assert "never rebinned" in svg_1
        assert "nonfinite" in svg_1
        _parse_svg(svg_1)

    @pytest.mark.smoke
    def test_contact_sheet_footer_ranking_pagination(self) -> None:
        view = _synthetic_view()
        sheet = render_contact_sheet(view)
        assert sheet == render_contact_sheet(view)
        assert "of 2 -- page 1 of 1" in sheet
        assert "nonfinite_first_drift_v1" in sheet
        ranked = rank_sites(view)
        assert ranked[0][0] == "m:b", "nonfinite site ranks first"
        assert "nonfinite" in ranked[0][2]
        assert contact_sheet_pages(view, tiles_per_page=1) == 2
        page_2 = render_contact_sheet(view, page=1, tiles_per_page=1)
        assert "sites 2-2 of 2" in page_2
        with pytest.raises(WatchRenderError) as excinfo:
            render_contact_sheet(view, page=7)
        assert excinfo.value.fields["code"] == "watch_page_out_of_range"
        _parse_svg(sheet)

    @pytest.mark.smoke
    def test_detail_sheet_discloses(self) -> None:
        view = _synthetic_view()
        detail = render_detail(view, "m:b")
        assert detail == render_detail(view, "m:b")
        assert "site detail" in detail
        assert "activation" in detail
        _parse_svg(detail)

    @pytest.mark.smoke
    def test_real_collector_views_render(self, collector_view: HistoryView) -> None:
        ranked = rank_sites(collector_view)
        assert ranked, "collector produced rankable sites"
        site_id = ranked[0][0]
        for document in (
            render_fan(collector_view, site_id),
            render_waterfall(collector_view, site_id),
            render_contact_sheet(collector_view),
            render_detail(collector_view, site_id),
        ):
            _parse_svg(document)
        # D11: the skipped optimizer step records skipped and NO fake update:
        # its block is stamped and carries zero param_delta observations.
        skipped_blocks = [b for b in collector_view.blocks if b.block.optimizer_status == "skipped"]
        assert len(skipped_blocks) == 1
        assert not [o for o in skipped_blocks[0].observations if o.stream == "param_delta"]

    def test_repr_html_is_contact_sheet(self) -> None:
        html = _synthetic_view()._repr_html_()
        assert html.startswith("<svg")


class TestCairosvgRoundTrip:
    def test_all_views_convert(self) -> None:
        cairosvg = pytest.importorskip("cairosvg")
        view = _synthetic_view()
        for document in (
            render_fan(view, "m:a"),
            render_waterfall(view, "m:a"),
            render_contact_sheet(view),
            render_detail(view, "m:a"),
        ):
            png = cairosvg.svg2png(bytestring=document.encode("utf-8"))
            assert png[:8] == b"\x89PNG\r\n\x1a\n"


class TestGraphJoin:
    @pytest.mark.smoke
    def test_color_by_watch_callable(self) -> None:
        from types import SimpleNamespace

        view = _synthetic_view()
        node_value = color_by_watch(view, step=1, stream="activation")
        matching = SimpleNamespace(module=("a", 1))
        missing = SimpleNamespace(module=("nope", 1))
        bare = SimpleNamespace(module=None)
        value = node_value(matching)
        assert value is not None and math.isfinite(value)
        assert node_value(missing) is None
        assert node_value(bare) is None
