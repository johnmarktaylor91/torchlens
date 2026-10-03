"""Vizfix wave-1 regression tests (lane C05).

Covers the treescope P0b `_nonfinite.py` inference-mode fix (a capture run
under ``torch.inference_mode()`` must never crash the repr/report surfaces,
with the coverage gap disclosed), plus the vizmech wave-1 render-seam
fixes: honest ``torch.cat`` argument labels (D14), the padded label
builders (D9), atomic render publishing (D20), exit-0 stderr surfacing and
the structured geometry record (D24), raster-only dpi (D23), and the
rank-path declared-endpoint invariant behind the stock densenet121 crash
(defect 1).
"""

from __future__ import annotations

import os

import pytest
import torch

import torchlens as tl
from torchlens.data_classes._nonfinite import (
    coverage_gap_note,
    inference_payload_count,
    nonfinite_coverage,
)


class _NaNModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.lin(x)
        return y / (y - y)


@pytest.fixture()
def inference_trace() -> tl.Trace:
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
    with torch.inference_mode():
        x = torch.randn(2, 4)
        return tl.trace(model, x)


def test_print_trace_under_inference_mode_does_not_raise(inference_trace: tl.Trace) -> None:
    # The shipped bug: ``tensor._version`` raises RuntimeError on inference
    # tensors, escaping getattr's AttributeError-only default and taking
    # ``print(trace)`` down. str() must succeed, twice (scan + revalidate).
    first = str(inference_trace)
    second = str(inference_trace)
    assert first
    assert second


def test_clean_answer_discloses_unknown_inference_coverage(inference_trace: tl.Trace) -> None:
    answer = inference_trace.first_nonfinite()
    assert "unknown (inference tensors)" in answer
    note = coverage_gap_note(inference_trace, kind="saved")
    assert "inference" in note
    assert "torch.inference_mode" in note


def test_coverage_record_counts_inference_payloads(inference_trace: tl.Trace) -> None:
    coverage = nonfinite_coverage(inference_trace)
    assert coverage.basis == "saved_payloads"
    assert coverage.inference > 0
    assert coverage.checked == 0
    # No verdict is claimed for unversioned payloads: the record stays empty
    # WITH the disclosure, never a silently-stale clean claim.
    assert inference_trace.nonfinite_ops == ()
    assert inference_payload_count(inference_trace, kind="saved") == coverage.inference


def test_report_explain_under_inference_mode_does_not_raise(inference_trace: tl.Trace) -> None:
    from torchlens import report

    # D14 (F09): bare explain never scans -- arm the saved-payload basis
    # through the explicit door (it persists the record), then the
    # inference-coverage gap note renders from the basis in hand.
    report.health_facts(inference_trace)
    text = str(report.explain(inference_trace))
    assert "inference tensors" in text


@pytest.mark.smoke
def test_capture_basis_still_checks_inference_tensors() -> None:
    # Capture-time verdicts settle at record time and need no revalidation,
    # so ``track_nonfinite=True`` is the remedy for real verdicts under
    # inference mode -- and it must catch a genuine NaN there.
    with torch.inference_mode():
        x = torch.randn(2, 4)
        log = tl.trace(
            _NaNModel(),
            x,
            capture=tl.options.CaptureOptions(track_nonfinite=True),
        )
    assert any("truediv" in label for label in log.nonfinite_ops)
    coverage = log.nonfinite_coverage
    assert coverage.basis == "capture"
    assert coverage.inference == 0
    assert coverage.nonfinite >= 1


def test_ordinary_capture_coverage_unchanged() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
    log = tl.trace(model, torch.randn(2, 4))
    coverage = nonfinite_coverage(log)
    assert coverage.inference == 0
    assert coverage.checked > 0
    assert coverage_gap_note(log, kind="saved") == ""


# ---------------------------------------------------------------------------
# vizmech D14: torch.cat is not commutative -- argument order gets labels
# ---------------------------------------------------------------------------


class _CatModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x * 2, x + 1, x - 1], dim=1)


def test_cat_left_commute_funcs() -> None:
    from torchlens.visualization._render_common import COMMUTE_FUNCS

    assert "cat" not in COMMUTE_FUNCS
    # The genuinely commutative rows stay.
    assert "add" in COMMUTE_FUNCS and "mul" in COMMUTE_FUNCS


def test_cat_edges_carry_argument_labels(tmp_path) -> None:
    # 3-way cat: the D14 regression row. Argument-order labels must SHOW on
    # cat's incoming edges now that the false commutativity claim is gone.
    log = tl.trace(_CatModel(), torch.randn(1, 4))
    source = log.draw(vis_outpath=str(tmp_path / "cat"), vis_save_only=True, vis_fileformat="svg")
    assert "arg (0, 0)" in source
    assert "arg (0, 1)" in source
    assert "arg (0, 2)" in source


def test_wide_cat_regression_row(tmp_path) -> None:
    # 40-way cat (compact stand-in for the corpus row): every slot labeled.
    class WideCat(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.cat([x + i for i in range(40)], dim=1)

    log = tl.trace(WideCat(), torch.randn(1, 2))
    source = log.draw(vis_outpath=str(tmp_path / "wcat"), vis_save_only=True, vis_fileformat="svg")
    assert "arg (0, 0)" in source
    assert "arg (0, 39)" in source


def test_argument_labels_use_padded_nonbold_builder(tmp_path) -> None:
    # D9: the argument channel routes through the tuned one-cell-table
    # builder (padding 6 / 8 pt / non-bold), not the raw bold 10-pt label.
    log = tl.trace(_CatModel(), torch.randn(1, 4))
    source = log.draw(vis_outpath=str(tmp_path / "pad"), vis_save_only=True, vis_fileformat="svg")
    assert 'CELLPADDING="6"' in source
    assert "<b>arg" not in source.lower().replace("</b>", "<b>")


# ---------------------------------------------------------------------------
# vizmech D20/D23/D24: atomic publish, raster-only dpi, geometry record
# ---------------------------------------------------------------------------


def _toy_trace() -> tl.Trace:
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
    return tl.trace(model, torch.randn(2, 4))


def test_failed_render_leaves_no_stub(tmp_path, monkeypatch) -> None:
    import subprocess as sp

    from torchlens.visualization import _render_utils

    log = _toy_trace()
    outpath = tmp_path / "boom"

    def exploding_subprocess(cmd, **kwargs):
        # Simulate graphviz dying mid-write: the temp target gets a partial
        # byte payload, then the process "fails".
        for index, token in enumerate(cmd):
            if token == "-o":
                with open(cmd[index + 1], "wb") as partial:
                    partial.write(b"partial")
        raise sp.CalledProcessError(1, cmd, output=b"", stderr=b"boom")

    from torchlens.visualization._render_common import GraphvizRenderError

    monkeypatch.setattr(_render_utils, "run_bounded_subprocess", exploding_subprocess)
    with pytest.raises(GraphvizRenderError):
        log.draw(vis_outpath=str(outpath), vis_save_only=True, vis_fileformat="svg")
    assert not (tmp_path / "boom.svg").exists()
    leftovers = [name for name in os.listdir(tmp_path) if "tl-partial" in name]
    assert leftovers == []


def test_geometry_record_populated_on_dot_path(tmp_path) -> None:
    log = _toy_trace()
    log.draw(vis_outpath=str(tmp_path / "geo"), vis_save_only=True, vis_fileformat="svg")
    record = log._last_render_geometry
    assert record.layout_path == "dot"
    assert record.engine == "dot"
    assert record.fileformat == "svg"
    assert record.declared_size_points is not None
    assert record.effective_text_scale == 1.0
    assert record.scaled_by is None


def test_dpi_is_raster_only_cross_format(tmp_path) -> None:
    # D23: dpi must not touch vector coordinate space. The SVG declared size
    # is invariant under dpi; PNG pixels scale with it.
    log = _toy_trace()
    log.draw(vis_outpath=str(tmp_path / "svg72"), vis_save_only=True, vis_fileformat="svg")
    base = log._last_render_geometry.declared_size_points
    log.draw(
        vis_outpath=str(tmp_path / "svg300"), vis_save_only=True, vis_fileformat="svg", dpi=300
    )
    scaled = log._last_render_geometry.declared_size_points
    assert base == scaled
    log.draw(vis_outpath=str(tmp_path / "png96"), vis_save_only=True, vis_fileformat="png", dpi=96)
    png_small = log._last_render_geometry.raster_size_pixels
    log.draw(
        vis_outpath=str(tmp_path / "png192"), vis_save_only=True, vis_fileformat="png", dpi=192
    )
    png_big = log._last_render_geometry.raster_size_pixels
    assert png_small is not None and png_big is not None
    assert png_big[0] > png_small[0] * 1.5


def test_geometry_record_is_runtime_only() -> None:
    import pickle

    log = _toy_trace()
    log.__dict__["_last_render_geometry"] = object()
    restored = pickle.loads(pickle.dumps(log))
    assert "_last_render_geometry" not in restored.__dict__


def test_scale_clamp_parsing() -> None:
    from torchlens.visualization.render_execution import parse_layout_scale

    stderr = "dot: graph is too large for cairo-renderer bitmaps. Scaling by 0.324492 to fit\n"
    assert parse_layout_scale(stderr) == pytest.approx(0.324492)
    assert parse_layout_scale("") is None


def test_exit_zero_stderr_is_surfaced(tmp_path, monkeypatch) -> None:
    import subprocess as sp

    from torchlens.errors._base import TorchLensWarning
    from torchlens.visualization import _render_utils

    log = _toy_trace()
    real_run = _render_utils.run_bounded_subprocess

    def noisy_subprocess(cmd, **kwargs):
        completed = real_run(cmd, **kwargs)
        return sp.CompletedProcess(
            completed.args,
            completed.returncode,
            completed.stdout,
            b"graph is too large for cairo-renderer bitmaps. Scaling by 0.5 to fit",
        )

    monkeypatch.setattr(_render_utils, "run_bounded_subprocess", noisy_subprocess)
    with pytest.warns(TorchLensWarning, match="Scaling by 0.5"):
        log.draw(vis_outpath=str(tmp_path / "noisy"), vis_save_only=True, vis_fileformat="svg")
    assert log._last_render_geometry.scaled_by == pytest.approx(0.5)
    assert log._last_render_geometry.effective_text_scale == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# vizmech defect 1 / D20: rank path -- hidden buffers, declared endpoints
# ---------------------------------------------------------------------------


def _bn_trace() -> tl.Trace:
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3), torch.nn.BatchNorm2d(4), torch.nn.ReLU()
    ).eval()
    return tl.trace(model, torch.randn(1, 3, 8, 8))


@pytest.mark.parametrize("mode", ["always", "never", "meaningful"])
def test_rank_layout_renders_batchnorm_buffers_all_modes(tmp_path, mode) -> None:
    # The stock-densenet121 crash class: hidden buffer WRITE edges pointed at
    # undeclared endpoints and neato -n hard-errored. Every visibility mode
    # must now produce a real artifact through the rank path.
    log = _bn_trace()
    out = tmp_path / f"bn_{mode}"
    log.draw(
        vis_outpath=str(out),
        vis_save_only=True,
        vis_fileformat="svg",
        vis_node_placement="rank",
        show_buffer_layers=mode,
    )
    artifact = tmp_path / f"bn_{mode}.svg"
    assert artifact.exists() and artifact.stat().st_size > 0


@pytest.mark.smoke
def test_hidden_buffer_edges_leave_the_typed_edge_list() -> None:
    from torchlens.visualization.node_universe import build_node_universe
    from torchlens.visualization.request import ResolvedRenderRequest
    from torchlens.visualization.source_graph import build_source_graph

    log = _bn_trace()
    request = ResolvedRenderRequest(show_buffer_layers="meaningful")
    universe = build_node_universe(build_source_graph(log, request), None, None)
    unit_names = {unit.unit_id for unit in universe.units}
    for occurrence in universe.projected_edges:
        assert occurrence.source_unit in unit_names
        assert occurrence.target_unit in unit_names


def test_rank_endpoint_invariant_raises_typed() -> None:
    from torchlens.errors import _base as errors_base  # noqa: F401  (import guard)
    from torchlens.visualization._rank_layout_internal.layout import (
        RankRenderEndpointError,
    )

    assert RankRenderEndpointError.code == "rank_render_endpoint_undeclared"
    assert issubclass(RankRenderEndpointError, RuntimeError)
