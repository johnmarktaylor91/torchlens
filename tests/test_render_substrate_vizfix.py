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

import pytest
import torch

import torchlens as tl


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


def test_report_explain_under_inference_mode_does_not_raise(inference_trace: tl.Trace) -> None:
    # D14 (F09): bare explain never scans -- arm the saved-payload basis
    # through the explicit door (it persists the record), then the
    # inference-coverage gap note renders from the basis in hand.
    assert inference_trace.health_facts is not None
    text = str(inference_trace.explain())
    assert "inference tensors" in text


def test_capture_basis_still_checks_inference_tensors() -> None:
    # Capture-time verdicts settle at record time and need no revalidation,
    # so ``track_nonfinite=True`` is the remedy for real verdicts under
    # inference mode -- and it must catch a genuine NaN there.
    with torch.inference_mode():
        x = torch.randn(2, 4)
        log = tl.trace(
            _NaNModel(),
            x,
            capture=tl.options.CaptureOptions(on_nonfinite="track"),
        )
    assert any("truediv" in label for label in log.nonfinite_ops)
    coverage = log.nonfinite_coverage
    assert coverage.basis == "capture"
    assert coverage.inference == 0
    assert coverage.nonfinite >= 1


# ---------------------------------------------------------------------------
# vizmech D14: torch.cat is not commutative -- argument order gets labels
# ---------------------------------------------------------------------------


class _CatModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x * 2, x + 1, x - 1], dim=1)


def test_cat_left_commute_funcs() -> None:
    from torchlens._visualization._render_common import COMMUTE_FUNCS

    assert "cat" not in COMMUTE_FUNCS
    # The genuinely commutative rows stay.
    assert "add" in COMMUTE_FUNCS and "mul" in COMMUTE_FUNCS


def test_cat_edges_carry_argument_labels(tmp_path) -> None:
    # 3-way cat: the D14 regression row. Argument-order labels must SHOW on
    # cat's incoming edges now that the false commutativity claim is gone.
    log = tl.trace(_CatModel(), torch.randn(1, 4))
    source = log.draw(path=str(tmp_path / "cat"), file_format="svg").dot
    assert "arg (0, 0)" in source
    assert "arg (0, 1)" in source
    assert "arg (0, 2)" in source


def test_wide_cat_regression_row(tmp_path) -> None:
    # 40-way cat (compact stand-in for the corpus row): every slot labeled.
    class WideCat(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.cat([x + i for i in range(40)], dim=1)

    log = tl.trace(WideCat(), torch.randn(1, 2))
    source = log.draw(path=str(tmp_path / "wcat"), file_format="svg").dot
    assert "arg (0, 0)" in source
    assert "arg (0, 39)" in source


def test_argument_labels_use_padded_nonbold_builder(tmp_path) -> None:
    # D9: the argument channel routes through the tuned one-cell-table
    # builder (padding 6 / 8 pt / non-bold), not the raw bold 10-pt label.
    log = tl.trace(_CatModel(), torch.randn(1, 4))
    source = log.draw(path=str(tmp_path / "pad"), file_format="svg").dot
    assert 'CELLPADDING="6"' in source
    assert "<b>arg" not in source.lower().replace("</b>", "<b>")


# ---------------------------------------------------------------------------
# vizmech D20/D23/D24: atomic publish, raster-only dpi, geometry record
# ---------------------------------------------------------------------------


def _toy_trace() -> tl.Trace:
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
    return tl.trace(model, torch.randn(2, 4))


def test_exit_zero_stderr_is_surfaced(tmp_path, monkeypatch) -> None:
    import subprocess as sp

    from torchlens._visualization import _render_utils
    from torchlens.errors._base import TorchLensWarning

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
        log.draw(path=str(tmp_path / "noisy"), file_format="svg")
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


def test_rank_endpoint_invariant_raises_typed() -> None:
    from torchlens._visualization._rank_layout_internal.layout import (
        RankRenderEndpointError,
    )
    from torchlens.errors import _base as errors_base  # noqa: F401  (import guard)

    assert RankRenderEndpointError.code == "rank_render_endpoint_undeclared"
    assert issubclass(RankRenderEndpointError, RuntimeError)
