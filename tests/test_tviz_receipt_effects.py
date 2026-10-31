"""Receipt grids show effects, and their disclosure footer never covers the matrix.

A patch grid stores the metric of each patched run; a receipt plots the
change from the unpatched corrupted run (the run every cell patches into),
so a cell that changes nothing reads exactly 0. The grid figure reserves the
footer's height below the matrix and its x tick labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

pytest.importorskip("matplotlib")

import torchlens.tviz as tviz  # noqa: E402
from torchlens.mechinterp import PatchGrid  # noqa: E402
from torchlens.tviz import _receipts  # noqa: E402

BASELINE = 0.75


def _grid(values: torch.Tensor, **receipts: Any) -> PatchGrid:
    """A [layer, head] grid with one fire per cell and the given receipts."""

    n_layers, n_heads = values.shape
    return PatchGrid(
        axes=("layer", "head"),
        coordinates={
            "layer": tuple(f"blk.{i}.attn" for i in range(n_layers)),
            "head": tuple(range(n_heads)),
        },
        values=values,
        receipts={
            "engine": "rerun",
            "fires": n_layers * n_heads,
            "n_cells": n_layers * n_heads,
            "metric_provenance": "logit_diff",
            **receipts,
        },
    )


def _receipt(grid: PatchGrid, layer: str = "blk.0.attn", **kwargs: Any) -> tviz.CausalReceipt:
    return tviz.receipt_from_patch_grid(
        grid, layer=layer, negative_control=0.0, positive_control=1.0, **kwargs
    )


def test_cell_equal_to_the_baseline_is_exactly_zero_and_prints_zero(tmp_path: Path) -> None:
    """Effect = patched metric minus the corrupted baseline; no change reads +0.00."""

    values = torch.tensor([[BASELINE, 1.25, 0.5]], dtype=torch.float32)
    grid = _grid(values, clean_metric=2.0, corrupted_metric=BASELINE)
    receipt = _receipt(grid, joint_effect=0.1)
    assert receipt.effects == (0.0, 0.5, -0.25)
    assert receipt.baseline_metric == BASELINE
    artifact = tviz.render_receipt_grid(receipt, tmp_path / "zero.svg", svg_fonttype="none")
    svg = artifact.paths[0].read_text()
    for cell in ("+0.00", "+0.50", "-0.25"):
        assert f">{cell}<" in svg, cell
    assert ">+0.75<" not in svg  # the raw baseline value is never a cell
    assert (
        "each cell: change in logit_diff from the unpatched corrupted run (+0.75); 0 = no change"
        in artifact.disclosure_lines
    )
    assert "measured head effects on logit_diff" in svg
    assert ">query head<" in svg


def test_small_effects_get_enough_decimals_to_read_nonzero(tmp_path: Path) -> None:
    """A grid of effects near 0.01 prints three decimals, so 0.002 never reads +0.00."""

    values = torch.tensor([[0.002, -0.019, 0.0]], dtype=torch.float32)
    receipt = _receipt(_grid(values, corrupted_metric=0.0), joint_effect=0.0)
    artifact = tviz.render_receipt_grid(receipt, tmp_path / "small.svg", svg_fonttype="none")
    svg = artifact.paths[0].read_text()
    for cell in ("+0.002", "-0.019", "+0.000"):
        assert f">{cell}<" in svg, cell


def test_known_patched_value_gives_the_known_difference() -> None:
    """A float32 cell and baseline subtract exactly as Python floats."""

    patched = torch.tensor([[0.3, -1.7]], dtype=torch.float32)
    baseline = float(torch.tensor(0.1, dtype=torch.float32))
    receipt = _receipt(_grid(patched, corrupted_metric=baseline))
    assert receipt.effects == (float(patched[0, 0]) - baseline, float(patched[0, 1]) - baseline)
    assert receipt.effects[0] == pytest.approx(0.2, abs=1e-7)
    assert receipt.effects[1] == pytest.approx(-1.8, abs=1e-7)


def test_grid_without_a_baseline_refuses_with_a_remedy() -> None:
    """Raw patched values never pass as effects."""

    with pytest.raises(tviz.RenderError) as caught:
        _receipt(_grid(torch.zeros(1, 2)))
    assert caught.value.fields["code"] == "tv_receipt_invalid"
    assert "corrupted_metric" in caught.value.fields["remedy"]
    assert "patch_query_heads_grid" in caught.value.fields["remedy"]


def test_default_wording_says_what_a_patch_grid_does() -> None:
    """The default intervention is a clean-into-corrupted patch, never an ablation."""

    grid = _grid(torch.zeros(1, 2), corrupted_metric=0.0)
    assert _receipt(grid).intervention == (
        "clean activation of one head patched into the corrupted run"
    )
    gqa = tviz.GqaInfo(n_query_heads=4, n_kv_heads=2)
    assert _receipt(grid, kv_groups=gqa).intervention == (
        "clean activation of one kv group patched into the corrupted run"
    )


def test_row_fires_are_the_row_cells_when_the_ledger_shows_one_per_cell(tmp_path: Path) -> None:
    """Rows from one grid never repeat the grid total in the figure's fire count."""

    grid = _grid(torch.zeros(3, 4), corrupted_metric=0.0)
    rows = [_receipt(grid, layer=f"blk.{i}.attn") for i in range(3)]
    assert [row.fires for row in rows] == [4, 4, 4]
    artifact = tviz.render_receipt_grid(rows, tmp_path / "fires.png", joint_effect=0.0)
    assert any(line.endswith("fires: 12") for line in artifact.disclosure_lines)
    uneven = _grid(torch.zeros(1, 4), corrupted_metric=0.0, fires=9)
    assert _receipt(uneven).fires == 9


def test_rows_from_different_baselines_refuse(tmp_path: Path) -> None:
    """Cells measured from different baseline runs are not on one scale."""

    first = _receipt(_grid(torch.zeros(1, 2), corrupted_metric=0.0))
    second = _receipt(_grid(torch.zeros(1, 2), corrupted_metric=1.0))
    with pytest.raises(tviz.RenderError) as caught:
        tviz.render_receipt_grid([first, second], tmp_path / "mixed.png", joint_effect=0.0)
    assert caught.value.fields["code"] == "tv_receipt_invalid"


def test_receipt_without_a_baseline_says_so() -> None:
    """A directly built receipt (no recorded baseline) names what its cells are."""

    receipt = tviz.CausalReceipt(
        layer="blk.0.attn",
        heads=(0,),
        effects=(1.0,),
        metric="m",
        intervention="head contribution zeroed",
        engine="rerun",
        disclosure="direct",
        fires=1,
        negative_control=0.0,
        positive_control=1.0,
        joint_effect=1.0,
    )
    assert receipt.baseline_run == "unpatched baseline run"
    lines = _receipts._receipt_disclosures([receipt], None)
    assert "each cell: change in m under the intervention (baseline value not recorded)" in lines


def test_non_finite_baseline_refuses() -> None:
    with pytest.raises(tviz.RenderError) as caught:
        _receipt(_grid(torch.zeros(1, 2), corrupted_metric=float("nan")))
    assert caught.value.fields["code"] == "tv_receipt_invalid"


def _capture_figure(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Keep the figure render_receipt_grid saves, for geometry checks."""

    figures: list[Any] = []
    real = _receipts.save_figure

    def keep(fig: Any, path: Any, **kwargs: Any) -> Path:
        figures.append(fig)
        return real(fig, path, **kwargs)

    monkeypatch.setattr(_receipts, "save_figure", keep)
    return figures


@pytest.mark.parametrize("kv", [False, True], ids=["query-heads", "kv-groups"])
@pytest.mark.parametrize("n_rows", [1, 2, 6])
def test_footer_never_overlaps_matrix_ticks_or_colorbar_label(
    n_rows: int, kv: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The footer's window extent misses the axes, every tick label and the bar label."""

    figures = _capture_figure(monkeypatch)
    gqa = tviz.GqaInfo(n_query_heads=8, n_kv_heads=4) if kv else None
    n_heads = 4
    values = torch.linspace(-1.0, 1.0, n_rows * n_heads).reshape(n_rows, n_heads)
    grid = _grid(values, corrupted_metric=0.0)
    rows = [_receipt(grid, layer=f"blk.{i}.attn", kv_groups=gqa) for i in range(n_rows)]
    artifact = tviz.render_receipt_grid(rows, tmp_path / "footer.png", joint_effect=0.5)
    assert len(artifact.disclosure_lines) >= 6

    (fig,) = figures
    renderer = fig.canvas.get_renderer()
    fig.draw(renderer)
    footer = next(text for text in fig.texts if "controls:" in text.get_text())
    footer_box = footer.get_window_extent(renderer)
    ax, bar_ax = fig.axes
    others = [
        ("axes", ax.get_window_extent(renderer)),
        ("colorbar", bar_ax.get_window_extent(renderer)),
        ("colorbar label", bar_ax.yaxis.label.get_window_extent(renderer)),
        ("title", fig._suptitle.get_window_extent(renderer)),
    ]
    labels = [*ax.get_xticklabels(), *ax.get_yticklabels(), *bar_ax.get_yticklabels()]
    others += [
        (f"tick {t.get_text()!r}", t.get_window_extent(renderer)) for t in labels if t.get_text()
    ]
    for name, box in others:
        assert not footer_box.overlaps(box), name
    # Numbers on the darkest cells are white, the rest black.
    colours = {t.get_text(): t.get_color() for t in ax.texts}
    assert colours[f"{values.max().item():+.2f}"] == "white"
    # Every text is at least 8 pt (about 11 CSS px at the pane target).
    texts = [footer, *labels, *ax.texts, bar_ax.yaxis.label]
    assert min(text.get_fontsize() for text in texts) >= 8
    # The footer is wrapped to end inside the figure.
    assert footer_box.x1 <= fig.bbox.x1
    # The colorbar label stays within the figure's vertical extent.
    label_box = bar_ax.yaxis.label.get_window_extent(renderer)
    assert label_box.y0 >= footer_box.y1 and label_box.y1 <= fig.bbox.y1
