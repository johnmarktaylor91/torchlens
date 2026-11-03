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

import torchlens
import torchlens.errors
import torchlens.transformer as tviz  # noqa: E402
from torchlens.mechinterp import PatchGrid  # noqa: E402
from torchlens.transformer import _receipts  # noqa: E402

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


def test_grid_without_a_baseline_refuses_with_a_remedy() -> None:
    """Raw patched values never pass as effects."""

    with pytest.raises(torchlens.errors.RenderError) as caught:
        _receipt(_grid(torch.zeros(1, 2)))
    assert caught.value.fields["code"] == "tv_receipt_invalid"
    assert "corrupted_metric" in caught.value.fields["remedy"]
    assert "patch_query_heads_grid" in caught.value.fields["remedy"]


def test_rows_from_different_baselines_refuse(tmp_path: Path) -> None:
    """Cells measured from different baseline runs are not on one scale."""

    first = _receipt(_grid(torch.zeros(1, 2), corrupted_metric=0.0))
    second = _receipt(_grid(torch.zeros(1, 2), corrupted_metric=1.0))
    with pytest.raises(torchlens.errors.RenderError) as caught:
        torchlens.viz.ablation_grid([first, second], tmp_path / "mixed.png", joint_effect=0.0)
    assert caught.value.fields["code"] == "tv_receipt_invalid"


def test_receipt_without_a_baseline_says_so() -> None:
    """A directly built receipt (no recorded baseline) names what its cells are."""

    receipt = tviz.CausalReceipt(
        module_address="blk.0.attn",
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
    with pytest.raises(torchlens.errors.RenderError) as caught:
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
    gqa = tviz.GqaInfo(num_query_heads=8, num_key_value_heads=4) if kv else None
    n_heads = 4
    values = torch.linspace(-1.0, 1.0, n_rows * n_heads).reshape(n_rows, n_heads)
    grid = _grid(values, corrupted_metric=0.0)
    rows = [_receipt(grid, layer=f"blk.{i}.attn", kv_groups=gqa) for i in range(n_rows)]
    artifact = torchlens.viz.ablation_grid(rows, tmp_path / "footer.png", joint_effect=0.5)
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
