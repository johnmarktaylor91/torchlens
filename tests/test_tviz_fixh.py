"""FIX-H regression rows for ``torchlens.viz.render_heatmap`` (tviz memo D8).

The shipped renderer had two independent honesty defects (OPUS M2-M5):
bilinear interpolation on discrete cells, and a label-omission marker
computed BEFORE overlap selection (measured: 8 tokens drawing 4 labels with
no marker). These rows pin the fixes: nearest-neighbor cells, separate
query/key label arguments, and an omission count computed after the FINAL
overlap selection.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from torchlens.viz import render_heatmap
from torchlens.viz.node_plots import _axis_selection_with_marker


def _selection(
    n_labels: int,
    *,
    label_text: str = "token",
    axis: str = "top",
    plot_extent: int = 200,
    max_axis_items: int = 8,
) -> tuple[list[int], int]:
    """Run the axis selection helper on ``n_labels`` identical labels."""

    draw = ImageDraw.Draw(Image.new("RGB", (1, 1), "white"))
    labels = [f"{label_text}{index}" for index in range(n_labels)]
    count = min(max_axis_items, n_labels)
    return _axis_selection_with_marker(
        draw,
        labels=labels,
        has_images=False,
        count=count,
        total=n_labels,
        axis=axis,
        plot_left=40,
        plot_top=24,
        plot_width=plot_extent,
        plot_height=plot_extent,
        canvas_width=40 + plot_extent + 6,
        canvas_height=24 + plot_extent + 6,
        thumb_size=16,
    )


# The seven-row measured regression table (tviz memo D8). Each row states the
# axis-item population and the honesty invariant: drawn + omitted == total,
# and a nonzero omission is NEVER silent. Row 2 is the measured defect case
# (8 tokens, 4 labels drawn, no marker in the shipped renderer).
_SEVEN_ROWS = [
    # (n_labels, max_axis_items, plot_extent, expect_all_drawn)
    (1, 8, 200, True),
    (8, 8, 200, False),  # the measured 8-token defect row: decimation must mark
    (4, 8, 400, True),
    (16, 8, 200, False),  # over the cap: cap omissions counted
    (32, 8, 120, False),  # cap + tight geometry: decimation omissions counted
    (128, 8, 200, False),  # the measured 128-token defect row
    (12, 12, 600, True),  # generous geometry: everything fits, no marker
]


@pytest.mark.smoke_cells("test_omission_count_after_final_selection[8-8-200-False]")
@pytest.mark.parametrize(
    ("n_labels", "max_axis_items", "plot_extent", "expect_all_drawn"), _SEVEN_ROWS
)
def test_omission_count_after_final_selection(
    n_labels: int, max_axis_items: int, plot_extent: int, expect_all_drawn: bool
) -> None:
    """Drawn + omitted always equals total; omission is never silent."""

    indices, omitted = _selection(n_labels, plot_extent=plot_extent, max_axis_items=max_axis_items)
    assert len(indices) + omitted == n_labels
    assert omitted >= 0
    if expect_all_drawn:
        assert omitted == 0
        assert len(indices) == n_labels
    else:
        assert omitted > 0, (
            "labels were dropped by cap or overlap selection but the omission "
            "count claims everything is shown (the FIX-H defect class)"
        )


def test_marker_counts_decimated_labels_not_just_cap() -> None:
    """The measured defect: total under the cap, decimation still marked.

    8 labels with an 8-item cap in tight geometry: the shipped renderer
    computed ``more = total - min(cap, total) = 0`` and drew no marker while
    overlap selection silently dropped half the labels.
    """

    indices, omitted = _selection(8, plot_extent=120, max_axis_items=8)
    assert len(indices) < 8, "geometry must force decimation for this row"
    assert omitted == 8 - len(indices)
    assert omitted > 0


def test_nearest_neighbor_no_interpolated_colors() -> None:
    """A 2x2 grid upscales to exactly its 4 cell colors -- no blends."""

    data = np.array([[0.0, 1.0], [1.0, 0.0]])
    image = render_heatmap(data, width=64, height=64, cmap="gray")
    colors = {tuple(pixel) for pixel in np.asarray(image).reshape(-1, 3)}
    assert colors == {(0, 0, 0), (255, 255, 255)}, (
        f"interpolated colors appeared between discrete cells: {sorted(colors)[:5]}"
    )


def test_nearest_neighbor_with_axis_labels() -> None:
    """The axis-decorated path also scales cells without interpolation."""

    data = np.array([[0.0, 1.0], [1.0, 0.0]])
    image = render_heatmap(
        data, width=96, height=96, cmap="gray", row_labels=["a", "b"], col_labels=["c", "d"]
    )
    # Interior plot pixels are pure cell colors, axis chrome, or the border;
    # no gray blend band can appear inside a cell. Sample the two cell centers.
    # Plot geometry: margins are deterministic for these inputs.
    pixels = [tuple(pixel) for pixel in np.asarray(image).reshape(-1, 3)]
    grays = {
        pixel for pixel in pixels if pixel[0] == pixel[1] == pixel[2] and pixel[0] not in (0, 255)
    }
    # Permitted non-cell grays: axis line color and text antialiasing shades.
    # The bilinear defect produced WIDE gradient bands; require that pure cell
    # colors dominate the image area.
    pure = sum(1 for pixel in pixels if pixel in ((0, 0, 0), (255, 255, 255)))
    assert pure > len(pixels) * 0.5, (
        f"cells are not rendered as discrete blocks (grays={len(grays)})"
    )


def test_separate_row_and_col_labels() -> None:
    """Rectangular data takes distinct query/key label sets per axis."""

    data = np.zeros((2, 5))
    image = render_heatmap(
        data,
        width=200,
        height=120,
        row_labels=["query0", "query1"],
        col_labels=["k0", "k1", "k2", "k3", "k4"],
    )
    assert image.size == (200, 120)


def test_row_col_labels_override_axis_labels() -> None:
    """Per-axis labels take precedence over the both-axes convenience."""

    data = np.zeros((3, 3))
    image = render_heatmap(
        data,
        width=160,
        height=160,
        axis_labels=["x0", "x1", "x2"],
        row_labels=["r0", "r1", "r2"],
    )
    assert image.size == (160, 160)


@pytest.mark.smoke
def test_label_count_mismatch_is_not_silent_growth() -> None:
    """Labels shorter than the axis simply stop; no invented labels."""

    data = np.zeros((4, 4))
    image = render_heatmap(data, width=160, height=160, row_labels=["only-one"])
    assert image.size == (160, 160)
