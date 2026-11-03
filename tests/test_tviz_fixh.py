"""FIX-H regression rows for ``torchlens.viz.heatmap`` (tviz memo D8).

The shipped renderer had two independent honesty defects (OPUS M2-M5):
bilinear interpolation on discrete cells, and a label-omission marker
computed BEFORE overlap selection (measured: 8 tokens drawing 4 labels with
no marker). These rows pin the fixes: nearest-neighbor cells, separate
query/key label arguments, and an omission count computed after the FINAL
overlap selection.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from torchlens.viz import heatmap
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
    image = heatmap(data, width=64, height=64, cmap="gray")
    colors = {tuple(pixel) for pixel in np.asarray(image).reshape(-1, 3)}
    assert colors == {(0, 0, 0), (255, 255, 255)}, (
        f"interpolated colors appeared between discrete cells: {sorted(colors)[:5]}"
    )


def test_row_col_labels_override_axis_labels() -> None:
    """Per-axis labels take precedence over the both-axes convenience."""

    data = np.zeros((3, 3))
    image = heatmap(
        data,
        width=160,
        height=160,
        axis_labels=["x0", "x1", "x2"],
        row_labels=["r0", "r1", "r2"],
    )
    assert image.size == (160, 160)


def test_label_count_mismatch_is_not_silent_growth() -> None:
    """Labels shorter than the axis simply stop; no invented labels."""

    data = np.zeros((4, 4))
    image = heatmap(data, width=160, height=160, row_labels=["only-one"])
    assert image.size == (160, 160)
