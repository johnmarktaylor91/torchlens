"""Tests for PIL-only node plot primitives."""

from __future__ import annotations

from io import BytesIO
import subprocess
import sys

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from torchlens.viz import render_heatmap, render_image_scatter, render_lineplot
from torchlens.viz.node_plots import (
    _coords_to_pixel_centers,
    _measure_text,
    _select_heatmap_axis_indices,
)
from torchlens.viz.node_plots import _spread_close_centers


def _png_bytes(image: Image.Image) -> bytes:
    """Return deterministic in-memory PNG bytes for a PIL image.

    Parameters
    ----------
    image:
        Image to serialize.

    Returns
    -------
    bytes
        PNG-encoded image bytes.
    """

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _solid_images(n_items: int) -> list[Image.Image]:
    """Return deterministic solid-color thumbnail images.

    Parameters
    ----------
    n_items:
        Number of images to create.

    Returns
    -------
    list[Image.Image]
        RGB thumbnail images.
    """

    images = []
    for index in range(n_items):
        image = Image.new(
            "RGB",
            (24, 20),
            color=((37 * index) % 255, (91 + 23 * index) % 255, (170 - 11 * index) % 255),
        )
        draw = ImageDraw.Draw(image)
        draw.text((3, 4), str(index), fill=(255, 255, 255))
        images.append(image)
    return images


def test_render_heatmap_deterministic_size_and_mode() -> None:
    """Heatmap rendering should be fixed-size RGB and same-process deterministic."""

    data = np.arange(25, dtype=np.float64).reshape(5, 5)

    first = render_heatmap(data, width=123, height=91)
    second = render_heatmap(data, width=123, height=91)

    assert first.mode == "RGB"
    assert first.size == (123, 91)
    assert _png_bytes(first) == _png_bytes(second)


def test_render_heatmap_constant_array_no_nan() -> None:
    """A constant finite heatmap should render low-color pixels, not nan-color pixels."""

    image = render_heatmap(np.full((4, 4), 7.0), nan_color=(235, 235, 235))

    assert image.getpixel((10, 10)) != (235, 235, 235)


def test_render_heatmap_unknown_cmap_raises() -> None:
    """Unknown colormap names should raise a clear ValueError."""

    with pytest.raises(ValueError, match="Unknown colormap"):
        render_heatmap(np.zeros((2, 2)), cmap="plasma")


def test_render_heatmap_axis_thumbnails_bounded() -> None:
    """Capped axis thumbnails should stay within the requested image bounds."""

    images = _solid_images(12)
    labels = [f"item-{index}" for index in range(12)]

    image = render_heatmap(
        np.arange(144, dtype=np.float64).reshape(12, 12),
        width=180,
        height=140,
        axis_images=images,
        axis_labels=labels,
        max_axis_items=5,
    )

    assert image.mode == "RGB"
    assert image.size == (180, 140)


def test_heatmap_top_axis_labels_decimate_without_overlap() -> None:
    """Crowded top-axis labels should be decimated to measured non-overlapping items."""

    canvas_width = 130
    plot_left = 28
    plot_width = 96
    labels = [f"state-{index}" for index in range(8)]
    draw = ImageDraw.Draw(Image.new("RGB", (canvas_width, 90), "white"))

    indices = _select_heatmap_axis_indices(
        draw,
        labels=labels,
        has_images=False,
        count=8,
        total=8,
        axis="top",
        plot_left=plot_left,
        plot_top=20,
        plot_width=plot_width,
        plot_height=64,
        canvas_width=canvas_width,
        canvas_height=90,
        thumb_size=12,
        more_text="+4 more",
    )

    assert len(indices) < len(labels)
    marker_width, _marker_height = _measure_text(draw, "+4 more")
    limit_high = canvas_width - marker_width - 8
    previous_end = -1.0
    for index in indices:
        text_width, _text_height = _measure_text(draw, labels[index])
        center = plot_left + (index + 0.5) * plot_width / len(labels)
        start = center - text_width / 2.0
        end = center + text_width / 2.0
        assert start >= 0.0
        assert end <= limit_high
        assert start - previous_end >= 4.0
        previous_end = end


def test_render_lineplot_single_and_multi_series() -> None:
    """Single and stacked series inputs should both render fixed-size images."""

    single = render_lineplot(np.linspace(1.0, 0.0, 12), width=180, height=110)
    multi = render_lineplot(
        np.stack([np.linspace(1.0, 0.0, 12), np.linspace(0.7, 0.2, 12)]),
        labels=["a", "b"],
        width=190,
        height=120,
    )

    assert single.mode == "RGB"
    assert single.size == (180, 110)
    assert multi.mode == "RGB"
    assert multi.size == (190, 120)


def test_render_image_scatter_points_and_thumbnails() -> None:
    """Scatter rendering should work with point fallback, thumbnails, and cap markers."""

    coords = np.column_stack([np.linspace(-1.0, 1.0, 20), np.cos(np.linspace(0.0, 3.0, 20))])
    points = render_image_scatter(coords, max_items=8, canvas_size=180, thumbnail_size=20)
    thumbnails = render_image_scatter(
        coords,
        images=_solid_images(20),
        max_items=8,
        canvas_size=180,
        thumbnail_size=20,
    )

    assert points.mode == "RGB"
    assert points.size == (180, 180)
    assert thumbnails.mode == "RGB"
    assert thumbnails.size == (180, 180)


def test_render_image_scatter_spreads_close_thumbnail_centers() -> None:
    """Near-coincident thumbnail centers should be spread to avoid full occlusion."""

    thumbnail_size = 24
    canvas_size = 220
    coords = np.array(
        [
            [0.00, 0.00],
            [0.01, 0.00],
            [0.00, 0.01],
            [0.01, 0.01],
            [0.02, 0.00],
            [0.00, 0.02],
            [0.02, 0.02],
            [0.03, 0.00],
            [0.00, 0.03],
            [0.03, 0.03],
        ],
        dtype=np.float64,
    )
    margin = thumbnail_size / 2.0 + 56.0
    centers = _coords_to_pixel_centers(coords, canvas_size=canvas_size, margin=margin)

    spread = _spread_close_centers(
        centers,
        canvas_size=canvas_size,
        margin=margin,
        min_distance=float(thumbnail_size),
    )

    for left, left_center in enumerate(spread):
        for right_center in spread[:left]:
            distance = np.hypot(
                left_center[0] - right_center[0],
                left_center[1] - right_center[1],
            )
            assert distance >= thumbnail_size - 1e-6

    image = render_image_scatter(
        coords,
        images=_solid_images(coords.shape[0]),
        max_items=coords.shape[0],
        canvas_size=canvas_size,
        thumbnail_size=thumbnail_size,
    )
    assert image.size == (canvas_size, canvas_size)


def test_render_image_scatter_accepts_torch_and_numpy() -> None:
    """Scatter coordinates should accept both torch tensors and NumPy arrays."""

    coords_np = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 0.5]], dtype=np.float32)
    coords_torch = torch.tensor(coords_np)

    numpy_image = render_image_scatter(coords_np, canvas_size=160, thumbnail_size=20)
    torch_image = render_image_scatter(coords_torch, canvas_size=160, thumbnail_size=20)

    assert numpy_image.size == (160, 160)
    assert torch_image.size == (160, 160)


def test_node_plots_no_matplotlib() -> None:
    """Importing node_plots should not import matplotlib."""

    script = """
import sys
import torchlens.viz.node_plots
print('matplotlib' in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False"
