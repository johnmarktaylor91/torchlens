"""Tests for PIL-only node plot primitives."""

from __future__ import annotations

import subprocess
import sys
from io import BytesIO

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from torchlens.viz import heatmap, image_scatter, lineplot
from torchlens.viz._tensor_display import show_tensor


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


def test_render_heatmap_unknown_cmap_raises() -> None:
    """Unknown colormap names should raise a clear ValueError."""

    with pytest.raises(ValueError, match="Unknown colormap"):
        heatmap(np.zeros((2, 2)), cmap="plasma")


def test_render_heatmap_axis_thumbnails_bounded() -> None:
    """Capped axis thumbnails should stay within the requested image bounds."""

    images = _solid_images(12)
    labels = [f"item-{index}" for index in range(12)]

    image = heatmap(
        np.arange(144, dtype=np.float64).reshape(12, 12),
        width=180,
        height=140,
        axis_images=images,
        axis_labels=labels,
        max_axis_items=5,
    )

    assert image.mode == "RGB"
    assert image.size == (180, 140)


def test_render_lineplot_swaps_inverted_y_bounds() -> None:
    """Accidentally swapped y bounds should render like their sorted equivalent."""

    values = np.linspace(0.0, 10.0, 12)
    sorted_bounds = lineplot(values, width=180, height=110, y_min=0.0, y_max=10.0)
    inverted_bounds = lineplot(values, width=180, height=110, y_min=10.0, y_max=0.0)

    assert _png_bytes(inverted_bounds) == _png_bytes(sorted_bounds)


def test_show_tensor_handles_zero_channel_3d_tensor() -> None:
    """Default tensor display should not crash on a zero-channel 3D tensor."""

    plt = pytest.importorskip("matplotlib.pyplot")
    fig = show_tensor(torch.zeros(0, 4, 4))
    try:
        assert fig is not None
        assert not isinstance(fig, str)
    finally:
        plt.close(fig)


def test_render_image_scatter_points_and_thumbnails() -> None:
    """Scatter rendering should work with point fallback, thumbnails, and cap markers."""

    coords = np.column_stack([np.linspace(-1.0, 1.0, 20), np.cos(np.linspace(0.0, 3.0, 20))])
    points = image_scatter(coords, max_items=8, canvas_size=180, thumbnail_size=20)
    thumbnails = image_scatter(
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


def test_render_image_scatter_accepts_torch_and_numpy() -> None:
    """Scatter coordinates should accept both torch tensors and NumPy arrays."""

    coords_np = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 0.5]], dtype=np.float32)
    coords_torch = torch.tensor(coords_np)

    numpy_image = image_scatter(coords_np, canvas_size=160, thumbnail_size=20)
    torch_image = image_scatter(coords_torch, canvas_size=160, thumbnail_size=20)

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


def test_scatter_displacement_is_disclosed_in_image() -> None:
    """Displaced thumbnails carry a quantified spread marker; exact ones do not.

    For a scatter the position IS the datum: when overlap-avoidance moves
    thumbnails away from their true coordinates, the image must say so.
    """

    from torchlens.viz.node_plots import _MORE_OUTLINE

    thumbs = [Image.new("RGB", (24, 24), (200, 40, 40)) for _ in range(2)]
    coincident = np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    displaced = image_scatter(coincident, images=thumbs, canvas_size=420)

    # Bottom-LEFT corner region carries the spread indicator's outline.
    corner = np.asarray(displaced)[-40:, :160]
    outline = np.asarray(_MORE_OUTLINE, dtype=np.uint8)
    assert (corner == outline).all(axis=-1).any(), "no spread disclosure marker drawn"

    separated = np.asarray([[-1.0, -1.0], [1.0, 1.0]], dtype=np.float64)
    exact = image_scatter(separated, images=thumbs, canvas_size=420)
    corner_exact = np.asarray(exact)[-40:, :160]
    assert not (corner_exact == outline).all(axis=-1).any(), (
        "spread marker must not appear when positions are exact"
    )
