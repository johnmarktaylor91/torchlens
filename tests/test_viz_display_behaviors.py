"""Behavioral tests for the lightweight viz display surfaces.

Covers the PIL visualizer factories (``tl.viz.heatmap`` / ``channel_grid`` /
``histogram``), the batch summary renderers, ``causal_trace_heatmap``, and the
``Layer.show()`` / ``Op.show()`` tensor display path.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

import torchlens as tl  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every matplotlib figure a test creates."""

    yield
    plt.close("all")


@pytest.mark.smoke
def test_heatmap_visualizer_renders_supported_shapes_and_rejects_others() -> None:
    """heatmap() renders 2-d/3-d/4-d tensors within max_size and skips 1-d."""

    visualizer = tl.viz.heatmap(max_size=32)

    for shape in [(8, 8), (3, 8, 8), (2, 3, 8, 8)]:
        image = visualizer(torch.randn(*shape))
        assert isinstance(image, Image.Image)
        assert max(image.size) <= 32

    big = visualizer(torch.randn(64, 128))
    assert big is not None
    assert max(big.size) <= 32

    assert visualizer(torch.randn(5)) is None


@pytest.mark.smoke
def test_heatmap_constant_tensor_renders_flat_image() -> None:
    """A zero-range tensor normalizes to a flat single-color heatmap."""

    visualizer = tl.viz.heatmap(max_size=16)
    image = visualizer(torch.ones(4, 4))
    assert image is not None
    colors = image.convert("RGB").getcolors()
    assert colors is not None
    assert len(colors) == 1


@pytest.mark.smoke
def test_channel_grid_renders_first_batch_element_and_validates_n() -> None:
    """channel_grid() tiles channels, uses tensor[0] for batches, refuses n<1."""

    with pytest.raises(ValueError, match="at least 1"):
        tl.viz.channel_grid(n=0)

    visualizer = tl.viz.channel_grid(n=4, max_size=64)
    image = visualizer(torch.randn(9, 5, 5))
    assert isinstance(image, Image.Image)
    # 4 channels tile as a 2x2 grid of equal square cells.
    assert image.size[0] == image.size[1]

    batched = visualizer(torch.randn(2, 3, 5, 5))
    assert isinstance(batched, Image.Image)

    assert visualizer(torch.randn(5, 5)) is None


@pytest.mark.smoke
def test_histogram_visualizer_draws_bars_and_skips_empty_input() -> None:
    """histogram() renders finite values and returns None with nothing to draw."""

    with pytest.raises(ValueError, match="at least 1"):
        tl.viz.histogram(bins=0)

    visualizer = tl.viz.histogram(bins=8, width=64, height=48)
    image = visualizer(torch.randn(100))
    assert isinstance(image, Image.Image)
    assert image.size == (64, 48)
    # Bars painted in the histogram blue over the white background.
    assert (66, 133, 244) in {color for _count, color in image.getcolors(maxcolors=4096)}

    all_nan = torch.full((10,), float("nan"))
    assert visualizer(all_nan) is None


@pytest.mark.smoke
def test_montage_tiles_images_and_validates_inputs() -> None:
    """montage() centers thumbnails on a square grid and refuses bad inputs."""

    with pytest.raises(ValueError, match="at least 1"):
        tl.viz.montage([Image.new("RGB", (4, 4))], max_n=0)
    with pytest.raises(ValueError, match="at least one image"):
        tl.viz.montage([], max_n=3)

    images = [Image.new("RGB", (10, 10), color) for color in ("red", "green", "blue")]
    grid = tl.viz.montage(images, max_n=2, max_size=20)
    assert isinstance(grid, Image.Image)
    # Two images tile as 2 columns x 1 row of 10px cells.
    assert grid.size == (20, 10)


@pytest.mark.smoke
def test_text_table_escapes_truncates_and_counts_overflow() -> None:
    """text_table() escapes markup, truncates long items, and adds a +N row."""

    label = tl.viz.text_table(["a<b>", "x" * 100, "third", "fourth"], max_n=2)
    assert label.startswith("<<TABLE")
    assert "a&lt;b&gt;" in label
    assert ("x" * 57 + "...") in label
    assert "x" * 58 not in label
    assert "+2 more" in label

    exact = tl.viz.text_table(["one"], max_n=1)
    assert "more" not in exact


@pytest.mark.smoke
def test_causal_trace_heatmap_renders_2d_scores_and_rejects_1d() -> None:
    """causal_trace_heatmap() plots clipped 2-d scores on matplotlib axes."""

    scores = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    axes = tl.viz.causal_trace_heatmap(scores.numpy(), outlier_perc=None)
    assert axes.images
    plotted = axes.images[0].get_array()
    assert plotted.shape == (3, 4)

    with pytest.raises(ValueError, match="2D"):
        tl.viz.causal_trace_heatmap(torch.ones(5).numpy())


def test_causal_trace_heatmap_sign_filter_clamps_displayed_data() -> None:
    """signs='positive' zeroes negative scores in the displayed array."""

    scores = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])
    axes = tl.viz.causal_trace_heatmap(scores.numpy(), signs="positive", outlier_perc=None)
    plotted = torch.as_tensor(axes.images[0].get_array()).float()
    assert torch.equal(plotted, torch.tensor([[0.0, 2.0], [3.0, 0.0]]))


class _ConvModel(nn.Module):
    """Tiny conv model producing 4-d activations."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.conv(x))


@pytest.mark.smoke
def test_layer_show_auto_routes_by_dimensionality() -> None:
    """Layer.show() picks hist/heatmap/channels/rgb from the payload shape."""

    torch.manual_seed(2)
    log = tl.trace(_ConvModel(), torch.randn(1, 3, 6, 6))

    conv_layer = log["conv2d_1_1"]
    fig = conv_layer.show()  # 4-d, channel dim 4 -> hist fallback
    assert fig is not None and hasattr(fig, "axes")

    fig_heatmap = conv_layer.show(method="heatmap")
    assert fig_heatmap.axes[0].images

    fig_channels = conv_layer.show(method="channels")
    assert len(fig_channels.axes) >= 2

    rgb_input = log["input_1"]
    fig_rgb = rgb_input.show()  # (1, 3, H, W) -> rgb
    assert fig_rgb.axes[0].images

    with pytest.raises(ValueError, match="method must be"):
        conv_layer.show(method="sideways")


@pytest.mark.smoke
def test_show_tensor_handles_raw_tensors_int_dtypes_and_missing_payloads() -> None:
    """show_tensor accepts raw tensors, integer dtypes, and reports no-payload."""

    from torchlens.viz._tensor_display import show_tensor

    fig_1d = show_tensor(torch.randn(16))
    assert fig_1d.axes[0].patches  # histogram bars

    fig_int = show_tensor(torch.arange(9).reshape(3, 3))
    assert fig_int.axes[0].images

    message = show_tensor(object())
    assert message == "No saved tensor out is available to display."


def test_show_on_recurrent_aggregate_layer_requires_pass_selection() -> None:
    """A multi-pass Layer refuses show() with pass-selection guidance."""

    class _Recurrent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(self.linear(x))

    torch.manual_seed(4)
    log = tl.trace(_Recurrent(), torch.randn(2, 3))
    aggregate = log["linear_1_1"]
    assert aggregate.num_passes == 2

    with pytest.raises(ValueError, match="recurrent"):
        aggregate.show()

    single_pass = log["linear_1_1:1"]
    fig = single_pass.show(method="heatmap")
    assert fig.axes[0].images


def test_channel_grid_discloses_hidden_channels() -> None:
    """A capped grid is visibly marked; it never renders as the full set."""

    from PIL import ImageChops

    base = torch.arange(2 * 3 * 3, dtype=torch.float32).reshape(2, 3, 3)
    capped_source = torch.cat([base, base + 1.0, base + 2.0], dim=0)  # 6 channels
    visualizer = tl.viz.channel_grid(n=2, max_size=64)

    capped = visualizer(capped_source)
    complete = visualizer(base)

    assert capped is not None and complete is not None
    # Same two tiles, but the capped render must disclose the 4 hidden channels.
    assert ImageChops.difference(capped, complete).getbbox() is not None


@pytest.mark.smoke
def test_normalize_image_tensor_is_nonfinite_robust() -> None:
    """One bad pixel or a constant image never blacks out the whole tile."""

    from torchlens.visualization._render_nodes import _normalize_image_tensor

    ramp = torch.linspace(0.0, 4.0, steps=12).reshape(1, 3, 4)
    poisoned = ramp.clone()
    poisoned[0, 0, 0] = float("nan")
    poisoned[0, 2, 3] = float("inf")

    normalized = _normalize_image_tensor(poisoned)
    finite_source = torch.isfinite(poisoned)
    # The finite pixels keep their contrast instead of collapsing to black.
    assert float(normalized[finite_source].max()) > 0.5
    assert float(normalized[finite_source].min()) == 0.0

    constant = torch.full((1, 3, 4), 7.5)
    uniform = _normalize_image_tensor(constant)
    # A constant image renders as a uniform mid-gray, not pure black.
    assert torch.allclose(uniform, torch.full_like(uniform, 0.5))
