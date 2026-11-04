"""Image views of a bare tensor: ``tl.viz.rgb`` and ``tl.viz.chans`` (views.4)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import torchlens as tl
from torchlens.viz._rgb_scale import rgb_unit_hwc


def _pixels(picture) -> np.ndarray:
    return np.asarray(picture.convert("RGB"))


def test_importing_torchlens_adds_nothing_to_tensors() -> None:
    assert not hasattr(torch.zeros(1), "rgb")
    assert not hasattr(torch.zeros(1), "chans")


def test_rgb_draws_channel_first_pixels_exactly() -> None:
    image = torch.zeros(3, 2, 2)
    image[0, 0, 0] = 1.0  # top-left red
    image[1, 1, 1] = 1.0  # bottom-right green
    picture = tl.viz.rgb(image, normalize="clip", max_size=2)
    assert picture.mode == "RGB" and picture.size == (2, 2)
    pixels = _pixels(picture)
    assert tuple(pixels[0, 0]) == (255, 0, 0)
    assert tuple(pixels[1, 1]) == (0, 255, 0)
    assert tuple(pixels[0, 1]) == (0, 0, 0)


def test_channels_last_and_batch_give_the_same_picture() -> None:
    torch.manual_seed(0)
    image = torch.rand(3, 5, 7)
    first = _pixels(tl.viz.rgb(image, normalize="clip", max_size=7))
    last = _pixels(
        tl.viz.rgb(image.permute(1, 2, 0), channels_last=True, normalize="clip", max_size=7)
    )
    batched = _pixels(tl.viz.rgb(torch.stack([image, -image]), normalize="clip", max_size=7))
    assert first.shape == (5, 7, 3)
    assert np.array_equal(first, last) and np.array_equal(first, batched)


def test_imagenet_normalization_round_trips() -> None:
    torch.manual_seed(1)
    image = torch.rand(3, 4, 4)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    normalized = (image - mean) / std
    back = rgb_unit_hwc(normalized, normalize="imagenet")
    torch.testing.assert_close(back, image.permute(1, 2, 0), rtol=0, atol=1e-6)


def test_minmax_matches_show_rgb_scaling() -> None:
    image = torch.tensor([[[-2.0, 0.0]], [[2.0, 1.0]], [[0.0, -1.0]]])  # (3, 1, 2)
    scaled = rgb_unit_hwc(image)
    assert float(scaled.min()) == 0.0 and float(scaled.max()) == 1.0
    torch.testing.assert_close(scaled[0, 0], torch.tensor([0.0, 1.0, 0.5]))


def test_small_images_grow_by_whole_pixels_and_large_ones_shrink() -> None:
    small = tl.viz.rgb(torch.rand(3, 8, 10))
    assert small.size == (300, 240)
    large = tl.viz.rgb(torch.rand(3, 400, 200), max_size=100)
    assert max(large.size) == 100


def test_rgb_refuses_a_tensor_without_three_channels() -> None:
    with pytest.raises(ValueError, match="chans"):
        tl.viz.rgb(torch.rand(4, 8, 8))
    with pytest.raises(ValueError, match="channels_last=True"):
        tl.viz.rgb(torch.rand(8, 8, 3))
    with pytest.raises(ValueError, match="normalize"):
        tl.viz.rgb(torch.rand(3, 8, 8), normalize="zscore")
    with pytest.raises(TypeError):
        tl.viz.rgb(np.zeros((3, 4, 4)))


def test_chans_is_channel_grid_on_the_tensor() -> None:
    torch.manual_seed(2)
    activation = torch.randn(1, 6, 8, 8)
    grid = tl.viz.chans(activation, n=4, max_size=64)
    expected = tl.viz.nodes.channel_grid(n=4, max_size=64)(activation)
    assert np.array_equal(_pixels(grid), _pixels(expected))
    with pytest.raises(ValueError, match="chans"):
        tl.viz.chans(torch.randn(8))


def test_show_rgb_draws_an_unbatched_image() -> None:
    pytest.importorskip("matplotlib")
    from torchlens.viz._tensor_display import show_tensor

    figure = show_tensor(torch.rand(3, 6, 6), method="rgb")
    assert hasattr(figure, "savefig")
