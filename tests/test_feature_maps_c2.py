"""Tests for C2 feature-map graph annotations."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image, ImageChops, ImageDraw
from torch import nn

import torchlens as tl
from torchlens.viz.feature_maps import (
    _feature_map_payload_for_node,
    _render_feature_map_grid,
    feature_map_evolution,
    feature_map_node_spec,
)


class _TinyConv(nn.Module):
    """Small deterministic conv model for spatial activation tests."""

    def __init__(self) -> None:
        """Initialize the conv and dense heads."""

        super().__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3 * 4 * 4, 2, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.tensor([[[[1.0]]], [[[2.0]]], [[[-1.0]]]]))
            self.fc.weight.fill_(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a conv layer followed by a dense projection."""

        return self.fc(self.flatten(self.conv(x)))


class _DenseOnly(nn.Module):
    """Small non-spatial model for validation tests."""

    def __init__(self) -> None:
        """Initialize a dense layer."""

        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the dense layer."""

        return self.fc(x)


class _TieConv(nn.Module):
    """Conv fixture with deterministic top-channel ties."""

    def __init__(self) -> None:
        """Initialize tied absolute-response channels."""

        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=1, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.tensor([[[[2.0]]], [[[-2.0]]], [[[1.0]]], [[[-0.5]]]]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the tied conv layer."""

        return self.conv(x)


def _input_batch(n_stimuli: int = 4) -> torch.Tensor:
    """Return deterministic image-like inputs.

    Parameters
    ----------
    n_stimuli:
        Batch size.

    Returns
    -------
    torch.Tensor
        Input tensor with shape ``[N, 1, 4, 4]``.
    """

    values = torch.arange(n_stimuli * 16, dtype=torch.float32).reshape(n_stimuli, 1, 4, 4)
    return values / values.max().clamp_min(1.0)


def _conv_trace(save: Any = None, *, n_stimuli: int = 4) -> tl.Trace:
    """Capture the tiny conv fixture.

    Parameters
    ----------
    save:
        Save selector.
    n_stimuli:
        Batch size.

    Returns
    -------
    tl.Trace
        Captured trace.
    """

    selector = tl.func("conv2d") if save is None else save
    return tl.trace(_TinyConv().eval(), _input_batch(n_stimuli), save=selector, random_seed=123)


def _pil_stimuli(n_stimuli: int = 4) -> list[Image.Image]:
    """Return deterministic PIL stimulus images.

    Parameters
    ----------
    n_stimuli:
        Number of images.

    Returns
    -------
    list[Image.Image]
        RGB images.
    """

    images = []
    for index in range(n_stimuli):
        image = Image.new("RGB", (18, 18), color=(40 + index * 30, 80, 170 - index * 20))
        draw = ImageDraw.Draw(image)
        draw.rectangle((index + 1, 2, 15, 15), outline=(255, 255, 255))
        images.append(image)
    return images


def _single_feature_node(trace: tl.Trace) -> Any:
    """Return the first annotated node from ``trace``.

    Parameters
    ----------
    trace:
        Trace with feature-map annotations.

    Returns
    -------
    Any
        Layer node with stored feature-map payloads.
    """

    for layer in trace.layers:
        key, payload = _feature_map_payload_for_node(trace, layer)
        if key is not None and payload is not None:
            return layer
    raise AssertionError("no annotated feature-map node found")


def test_aggregate_default_stores_maps_ids_and_counts() -> None:
    """Default feature maps should store aggregate maps and exact metadata."""

    trace = _conv_trace(n_stimuli=3)
    maps_by_key = feature_map_evolution(trace)

    assert list(maps_by_key) == ["layer:conv2d_1_1"]
    stored = trace._annotation_blobs
    assert stored is not None
    assert sorted(stored) == [
        "featmap:layer:conv2d_1_1:channels",
        "featmap:layer:conv2d_1_1:counts",
        "featmap:layer:conv2d_1_1:maps",
        "featmap:layer:conv2d_1_1:stimuli",
    ]
    assert stored["featmap:layer:conv2d_1_1:maps"].shape == (3, 1, 4, 4)
    assert torch.equal(stored["featmap:layer:conv2d_1_1:channels"], torch.full((3, 1), -1))
    assert torch.equal(stored["featmap:layer:conv2d_1_1:stimuli"], torch.tensor([0, 1, 2]))
    assert torch.equal(
        stored["featmap:layer:conv2d_1_1:counts"],
        torch.tensor([3, 1, 3, 1, 0, 0]),
    )


def test_explicit_and_top_channels_store_deterministic_ids() -> None:
    """Explicit and top channel modes should store deterministic channel ids."""

    trace = _conv_trace(n_stimuli=2)
    feature_map_evolution(trace, channels=[2, 0], max_channels=2)
    stored = trace._annotation_blobs
    assert stored is not None
    assert torch.equal(
        stored["featmap:layer:conv2d_1_1:channels"],
        torch.tensor([[2, 0], [2, 0]]),
    )
    assert torch.equal(
        stored["featmap:layer:conv2d_1_1:counts"],
        torch.tensor([2, 3, 2, 2, 1, 0]),
    )

    tie_trace = tl.trace(_TieConv().eval(), _input_batch(2), save=tl.func("conv2d"))
    feature_map_evolution(tie_trace, channels="top", top_k=3, max_channels=3)

    assert torch.equal(
        tie_trace._annotation_blobs["featmap:layer:conv2d_1_1:channels"],
        torch.tensor([[0, 1, 2], [0, 1, 2]]),
    )
    assert torch.equal(
        tie_trace._annotation_blobs["featmap:layer:conv2d_1_1:counts"],
        torch.tensor([2, 4, 2, 3, 2, 0]),
    )


def test_non_spatial_selection_raises_and_mixed_selection_skips_dense() -> None:
    """Only spatial sites should be annotated, with dense-only guidance."""

    dense_trace = tl.trace(_DenseOnly().eval(), torch.ones(2, 4), save=tl.func("linear"))
    with pytest.raises(ValueError, match=r"spatial \[N, C, H, W\].*saw layer:linear.*save=.*conv"):
        feature_map_evolution(dense_trace)

    mixed_trace = _conv_trace(save=tl.func("conv2d") | tl.func("linear"), n_stimuli=2)
    maps = feature_map_evolution(mixed_trace)

    assert list(maps) == ["layer:conv2d_1_1"]
    assert mixed_trace._annotation_blobs is not None
    assert "featmap:layer:linear_1_1:maps" not in mixed_trace._annotation_blobs


def test_feature_map_node_spec_renders_one_image_and_overlay_differs_from_fallback(
    tmp_path: Path,
) -> None:
    """Node spec should render one contained image and support overlay fallback."""

    trace = _conv_trace(n_stimuli=3)
    trace.raw_input = _pil_stimuli(3)
    feature_map_evolution(trace, max_stimuli=2)
    dot = trace.draw(
        node_spec_fn=feature_map_node_spec(max_stimuli=2, max_channels=1),
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / "feature_graph"),
    )

    image_paths = list((Path(trace._visualizer_dir) / "feature_maps").glob("*.png"))
    assert len(image_paths) == 1
    assert str(image_paths[0]) in dot
    image = Image.open(image_paths[0])
    assert image.size == (72, 148)

    node = _single_feature_node(trace)
    _, payload = _feature_map_payload_for_node(trace, node)
    assert payload is not None
    maps, stimulus_indices, channel_ids, counts = payload
    overlay_image = _render_feature_map_grid(
        maps[:1],
        stimulus_indices[:1],
        channel_ids[:1],
        raw_images=trace.raw_input,
        overlay=True,
        alpha=0.55,
        cmap="magma",
        cell_size=72,
        more_count=0,
    )
    fallback_image = _render_feature_map_grid(
        maps[:1],
        stimulus_indices[:1],
        channel_ids[:1],
        raw_images=None,
        overlay=False,
        alpha=0.55,
        cmap="magma",
        cell_size=72,
        more_count=0,
    )
    assert ImageChops.difference(overlay_image, fallback_image).getbbox() is not None
    assert int(counts[2].item()) == 2


def test_tlspec_roundtrip_preserves_maps_and_renders_after_load(tmp_path: Path) -> None:
    """Feature-map tensors should survive portable save/load and render after load."""

    trace = _conv_trace(n_stimuli=2)
    feature_map_evolution(trace)
    original = trace._annotation_blobs["featmap:layer:conv2d_1_1:maps"].clone()
    bundle_path = tmp_path / "feature_maps.tlspec"

    tl.save(trace, bundle_path)
    restored = tl.load(bundle_path)

    assert restored._annotation_blobs is not None
    assert torch.equal(restored._annotation_blobs["featmap:layer:conv2d_1_1:maps"], original)
    dot = restored.draw(
        node_spec_fn=feature_map_node_spec(),
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / "restored_feature_graph"),
    )
    assert "feature_maps" in dot


def test_default_draw_without_hook_is_byte_identical_and_plain_trace_has_no_blobs(
    tmp_path: Path,
) -> None:
    """Default draw should not change when no feature-map hook is installed."""

    trace = _conv_trace(n_stimuli=1)
    assert trace._annotation_blobs is None

    first = trace.draw(
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / "first"),
    )
    second = trace.draw(
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / "second"),
    )

    assert trace._annotation_blobs is None
    assert first == second


def test_import_feature_maps_does_not_import_matplotlib() -> None:
    """Importing feature maps should not import matplotlib."""

    code = (
        "import sys; import torchlens.viz.feature_maps; "
        "raise SystemExit(1 if 'matplotlib' in sys.modules else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], check=False)

    assert result.returncode == 0


def test_render_grid_cap_marker_is_contained() -> None:
    """The grid renderer should keep cap markers within a bounded image."""

    maps = torch.arange(5 * 3 * 4 * 4, dtype=torch.float32).reshape(5, 3, 4, 4)
    image = _render_feature_map_grid(
        maps[:2, :2],
        torch.tensor([0, 1]),
        torch.tensor([[0, 1], [0, 1]]),
        raw_images=None,
        overlay=False,
        alpha=0.55,
        cmap="magma",
        cell_size=32,
        more_count=4,
    )

    assert image.size == (68, 68)
    assert np.asarray(image).shape == (68, 68, 3)
