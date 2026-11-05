"""Tests for C2 feature-map graph annotations."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
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
    feature_map_node_spec,
    feature_maps,
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
    return tl.trace(
        _TinyConv().eval(),
        _input_batch(n_stimuli),
        save=selector,
        capture=tl.options.CaptureOptions(seed=123),
    )


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


def test_non_spatial_selection_raises_and_mixed_selection_skips_dense() -> None:
    """Only spatial sites should be annotated, with dense-only guidance."""

    dense_trace = tl.trace(_DenseOnly().eval(), torch.ones(2, 4), save=tl.func("linear"))
    with pytest.raises(ValueError, match=r"spatial \[N, C, H, W\].*saw layer:linear.*save=.*conv"):
        feature_maps(dense_trace)

    mixed_trace = _conv_trace(save=tl.func("conv2d") | tl.func("linear"), n_stimuli=2)
    maps = feature_maps(mixed_trace)

    assert list(maps) == ["layer:conv2d_1_1"]
    assert mixed_trace._annotation_blobs is not None
    assert "featmap:layer:linear_1_1:maps" not in mixed_trace._annotation_blobs


def test_feature_map_node_spec_renders_one_image_and_overlay_differs_from_fallback(
    tmp_path: Path,
) -> None:
    """Node spec should render one contained image and support overlay fallback."""

    trace = _conv_trace(n_stimuli=3)
    trace.input = _pil_stimuli(3)
    feature_maps(trace, max_stimuli=2)
    dot = trace.draw(
        node_spec_fn=feature_map_node_spec(max_stimuli=2, max_channels=1),
        file_format="svg",
        path=str(tmp_path / "feature_graph"),
    ).dot

    image_paths = list((Path(trace._visualizer_dir) / "feature_maps").glob("*.png"))
    assert len(image_paths) == 1
    # R19-6: node image attrs are visualizer-relative. REVIEWED rebaseline
    # (T9, grind-p3): the mkdtemp root no longer appears in the source AT
    # ALL — it reaches Graphviz as the render subprocess cwd, so user-saved
    # DOT stays renderable after the trace's scratch dir is GC'd.
    relative_image = image_paths[0].relative_to(trace._visualizer_dir)
    assert f'image="{relative_image}"' in dot
    assert "imagepath" not in dot
    assert f'image="{image_paths[0]}"' not in dot
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
        raw_images=trace.input,
        overlay=True,
        alpha=0.55,
        cmap="magma",
        cell_size=72,
        cap_text=None,
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
        cap_text=None,
    )
    assert ImageChops.difference(overlay_image, fallback_image).getbbox() is not None
    assert int(counts[2].item()) == 2


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
        cap_text="+4 more",
    )

    assert image.size == (68, 68)
    assert np.asarray(image).shape == (68, 68, 3)


def test_evolution_preserves_nan_from_activations() -> None:
    """End-to-end: a NaN activation survives into the stored annotation blob."""

    poisoned = _input_batch(2)
    poisoned[0, 0, 1, 1] = float("nan")
    trace = tl.trace(_TinyConv().eval(), poisoned, save=tl.func("conv2d"))
    feature_maps(trace, save=tl.func("conv2d"), channels=[0])

    stored = trace._annotation_blobs["featmap:layer:conv2d_1_1:maps"]
    assert torch.isnan(stored[0, 0, 1, 1])


def test_constant_cells_are_labeled_and_distinct_from_zero() -> None:
    """Constant maps carry their value; constant-5 never renders as all-zero."""

    shape = (1, 1, 4, 4)
    constant_five = torch.full(shape, 5.0)
    all_zero = torch.zeros(shape)
    kwargs: dict[str, Any] = {
        "raw_images": None,
        "overlay": False,
        "alpha": 0.55,
        "cmap": "magma",
        "cell_size": 72,
        "cap_text": None,
    }
    stimuli = torch.tensor([0])
    channels = torch.tensor([[0]])

    five_image = _render_feature_map_grid(constant_five, stimuli, channels, **kwargs)
    zero_image = _render_feature_map_grid(all_zero, stimuli, channels, **kwargs)

    assert ImageChops.difference(five_image, zero_image).getbbox() is not None


def test_tooltip_reports_true_totals_not_capped_counts(tmp_path: Path) -> None:
    """A 4-of-6 stimuli x 4-of-8 channel grid must say so, not '4 stimuli'."""

    wide = nn.Sequential(nn.Conv2d(1, 8, kernel_size=1, bias=False)).eval()
    with torch.no_grad():
        wide[0].weight.copy_(torch.arange(1.0, 9.0).reshape(8, 1, 1, 1))
    trace = tl.trace(wide, _input_batch(6), save=tl.func("conv2d"))
    feature_maps(trace, channels="top", k=8, max_stimuli=4, max_channels=4)

    dot = trace.draw(
        node_spec_fn=feature_map_node_spec(),
        file_format="svg",
        path=str(tmp_path / "true_totals"),
    ).dot

    assert "4 of 6 stimuli" in dot
    assert "4 of 8 channels" in dot
    # The capped count must never masquerade as the total.
    assert "4 stimuli" not in dot


def test_aggregate_labels_report_actual_reduction(tmp_path: Path) -> None:
    """A max-projection is captioned max, never avg (reduce_id is honored)."""

    trace = _conv_trace(n_stimuli=2)
    feature_maps(trace, reduce="max")
    dot = trace.draw(
        node_spec_fn=feature_map_node_spec(),
        file_format="svg",
        path=str(tmp_path / "reduce_label"),
    ).dot

    assert "aggregate (max)" in dot

    maps = trace._annotation_blobs["featmap:layer:conv2d_1_1:maps"]
    stimuli = trace._annotation_blobs["featmap:layer:conv2d_1_1:stimuli"]
    channels = trace._annotation_blobs["featmap:layer:conv2d_1_1:channels"]
    kwargs: dict[str, Any] = {
        "raw_images": None,
        "overlay": False,
        "alpha": 0.55,
        "cmap": "magma",
        "cell_size": 72,
        "cap_text": None,
    }
    as_max = _render_feature_map_grid(maps, stimuli, channels, reduce_label="max", **kwargs)
    as_avg = _render_feature_map_grid(maps, stimuli, channels, reduce_label="avg", **kwargs)
    assert ImageChops.difference(as_max, as_avg).getbbox() is not None


class _RecurrentBlockModel(nn.Module):
    """Stem conv (annotated) plus a block conv called twice (multi-pass)."""

    def __init__(self) -> None:
        """Initialize the stem and the reused block."""

        super().__init__()
        self.stem = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        self.block = nn.Conv2d(3, 3, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stem, then the block twice (a recurrent layer)."""

        h = self.stem(x)
        h = self.block(h)
        return self.block(h)


def test_rolled_draw_with_annotations_survives_multipass_layers(tmp_path: Path) -> None:
    """An UNRELATED multi-pass layer must never crash an annotated draw().

    ``Layer.label`` raises the multi-pass ValueError tripwire on rolled
    recurrent aggregates; ``getattr(node, "label", None)`` does not swallow
    it. The annotation lookups must use the ``get_multipass_attr`` idiom.
    """

    trace = tl.trace(_RecurrentBlockModel().eval(), _input_batch(2), save=tl.in_module("stem"))
    feature_maps(trace, save=tl.in_module("stem"))

    dot = trace.draw(
        rolled=True,
        node_spec_fn=feature_map_node_spec(),
        file_format="svg",
        path=str(tmp_path / "rolled_annotated"),
    ).dot
    assert "Feature maps for" in dot

    from torchlens.repgeom import (
        _mds_scatter_coords_for_node,
        _rdm_matrix_for_node,
        _scree_eigenvalues_for_node,
    )

    multipass_layer = trace.layers["conv2d_2_2"]
    trace._annotation_blobs = trace._annotation_blobs or {}
    # Each repgeom lookup shares the same candidate-building block; a
    # multi-pass node must skip the ambiguous op label, not raise.
    assert _scree_eigenvalues_for_node(trace, multipass_layer) == (None, None)
    assert _rdm_matrix_for_node(trace, multipass_layer) == (None, None)
    assert _mds_scatter_coords_for_node(trace, multipass_layer) == (None, None)
