"""Visualization convenience namespace for TorchLens."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
from PIL import Image, ImageDraw

from . import batch_summary
from ..visualization.bundle_diff import bundle_diff

__all__ = [
    "batch_summary",
    "bundle_diff",
    "causal_trace_heatmap",
    "channel_grid",
    "heatmap",
    "histogram",
]

_HEATMAP_COLORS = (
    (68, 1, 84),
    (59, 82, 139),
    (33, 145, 140),
    (94, 201, 98),
    (253, 231, 37),
)


def heatmap(max_size: int = 200) -> Callable[[torch.Tensor], Image.Image | None]:
    """Create a channel-averaged 2D activation heatmap visualizer.

    Parameters
    ----------
    max_size:
        Maximum width or height of the rendered image.

    Returns
    -------
    Callable[[torch.Tensor], Image.Image | None]
        Visualizer callable.
    """

    def visualizer(tensor: torch.Tensor, *, layer_label: str | None = None) -> Image.Image | None:
        """Render a tensor as a channel-averaged heatmap.

        Parameters
        ----------
        tensor:
            Tensor with shape ``(C, H, W)``, ``(B, C, H, W)``, or ``(H, W)``.
        layer_label:
            Optional layer label, accepted for the visualizer contract.

        Returns
        -------
        Image.Image | None
            Rendered heatmap, or ``None`` when the tensor is incompatible.
        """

        _ = layer_label
        data = _to_2d_activation(tensor)
        if data is None:
            return None
        return _resize_image(_array_to_heatmap_image(data), max_size=max_size)

    return visualizer


def channel_grid(n: int = 16, max_size: int = 300) -> Callable[[torch.Tensor], Image.Image | None]:
    """Create a grid visualizer for the first ``n`` activation channels.

    Parameters
    ----------
    n:
        Maximum number of channels to render.
    max_size:
        Maximum width or height of the rendered grid.

    Returns
    -------
    Callable[[torch.Tensor], Image.Image | None]
        Visualizer callable.
    """

    if n < 1:
        raise ValueError("channel_grid n must be at least 1.")

    def visualizer(tensor: torch.Tensor, *, layer_label: str | None = None) -> Image.Image | None:
        """Render a tensor's leading channels in a mosaic.

        Parameters
        ----------
        tensor:
            Tensor with shape ``(C, H, W)`` or ``(B, C, H, W)``.
        layer_label:
            Optional layer label, accepted for the visualizer contract.

        Returns
        -------
        Image.Image | None
            Rendered channel grid, or ``None`` when the tensor is incompatible.
        """

        _ = layer_label
        channels = _to_channel_stack(tensor)
        if channels is None:
            return None
        count = min(n, int(channels.shape[0]))
        cols = int(math.ceil(math.sqrt(count)))
        rows = int(math.ceil(count / cols))
        cell_images = [_array_to_grayscale_image(channels[index]) for index in range(count)]
        cell_size = max(1, max_size // max(rows, cols))
        grid = Image.new("RGB", (cols * cell_size, rows * cell_size), "white")
        for index, image in enumerate(cell_images):
            tile = image.resize((cell_size, cell_size), Image.Resampling.BILINEAR).convert("RGB")
            x = (index % cols) * cell_size
            y = (index // cols) * cell_size
            grid.paste(tile, (x, y))
        return _resize_image(grid, max_size=max_size)

    return visualizer


def histogram(
    bins: int = 30, width: int = 240, height: int = 160
) -> Callable[[torch.Tensor], Image.Image | None]:
    """Create a compact value-distribution histogram visualizer.

    Parameters
    ----------
    bins:
        Number of histogram bins.
    width:
        Rendered image width.
    height:
        Rendered image height.

    Returns
    -------
    Callable[[torch.Tensor], Image.Image | None]
        Visualizer callable.
    """

    if bins < 1:
        raise ValueError("histogram bins must be at least 1.")

    def visualizer(tensor: torch.Tensor, *, layer_label: str | None = None) -> Image.Image | None:
        """Render a tensor's value distribution as a small bar chart.

        Parameters
        ----------
        tensor:
            Tensor whose finite values should be histogrammed.
        layer_label:
            Optional layer label, accepted for the visualizer contract.

        Returns
        -------
        Image.Image | None
            Rendered histogram, or ``None`` when no finite values exist.
        """

        _ = layer_label
        values = _finite_float_tensor(tensor).flatten()
        if values.numel() == 0:
            return None
        counts = torch.histc(values, bins=bins, min=float(values.min()), max=float(values.max()))
        max_count = float(counts.max())
        if max_count <= 0:
            return None
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        margin = 12
        chart_w = max(1, width - 2 * margin)
        chart_h = max(1, height - 2 * margin)
        bar_w = max(1, chart_w / bins)
        draw.line((margin, height - margin, width - margin, height - margin), fill=(60, 60, 60))
        for index, count in enumerate(counts.tolist()):
            bar_h = int((count / max_count) * chart_h)
            x0 = margin + int(index * bar_w)
            x1 = margin + max(x0 + 1, int((index + 1) * bar_w) - 1)
            y0 = height - margin - bar_h
            draw.rectangle((x0, y0, x1, height - margin), fill=(66, 133, 244))
        return image

    return visualizer


def _finite_float_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return finite detached tensor values as CPU float32.

    Parameters
    ----------
    tensor:
        Tensor to normalize.

    Returns
    -------
    torch.Tensor
        One-dimensional finite float tensor.
    """

    with torch.no_grad():
        data = tensor.detach().to(device="cpu", dtype=torch.float32)
        return data[torch.isfinite(data)]


def _to_2d_activation(tensor: torch.Tensor) -> torch.Tensor | None:
    """Reduce a tensor to one finite 2D activation map.

    Parameters
    ----------
    tensor:
        Tensor to reduce.

    Returns
    -------
    torch.Tensor | None
        Two-dimensional finite tensor, or ``None`` for incompatible shapes.
    """

    with torch.no_grad():
        data = tensor.detach().to(device="cpu", dtype=torch.float32)
        if data.ndim == 4:
            data = data.mean(dim=(0, 1))
        elif data.ndim == 3:
            data = data.mean(dim=0)
        elif data.ndim != 2:
            return None
        data = torch.nan_to_num(data)
    return data


def _to_channel_stack(tensor: torch.Tensor) -> torch.Tensor | None:
    """Return a ``(C, H, W)`` activation stack.

    Parameters
    ----------
    tensor:
        Tensor to normalize.

    Returns
    -------
    torch.Tensor | None
        Channel stack or ``None`` for incompatible shapes.
    """

    with torch.no_grad():
        data = tensor.detach().to(device="cpu", dtype=torch.float32)
        if data.ndim == 4:
            data = data[0]
        if data.ndim != 3:
            return None
        return torch.nan_to_num(data)


def _normalize_2d(data: torch.Tensor) -> torch.Tensor:
    """Normalize a 2D tensor to the ``[0, 1]`` interval.

    Parameters
    ----------
    data:
        Two-dimensional tensor.

    Returns
    -------
    torch.Tensor
        Normalized tensor.
    """

    low = float(data.min())
    high = float(data.max())
    if high <= low:
        return torch.zeros_like(data)
    return (data - low) / (high - low)


def _array_to_grayscale_image(data: torch.Tensor) -> Image.Image:
    """Convert one 2D tensor to a grayscale PIL image.

    Parameters
    ----------
    data:
        Two-dimensional tensor.

    Returns
    -------
    Image.Image
        Grayscale image.
    """

    scaled = (_normalize_2d(data) * 255).to(dtype=torch.uint8).numpy()
    return Image.fromarray(scaled, mode="L")


def _array_to_heatmap_image(data: torch.Tensor) -> Image.Image:
    """Convert one 2D tensor to a color heatmap image.

    Parameters
    ----------
    data:
        Two-dimensional tensor.

    Returns
    -------
    Image.Image
        RGB heatmap image.
    """

    normalized = _normalize_2d(data)
    scaled = normalized * (len(_HEATMAP_COLORS) - 1)
    low_indices = torch.floor(scaled).to(dtype=torch.long).clamp(0, len(_HEATMAP_COLORS) - 1)
    high_indices = (low_indices + 1).clamp(0, len(_HEATMAP_COLORS) - 1)
    weights = (scaled - low_indices.to(dtype=torch.float32)).unsqueeze(-1)
    palette = torch.tensor(_HEATMAP_COLORS, dtype=torch.float32)
    colors = palette[low_indices] * (1 - weights) + palette[high_indices] * weights
    return Image.fromarray(colors.to(dtype=torch.uint8).numpy(), mode="RGB")


def _resize_image(image: Image.Image, *, max_size: int) -> Image.Image:
    """Resize an image in-place proportionally within ``max_size``.

    Parameters
    ----------
    image:
        Image to resize.
    max_size:
        Maximum output width or height.

    Returns
    -------
    Image.Image
        Resized image.
    """

    if max(image.size) <= max_size:
        return image
    resized = image.copy()
    resized.thumbnail((max_size, max_size), Image.Resampling.BILINEAR)
    return resized


def causal_trace_heatmap(
    scores: object,
    *,
    signs: str = "all",
    outlier_perc: float | None = 2.0,
    cmap: str = "viridis",
    ax: object | None = None,
) -> object:
    """Render a 2D causal-trace score heatmap.

    Parameters
    ----------
    scores:
        2D array-like patching scores.
    signs:
        Captum-style sign selector: ``"positive"``, ``"negative"``,
        ``"absolute_value"``, or ``"all"``.
    outlier_perc:
        Percentage clipped from each tail before rendering.
    cmap:
        Matplotlib colormap.
    ax:
        Optional matplotlib axes.

    Returns
    -------
    object
        Matplotlib axes containing the heatmap.
    """

    from typing import Any

    import numpy as np

    from ._tensor_display import _clip_outliers, _prepare_signed_data

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for causal_trace_heatmap.") from exc
    import torch

    data = torch.as_tensor(np.asarray(scores), dtype=torch.float32)
    if data.ndim != 2:
        raise ValueError("causal_trace_heatmap expects a 2D score array.")
    data = _prepare_signed_data(data, signs=signs)  # type: ignore[arg-type]
    data = _clip_outliers(data, outlier_perc)
    if ax is None:
        _fig, ax = plt.subplots()
    axes: Any = ax
    image = axes.imshow(data.detach().cpu().numpy(), cmap=cmap, aspect="auto")
    axes.figure.colorbar(image, ax=axes)
    return axes
