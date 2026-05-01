"""Minimal tensor display helpers for ``LayerLog.show`` and ``LayerPassLog.show``."""

from __future__ import annotations

from typing import Any, Literal

import torch

TensorShowMethod = Literal["auto", "heatmap", "channels", "rgb", "hist"]
HeatmapSigns = Literal["positive", "negative", "absolute_value", "all"]


def _tensor_from_log(log_entry: Any) -> torch.Tensor | None:
    """Extract a display tensor from a TorchLens log entry.

    Parameters
    ----------
    log_entry:
        ``LayerLog``, ``LayerPassLog``, or tensor-like object.

    Returns
    -------
    torch.Tensor | None
        Tensor to display, if available.
    """

    if isinstance(log_entry, torch.Tensor):
        return log_entry
    activation = getattr(log_entry, "transformed_activation", None)
    if isinstance(activation, torch.Tensor):
        return activation
    activation = getattr(log_entry, "activation", None)
    if isinstance(activation, torch.Tensor):
        return activation
    passes = getattr(log_entry, "passes", None)
    if isinstance(passes, dict) and 1 in passes:
        return _tensor_from_log(passes[1])
    return None


def _auto_method(tensor: torch.Tensor) -> TensorShowMethod:
    """Choose a display method from tensor dimensionality.

    Parameters
    ----------
    tensor:
        Tensor to display.

    Returns
    -------
    TensorShowMethod
        Concrete display method.
    """

    ndim = tensor.ndim
    shape = tuple(tensor.shape)
    if ndim == 1:
        return "hist"
    if ndim == 2:
        return "heatmap"
    if ndim == 3:
        return "channels"
    if ndim == 4 and len(shape) == 4 and shape[1] == 3:
        return "rgb"
    return "hist"


def _prepare_signed_data(
    tensor: torch.Tensor,
    *,
    signs: HeatmapSigns,
) -> torch.Tensor:
    """Apply Captum-style sign filtering to display data.

    Parameters
    ----------
    tensor:
        Tensor to transform.
    signs:
        Captum-style sign selector.

    Returns
    -------
    torch.Tensor
        Transformed tensor.
    """

    if signs == "positive":
        return torch.clamp(tensor, min=0)
    if signs == "negative":
        return torch.clamp(tensor, max=0)
    if signs == "absolute_value":
        return tensor.abs()
    if signs == "all":
        return tensor
    raise ValueError("signs must be 'positive', 'negative', 'absolute_value', or 'all'.")


def _to_2d(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce a tensor to two display dimensions.

    Parameters
    ----------
    tensor:
        Tensor to reshape/slice.

    Returns
    -------
    torch.Tensor
        2D tensor.
    """

    data = tensor.detach().cpu()
    if data.ndim == 0:
        return data.reshape(1, 1)
    if data.ndim == 1:
        return data.reshape(1, -1)
    while data.ndim > 2:
        data = data[0]
    return data


def _clip_outliers(data: torch.Tensor, outlier_perc: float | None) -> torch.Tensor:
    """Clip display data by percentile.

    Parameters
    ----------
    data:
        Display tensor.
    outlier_perc:
        Percentage to clip from each tail.

    Returns
    -------
    torch.Tensor
        Clipped tensor.
    """

    if outlier_perc is None or outlier_perc <= 0 or data.numel() == 0:
        return data
    finite = data[torch.isfinite(data)]
    if finite.numel() == 0:
        return data
    lower = torch.quantile(finite, outlier_perc / 100.0)
    upper = torch.quantile(finite, 1.0 - outlier_perc / 100.0)
    return torch.clamp(data, min=float(lower.item()), max=float(upper.item()))


def _figure_or_text() -> tuple[Any | None, Any | None]:
    """Import matplotlib lazily.

    Returns
    -------
    tuple[Any | None, Any | None]
        ``(plt, None)`` when available, otherwise ``(None, error_text)``.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        return None, f"matplotlib is required for tensor plotting ({exc})"
    return plt, None


def show_tensor(
    log_entry: Any,
    *,
    method: TensorShowMethod = "auto",
    signs: HeatmapSigns = "all",
    outlier_perc: float | None = 2.0,
    cmap: str = "viridis",
    **kwargs: Any,
) -> Any:
    """Display a TorchLens tensor payload.

    Parameters
    ----------
    log_entry:
        ``LayerLog``, ``LayerPassLog``, or tensor.
    method:
        ``"auto"``, ``"heatmap"``, ``"channels"``, ``"rgb"``, or ``"hist"``.
    signs:
        Captum-style sign selector for heatmap-like displays.
    outlier_perc:
        Percentile clipping for heatmaps.
    cmap:
        Matplotlib colormap name.
    **kwargs:
        Forwarded to the relevant matplotlib call.

    Returns
    -------
    Any
        Matplotlib figure when plotting is available, otherwise a text message.
    """

    tensor = _tensor_from_log(log_entry)
    if tensor is None:
        return "No saved tensor activation is available to display."
    resolved_method = _auto_method(tensor) if method == "auto" else method
    if resolved_method not in {"heatmap", "channels", "rgb", "hist"}:
        raise ValueError("method must be 'auto', 'heatmap', 'channels', 'rgb', or 'hist'.")

    plt, error_text = _figure_or_text()
    if plt is None:
        return error_text

    data = tensor.detach().cpu()
    if not data.is_floating_point() and not data.is_complex():
        data = data.to(torch.float32)
    if data.is_complex():
        data = data.abs()
    data = _prepare_signed_data(data.to(torch.float32), signs=signs)

    if resolved_method == "hist":
        fig, ax = plt.subplots()
        ax.hist(data.flatten().numpy(), **kwargs)
        ax.set_title(getattr(log_entry, "layer_label", "Tensor"))
        return fig

    if resolved_method == "rgb":
        image = data[0]
        image = image.permute(1, 2, 0)
        image = image - image.min()
        max_value = image.max()
        if max_value > 0:
            image = image / max_value
        fig, ax = plt.subplots()
        ax.imshow(image.numpy(), **kwargs)
        ax.axis("off")
        return fig

    if resolved_method == "channels":
        channel_data = data if data.ndim == 3 else data.reshape(1, *tuple(_to_2d(data).shape))
        channels = min(int(channel_data.shape[0]), 8)
        fig, axes = plt.subplots(1, channels, squeeze=False)
        for index in range(channels):
            ax = axes[0][index]
            ax.imshow(
                _clip_outliers(channel_data[index], outlier_perc).numpy(), cmap=cmap, **kwargs
            )
            ax.axis("off")
            ax.set_title(str(index))
        return fig

    heatmap_data = _clip_outliers(_to_2d(data), outlier_perc)
    fig, ax = plt.subplots()
    ax.imshow(heatmap_data.numpy(), cmap=cmap, aspect="auto", **kwargs)
    ax.set_title(getattr(log_entry, "layer_label", "Tensor"))
    return fig
