"""Visualization convenience namespace reserved for TorchLens 2.0."""

__all__: list[str] = []


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
