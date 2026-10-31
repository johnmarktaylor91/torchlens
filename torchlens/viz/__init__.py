"""Visualization convenience namespace for TorchLens.

Node visualizer factories for ``trace.draw(node_spec_fn=...)`` live in
``torchlens.viz.nodes``.
"""

from __future__ import annotations

from typing import Any

import torch

from ..visualization.bundle_diff import bundle_diff
from . import batch_summary as _batch_summary, nodes
from .batch_summary import montage
from .feature_maps import feature_maps
from .node_plots import heatmap, image_scatter, lineplot

__all__ = [
    "bundle_diff",
    "causal_trace_heatmap",
    "feature_maps",
    "heatmap",
    "image_scatter",
    "lineplot",
    "montage",
    "nodes",
    # Transformer renderers, resolved by __getattr__ below.
    "ablation_grid",  # noqa: F822 -- resolved by __getattr__
    "attention",  # noqa: F822 -- resolved by __getattr__
    "attention_atlas",  # noqa: F822 -- resolved by __getattr__
    "logit_lens",  # noqa: F822 -- resolved by __getattr__
    "metric_strip",  # noqa: F822 -- resolved by __getattr__
    "prediction_table",  # noqa: F822 -- resolved by __getattr__
    "token_strip",  # noqa: F822 -- resolved by __getattr__
]

#: Transformer renderers (attention, logit lens, predictions, receipts), resolved on
#: first access so importing this namespace does not import the transformer kit.
_TRANSFORMER_RENDERERS: dict[str, tuple[str, str]] = {
    "ablation_grid": ("torchlens.tviz._receipts", "render_receipt_grid"),
    "attention": ("torchlens.tviz._render_attention", "render_attention"),
    "attention_atlas": ("torchlens.tviz._render_attention", "render_attention_atlas"),
    "logit_lens": ("torchlens.tviz._lens_heatmap", "render_logit_lens_heatmap"),
    "metric_strip": ("torchlens.tviz._metrics", "render_metric_strip"),
    "prediction_table": ("torchlens.tviz._predictions", "render_prediction_table"),
    "token_strip": ("torchlens.tviz._strip", "render_token_strip"),
}


def __getattr__(name: str) -> Any:
    """Resolve the transformer renderers lazily.

    Parameters
    ----------
    name:
        Attribute requested from ``torchlens.viz``.

    Returns
    -------
    Any
        The requested renderer.

    Raises
    ------
    AttributeError
        If ``name`` is not a ``torchlens.viz`` name.
    """

    if name in _TRANSFORMER_RENDERERS:
        import importlib

        module_path, attr = _TRANSFORMER_RENDERERS[name]
        value = getattr(importlib.import_module(module_path), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'torchlens.viz' has no attribute {name!r}")


batch_summary = _batch_summary


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

    import numpy as np

    from ._tensor_display import _clip_outliers, _prepare_signed_data

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for causal_trace_heatmap.") from exc

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
