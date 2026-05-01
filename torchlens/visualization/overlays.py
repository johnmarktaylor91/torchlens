"""Per-node overlay helpers for TorchLens graph rendering."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from ..utils.display import format_flops, human_readable_size

OverlayScores = Mapping[str, Any]

SUPPORTED_OVERLAYS = frozenset(
    {
        "flops",
        "time",
        "bytes",
        "magnitude",
        "grad_norm",
        "grad-norm",
        "nan",
        "intervention",
        "bundle_delta",
        "bundle delta",
    }
)


def normalize_overlay_name(name: str) -> str:
    """Return the canonical name for an overlay preset.

    Parameters
    ----------
    name:
        User-facing overlay name.

    Returns
    -------
    str
        Canonical overlay name.

    Raises
    ------
    ValueError
        If ``name`` is not a supported overlay.
    """

    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in {
        overlay.replace("-", "_").replace(" ", "_") for overlay in SUPPORTED_OVERLAYS
    }:
        supported = ", ".join(sorted(SUPPORTED_OVERLAYS))
        raise ValueError(f"Unsupported node overlay {name!r}; choose one of {supported}.")
    return normalized


def external_overlay_value(node: Any, scores: OverlayScores) -> Any:
    """Return an externally supplied overlay value for ``node``.

    Parameters
    ----------
    node:
        Layer log or layer-pass log.
    scores:
        Mapping from node labels to overlay values.

    Returns
    -------
    Any
        Overlay value, or ``None`` when no matching key is present.
    """

    candidates = (
        getattr(node, "layer_label", None),
        getattr(node, "layer_label_w_pass", None),
        getattr(node, "layer_label_no_pass", None),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate in scores:
            return scores[candidate]
    return None


def builtin_overlay_value(node: Any, overlay: str) -> Any:
    """Compute one built-in overlay value for ``node``.

    Parameters
    ----------
    node:
        Layer log or layer-pass log.
    overlay:
        Overlay preset name.

    Returns
    -------
    Any
        Computed overlay value.
    """

    name = normalize_overlay_name(overlay)
    if name == "flops":
        return int(getattr(node, "flops_forward", 0) or 0)
    if name == "time":
        return float(getattr(node, "func_time", 0.0) or 0.0)
    if name == "bytes":
        return int(getattr(node, "tensor_memory", 0) or 0)
    if name == "magnitude":
        return _tensor_magnitude(getattr(node, "activation", None))
    if name == "grad_norm":
        return _tensor_norm(getattr(node, "gradient", None))
    if name == "nan":
        return _has_nonfinite(getattr(node, "activation", None))
    if name == "intervention":
        return len(getattr(node, "intervention_log", ()) or ())
    if name == "bundle_delta":
        return getattr(node, "bundle_delta", None)
    return None


def format_overlay_value(name: str, value: Any) -> str:
    """Format one overlay value for a node label.

    Parameters
    ----------
    name:
        Overlay display name.
    value:
        Raw overlay value.

    Returns
    -------
    str
        Compact display line.
    """

    display_name = name.replace("_", "-")
    if value is None:
        return f"{display_name}: n/a"
    if isinstance(value, bool):
        return f"{display_name}: {'yes' if value else 'no'}"
    if display_name == "flops":
        return f"flops: {format_flops(int(value or 0))}"
    if display_name == "time":
        return f"time: {float(value or 0.0) * 1000:.3g} ms"
    if display_name == "bytes":
        return f"bytes: {human_readable_size(int(value or 0))}"
    if isinstance(value, float):
        if math.isnan(value):
            return f"{display_name}: nan"
        return f"{display_name}: {value:.4g}"
    return f"{display_name}: {value}"


def overlay_line(node: Any, overlay: str | OverlayScores | None) -> str | None:
    """Return a rendered overlay line for ``node``.

    Parameters
    ----------
    node:
        Layer log or layer-pass log.
    overlay:
        Built-in overlay name or external score mapping.

    Returns
    -------
    str | None
        Overlay label line, if an overlay is active.
    """

    if overlay is None:
        return None
    if isinstance(overlay, str):
        value = builtin_overlay_value(node, overlay)
        return format_overlay_value(normalize_overlay_name(overlay), value)
    value = external_overlay_value(node, overlay)
    return format_overlay_value("overlay", value)


def overlay_border_attrs(node: Any, overlay: str | OverlayScores | None) -> dict[str, str]:
    """Return graph node attributes implied by an overlay.

    Parameters
    ----------
    node:
        Layer log or layer-pass log.
    overlay:
        Built-in overlay name or external score mapping.

    Returns
    -------
    dict[str, str]
        Graphviz node attribute overrides.
    """

    if overlay is None:
        return {}
    if isinstance(overlay, str) and normalize_overlay_name(overlay) == "nan":
        if builtin_overlay_value(node, overlay):
            return {"color": "#D55E00", "penwidth": "3"}
    if isinstance(overlay, str) and normalize_overlay_name(overlay) == "intervention":
        if builtin_overlay_value(node, overlay):
            return {"color": "#CC79A7", "penwidth": "3"}
    value = (
        builtin_overlay_value(node, overlay)
        if isinstance(overlay, str)
        else external_overlay_value(node, overlay)
    )
    if isinstance(value, (int, float)) and float(value) != 0.0:
        return {"penwidth": "2"}
    return {}


def _tensor_magnitude(value: Any) -> float | None:
    """Return mean absolute magnitude for a tensor-like value.

    Parameters
    ----------
    value:
        Candidate tensor value.

    Returns
    -------
    float | None
        Mean absolute value, or ``None`` for unavailable tensors.
    """

    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return None
    return float(value.detach().abs().float().mean().item())


def _tensor_norm(value: Any) -> float | None:
    """Return L2 norm for a tensor-like value.

    Parameters
    ----------
    value:
        Candidate tensor value.

    Returns
    -------
    float | None
        Tensor norm, or ``None`` for unavailable tensors.
    """

    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return None
    return float(value.detach().float().norm().item())


def _has_nonfinite(value: Any) -> bool:
    """Return whether a tensor-like value contains NaN or Inf.

    Parameters
    ----------
    value:
        Candidate tensor value.

    Returns
    -------
    bool
        Whether any element is non-finite.
    """

    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return False
    return bool((~torch.isfinite(value.detach())).any().item())
