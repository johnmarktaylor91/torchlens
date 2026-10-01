"""Per-node overlay helpers for TorchLens graph rendering."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping
from typing import Any

import torch

from .._errors import InvalidArgumentError
from ..errors._base import TorchLensWarning
from ..utils._multipass_access import get_multipass_attr
from ..utils.display import format_flops, human_readable_size

OverlayScores = Mapping[str, Any]


def _overlay_candidate_labels(trace: Any) -> set[str]:
    """Every label spelling the per-node overlay lookup can realize on ``trace``."""

    candidates: set[str] = set(getattr(trace, "op_labels", ()) or ())
    for node in getattr(trace, "layer_list", ()) or ():
        for attr in ("layer_label", "label", "layer_label_short", "label_short"):
            value = getattr(node, attr, None)
            if isinstance(value, str):
                candidates.add(value)
    return candidates


def _lower_selection_overlay(trace: Any, overlay: Any) -> Any:
    """Lower a Selection-shaped overlay source to per-site SELECTED COUNTS.

    The element-mask law (leverage D-4): a consumer of a Selection-shaped
    source reads element masks / ``selected_count``, never family membership
    — painting whole sites for a sparse element selection is the measured
    2.7x overpaint. Touched sites with zero selected elements stay in the
    mapping as an honest ``0``.
    """

    from ..selection import ResolvedSelection, Selection

    if isinstance(overlay, Selection):
        overlay = overlay.resolve(trace)
    elif not isinstance(overlay, ResolvedSelection):
        selection_hook = getattr(overlay, "__selection__", None)
        if selection_hook is None:
            return overlay
        overlay = selection_hook().resolve(trace)
    if overlay._trace is not trace:
        from ..selection import SelectionError

        raise SelectionError(
            "the node_overlay selection is resolved against a DIFFERENT "
            "trace; re-resolve it against the trace being drawn.",
            code="selection_trace_mismatch",
        )
    scores: dict[str, int] = {}
    for entry in overlay:
        layer_label = entry.site_key[0]
        scores[layer_label] = scores.get(layer_label, 0) + int(entry._mask._dense_ro().sum())
    return scores


def resolve_overlay_request(trace: Any, overlay: Any) -> Any:
    """Validate and lower one draw request's ``node_overlay`` (leverage B8/D-4).

    Selection-shaped sources lower to per-site selected counts (element-mask
    law). Mapping sources are coverage-checked against the trace's realizable
    label spellings: a mapping that matches NOTHING refuses typed — the
    shipped door used to label-join such a table to zero nodes, render a
    blank picture, and report success — and a partial match disclosed with
    the realized/requested counts. Builtin names and callables pass through.

    Raises
    ------
    InvalidArgumentError
        ``node_overlay_zero_match`` when a non-empty mapping matches zero
        realizable node labels on this trace.
    """

    if overlay is None or isinstance(overlay, str) or callable(overlay):
        return overlay
    if not isinstance(overlay, Mapping):
        overlay = _lower_selection_overlay(trace, overlay)
        if not isinstance(overlay, Mapping):
            return overlay
    if not overlay:
        return overlay
    candidates = _overlay_candidate_labels(trace)
    realized = [key for key in overlay if isinstance(key, str) and key in candidates]
    if not realized:
        examples = sorted(candidates)[:4]
        raise InvalidArgumentError(
            f"node_overlay matched 0 of {len(overlay)} requested keys on this "
            "trace: nothing would be painted, and a blank overlay reporting "
            "success is a lie, not a picture. Overlay keys join on node "
            f"labels (e.g. {examples}); structural site keys and foreign "
            "labels do not join here.",
            code="node_overlay_zero_match",
            remedy="key the overlay table by this trace's layer labels",
        )
    if len(realized) < len(overlay):
        missing = [key for key in overlay if key not in set(realized)]
        warnings.warn(
            TorchLensWarning(
                f"node_overlay matched {len(realized)} of {len(overlay)} "
                f"requested keys; unmatched keys (first 5): {missing[:5]}. "
                "Painted nodes reflect only the realized rows. "
                "Remedy: key the overlay table by this trace's layer labels.",
                code="node_overlay_partial_match",
            ),
            stacklevel=3,
        )
    return overlay


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
        raise InvalidArgumentError(
            f"Unsupported node overlay {name!r}; choose one of {supported}",
            code="node_overlay_invalid",
            remedy=f"pass one of the supported overlays ({supported})",
            argument="overlay",
        )
    return normalized


def external_overlay_value(node: Any, scores: OverlayScores | Callable[[Any], Any]) -> Any:
    """Return an externally supplied overlay value for ``node``.

    Parameters
    ----------
    node:
        Layer log or layer-pass log.
    scores:
        Either a mapping from node labels to overlay values, or a callable invoked as
        ``scores(node)`` to compute the value per node.

    Returns
    -------
    Any
        Overlay value, or ``None`` when no matching key is present.
    """

    if callable(scores):
        return scores(node)

    candidates = (
        getattr(node, "layer_label", None),
        getattr(node, "label", None),
        getattr(node, "layer_label_short", None),
        getattr(node, "label_short", None),
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

    # Overlays render one scalar per node. On a rolled recurrent node ``node`` is an
    # aggregate multi-pass Layer, so a per-pass read (out/func_duration/grad/
    # interventions) trips the multi-pass ValueError tripwire -- historically
    # crashing draw() with any of these overlays. Route every read through the
    # shared helper with ``multipass=None`` so an ambiguous aggregate degrades to an
    # honest "n/a" (format_overlay_value maps None -> "n/a") instead of crashing or
    # fabricating a per-pass value. Aggregate-stable fields (flops_forward,
    # activation_memory) resolve normally and are unaffected.
    name = normalize_overlay_name(overlay)
    if name == "flops":
        value = get_multipass_attr(node, "flops_forward", 0, multipass=None)
        return None if value is None else int(value or 0)
    if name == "time":
        value = get_multipass_attr(node, "func_duration", 0.0, multipass=None)
        return None if value is None else float(value or 0.0)
    if name == "bytes":
        value = get_multipass_attr(node, "activation_memory", 0, multipass=None)
        return None if value is None else int(value or 0)
    if name == "magnitude":
        return _tensor_magnitude(get_multipass_attr(node, "out", None, multipass=None))
    if name == "grad_norm":
        return _tensor_norm(get_multipass_attr(node, "grad", None, multipass=None))
    if name == "nan":
        tensor = get_multipass_attr(node, "out", None, multipass=None)
        # F15: distinguish "nothing was checked" (no available tensor -- missing,
        # None, or an ambiguous aggregate) -> None -> "nan: n/a" from "checked,
        # none found" -> False -> "nan: no". The old code returned _has_nonfinite
        # of a missing/None tensor, i.e. False, and so asserted "nan: no" on nodes
        # whose output was never inspected -- a false all-clear.
        if not isinstance(tensor, torch.Tensor):
            return None
        return _has_nonfinite(tensor)
    if name == "intervention":
        value = get_multipass_attr(node, "interventions", (), multipass=None)
        return None if value is None else len(value or ())
    if name == "bundle_delta":
        return get_multipass_attr(node, "bundle_delta", None, multipass=None)
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


def overlay_line(
    node: Any, overlay: str | OverlayScores | Callable[[Any], Any] | None
) -> str | None:
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


def overlay_border_attrs(
    node: Any, overlay: str | OverlayScores | Callable[[Any], Any] | None
) -> dict[str, str]:
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
