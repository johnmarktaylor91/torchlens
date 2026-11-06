"""Labels, colours, caption and key of the bundle diff picture, and its style choices."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .bundle_diff import DiffTensorField


@dataclass(frozen=True)
class DiffStyle:
    """Direction, overrides and key choice for the diff picture (``Bundle.draw`` keywords).

    ``rankdir`` is the Graphviz rank direction (``"TB"`` draws the members as two
    columns); ``graph_overrides`` apply over the graph defaults; ``node_overrides``
    key node attributes by op label (``relu_1_1:2`` for one pass) and
    ``edge_overrides`` key edge attributes by ``(source, target)`` op labels, both
    applied in both members; ``show_legend=False`` drops the colour key.
    """

    rankdir: str = "TB"
    graph_overrides: Mapping[str, Any] = field(default_factory=dict)
    node_overrides: Mapping[str, Any] | None = None
    edge_overrides: Mapping[tuple[str, str], Any] | None = None
    show_legend: bool | None = None


def _caption(
    left_name: str,
    right_name: str,
    metric_name: str,
    tensor_field: DiffTensorField,
) -> str:
    """Return the graph caption for a paired bundle diff.

    Parameters
    ----------
    left_name:
        Left bundle member name.
    right_name:
        Right bundle member name.
    metric_name:
        Human-readable metric name.
    tensor_field:
        Tensor field used for comparison.

    Returns
    -------
    str
        Graphviz label text.
    """

    # Left-justified lines (Graphviz ``\l`` ends), each shorter than the old
    # centred pair: a centred line wider than the drawing ran off its left edge.
    return (
        f"{left_name} vs {right_name} - columns: {left_name}, {right_name}.\\l"
        f"Color: per-node {metric_name} delta on {tensor_field}.\\l"
        "Legend: white to red = increasing delta.\\l"
    )


def _aria_label(
    left_name: str,
    right_name: str,
    metric_name: str,
    tensor_field: DiffTensorField,
) -> str:
    """Return the SVG accessibility label for a paired bundle diff.

    Parameters
    ----------
    left_name:
        Left bundle member name.
    right_name:
        Right bundle member name.
    metric_name:
        Human-readable metric name.
    tensor_field:
        Tensor field used for comparison.

    Returns
    -------
    str
        Accessibility label for the SVG root.
    """

    return (
        f"TorchLens bundle diff: {left_name} versus {right_name}; "
        f"{metric_name} delta on {tensor_field}. "
        "White means zero delta; red intensity scales with delta magnitude."
    )


def _node_id(side: str, layer: Any) -> str:
    """Return a stable Graphviz node ID for one side/layer pair.

    Parameters
    ----------
    side:
        Side key.
    layer:
        Layer-like object.

    Returns
    -------
    str
        Graphviz-safe node ID.
    """

    safe = re.sub(r"[^0-9A-Za-z_]", "_", _site_label(layer).replace(":", "_p"))
    return f"{side}_{safe}"


def _site_label(layer: Any) -> str:
    """Return a drawn op's label: ``"<layer>:<k>"`` for pass ``k`` of a multi-pass layer.

    The row key of ``Bundle.compare`` (``bundle._aligned_sites.site_row_key``),
    spelled here so the renderer adds no import of ``bundle``.
    """

    if not hasattr(layer, "layer_label"):
        return "node"
    if int(getattr(layer, "num_passes", 1) or 1) > 1:
        return f"{layer.layer_label}:{layer.pass_number}"
    return str(layer.layer_label)


def _node_label(layer: Any, delta: float) -> str:
    """Return a compact display label for a layer node.

    Parameters
    ----------
    layer:
        Layer-like object.
    delta:
        Metric value for the paired node.

    Returns
    -------
    str
        Graphviz node label.
    """

    label = _site_label(layer)
    func_name = str(getattr(layer, "func_name", "op"))
    module = getattr(layer, "module", None)
    module_line = f"@{module}" if module else ""
    shape = getattr(layer, "shape", None)
    shape_line = str(tuple(shape)) if isinstance(shape, (list, tuple)) else str(shape or "")
    parts = [label, func_name]
    if module_line:
        parts.append(module_line)
    if shape_line:
        parts.append(shape_line)
    parts.append(f"delta={delta:.3g}")
    return "\n".join(parts)


def _node_title(member_name: str, layer: Any, delta: float) -> str:
    """Return an accessible title/tooltip for a layer node.

    Parameters
    ----------
    member_name:
        Bundle member name.
    layer:
        Layer-like object.
    delta:
        Metric value.

    Returns
    -------
    str
        Node title text.
    """

    label = _site_label(layer)
    func_name = str(getattr(layer, "func_name", "op"))
    return f"{member_name} {label}; operation {func_name}; L2 delta {delta:.6g}"


def _node_border_color(layer: Any, unmatched_layer_ids: set[int]) -> str:
    """Return the node border color, gray for unmatched layers.

    Parameters
    ----------
    layer:
        Layer-like object.
    unmatched_layer_ids:
        ``id(layer)`` values whose aligned counterpart is missing.

    Returns
    -------
    str
        Hex color.
    """

    return "#808080" if id(layer) in unmatched_layer_ids else "#222222"


def _node_penwidth(layer: Any) -> str:
    """Return the node border width.

    Parameters
    ----------
    layer:
        Layer-like object.

    Returns
    -------
    str
        Graphviz penwidth value.
    """

    return "1.4" if layer is not None else "2.2"
