"""Bundle diff renderer for paired TorchLens traces."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import graphviz
import torch

from ._render_utils import html_escape, render_dot_to_file, strip_known_extension
from .themes import resolve_theme, theme_edge_attrs, theme_graph_attrs, theme_node_attrs

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from ..data_classes.layer_pass_log import LayerPassLog
    from ..data_classes.model_log import ModelLog
    from ..intervention.bundle import Bundle


DiffLayout = Literal["paired"]
DiffTensorField = Literal["activation", "gradient"]

_CAPTION = (
    "Clean vs zero_ablate(layer1.0.relu) — top: clean, bottom: ablated. "
    "Color: per-node L2 norm delta."
)
_ARIA_LABEL = (
    "TorchLens bundle diff: clean versus zero ablate layer1.0.relu. "
    "Blue means lower delta, white means no or middle delta, red means higher delta."
)


def bundle_diff(
    bundle: "Bundle",
    *,
    metric: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "relative_l2",
    layout: DiffLayout = "paired",
    left: str | "ModelLog" | None = None,
    right: str | "ModelLog" | None = None,
    baseline: str | "ModelLog" | None = None,
    on: DiffTensorField = "activation",
    vis_outpath: str = "bundle_diff",
    vis_save_only: bool = False,
    vis_fileformat: str = "svg",
    theme: str = "paper",
    max_pairs: int | None = 16,
    include_unmatched: bool = False,
) -> str:
    """Render a two-column bundle diff SVG using aligned node pairs.

    Parameters
    ----------
    bundle:
        Bundle containing at least two traces.
    metric:
        Metric forwarded to ``bundle.delta_map``.
    layout:
        Layout strategy. Only ``"paired"`` is currently supported.
    left:
        Left member name or log. Defaults to the first bundle member.
    right:
        Right member name or log. Defaults to the second bundle member.
    baseline:
        Baseline forwarded to ``bundle.delta_map``. Defaults to ``left``.
    on:
        Tensor field forwarded to ``bundle.delta_map``.
    vis_outpath:
        Output path, with or without ``.svg``.
    vis_save_only:
        Whether to suppress opening the rendered artifact.
    vis_fileformat:
        Output format. Phase 9 supports ``"svg"``.
    theme:
        Visualization theme preset.
    max_pairs:
        Maximum aligned pairs to draw. ``None`` draws every pair.
    include_unmatched:
        Whether to add unpaired nodes with gray borders.

    Returns
    -------
    str
        Graphviz DOT source for the rendered bundle diff.
    """

    if layout != "paired":
        raise ValueError("bundle_diff layout must be 'paired'.")
    if vis_fileformat != "svg":
        raise ValueError("bundle_diff currently renders SVG only.")

    left_name, right_name = _resolve_side_names(bundle, left=left, right=right)
    baseline_ref = baseline if baseline is not None else left_name
    delta_map = bundle.delta_map(metric, baseline=baseline_ref, on=on)
    layer_to_node = _layer_to_supergraph_node(bundle)
    pairs = _select_pairs(
        bundle.aligned_pairs(left_name, right_name),
        right_name=right_name,
        delta_map=delta_map,
        layer_to_node=layer_to_node,
        max_pairs=max_pairs,
    )
    dot = _build_dot(
        pairs=pairs,
        bundle=bundle,
        left_name=left_name,
        right_name=right_name,
        delta_map=delta_map,
        layer_to_node=layer_to_node,
        theme_name=theme,
        include_unmatched=include_unmatched,
    )
    outpath = strip_known_extension(vis_outpath)
    source = render_dot_to_file(dot, outpath, vis_fileformat, vis_save_only)
    _add_svg_accessibility(f"{outpath}.{vis_fileformat}")
    return source


def _resolve_side_names(
    bundle: "Bundle",
    *,
    left: str | "ModelLog" | None,
    right: str | "ModelLog" | None,
) -> tuple[str, str]:
    """Resolve the left and right bundle member names.

    Parameters
    ----------
    bundle:
        Bundle being rendered.
    left:
        Optional left member name or log.
    right:
        Optional right member name or log.

    Returns
    -------
    tuple[str, str]
        Resolved ``(left_name, right_name)``.
    """

    names = bundle.names
    if len(names) < 2 and (left is None or right is None):
        raise ValueError("bundle_diff requires at least two bundle members.")
    left_name = _resolve_member_name(bundle, left if left is not None else names[0])
    right_name = _resolve_member_name(bundle, right if right is not None else names[1])
    if left_name == right_name:
        raise ValueError("bundle_diff requires distinct left and right members.")
    return left_name, right_name


def _resolve_member_name(bundle: "Bundle", member: str | "ModelLog") -> str:
    """Resolve a member reference inside a bundle.

    Parameters
    ----------
    bundle:
        Bundle being queried.
    member:
        Member name or ModelLog reference.

    Returns
    -------
    str
        Resolved member name.
    """

    if isinstance(member, str):
        if member not in bundle:
            raise KeyError(f"Unknown bundle member {member!r}.")
        return member
    for name, candidate in bundle.members.items():
        if candidate is member:
            return name
    raise KeyError("ModelLog is not a member of this Bundle.")


def _layer_to_supergraph_node(bundle: "Bundle") -> dict[int, str]:
    """Map concrete layer objects to their supergraph node names.

    Parameters
    ----------
    bundle:
        Bundle with a built supergraph.

    Returns
    -------
    dict[int, str]
        ``id(layer)`` to supergraph node name.
    """

    lookup: dict[int, str] = {}
    for node_name, node in bundle.supergraph.nodes.items():
        for layer in getattr(node, "layer_refs", {}).values():
            lookup[id(layer)] = str(node_name)
    return lookup


def _build_dot(
    *,
    pairs: list[tuple[Any, Any]],
    bundle: "Bundle",
    left_name: str,
    right_name: str,
    delta_map: dict[str, dict[str, float]],
    layer_to_node: dict[int, str],
    theme_name: str,
    include_unmatched: bool,
) -> graphviz.Digraph:
    """Build the Graphviz object for a paired bundle diff.

    Parameters
    ----------
    pairs:
        Aligned layer pairs from ``bundle.aligned_pairs``.
    bundle:
        Bundle being rendered.
    left_name:
        Left member name.
    right_name:
        Right member name.
    delta_map:
        Per-node metric values from ``bundle.delta_map``.
    layer_to_node:
        Mapping from concrete layers to supergraph node keys.
    theme_name:
        Visualization theme name.
    include_unmatched:
        Whether to draw nodes without aligned counterparts.

    Returns
    -------
    graphviz.Digraph
        Configured Graphviz graph.
    """

    theme = resolve_theme(theme_name)
    dot = graphviz.Digraph(
        name="TorchLens_Bundle_Diff",
        comment="TorchLens bundle diff",
        format="svg",
    )
    graph_attrs = theme_graph_attrs(theme, font_size=18, dpi=100)
    graph_attrs.update(
        {
            "rankdir": "TB",
            "compound": "true",
            "splines": "ortho",
            "nodesep": "0.32",
            "ranksep": "0.34",
            "size": "12,8!",
            "ratio": "fill",
            "label": _CAPTION + "\\nLegend: blue→white→red = increasing L2 delta.",
            "labelloc": "b",
            "labeljust": "l",
            "fontname": "Helvetica",
        }
    )
    dot.graph_attr.update(graph_attrs)
    dot.node_attr.update(theme_node_attrs(theme, font_size=10))
    dot.edge_attr.update(theme_edge_attrs(theme, font_size=8))

    paired_left_ids = {id(left_layer) for left_layer, _right_layer in pairs}
    paired_right_ids = {id(right_layer) for _left_layer, right_layer in pairs}
    left_unmatched = (
        [
            layer
            for layer in getattr(bundle[left_name], "layer_list", [])
            if id(layer) not in paired_left_ids
        ]
        if include_unmatched
        else []
    )
    right_unmatched = (
        [
            layer
            for layer in getattr(bundle[right_name], "layer_list", [])
            if id(layer) not in paired_right_ids
        ]
        if include_unmatched
        else []
    )
    left_layers = [left_layer for left_layer, _right_layer in pairs] + left_unmatched
    right_layers = [right_layer for _left_layer, right_layer in pairs] + right_unmatched
    values = _right_delta_values(
        pairs=pairs,
        right_name=right_name,
        delta_map=delta_map,
        layer_to_node=layer_to_node,
    )
    max_delta = max(values) if values else 0.0
    _add_side_cluster(
        dot,
        layers=left_layers,
        side="clean",
        member_name=left_name,
        compared_member_name=right_name,
        delta_map=delta_map,
        layer_to_node=layer_to_node,
        max_delta=max_delta,
        unmatched_layer_ids={id(layer) for layer in left_unmatched},
    )
    _add_side_cluster(
        dot,
        layers=right_layers,
        side="intervention",
        member_name=right_name,
        compared_member_name=right_name,
        delta_map=delta_map,
        layer_to_node=layer_to_node,
        max_delta=max_delta,
        unmatched_layer_ids={id(layer) for layer in right_unmatched},
    )
    _add_side_edges(dot, layers=left_layers, side="clean")
    _add_side_edges(dot, layers=right_layers, side="intervention")
    _add_pair_alignment(dot, pairs)
    _add_legend(dot)
    return dot


def _right_delta_values(
    *,
    pairs: list[tuple[Any, Any]],
    right_name: str,
    delta_map: dict[str, dict[str, float]],
    layer_to_node: dict[int, str],
) -> list[float]:
    """Return right-side delta values for color normalization.

    Parameters
    ----------
    pairs:
        Aligned layer pairs.
    right_name:
        Right member name.
    delta_map:
        Per-node metric values.
    layer_to_node:
        Mapping from layers to supergraph nodes.

    Returns
    -------
    list[float]
        Non-negative finite delta values.
    """

    values: list[float] = []
    for left_layer, right_layer in pairs:
        node_name = layer_to_node.get(id(left_layer), layer_to_node.get(id(right_layer)))
        if node_name is None:
            continue
        value = float(delta_map.get(node_name, {}).get(right_name, 0.0))
        if value >= 0.0:
            values.append(value)
    return values


def _select_pairs(
    pairs: list[tuple[Any, Any]],
    *,
    right_name: str,
    delta_map: dict[str, dict[str, float]],
    layer_to_node: dict[int, str],
    max_pairs: int | None,
) -> list[tuple[Any, Any]]:
    """Select high-signal aligned pairs for a compact hero diff.

    Parameters
    ----------
    pairs:
        Candidate aligned pairs.
    right_name:
        Right member name.
    delta_map:
        Per-node metric values.
    layer_to_node:
        Mapping from layers to supergraph nodes.
    max_pairs:
        Maximum pair count, or ``None`` for all pairs.

    Returns
    -------
    list[tuple[Any, Any]]
        Selected pairs in original trace order.
    """

    if max_pairs is None or len(pairs) <= max_pairs:
        return pairs
    if max_pairs < 1:
        raise ValueError("max_pairs must be at least 1 or None.")
    scored: list[tuple[float, int, tuple[Any, Any]]] = []
    for index, (left_layer, right_layer) in enumerate(pairs):
        node_name = layer_to_node.get(id(left_layer), layer_to_node.get(id(right_layer)))
        value = float(delta_map.get(str(node_name), {}).get(right_name, 0.0))
        scored.append((value, index, (left_layer, right_layer)))
    chosen_indexes = {
        index
        for _value, index, _pair in sorted(scored, key=lambda row: row[0], reverse=True)[:max_pairs]
    }
    return [pair for index, pair in enumerate(pairs) if index in chosen_indexes]


def _add_side_cluster(
    dot: graphviz.Digraph,
    *,
    layers: list[Any],
    side: str,
    member_name: str,
    compared_member_name: str,
    delta_map: dict[str, dict[str, float]],
    layer_to_node: dict[int, str],
    max_delta: float,
    unmatched_layer_ids: set[int],
) -> None:
    """Add one clean or intervention cluster to the diff graph.

    Parameters
    ----------
    dot:
        Graphviz graph to mutate.
    layers:
        Layers to render in this side.
    side:
        Stable side key used in Graphviz node IDs.
    member_name:
        Bundle member represented by this side.
    compared_member_name:
        Member used for delta color values.
    delta_map:
        Per-node metric values.
    layer_to_node:
        Mapping from layers to supergraph nodes.
    max_delta:
        Maximum delta value used for normalization.
    unmatched_layer_ids:
        ``id(layer)`` values whose aligned counterpart is missing.

    Returns
    -------
    None
        ``dot`` is mutated in place.
    """

    with dot.subgraph(name=f"cluster_{side}") as subgraph:
        subgraph.attr(
            label=f"{member_name}",
            labelloc="t",
            color="#222222",
            penwidth="2",
            style="rounded",
        )
        for layer in layers:
            node_name = layer_to_node.get(id(layer), str(getattr(layer, "layer_label", "node")))
            value = float(delta_map.get(node_name, {}).get(compared_member_name, 0.0))
            subgraph.node(
                _node_id(side, layer),
                label=_node_label(layer, value),
                tooltip=_node_title(member_name, layer, value),
                fillcolor=_delta_color(value, max_delta),
                color=_node_border_color(layer, unmatched_layer_ids),
                fontcolor="#111111",
                shape="box",
                style="filled,rounded",
                penwidth=_node_penwidth(layer),
            )


def _add_side_edges(dot: graphviz.Digraph, *, layers: list[Any], side: str) -> None:
    """Add intra-side parent-child edges among displayed nodes.

    Parameters
    ----------
    dot:
        Graphviz graph to mutate.
    layers:
        Layers displayed for one side.
    side:
        Stable side key used in Graphviz node IDs.

    Returns
    -------
    None
        ``dot`` is mutated in place.
    """

    visible = {str(getattr(layer, "layer_label", "")): layer for layer in layers}
    for layer in layers:
        target_id = _node_id(side, layer)
        for parent_label in getattr(layer, "parent_layers", []) or []:
            parent = visible.get(str(parent_label))
            if parent is not None:
                dot.edge(_node_id(side, parent), target_id, color="#707070", arrowsize="0.5")


def _add_pair_alignment(dot: graphviz.Digraph, pairs: list[tuple[Any, Any]]) -> None:
    """Add same-rank constraints for aligned layer pairs.

    Parameters
    ----------
    dot:
        Graphviz graph to mutate.
    pairs:
        Aligned layer pairs.

    Returns
    -------
    None
        ``dot`` is mutated in place.
    """

    for left_layer, right_layer in pairs:
        left_id = _node_id("clean", left_layer)
        right_id = _node_id("intervention", right_layer)
        with dot.subgraph() as rank:
            rank.attr(rank="same")
            rank.node(left_id)
            rank.node(right_id)
        dot.edge(left_id, right_id, style="invis", weight="8")


def _add_legend(dot: graphviz.Digraph) -> None:
    """Add a compact color legend to the diff graph.

    Parameters
    ----------
    dot:
        Graphviz graph to mutate.

    Returns
    -------
    None
        ``dot`` is mutated in place.
    """

    with dot.subgraph(name="cluster_delta_legend") as legend:
        legend.attr(label="delta_map", color="#D0D0D0", style="rounded", labelloc="t")
        legend.node(
            "legend_low",
            label="low",
            shape="box",
            style="filled,rounded",
            fillcolor="#3B6FB6",
            color="#222222",
        )
        legend.node(
            "legend_mid",
            label="mid",
            shape="box",
            style="filled,rounded",
            fillcolor="#FFFFFF",
            color="#222222",
        )
        legend.node(
            "legend_high",
            label="high",
            shape="box",
            style="filled,rounded",
            fillcolor="#C93F3F",
            color="#222222",
        )
        legend.edge("legend_low", "legend_mid", style="invis")
        legend.edge("legend_mid", "legend_high", style="invis")


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

    label = str(getattr(layer, "layer_label", "node"))
    safe = re.sub(r"[^0-9A-Za-z_]", "_", label)
    return f"{side}_{safe}"


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

    label = str(getattr(layer, "layer_label", "node"))
    func_name = str(getattr(layer, "func_name", "op"))
    module = getattr(layer, "containing_module", None)
    module_line = f"@{module}" if module else ""
    shape = getattr(layer, "tensor_shape", None)
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

    label = str(getattr(layer, "layer_label", "node"))
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


def _delta_color(value: float, max_delta: float) -> str:
    """Map a non-negative delta value to a blue-white-red color.

    Parameters
    ----------
    value:
        Delta value.
    max_delta:
        Maximum delta used for normalization.

    Returns
    -------
    str
        Hex color.
    """

    if max_delta <= 0.0:
        return "#FFFFFF"
    normalized = max(0.0, min(1.0, value / max_delta))
    if normalized <= 0.5:
        frac = normalized / 0.5
        return _interpolate("#3B6FB6", "#FFFFFF", frac)
    frac = (normalized - 0.5) / 0.5
    return _interpolate("#FFFFFF", "#C93F3F", frac)


def _interpolate(start: str, end: str, fraction: float) -> str:
    """Linearly interpolate between two hex colors.

    Parameters
    ----------
    start:
        Start color.
    end:
        End color.
    fraction:
        Interpolation fraction in ``[0, 1]``.

    Returns
    -------
    str
        Interpolated hex color.
    """

    start_rgb = _hex_to_rgb(start)
    end_rgb = _hex_to_rgb(end)
    values = [
        round(start_value + (end_value - start_value) * fraction)
        for start_value, end_value in zip(start_rgb, end_rgb)
    ]
    return "#" + "".join(f"{value:02X}" for value in values)


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    """Convert a ``#RRGGBB`` color to RGB integers.

    Parameters
    ----------
    value:
        Hex color.

    Returns
    -------
    tuple[int, int, int]
        RGB values.
    """

    raw = value.lstrip("#")
    return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)


def _add_svg_accessibility(path: str) -> None:
    """Add figure-level accessibility attributes to a rendered SVG.

    Parameters
    ----------
    path:
        Rendered SVG path.

    Returns
    -------
    None
        The SVG is updated in place when it exists.
    """

    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as handle:
        svg = handle.read()
    if "aria-label=" not in svg:
        svg = re.sub(
            r"<svg\b",
            f'<svg role="img" aria-label="{html_escape(_ARIA_LABEL)}"',
            svg,
            count=1,
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(svg)


__all__ = ["bundle_diff"]
