"""Theme presets for TorchLens graph rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .node_spec import NodeSpec


@dataclass(frozen=True)
class VisualizationTheme:
    """Resolved visual theme values for graph renderers.

    Parameters
    ----------
    name:
        Public theme preset name.
    graph:
        Graphviz graph attributes.
    node:
        Graphviz node attributes.
    edge:
        Graphviz edge attributes.
    default_fill:
        Default operation-node fill color.
    default_border:
        Default operation-node border color.
    default_font:
        Default operation-node font color.
    legend_items:
        Color legend entries as ``(label, color)`` pairs.
    """

    name: str
    graph: dict[str, str]
    node: dict[str, str]
    edge: dict[str, str]
    default_fill: str
    default_border: str
    default_font: str
    legend_items: tuple[tuple[str, str], ...]


_LEGEND_ITEMS: tuple[tuple[str, str], ...] = (
    ("input", "#009E73"),
    ("output", "#D55E00"),
    ("parameterized", "#56B4E9"),
    ("buffer", "#E69F00"),
    ("boolean", "#F0E442"),
    ("intervention/cone", "#CC79A7"),
)


THEME_PRESETS: dict[str, VisualizationTheme] = {
    "torchlens": VisualizationTheme(
        name="torchlens",
        graph={"bgcolor": "white"},
        node={},
        edge={},
        default_fill="white",
        default_border="black",
        default_font="black",
        legend_items=_LEGEND_ITEMS,
    ),
    "paper": VisualizationTheme(
        name="paper",
        graph={"bgcolor": "white", "colorscheme": "paired12"},
        node={"fontname": "Helvetica"},
        edge={"fontname": "Helvetica"},
        default_fill="#F7F7F7",
        default_border="#222222",
        default_font="#111111",
        legend_items=_LEGEND_ITEMS,
    ),
    "dark": VisualizationTheme(
        name="dark",
        graph={"bgcolor": "#111827"},
        node={"fontname": "Helvetica"},
        edge={"color": "#9CA3AF", "fontcolor": "#D1D5DB", "fontname": "Helvetica"},
        default_fill="#1F2937",
        default_border="#E5E7EB",
        default_font="#F9FAFB",
        legend_items=_LEGEND_ITEMS,
    ),
    "colorblind": VisualizationTheme(
        name="colorblind",
        graph={"bgcolor": "white"},
        node={"fontname": "Helvetica"},
        edge={"fontname": "Helvetica"},
        default_fill="#F0F0F0",
        default_border="#0072B2",
        default_font="#111111",
        legend_items=_LEGEND_ITEMS,
    ),
    "high_contrast": VisualizationTheme(
        name="high_contrast",
        graph={"bgcolor": "white"},
        node={"fontname": "Helvetica-Bold"},
        edge={"color": "black", "fontcolor": "black", "penwidth": "2"},
        default_fill="white",
        default_border="black",
        default_font="black",
        legend_items=_LEGEND_ITEMS,
    ),
}


def resolve_theme(theme: str, *, for_paper: bool = False) -> VisualizationTheme:
    """Return a supported visualization theme preset.

    Parameters
    ----------
    theme:
        Theme preset name.
    for_paper:
        Whether to force the paper preset.

    Returns
    -------
    VisualizationTheme
        Resolved theme preset.

    Raises
    ------
    ValueError
        If the requested theme is unknown.
    """

    resolved_name = "paper" if for_paper else theme
    if resolved_name not in THEME_PRESETS:
        supported = ", ".join(sorted(THEME_PRESETS))
        raise ValueError(
            f"Unsupported visualization theme {resolved_name!r}; choose one of {supported}."
        )
    return THEME_PRESETS[resolved_name]


def apply_theme_to_spec(spec: NodeSpec, theme: VisualizationTheme) -> NodeSpec:
    """Apply theme defaults to a node spec without replacing explicit styles.

    Parameters
    ----------
    spec:
        Node spec to style.
    theme:
        Resolved theme preset.

    Returns
    -------
    NodeSpec
        Themed copy of ``spec``.
    """

    fillcolor = theme.default_fill if spec.fillcolor in {None, "white"} else spec.fillcolor
    fontcolor = theme.default_font if spec.fontcolor in {None, "black"} else spec.fontcolor
    color = theme.default_border if spec.color in {None, "black"} else spec.color
    return spec.replace(fillcolor=fillcolor, fontcolor=fontcolor, color=color)


def theme_graph_attrs(
    theme: VisualizationTheme,
    *,
    font_size: int | None = None,
    dpi: int | None = None,
) -> dict[str, str]:
    """Build graph-level attributes for a theme and convenience knobs.

    Parameters
    ----------
    theme:
        Resolved theme preset.
    font_size:
        Optional graph font size.
    dpi:
        Optional output DPI.

    Returns
    -------
    dict[str, str]
        Graphviz graph attributes.
    """

    attrs: dict[str, str] = dict(theme.graph)
    if font_size is not None:
        attrs["fontsize"] = str(font_size)
    if dpi is not None:
        attrs["dpi"] = str(dpi)
    return attrs


def theme_node_attrs(theme: VisualizationTheme, *, font_size: int | None = None) -> dict[str, str]:
    """Build node-level attributes for a theme and font-size knob.

    Parameters
    ----------
    theme:
        Resolved theme preset.
    font_size:
        Optional node font size.

    Returns
    -------
    dict[str, str]
        Graphviz node attributes.
    """

    attrs: dict[str, str] = dict(theme.node)
    if font_size is not None:
        attrs["fontsize"] = str(font_size)
    return attrs


def theme_edge_attrs(theme: VisualizationTheme, *, font_size: int | None = None) -> dict[str, str]:
    """Build edge-level attributes for a theme and font-size knob.

    Parameters
    ----------
    theme:
        Resolved theme preset.
    font_size:
        Optional edge font size.

    Returns
    -------
    dict[str, str]
        Graphviz edge attributes.
    """

    attrs: dict[str, str] = dict(theme.edge)
    if font_size is not None:
        attrs["fontsize"] = str(font_size)
    return attrs


def legend_lines(theme: VisualizationTheme) -> list[str]:
    """Return human-readable legend lines for ``theme``.

    Parameters
    ----------
    theme:
        Resolved theme preset.

    Returns
    -------
    list[str]
        Legend rows.
    """

    return [f"{label}: {color}" for label, color in theme.legend_items]


def semantic_class_attrs(node_kind: str) -> dict[str, Any]:
    """Return semantic SVG/CSS attributes for a TorchLens node kind.

    Parameters
    ----------
    node_kind:
        Semantic node kind.

    Returns
    -------
    dict[str, Any]
        Attribute mapping for exporters.
    """

    safe_kind = node_kind.replace("_", "-")
    return {"class": f"tl-node tl-node-{safe_kind}"}
