"""One compact legend table in a dedicated rank (vizmech item 13, D28).

The historical legend was six DISCONNECTED Graphviz nodes in a cluster. dot
packs disconnected components side by side, so the strip (839 x 77 pt on the
memo's resnet50 depth-1 measurement) drove the page from 366 pt to 1048 pt
wide -- 2.86x -- while the collision audit scored the render CLEAN (the
two-oracles worked example, memo D2). The channel disclosure legend had the
same disconnected-nodes form.

This module renders every legend as ONE HTML-table plaintext node:

- rows carry a color swatch cell (the REAL rendered fill, resolved through
  ``apply_theme_to_spec`` -- the legend must show what was painted, never an
  aspirational palette hex) plus a text cell;
- the node is placed in a DEDICATED RANK (``rank=sink``) and tied into the
  main component with an invisible non-constraint edge when an anchor node is
  known, so it stops being a disconnected component dot packs BESIDE the
  model;
- the theme role legend, the channel (AUTO) disclosure legend, and the
  backward-vocabulary key (vizmech item 17, D30) are SECTIONS of the same
  table, so a render never carries two competing legend topologies.

The bounded legend-to-content ratio is enforced by the usability-envelope
oracle (``_geometry_audit``), not at build time. Spellings here are
DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ._render_utils import html_escape

if TYPE_CHECKING:
    import graphviz

    from .themes import VisualizationTheme

__all__ = [
    "LEGEND_NODE_NAME",
    "LegendRow",
    "LegendSection",
    "add_legend_table_to_graphviz",
    "backward_key_sections",
    "build_legend_table_label",
    "encoding_sections",
    "legend_table_lines_for_rank_path",
    "theme_role_sections",
]

#: The one legend node name. Tests and the audit key on this marker.
LEGEND_NODE_NAME = "tl_legend"


@dataclass(frozen=True)
class LegendRow:
    """One legend table row: an optional swatch plus explanatory text.

    Attributes
    ----------
    text:
        Plain explanatory text (escaped at build time).
    swatch_fill:
        Fill color for the swatch cell; ``None`` renders a text-only row.
    swatch_border:
        Border color for the swatch cell (defaults to the theme border).
    hint:
        Optional short suffix rendered after the text, e.g. the non-oval
        shape vocabulary (``"cylinder node"``) or an edge-style note.
    """

    text: str
    swatch_fill: str | None = None
    swatch_border: str | None = None
    hint: str | None = None


@dataclass(frozen=True)
class LegendSection:
    """A titled group of legend rows (role legend, encoding, backward key)."""

    title: str
    rows: tuple[LegendRow, ...]


def theme_role_sections(theme: VisualizationTheme) -> tuple[LegendSection, ...]:
    """Build the node-role legend section from the RENDERED specs.

    The swatches resolve through the same ``NodeSpec`` + ``apply_theme_to_spec``
    path the graph nodes use, so the legend shows the colors actually painted
    (the dark theme remaps parameterized fills, for example) -- never the
    aspirational palette hexes.
    """

    from ._render_common import (
        BOOL_NODE_COLOR,
        DEFAULT_BG_COLOR,
        INPUT_COLOR,
        OUTPUT_COLOR,
        TRAINABLE_PARAMS_BG_COLOR,
    )
    from .node_spec import INTERVENTION_CONE_COLOR, INTERVENTION_SITE_COLOR, NodeSpec
    from .themes import apply_theme_to_spec

    role_specs = (
        ("input", NodeSpec(["input"], shape="oval", fillcolor=INPUT_COLOR), None),
        ("output", NodeSpec(["output"], shape="oval", fillcolor=OUTPUT_COLOR), None),
        (
            "parameterized",
            NodeSpec(["parameterized"], shape="oval", fillcolor=TRAINABLE_PARAMS_BG_COLOR),
            None,
        ),
        ("buffer", NodeSpec(["buffer"], shape="cylinder", fillcolor=DEFAULT_BG_COLOR), "cylinder"),
        ("boolean", NodeSpec(["boolean"], shape="oval", fillcolor=BOOL_NODE_COLOR), None),
        (
            "intervention/cone",
            NodeSpec(
                ["intervention/cone"],
                shape="oval",
                fillcolor=INTERVENTION_CONE_COLOR,
                color=INTERVENTION_SITE_COLOR,
            ),
            None,
        ),
    )
    rows = []
    for text, spec, hint in role_specs:
        themed = apply_theme_to_spec(spec, theme)
        rows.append(
            LegendRow(
                text=text,
                swatch_fill=str(themed.fillcolor),
                swatch_border=str(themed.color),
                hint=hint,
            )
        )
    return (LegendSection(title="TorchLens legend", rows=tuple(rows)),)


def encoding_sections(state: Any) -> tuple[LegendSection, ...]:
    """Build the channel disclosure sections from an ``EncodingState``.

    Every disclosure line the disconnected-node form carried is preserved
    (AUTO disclosures are contract text, memo 2.3); ramp stop rows keep their
    interpolated swatches.
    """

    from ._encoding import _color_legend_rows, _non_color_legend_rows

    rows: list[LegendRow] = []
    for spec in (*_color_legend_rows(state), *_non_color_legend_rows(state)):
        fill = spec.fillcolor if isinstance(spec.fillcolor, str) else None
        for index, line in enumerate(spec.lines):
            rows.append(LegendRow(text=str(line), swatch_fill=fill if index == 0 else None))
    if not rows:
        return ()
    return (LegendSection(title="TorchLens encoding", rows=tuple(rows)),)


def backward_key_sections(
    *,
    has_higher_order: bool,
    has_intervening: bool,
    has_accumulation: bool,
    has_custom: bool,
    num_backward_passes: int,
) -> tuple[LegendSection, ...]:
    """Build the backward-vocabulary key from the styles ACTUALLY painted.

    The WGAN-GP acceptance contract (memo section 7) requires an automatic
    legend row for every active style and no unexplained encodings; a key row
    for a style the render does not use would itself be an unexplained claim,
    so each row is gated on the corresponding inventory flag.
    """

    from ._render_common import (
        BACKWARD_HIGHER_ORDER_COLOR,
        BACKWARD_NODE_COLOR,
        GRADIENT_ARROW_COLOR,
    )

    rows: list[LegendRow] = [
        LegendRow(text="backward op (grad_fn)", swatch_fill=BACKWARD_NODE_COLOR)
    ]
    if has_higher_order:
        rows.append(
            LegendRow(
                text="order 2+: grad-of-grad (double backprop)",
                swatch_fill=BACKWARD_HIGHER_ORDER_COLOR,
            )
        )
    if has_intervening:
        rows.append(LegendRow(text="[i] = intervening grad_fn (no forward op)"))
    if has_custom:
        rows.append(LegendRow(text="[custom] = custom autograd function"))
    if has_accumulation:
        rows.append(
            LegendRow(
                text="accum = gradient accumulation into a leaf",
                swatch_fill=GRADIENT_ARROW_COLOR,
                hint="dotted edge",
            )
        )
    if num_backward_passes > 1:
        rows.append(LegendRow(text="bwd N = backward pass N; order N = derivative order"))
    else:
        rows.append(LegendRow(text="order N = derivative order"))
    return (LegendSection(title="backward key", rows=tuple(rows)),)


def build_legend_table_label(sections: tuple[LegendSection, ...], theme: VisualizationTheme) -> str:
    """Return the Graphviz HTML label for the one legend table node."""

    font_color = theme.default_font
    border_color = theme.default_border
    size = theme.typography.secondary_pt
    cells: list[str] = []
    for section in sections:
        cells.append(
            f'<TR><TD COLSPAN="2" ALIGN="LEFT"><FONT POINT-SIZE="{size}" '
            f'COLOR="{font_color}"><B>{html_escape(section.title)}</B></FONT></TD></TR>'
        )
        for row in section.rows:
            text = html_escape(row.text)
            if row.hint is not None:
                text += f" ({html_escape(row.hint)})"
            if row.swatch_fill is not None:
                swatch_border = row.swatch_border or border_color
                swatch = (
                    f'<TD FIXEDSIZE="TRUE" WIDTH="16" HEIGHT="10" '
                    f'BGCOLOR="{row.swatch_fill}" BORDER="1" COLOR="{swatch_border}"></TD>'
                )
            else:
                swatch = "<TD></TD>"
            cells.append(
                f"<TR>{swatch}"
                f'<TD ALIGN="LEFT"><FONT POINT-SIZE="{size}" COLOR="{font_color}">'
                f"{text}</FONT></TD></TR>"
            )
    return (
        f'<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="3" '
        f'COLOR="{border_color}">' + "".join(cells) + "</TABLE>>"
    )


def add_legend_table_to_graphviz(
    dot: graphviz.Digraph,
    theme: VisualizationTheme,
    sections: tuple[LegendSection, ...],
    *,
    anchor: str | None = None,
) -> None:
    """Emit the one legend table node in a dedicated rank.

    Parameters
    ----------
    dot:
        Graph under construction.
    theme:
        Resolved theme (fonts, border, text color).
    sections:
        Ordered legend sections; empty emits nothing.
    anchor:
        Name of a real graph node. When given, an invisible non-constraint
        edge ties the legend into the anchor's component so dot stops packing
        it BESIDE the model (the 2.86x page-width defect); the dedicated
        ``rank=sink`` keeps it at the terminal rank.
    """

    if not sections:
        return
    label = build_legend_table_label(sections, theme)
    with dot.subgraph() as legend_rank:
        legend_rank.attr(rank="sink")
        legend_rank.node(
            LEGEND_NODE_NAME,
            label=label,
            shape="plaintext",
            margin="0",
            fontname=theme.typography.family,
        )
    if anchor is not None:
        dot.edge(anchor, LEGEND_NODE_NAME, style="invis", constraint="false", arrowhead="none")


def legend_table_lines_for_rank_path(
    theme: Any,
    max_y: float,
) -> list[str]:
    """Return raw-DOT lines for the rank path's pinned legend table.

    The rank path builds DOT text directly (no ``graphviz.Digraph``), and
    ``neato -n`` needs every node positioned, so the one table node is pinned
    left of the graph origin like the historical pinned cluster -- but as ONE
    compact node instead of six.
    """

    from .themes import THEME_PRESETS

    resolved = theme if theme is not None else THEME_PRESETS["torchlens"]
    label = build_legend_table_label(theme_role_sections(resolved), resolved)
    return [
        f"  {LEGEND_NODE_NAME} [label={label} shape=plaintext margin=0 "
        f'fontname="{resolved.typography.family}" pos="-200.0,{max(max_y - 40.0, 40.0):.1f}!"]'
    ]
