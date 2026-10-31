"""Legend helpers: drawn-role detection and the one compact legend table.

The forward picture's key is drawn by ``_header_block`` inside the header
label (JMT, 29 Oct 2026: header, then key, then graph, all top left); this
module still detects which roles a picture drew, builds the channel
disclosure and backward-key sections, and emits the backward key's table.

The backward key is ONE HTML-table plaintext node (vizmech item 13, D28;
the older six disconnected nodes were packed beside the model and drove a
366 pt page to 1048 pt): rows carry the rendered fill as a swatch plus a
text cell, and the node sits in a dedicated ``rank=sink`` group tied into the
main component with an invisible non-constraint edge.

The bounded legend-to-content ratio is enforced by the usability-envelope
oracle (``_geometry_audit``), not at build time. Spellings here are
DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ._render_utils import html_escape

if TYPE_CHECKING:
    import graphviz

    from .themes import VisualizationTheme

__all__ = [
    "LEGEND_NODE_NAME",
    "LEGEND_ROLES",
    "LegendRow",
    "LegendSection",
    "add_legend_table_to_graphviz",
    "backward_key_sections",
    "build_legend_table_label",
    "drawn_legend_roles",
    "encoding_sections",
    "statement_node_attrs",
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


#: Legend role rows in display order. ``drawn_legend_roles`` returns these keys.
LEGEND_ROLES = (
    "input",
    "output",
    "plain op",
    "parameterized",
    "buffer",
    "mutated parameter",
    "boolean",
    "edited",
    "changed by the edit",
)

#: Shapes that never stand for a legend role (layout helpers, image nodes).
_NON_ROLE_SHAPES = frozenset({"plaintext", "plain", "none", "point"})


def _role_fills(theme: VisualizationTheme) -> dict[str, frozenset[str]]:
    """Return each fill-coded role's rendered fills under ``theme``, lowercased."""

    from ._render_common import (
        BOOL_NODE_COLOR,
        BUFFER_NODE_COLOR,
        DEFAULT_BG_COLOR,
        FROZEN_PARAMS_BG_COLOR,
        INPUT_COLOR,
        OUTPUT_COLOR,
        PARAMS_NODE_BG_COLOR,
        TRAINABLE_PARAMS_BG_COLOR,
    )
    from .node_spec import NodeSpec
    from .themes import apply_theme_to_spec

    def themed(*constants: str) -> frozenset[str]:
        """Return the lowercased fills ``theme`` paints for ``constants``."""
        return frozenset(
            str(apply_theme_to_spec(NodeSpec(["x"], fillcolor=constant), theme).fillcolor).lower()
            for constant in constants
        )

    return {
        "input": themed(INPUT_COLOR),
        "output": themed(OUTPUT_COLOR),
        "plain op": themed(DEFAULT_BG_COLOR),
        "parameterized": themed(
            TRAINABLE_PARAMS_BG_COLOR,
            FROZEN_PARAMS_BG_COLOR,
            PARAMS_NODE_BG_COLOR,
            f"{TRAINABLE_PARAMS_BG_COLOR}:{FROZEN_PARAMS_BG_COLOR}",
        ),
        "buffer": themed(BUFFER_NODE_COLOR),
        "boolean": themed(BOOL_NODE_COLOR),
    }


def statement_node_attrs(statements: Iterable[Any]) -> Iterator[dict[str, Any]]:
    """Yield the attributes of every node statement in a RenderIR statement tree."""

    for statement in statements:
        if statement.kind == "node":
            yield dict(statement.attrs)
        yield from statement_node_attrs(statement.children)


def drawn_legend_roles(
    node_attrs: Iterable[Mapping[str, Any]], theme: VisualizationTheme
) -> frozenset[str]:
    """Return the legend roles some drawn node actually carries.

    A role counts only when a node was painted with it: its rendered fill
    (and, for buffers and mutated Parameters, the cylinder), or for
    interventions the ``tl_edited`` and ``tl_downstream`` SVG classes the
    edit marks carry.
    Collapsed module boxes (``box3d``) are not plain ops; a parameterized one
    counts as parameterized.
    """

    from ._edit_marks import DOWNSTREAM_NODE_CLASS, EDITED_NODE_CLASS

    fills = _role_fills(theme)
    roles: set[str] = set()
    for attrs in node_attrs:
        shape = str(attrs.get("shape", "")).lower()
        if shape in _NON_ROLE_SHAPES or "invis" in str(attrs.get("style", "")):
            continue
        fill = str(attrs.get("fillcolor", "")).lower()
        classes = str(attrs.get("class", "")).split()
        if EDITED_NODE_CLASS in classes:
            roles.add("edited")
        if DOWNSTREAM_NODE_CLASS in classes:
            roles.add("changed by the edit")
        if shape == "cylinder":
            if fill in fills["buffer"]:
                roles.add("buffer")
            elif fill in fills["parameterized"]:
                roles.add("mutated parameter")
            continue
        for role, role_fills in fills.items():
            if fill in role_fills and not (role == "plain op" and shape == "box3d"):
                roles.add(role)
    return frozenset(roles)


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
