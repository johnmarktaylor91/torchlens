"""Legend helpers: drawn-role detection and the one compact legend table.

The forward picture's key is drawn by ``_header_block`` inside the header
label (JMT, 29 Oct 2026: header, then key, then graph, all top left); this
module still detects which roles a picture drew, builds the channel
disclosure and backward-key sections, and emits the backward key's table.

The backward key is ONE HTML-table plaintext node (vizmech item 13;
the older six disconnected nodes were packed beside the model and drove a
366 pt page to 1048 pt): rows carry the rendered fill as a swatch plus a
text cell, and the node sits in a dedicated ``rank=sink`` group tied into the
main component with an invisible non-constraint edge.

The bounded legend-to-content ratio is enforced by the usability-envelope
oracle (``_geometry_audit``), not at build time. Spellings here are
DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

from ._bool_outcomes import BOOL_FALSE_NODE_COLOR
from ._edit_change import CHANGED_NODE_CLASS
from ._edit_marks import DOWNSTREAM_NODE_CLASS, EDITED_NODE_CLASS
from ._encoding_key import (
    RampKey,
    color_key,
    ramp_row_extent,
    ramp_row_html,
    size_key_line,
    stack_key_line,
)
from ._endpoint_rings import endpoint_ring_role
from ._header_block import _text_width
from ._render_common import (
    BACKWARD_HIGHER_ORDER_COLOR,
    BACKWARD_NODE_COLOR,
    BOOL_NODE_COLOR,
    BUFFER_NODE_COLOR,
    DEFAULT_BG_COLOR,
    FROZEN_PARAMS_BG_COLOR,
    GRADIENT_ARROW_COLOR,
    INPUT_COLOR,
    OUTPUT_COLOR,
    PARAMS_NODE_BG_COLOR,
    TRAINABLE_PARAMS_BG_COLOR,
)
from ._render_utils import html_escape
from .node_spec import NodeSpec
from .overlays import legend_domain_text
from .themes import apply_theme_to_spec

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
        shape vocabulary (``"cylinder node"``) or an edge-style note; on a
        ramp row, the label at the bar's high end.
    ramp:
        Colour-bar segment fills, low value first; non-empty draws ``text``,
        the bar, then ``hint`` on one row.
    """

    text: str
    swatch_fill: str | None = None
    swatch_border: str | None = None
    hint: str | None = None
    ramp: tuple[str, ...] = ()


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
    "changed value",
    "changed by the edit",
)

#: Shapes that never stand for a legend role (layout helpers, image nodes).
_NON_ROLE_SHAPES = frozenset({"plaintext", "plain", "none", "point"})


def _role_fills(theme: VisualizationTheme) -> dict[str, frozenset[str]]:
    """Return each fill-coded role's rendered fills under ``theme``, lowercased."""

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
        "boolean": themed(
            BOOL_NODE_COLOR, BOOL_FALSE_NODE_COLOR, f"{BOOL_NODE_COLOR}:{BOOL_FALSE_NODE_COLOR}"
        ),
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
    interventions the ``tl_edited``, ``tl_downstream`` and ``tl_changed`` SVG
    classes the edit marks carry.
    Collapsed module boxes (``box3d``) are not plain ops; a parameterized one
    counts as parameterized.
    """

    fills = _role_fills(theme)
    roles: set[str] = set()
    for attrs in node_attrs:
        shape = str(attrs.get("shape", "")).lower()
        if shape in _NON_ROLE_SHAPES or "invis" in str(attrs.get("style", "")):
            continue
        # Drop Graphviz segment weights: a mixed if-test's split is weighted.
        fill = re.sub(r";[0-9.]+", "", str(attrs.get("fillcolor", "")).lower())
        classes = str(attrs.get("class", "")).split()
        if EDITED_NODE_CLASS in classes:
            roles.add("edited")
        if DOWNSTREAM_NODE_CLASS in classes:
            roles.add("changed by the edit")
        if CHANGED_NODE_CLASS in classes:
            roles.add("changed value")
        if shape == "cylinder":
            if fill in fills["buffer"]:
                roles.add("buffer")
            elif fill in fills["parameterized"]:
                roles.add("mutated parameter")
            continue
        ring_role = endpoint_ring_role(attrs, theme)
        if ring_role is not None:
            roles.add(ring_role)  # an endpoint under a colour scale: role on the ring
        # A fill the skin shares with plain ops (the default skin's white
        # tests and buffers) names no other role on a non-cylinder node.
        plain_fill = fill in fills["plain op"]
        for role, role_fills in fills.items():
            if role == "buffer":
                continue  # a buffer is cued by its cylinder, matched above, never by fill alone
            if plain_fill and role != "plain op":
                continue
            if fill in role_fills and not (role == "plain op" and shape == "box3d"):
                roles.add(role)
    return frozenset(roles)


def encoding_sections(state: Any) -> tuple[LegendSection, ...]:
    """Build the untitled channel key section from an ``EncodingState``.

    The drawn key is plain: a labelled colour bar and one line per other
    channel, nothing else.
    """

    rows: list[LegendRow] = []
    color = color_key(state, legend_domain_text)
    if isinstance(color, RampKey):
        rows.append(LegendRow(text=color.low_text, ramp=color.fills, hint=color.high_text))
    elif color is not None:
        rows.append(LegendRow(text=color))
    for line in (size_key_line(state, legend_domain_text), stack_key_line(state)):
        if line is not None:
            rows.append(LegendRow(text=line))
    if not rows:
        return ()
    return (LegendSection(title="", rows=tuple(rows)),)


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


def build_legend_table_label(
    sections: tuple[LegendSection, ...], theme: VisualizationTheme, *, fit: bool = False
) -> str:
    """Return the Graphviz HTML label for the one legend table node.

    Parameters
    ----------
    sections:
        Ordered legend sections.
    theme:
        Resolved theme (fonts, border, text color).
    fit:
        Fix every text cell to its measured size and hold the table in a
        fixed wrapper of exactly its content size. Graphviz stretches a nested
        table to fill its cell, so a key under a wider header drew its border
        out to the header's width with empty space on the right; the header
        block passes ``True``. A standalone
        legend node is never stretched and keeps the natural layout.
    """

    cells: list[str] = []
    extents: list[tuple[int, int, int]] = []  # (swatch column, other column, height)
    for section in sections:
        if section.title:
            title = f"<B>{html_escape(section.title)}</B>"
            cell, extent = _text_cell(
                title, section.title, theme, _CellLayout(span=True, bold=True, fit=fit)
            )
            cells.append(f"<TR>{cell}</TR>")
            extents.append(extent)
        for row in section.rows:
            cell, extent = _legend_row(row, theme, titled=bool(section.title), fit=fit)
            cells.append(cell)
            extents.append(extent)
    table = (
        f'<TABLE BORDER="{_LEGEND_BORDER}" CELLBORDER="0" CELLSPACING="0" '
        f'CELLPADDING="{_LEGEND_PADDING}" COLOR="{theme.default_border}">'
        + "".join(cells)
        + "</TABLE>"
    )
    if fit:
        # The outlined table stays natural inside a borderless fixed, left-aligned
        # wrapper of exactly its size: Graphviz 2.43 draws the outline of a fixed
        # table that is also ALIGN="LEFT" at the wrong width.
        width, height = _fitted_table_size(extents)
        table = (
            '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0" '
            f'FIXEDSIZE="TRUE" ALIGN="LEFT" WIDTH="{width}" HEIGHT="{height}">'
            f"<TR><TD>{table}</TD></TR></TABLE>"
        )
    return f"<{table}>"


#: Outline and cell padding of the legend table, in points.
_LEGEND_BORDER = 1
_LEGEND_PADDING = 3
#: The swatch cell's fixed size, in points (its 1 pt border included).
_SWATCH_SIZE = (16, 10)


def _legend_row(
    row: LegendRow, theme: VisualizationTheme, *, titled: bool, fit: bool
) -> tuple[str, tuple[int, int, int]]:
    """Return one legend row's ``<TR>`` and its (swatch, rest, height) extent.

    A swatch extent of 0 marks a row that spans both columns.
    """

    if row.ramp:
        width, height = _ramp_extent(row, theme)
        pad = 2 * _LEGEND_PADDING
        return _ramp_row(row, theme), (0, width + pad, height + pad)
    text = html_escape(row.text)
    plain = row.text
    if row.hint is not None:
        text += f" ({html_escape(row.hint)})"
        plain += f" ({row.hint})"
    if not titled:
        # An untitled section (the encoding key) has no swatch column: the
        # line spans the table, so a stretched table cannot push it right.
        cell, extent = _text_cell(text, plain, theme, _CellLayout(span=True, bold=False, fit=fit))
        return f"<TR>{cell}</TR>", extent
    cell, (_, text_w, text_h) = _text_cell(
        text, plain, theme, _CellLayout(span=False, bold=False, fit=fit)
    )
    if row.swatch_fill is None:
        swatch, swatch_w, swatch_h = "<TD></TD>", 2 * _LEGEND_PADDING, 2 * _LEGEND_PADDING
    else:
        swatch_w, swatch_h = _SWATCH_SIZE
        swatch = (
            f'<TD FIXEDSIZE="TRUE" WIDTH="{swatch_w}" HEIGHT="{swatch_h}" '
            f'BGCOLOR="{row.swatch_fill}" BORDER="1" '
            f'COLOR="{row.swatch_border or theme.default_border}"></TD>'
        )
    return f"<TR>{swatch}{cell}</TR>", (swatch_w, text_w, max(swatch_h, text_h))


class _CellLayout(NamedTuple):
    """How one legend text cell is laid out: spanning both columns, bold, fixed size."""

    span: bool
    bold: bool
    fit: bool


def _text_cell(
    text: str, plain: str, theme: VisualizationTheme, layout: _CellLayout
) -> tuple[str, tuple[int, int, int]]:
    """Return a left-aligned text ``<TD>`` and its (0, width, height) extent.

    ``text`` is the escaped markup and ``plain`` the same words unescaped for
    measuring; under ``layout.fit`` the cell is fixed to the measured size.
    """

    size = theme.typography.secondary_size
    pad = 2 * _LEGEND_PADDING
    width = math.ceil(_text_width(plain, size, layout.bold) * 1.06 + 2.0) + pad
    height = math.ceil(size * 1.25) + pad
    attrs = ' COLSPAN="2"' if layout.span else ""
    if layout.fit:
        attrs += f' FIXEDSIZE="TRUE" WIDTH="{width}" HEIGHT="{height}"'
    cell = (
        f'<TD{attrs} ALIGN="LEFT"><FONT POINT-SIZE="{theme.typography.secondary_pt}" '
        f'COLOR="{theme.default_font}">{text}</FONT></TD>'
    )
    return cell, (0, width, height)


def _fitted_table_size(extents: list[tuple[int, int, int]]) -> tuple[int, int]:
    """Return the legend table's whole-point width and height from its row extents.

    Two-column rows share the swatch and text columns; a spanning row needs
    only its own width. The 1 pt outline adds on each side (cell spacing 0).
    """

    swatch = max((extent[0] for extent in extents), default=0)
    paired = [extent[1] for extent in extents if extent[0]]
    spans = [extent[1] for extent in extents if not extent[0]]
    width = max([swatch + max(paired, default=0), *spans])
    height = sum(extent[2] for extent in extents)
    return width + 2 * _LEGEND_BORDER, height + 2 * _LEGEND_BORDER


def _ramp_extent(row: LegendRow, theme: VisualizationTheme) -> tuple[int, int]:
    """Return the colour-bar row's fixed table size, measured in the key's face."""

    size = theme.typography.secondary_size
    bold = "bold" in theme.typography.family.lower()
    return ramp_row_extent(
        row, theme.typography.secondary_pt, lambda text: _text_width(text, size, bold)
    )


def _ramp_row(row: LegendRow, theme: VisualizationTheme) -> str:
    """Return the key's colour-bar row, measured in the skin's key face."""

    size = theme.typography.secondary_size
    bold = "bold" in theme.typography.family.lower()
    return ramp_row_html(
        row,
        theme.typography.secondary_pt,
        (theme.default_font, theme.default_border),
        lambda text: _text_width(text, size, bold),
    )


def add_legend_table_to_graphviz(
    dot: graphviz.Digraph,
    theme: VisualizationTheme,
    sections: tuple[LegendSection, ...],
    *,
    anchor: str | None = None,
    fit: bool = False,
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
    fit:
        Size every text cell from the bundled face's measured width plus the
        key's margin (:func:`build_legend_table_label`). Without it a cell is
        exactly as wide as Graphviz measured its words, so any viewer face a
        little wider than that measurement runs the words past the border.
    """

    if not sections:
        return
    label = build_legend_table_label(sections, theme, fit=fit)
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
