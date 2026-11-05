"""The drawn encoding key and the non-drawn record of how values were encoded.

A drawing carries no explanation lines, so the key for ``color_by`` /
``size_by`` / ``stack_by`` is plain: a small colour bar labelled with its
quantity and units at both ends (``time: 0.098 ms [bar] 1.46 ms``) and one
line per other channel (``size: output bytes, 15 KB ... 60 KB``).

The method details the key used to print (the transform, how many nodes were
encoded, cross-pass and module-box aggregation, unencoded-node notes, the size
scale) are kept as text in the graph's ``comment`` attribute: it rides in the
DOT source ``draw()`` returns and in a written SVG, and is never drawn
(:func:`record_encoding_method`). The live ``EncodingState`` keeps the same
facts as fields.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ._render_utils import html_escape

if TYPE_CHECKING:
    from ._encoding import EncodingChannelSpec, EncodingState
    from ._legend import LegendRow

#: Formats one value with its source's units (``overlays.legend_domain_text``).
#: Callers pass it in so this module imports nothing from the render core.
ValueText = Callable[["EncodingChannelSpec", float], str]

#: Number of solid segments in the key's colour bar (first and last are the
#: ramp's end colours).
KEY_BAR_SEGMENTS = 16
#: Width and height of one bar segment, in points.
KEY_BAR_SEGMENT_PT = (5, 10)
#: Spacing between the key row's three cells, in points.
_ROW_SPACING = 3
#: Prefix of the graph comment that holds the method record.
METHOD_COMMENT_PREFIX = "encoding method: "

#: Builtin source tokens whose key name says more than the token.
_KEY_QUANTITY_NAMES = {
    "bytes": "output bytes",
    "dims": "output elements",
    "grad_norm": "gradient norm",
    "device_time": "device time",
}

_TRANSFORM_WORDING = {
    "linear": "linear min-max",
    "rank": "rank mapping (ordinal, not ratio)",
    "log": "log scale (scale-invariant; values <= 0 unencoded)",
}


@dataclass(frozen=True)
class RampKey:
    """The colour bar row: low label, segment fills, high label."""

    low_text: str
    fills: tuple[str, ...]
    high_text: str


def quantity_name(spec: EncodingChannelSpec) -> str:
    """Return the plain quantity name the key prints for ``spec``.

    Parameters
    ----------
    spec:
        A resolved channel spec.

    Returns
    -------
    str
        For example ``"time"`` or ``"output bytes"``.
    """

    if spec.source_kind in ("builtin", "dims"):
        return _KEY_QUANTITY_NAMES.get(spec.display_name, spec.display_name)
    if spec.source_kind == "callable":
        return spec.display_name.removeprefix("callable ")
    if spec.source_kind == "auto":  # stack_by=True: the derived pass number
        return "pass"
    return spec.display_name


def _no_range_text(
    spec: EncodingChannelSpec, values: dict[str, float], value_text: ValueText
) -> str:
    """Key wording when nothing could be spread over a range."""

    if not values:
        return "no values"
    distinct = set(values.values())
    if len(distinct) == 1:
        return f"all {value_text(spec, distinct.pop())}"
    # Only the log transform leaves differing values uncoloured.
    return "too few values > 0 to compare"


def color_key(state: EncodingState, value_text: ValueText) -> RampKey | str | None:
    """Return the colour channel's key: a bar row, a text line, or ``None``.

    Parameters
    ----------
    state:
        The populated per-draw encoding state.
    value_text:
        Formats one value with its units.

    Returns
    -------
    RampKey | str | None
        The bar when nodes were coloured, a plain line when no value range
        exists (``"time: all 0.1 ms"``), ``None`` when color_by is off.
    """

    spec = state.spec
    if spec is None:
        return None
    name = quantity_name(spec)
    if state.domain is None or not state.colors:
        return f"{name}: {_no_range_text(spec, state.values, value_text)}"
    low, high = state.domain
    fills = tuple(
        state.ramp_color(index / (KEY_BAR_SEGMENTS - 1)) for index in range(KEY_BAR_SEGMENTS)
    )
    return RampKey(f"{name}: {value_text(spec, low)}", fills, value_text(spec, high))


def size_key_line(state: EncodingState, value_text: ValueText) -> str | None:
    """Return the size channel's key line, or ``None`` when size_by is off.

    Parameters
    ----------
    state:
        The populated per-draw encoding state.
    value_text:
        Formats one value with its units.

    Returns
    -------
    str | None
        For example ``"size: output bytes, 15 KB ... 60 KB"``.
    """

    spec = state.size_spec
    if spec is None:
        return None
    name = quantity_name(spec)
    if state.size_domain is None or not state.sizes:
        return f"size: {name}, {_no_range_text(spec, state.size_values, value_text)}"
    low, high = state.size_domain
    return f"size: {name}, {value_text(spec, low)} ... {value_text(spec, high)}"


def stack_key_line(state: EncodingState) -> str | None:
    """Return the stack channel's key line, or ``None`` when stack_by is off.

    Parameters
    ----------
    state:
        The populated per-draw encoding state.

    Returns
    -------
    str | None
        For example ``"aligned: same stage"``.
    """

    if state.stack_spec is None:
        return None
    return f"aligned: same {quantity_name(state.stack_spec)}"


def ramp_row_html(
    row: LegendRow,
    size: str,
    colors: tuple[str, str],
    text_width: Callable[[str], float],
) -> str:
    """Return one key row holding ``text``, the colour bar, then ``hint``.

    Graphviz spreads a stretched table's extra width over its columns, which
    pushed the labels away from the bar; the row's table is therefore fixed
    to its content (``text_width`` measures the labels) and left-aligned.

    Parameters
    ----------
    row:
        A row whose ``ramp`` holds the bar's segment fills and whose ``hint``
        is the high-end label.
    size:
        Text point size.
    colors:
        Text ink and bar outline colour.
    text_width:
        Width in points of a label in the key's face.

    Returns
    -------
    str
        A full-width ``<TR>`` for the key table.
    """

    font_color, border_color = colors
    width, height = KEY_BAR_SEGMENT_PT
    segments = "".join(
        f'<TD FIXEDSIZE="TRUE" WIDTH="{width}" HEIGHT="{height}" BGCOLOR="{fill}"></TD>'
        for fill in row.ramp
    )
    bar_width = width * len(row.ramp) + 2  # plus the 1 pt outline each side
    bar = (
        f'<TD FIXEDSIZE="TRUE" WIDTH="{bar_width}" HEIGHT="{height + 2}">'
        f'<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0" '
        f'COLOR="{border_color}"><TR>{segments}</TR></TABLE></TD>'
    )
    texts = (row.text, row.hint or "")
    # Graphviz takes whole points for WIDTH and HEIGHT.
    widths = [math.ceil(text_width(text) * 1.06 + 2.0) for text in texts]
    row_height = max(height + 2, math.ceil(float(size) * 1.25))
    cells = [
        f'<TD FIXEDSIZE="TRUE" WIDTH="{cell_width}" HEIGHT="{row_height}" ALIGN="LEFT">'
        f'<FONT POINT-SIZE="{size}" COLOR="{font_color}">{html_escape(text)}</FONT></TD>'
        for text, cell_width in zip(texts, widths, strict=True)
    ]
    table_width = sum(widths) + bar_width + _ROW_SPACING * 4
    return (
        f'<TR><TD COLSPAN="2" ALIGN="LEFT"><TABLE BORDER="0" CELLBORDER="0" '
        f'CELLSPACING="{_ROW_SPACING}" CELLPADDING="0" FIXEDSIZE="TRUE" ALIGN="LEFT" '
        f'WIDTH="{table_width}" HEIGHT="{row_height + 2 * _ROW_SPACING}">'
        f"<TR>{cells[0]}{bar}{cells[1]}</TR></TABLE></TD></TR>"
    )


def _color_method_lines(state: EncodingState, value_text: ValueText) -> list[str]:
    """Return the colour channel's method details."""

    spec = state.spec
    if spec is None:
        return []
    lines = [
        f"color_by: {spec.display_name}",
        _TRANSFORM_WORDING[spec.transform],
        f"encoded {len(state.colors)} of {state.eligible_count} eligible nodes",
        *state.aggregation_lines,
        *state.notes,
    ]
    if state.domain is not None and state.colors:
        low, high = state.domain
        lines.append(f"min {value_text(spec, low)}, max {value_text(spec, high)}")
        lines.append(f"ramp {' '.join(state.ramp)} (OKLab)")
    return lines


def encoding_method_lines(state: EncodingState, value_text: ValueText) -> list[str]:
    """Return every method detail for the active channels, one fact per line.

    Parameters
    ----------
    state:
        The populated per-draw encoding state.
    value_text:
        Formats one value with its units.

    Returns
    -------
    list[str]
        The transform, coverage, aggregation, notes and size-scale lines.
    """

    lines = _color_method_lines(state, value_text)
    size_spec = state.size_spec
    if size_spec is not None:
        lines += [
            f"size_by: {size_spec.display_name}",
            f"size ~ {state.size_scale}({size_spec.display_name}), min-max, clamped",
            *state.size_aggregation_lines,
            *state.size_notes,
        ]
        if state.size_domain is not None:
            low, high = state.size_domain
            lines.append(
                f"size min {value_text(size_spec, low)}, max {value_text(size_spec, high)}"
            )
    if state.stack_spec is not None:
        lines += [
            f"stack_by: {state.stack_spec.display_name}",
            "same rank = same annotation value",
            *state.stack_notes,
        ]
    return lines


def record_encoding_method(dot: Any, state: EncodingState, value_text: ValueText) -> None:
    """Write the method details into ``dot``'s graph comment (never drawn).

    Parameters
    ----------
    dot:
        The Graphviz graph under construction.
    state:
        The populated per-draw encoding state.
    value_text:
        Formats one value with its units.
    """

    lines = encoding_method_lines(state, value_text)
    if lines:
        # An SVG carries the comment as an XML comment, where "--" is illegal.
        text = " | ".join(line.replace(" -- ", ": ") for line in lines)
        dot.graph_attr["comment"] = METHOD_COMMENT_PREFIX + text
