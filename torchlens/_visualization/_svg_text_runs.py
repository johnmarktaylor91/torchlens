"""Flow the mixed-font text runs of a header line or an edit row into one SVG text.

Graphviz writes each font change on an HTML-label line as its own ``<text>``
at an absolute ``x`` taken from its own text metrics. A browser draws the
runs with its own font metrics, so a bold run that comes out wider than
Graphviz measured covers the start of the next run (the header's
``Δ zero_ablate on 2 layers`` read ``zero_ablateon`` in Chromium). Joining
one line's runs into a single ``<text>`` of ``<tspan>`` pieces lets the
browser place each piece after the last, whatever font it resolves.

Graphviz-side padding at a run's edge (the header's two non-breaking spaces
after the coloured edit token, there to absorb Graphviz's short width
estimate) would read as a double gap once the browser lays the pieces out in
order, so the whitespace where two pieces meet collapses to one character.

Only the root graph label (the header block) and edited nodes (``tl_edited``)
are rewritten; every other line keeps Graphviz's bytes. Raster output comes
from Graphviz itself and is unaffected.

Leaf module: imports nothing from the visualization package.
"""

from __future__ import annotations

import re

__all__ = ("flow_text_runs",)

#: One Graphviz text element on its own line: attributes and escaped content.
_TEXT_LINE_RE = re.compile(r"<text (?P<attrs>[^>]*)>(?P<body>[^<]*)</text>\n")
_ATTR_RE = re.compile(r'([a-zA-Z_:][-a-zA-Z0-9_:.]*)="([^"]*)"')
_GRAPH_OPEN_RE = re.compile(r'<g id="graph0" class="graph"[^>]*>\n')
_EDITED_NODE_RE = re.compile(r'<g id="[^"]+" class="node[^"]*\btl_edited\b[^"]*">\n')
#: Attributes that place a run; the rest style it and move onto its tspan.
_PLACEMENT = ("text-anchor", "x", "y")
#: Whitespace Graphviz writes at a run's edge: a space or a non-breaking space.
_EDGE_SPACE = " \u00a0"


def flow_text_runs(svg: str) -> str:
    """Return ``svg`` with each multi-run line of the header and edited nodes as one text.

    Parameters
    ----------
    svg:
        SVG text written by Graphviz.

    Returns
    -------
    str
        The SVG with those lines' runs joined into ``<tspan>`` pieces; other
        content is unchanged.

    Examples
    --------
    >>> from torchlens._visualization._svg_text_runs import flow_text_runs
    >>> svg = ('<g id="graph0" class="graph">\\n'
    ...        '<text text-anchor="start" x="8" y="-5" font-size="14.00">2 edits: </text>\\n'
    ...        '<text text-anchor="start" x="56" y="-5" font-weight="bold">Δ zero_ablate</text>\\n'
    ...        '<!-- n -->\\n')
    >>> flow_text_runs(svg).count("<tspan")
    2
    """

    regions: list[tuple[int, tuple[str, ...]]] = []
    graph_open = _GRAPH_OPEN_RE.search(svg)
    if graph_open is not None:
        regions.append((graph_open.end(), ("<!--", "<g ")))
    regions.extend((match.end(), ("<!--", "</g>")) for match in _EDITED_NODE_RE.finditer(svg))
    if not regions:
        return svg
    pieces: list[str] = []
    cursor = 0
    for start, stop_marks in sorted(regions):
        end = _text_block_end(svg, start, stop_marks)
        pieces.append(svg[cursor:start])
        pieces.append(_flow_block(svg[start:end]))
        cursor = end
    pieces.append(svg[cursor:])
    return "".join(pieces)


def _text_block_end(svg: str, start: int, stop_marks: tuple[str, ...]) -> int:
    """Return where the block of label lines that opens at ``start`` ends.

    The header block stops at the first comment or group (the first node, edge
    or cluster); a node's block stops at its closing tag or the next comment.
    """

    stops = [index for index in (svg.find(mark, start) for mark in stop_marks) if index >= 0]
    return min(stops) if stops else len(svg)


def _flow_block(block: str) -> str:
    """Join consecutive same-baseline, start-anchored text lines of one block."""

    out: list[str] = []
    group: list[dict[str, str]] = []
    cursor = 0
    for match in _TEXT_LINE_RE.finditer(block):
        run = dict(_ATTR_RE.findall(match.group("attrs")))
        run["__body__"] = match.group("body")
        run["__raw__"] = match.group(0)
        adjacent = match.start() == cursor
        if group and adjacent and _same_line(group[-1], run):
            group.append(run)
        else:
            out.append(_emit(group))
            out.append(block[cursor : match.start()])
            group = [run]
        cursor = match.end()
    out.append(_emit(group))
    out.append(block[cursor:])
    return "".join(out)


def _same_line(previous: dict[str, str], run: dict[str, str]) -> bool:
    """Return whether two runs sit on one start-anchored baseline."""

    return (
        previous.get("text-anchor") == run.get("text-anchor") == "start"
        and previous.get("y") == run.get("y")
        and previous.get("y") is not None
    )


def _emit(group: list[dict[str, str]]) -> str:
    """Return one line's SVG: the run unchanged, or a text of tspans for several runs."""

    if not group:
        return ""
    if len(group) == 1:
        return group[0]["__raw__"]
    first = group[0]
    bodies = _collapse_edge_gaps([run["__body__"] for run in group])
    spans = "".join(f"<tspan{_style_attrs(run)}>{body}</tspan>" for run, body in zip(group, bodies))
    return f'<text text-anchor="start" x="{first["x"]}" y="{first["y"]}">{spans}</text>\n'


def _collapse_edge_gaps(bodies: list[str]) -> list[str]:
    """Return ``bodies`` with the whitespace where two pieces meet cut to one character.

    Graphviz places each run at its own ``x``, so padding at a run's edge only
    moves the next run; in one flowed text every character is drawn, so the
    gap keeps its first whitespace character (a non-breaking space survives
    SVG whitespace handling) and drops the rest.
    """

    out = list(bodies)
    for index in range(len(out) - 1):
        left, right = out[index], out[index + 1]
        left_core, right_core = left.rstrip(_EDGE_SPACE), right.lstrip(_EDGE_SPACE)
        gap = left[len(left_core) :] + right[: len(right) - len(right_core)]
        if len(gap) > 1:
            if len(left_core) < len(left):
                out[index], out[index + 1] = left_core + gap[0], right_core
            else:
                out[index + 1] = gap[0] + right_core
    return out


def _style_attrs(run: dict[str, str]) -> str:
    """Return a run's styling attributes (font, weight, fill) for its tspan."""

    return "".join(
        f' {key}="{value}"'
        for key, value in run.items()
        if key not in _PLACEMENT and not key.startswith("__")
    )
