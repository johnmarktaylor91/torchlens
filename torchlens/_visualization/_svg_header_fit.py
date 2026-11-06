"""Keep an SVG's header inside the page when a browser draws it wider than Graphviz measured.

Graphviz sizes the graph to its header (the root graph label) with its own text
metrics, which snap glyph advances to a pixel grid and come out a few percent
narrower than the face's real advances. A browser draws the real advances, so
a header line that set the drawing's width ran past the right edge and was cut
off ("... on interventionreplacement_1_5"). This step measures each header line
in the bundled face at its real advances and, when one would pass the right
margin, widens the page (the ``<svg>`` width, its viewBox and the page fill) so
the line fits. The graph itself does not move; every other byte is unchanged.

Raster and PDF output come from Graphviz itself, which draws with the metrics it
measured, so only the SVG needs this.

Leaf module: imports nothing from the visualization package but the text metrics.
"""

from __future__ import annotations

import html
import math
import re

from ._text_metrics import text_width

__all__ = ("fit_header_width",)

#: Space kept between the widest header line and the page edge, in points
#: (Graphviz's own page margin).
_PAGE_MARGIN = 4.0
#: The root graph group and the page fill Graphviz draws first inside it.
_GRAPH_OPEN_RE = re.compile(r'<g id="graph0" class="graph" transform="[^"]*">\n')
_PAGE_FILL_RE = re.compile(r'<polygon fill="[^"]*" stroke="transparent" points="([^"]*)"/>\n')
#: The first node, cluster or edge group: the header block ends before it.
_FIRST_ELEMENT_RE = re.compile(r'<g id="(?:node|clust|edge)\d+" class=')
_TEXT_RE = re.compile(r"<text (?P<attrs>[^>]*)>(?P<body>.*?)</text>", re.S)
_TSPAN_RE = re.compile(r"<tspan (?P<attrs>[^>]*)>(?P<body>[^<]*)</tspan>")
_ATTR_RE = re.compile(r'([a-zA-Z_:][-a-zA-Z0-9_:.]*)="([^"]*)"')
_SVG_ROOT_RE = re.compile(r'<svg width="(?P<w>[\d.]+)pt"(?P<mid>[^>]*?)viewBox="(?P<vb>[^"]*)"')


def _run_width(attrs: dict[str, str], body: str, inherited: dict[str, str]) -> float:
    """Return one text run's width from its own font attributes, else the enclosing text's."""

    merged = {**inherited, **attrs}
    try:
        size = float(merged.get("font-size", "14"))
    except ValueError:
        size = 14.0
    bold = merged.get("font-weight", "") == "bold"
    return text_width(html.unescape(body), size, bold)


def _line_right(attrs_text: str, body: str) -> float | None:
    """Return a left-anchored header line's right edge in graph units, else ``None``."""

    attrs = dict(_ATTR_RE.findall(attrs_text))
    if attrs.get("text-anchor", "start") != "start":
        return None
    try:
        left = float(attrs["x"])
    except (KeyError, ValueError):
        return None
    runs = list(_TSPAN_RE.finditer(body))
    if not runs:
        return left + _run_width({}, body, attrs)
    return left + sum(
        _run_width(dict(_ATTR_RE.findall(run.group("attrs"))), run.group("body"), attrs)
        for run in runs
    )


def _widened_fill(match: re.Match[str], shift: float) -> str:
    """Return the page fill polygon with its right-hand corners moved by ``shift``."""

    points = [pair.split(",") for pair in match.group(1).split()]
    right = max(float(x) for x, _y in points)
    moved = " ".join(
        f"{float(x) + shift:g},{y}" if float(x) == right else f"{x},{y}" for x, y in points
    )
    return match.group(0).replace(match.group(1), moved)


def fit_header_width(svg: str) -> str:
    """Return ``svg`` widened so every header line fits inside the page at real advances.

    Parameters
    ----------
    svg:
        SVG text written by Graphviz.

    Returns
    -------
    str
        The SVG, with a wider page only when a header line would pass the right margin.

    Examples
    --------
    >>> from torchlens._visualization._svg_header_fit import fit_header_width
    >>> svg = ('<svg width="60pt" height="20pt"\\n viewBox="0.00 0.00 60.00 20.00">\\n'
    ...        '<g id="graph0" class="graph" transform="scale(1 1) rotate(0) translate(4 16)">\\n'
    ...        '<polygon fill="white" stroke="transparent" points="-4,4 -4,-16 56,-16 56,4 -4,4"/>\\n'
    ...        '<text text-anchor="start" x="0" y="-5" font-size="14.00">a long header line</text>\\n'
    ...        '</g>\\n</svg>\\n')
    >>> 'width="60pt"' in fit_header_width(svg)
    False
    """

    graph_open = _GRAPH_OPEN_RE.search(svg)
    root = _SVG_ROOT_RE.search(svg)
    if graph_open is None or root is None:
        return svg
    first = _FIRST_ELEMENT_RE.search(svg, graph_open.end())
    block = svg[graph_open.end() : first.start() if first else len(svg)]
    rights = [_line_right(m.group("attrs"), m.group("body")) for m in _TEXT_RE.finditer(block)]
    known = [right for right in rights if right is not None]
    fill = _PAGE_FILL_RE.match(svg, graph_open.end())
    if not known or fill is None:
        return svg
    try:
        min_x, min_y, vb_width, vb_height = (float(part) for part in root.group("vb").split())
    except ValueError:
        return svg
    # Graph units start one page margin in from the page's left edge.
    if max(known) + 2 * _PAGE_MARGIN <= vb_width:
        return svg
    new_width = float(math.ceil(max(known) + 2 * _PAGE_MARGIN))
    shift = new_width - vb_width
    new_root = (
        f'<svg width="{new_width:.0f}pt"{root.group("mid")}'
        f'viewBox="{min_x:.2f} {min_y:.2f} {new_width:.2f} {vb_height:.2f}"'
    )
    return (
        svg[: root.start()]
        + new_root
        + svg[root.end() : fill.start()]
        + _widened_fill(fill, shift)
        + svg[fill.end() :]
    )
