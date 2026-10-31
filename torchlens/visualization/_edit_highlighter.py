"""The downstream highlighter: a translucent purple stroke under changed arrows and nodes.

With ``vis_show_cone=True`` every arrow leaving the edited node or a node the
edit changed keeps its normal stroke and gets a wide translucent purple
underlay, and so does each changed node's outline (option F2, JMT 29 Oct
2026). Graphviz cannot stack two strokes on one edge, so this is an SVG step
after layout: the node callback tags edited nodes ``tl_edited`` and changed
ones ``tl_downstream`` (Graphviz writes the ``class`` attribute onto each SVG
group), and :func:`highlight_svg` duplicates the tagged shapes underneath
with the highlighter stroke. PDF and PNG are produced from the processed SVG
with ``cairosvg``; without it (or for another raster format) the picture is
drawn unhighlighted with a warning, and the edited node is still marked.

Spellings here are DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

import html
import os
import re
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .._utils.display import atomic_write_text
from ._edit_marks import DOWNSTREAM_NODE_CLASS, EDITED_NODE_CLASS, edit_ink

if TYPE_CHECKING:
    from .._data_classes.trace import Trace
    from .themes import VisualizationTheme

__all__ = [
    "HighlightJob",
    "HighlighterStyle",
    "downstream_highlighter",
    "highlight_svg",
    "highlighter_rgba",
    "highlighter_supported",
    "render_highlighted",
    "write_highlighted",
]

#: Underlay width in SVG user units (points), and its opacity on light and dark pages.
_UNDERLAY_WIDTH = 8.0
_LIGHT_OPACITY = 0.25
_DARK_OPACITY = 0.35
#: Formats the highlighter can produce (PDF and PNG need ``cairosvg``).
_CAIRO_FORMATS = frozenset({"pdf", "png"})
#: Text formats that carry the DOT source, not a picture: nothing to highlight.
_SOURCE_FORMATS = frozenset({"dot", "gv", "canon", "xdot", "plain", "plain-ext", "json"})

_GROUP_TOKEN_RE = re.compile(r"<g\b[^>]*>|</g>")
_CLASS_RE = re.compile(r'\bclass="([^"]*)"')
_TITLE_RE = re.compile(r"<title>(.*?)</title>", re.S)
_SHAPE_RE = re.compile(r"<(?:path|polygon|ellipse)\b[^>]*/>")
_GEOMETRY_RE = re.compile(r'\b(?:d|points|cx|cy|rx|ry|transform)="[^"]*"')
_DPI_RE = re.compile(r"\bdpi\s*=\s*\"?([0-9.]+)")


@dataclass(frozen=True)
class HighlighterStyle:
    """The highlighter stroke: colour, opacity and width.

    Attributes
    ----------
    color:
        Stroke colour (the skin's edit purple, or ``intervention_cone_color``).
    opacity:
        Stroke opacity (0.25 on light pages, 0.35 on the dark skin).
    width:
        Stroke width in points (``intervention_cone_penwidth``, default 8).
    """

    color: str
    opacity: float
    width: float = _UNDERLAY_WIDTH


def _is_dark(theme: VisualizationTheme | None) -> bool:
    """Return whether the skin draws on a dark page."""

    return theme is not None and edit_ink(theme) != edit_ink(None)


def highlighter_rgba(theme: VisualizationTheme | None) -> str:
    """Return the highlighter colour with its opacity as ``#RRGGBBAA`` (for the key)."""

    opacity = _DARK_OPACITY if _is_dark(theme) else _LIGHT_OPACITY
    return f"{edit_ink(theme)}{round(opacity * 255):02X}"


def downstream_highlighter(
    trace: Trace,
    *,
    show_cone: bool,
    theme: VisualizationTheme | None,
    graph_overrides: Mapping[str, Any] | None,
    fileformat: str,
) -> HighlighterStyle | None:
    """Return the highlighter for a forward picture, or ``None`` when none is drawn.

    It is drawn only with ``show_cone`` on, a non-empty downstream set and an
    output format that can carry it (:func:`highlighter_supported`).
    """

    if not show_cone:
        return None
    from .node_spec import intervention_graph_override, intervention_site_and_cone_labels

    _, cone_labels = intervention_site_and_cone_labels(trace, show_cone=True)
    if not cone_labels:
        return None
    overrides = dict(graph_overrides) if graph_overrides is not None else None
    color = str(intervention_graph_override(overrides, "intervention_cone_color", edit_ink(theme)))
    width = float(
        intervention_graph_override(overrides, "intervention_cone_penwidth", _UNDERLAY_WIDTH)
    )
    if not highlighter_supported(fileformat):
        return None
    opacity = _DARK_OPACITY if _is_dark(theme) else _LIGHT_OPACITY
    return HighlighterStyle(color=color, opacity=opacity, width=width)


def highlighter_supported(fileformat: str) -> bool:
    """Return whether ``fileformat`` can carry the highlighter, warning once when it cannot."""

    if fileformat == "svg" or fileformat in _SOURCE_FORMATS:
        return fileformat == "svg"
    if fileformat in _CAIRO_FORMATS:
        try:
            import cairosvg  # noqa: F401
        except (ImportError, OSError):
            _warn_unsupported(
                f"vis_fileformat={fileformat!r} needs the cairosvg package (pip install cairosvg)"
            )
            return False
        return True
    _warn_unsupported(f"vis_fileformat={fileformat!r} is not svg, pdf or png")
    return False


def _warn_unsupported(reason: str) -> None:
    """Warn that the downstream highlighter was left out, and why."""

    from .._utils.display import user_stacklevel
    from ..errors import TorchLensWarning

    warnings.warn(
        TorchLensWarning(
            f"The downstream highlighter (vis_show_cone=True) was not drawn: {reason}. "
            "The edited node is still marked. "
            "Remedy: use vis_fileformat='svg', or install cairosvg for PDF and PNG",
            code="viz_highlighter_not_drawn",
        ),
        stacklevel=user_stacklevel(),
    )


def _groups(svg: str) -> list[tuple[int, int, str]]:
    """Return ``(start, end, open_tag)`` for every node and edge group, nesting-aware."""

    stack: list[tuple[int, str]] = []
    found: list[tuple[int, int, str]] = []
    for match in _GROUP_TOKEN_RE.finditer(svg):
        token = match.group(0)
        if token != "</g>":
            stack.append((match.start(), token))
            continue
        if not stack:
            continue
        start, open_tag = stack.pop()
        classes = _CLASS_RE.search(open_tag)
        if classes is not None and classes.group(1).split()[:1] in (["node"], ["edge"]):
            found.append((start, match.end(), open_tag))
    return sorted(found)


def _underlay(shape: str, style: HighlighterStyle) -> str:
    """Return a copy of one shape carrying only its geometry and the highlighter stroke."""

    tag = shape[1 : shape.index(" ")] if " " in shape else shape[1:-2]
    geometry = " ".join(_GEOMETRY_RE.findall(shape))
    return (
        f'<{tag} {geometry} fill="none" stroke="{style.color}" '
        f'stroke-opacity="{style.opacity:g}" stroke-width="{style.width:g}" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
    )


def _edge_tail(title: str, node_titles: set[str]) -> str | None:
    """Return the tail node of an edge title (``a->b`` or ``a:port->b``)."""

    tail = title.split("->", 1)[0]
    if tail in node_titles:
        return tail
    head, _, _ = tail.rpartition(":")
    return head if head in node_titles else None


@dataclass(frozen=True)
class _ScannedGroup:
    """One node or edge group: where its title ends, its shapes, and what it is."""

    title_end: int
    shapes: tuple[str, ...]
    title: str
    is_node: bool


def _scan(svg: str) -> tuple[list[_ScannedGroup], set[str], set[str], set[str]]:
    """Return the groups, every node title, the marked node titles and the downstream ones."""

    scanned: list[_ScannedGroup] = []
    node_titles: set[str] = set()
    marked: set[str] = set()
    downstream: set[str] = set()
    for start, end, open_tag in _groups(svg):
        body = svg[start:end]
        title_match = _TITLE_RE.search(body)
        if title_match is None:
            continue
        title = html.unescape(title_match.group(1))
        classes = _CLASS_RE.search(open_tag)
        names = classes.group(1).split() if classes is not None else []
        is_node = names[:1] == ["node"]
        if is_node:
            node_titles.add(title)
            if EDITED_NODE_CLASS in names or DOWNSTREAM_NODE_CLASS in names:
                marked.add(title)
            if DOWNSTREAM_NODE_CLASS in names:
                downstream.add(title)
        shapes = tuple(_SHAPE_RE.findall(body[title_match.end() :]))
        scanned.append(_ScannedGroup(start + title_match.end(), shapes, title, is_node))
    return scanned, node_titles, marked, downstream


def highlight_svg(svg: str, style: HighlighterStyle) -> str:
    """Lay the highlighter under changed nodes and every arrow leaving a marked node.

    A ``tl_downstream`` node gets an underlay copy of its outline (the first
    path, polygon or ellipse in its group); an edge whose tail node is
    ``tl_edited`` or ``tl_downstream`` gets one for its shaft and arrowhead.
    The copies are inserted right after the group's ``<title>``, so they are
    painted under the originals. Everything else is left byte for byte.
    """

    scanned, node_titles, marked, downstream = _scan(svg)
    insertions: list[tuple[int, str]] = []
    for group in scanned:
        if not group.shapes:
            continue
        if group.is_node and group.title in downstream:
            insertions.append((group.title_end, _underlay(group.shapes[0], style)))
        elif not group.is_node and _edge_tail(group.title, node_titles) in marked:
            underlays = "".join(_underlay(shape, style) for shape in group.shapes)
            insertions.append((group.title_end, underlays))
    pieces = []
    cursor = 0
    for position, text in sorted(insertions):
        pieces.append(svg[cursor:position])
        pieces.append(text)
        cursor = position
    pieces.append(svg[cursor:])
    return "".join(pieces)


def _source_dpi(dot_source: str) -> float | None:
    """Return the graph ``dpi`` the DOT source sets, if any."""

    match = _DPI_RE.search(dot_source)
    return float(match.group(1)) if match is not None else None


def write_highlighted(
    svg: str,
    output_path: str,
    fileformat: str,
    style: HighlighterStyle,
    *,
    dot_source: str = "",
) -> None:
    """Highlight ``svg`` and write it to ``output_path`` as SVG, PDF or PNG.

    PNG keeps the size Graphviz would give it: the graph ``dpi`` from the DOT
    source, else Graphviz's 96.
    """

    processed = highlight_svg(svg, style)
    if fileformat == "svg":
        atomic_write_text(output_path, processed)
        return
    import cairosvg

    data = processed.encode("utf-8")
    if fileformat == "pdf":
        cairosvg.svg2pdf(bytestring=data, write_to=output_path)
    else:
        cairosvg.svg2png(bytestring=data, write_to=output_path, dpi=_source_dpi(dot_source) or 96)


@dataclass(frozen=True)
class HighlightJob:
    """What one highlighted render needs besides the layout call.

    Attributes
    ----------
    fileformat:
        Output format (``svg``, ``pdf`` or ``png``).
    style:
        Highlighter stroke.
    dot_source:
        The DOT text being laid out (its ``dpi`` sizes a PNG).
    image_root:
        Directory relative image references resolve against, for inlining.
    """

    fileformat: str
    style: HighlighterStyle
    dot_source: str = ""
    image_root: Path | None = None


def render_highlighted(run_svg: Callable[[str], Any], output_path: str, job: HighlightJob) -> Any:
    """Lay out into a sibling SVG, add the highlighter and write ``output_path``.

    ``run_svg(path)`` runs the layout engine with ``-Tsvg`` into ``path`` and
    returns its process result, which is returned unchanged (its stderr keeps
    the layout warnings). Local images are inlined before conversion, so PDF
    and PNG keep raw-input pictures and key swatches.
    """

    from ._svg_compose import _inline_svg_local_images

    svg_path = f"{output_path}.highlight.svg"
    try:
        result = run_svg(svg_path)
        svg = Path(svg_path).read_text(encoding="utf-8")
    finally:
        if os.path.exists(svg_path):
            os.remove(svg_path)
    write_highlighted(
        _inline_svg_local_images(svg, job.image_root),
        output_path,
        job.fileformat,
        job.style,
        dot_source=job.dot_source,
    )
    return result
