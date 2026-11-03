"""SVG post-processing and code-panel composition for rendered graphs.

Extracted verbatim from ``_render_dot.py`` (renderer-thinning pass): these
helpers run the bounded Graphviz subprocess for SVG-only renders, inline
local image references as data URIs, normalize negative-origin viewBoxes,
and compose the optional code panel beside the rendered graph.
"""

from __future__ import annotations

import base64
import html
import os
import re
from collections.abc import Mapping
from pathlib import Path

from .._utils.display import atomic_write_text
from . import _render_utils
from ._edit_highlighter import HighlighterStyle, highlight_svg
from ._render_common import (
    _SVG_ATTR_RE,
    _SVG_IMAGE_TAG_RE,
    _SVG_ROOT_RE,
    _SVG_VIEWBOX_RE,
)
from ._render_flow import _resolve_svg_image_path
from .code_panel import compose_graph_with_code_panel
from .render_execution import surface_layout_stderr

#: Upper bound for one detached Graphviz render, matching a draw's forward render.
_DETACHED_RENDER_TIMEOUT_S = 120


def _render_graph_only_svg(
    engine: str, source_path: str, timeout: int, image_root: Path | None = None
) -> str:
    """Render a saved DOT source to an SVG string (no code panel).

    T9 (grind-p3): ``image_root`` doubles as the subprocess working
    directory so relative node image refs resolve without an in-source
    ``imagepath`` (the saved DOT must not carry the per-run temp path).
    """

    completed = _render_utils.run_bounded_subprocess(
        [engine, "-Tsvg", os.path.abspath(source_path)],
        timeout=timeout,
        cwd=str(image_root) if image_root else None,
    )
    return _inline_svg_local_images(completed.stdout.decode("utf-8"), image_root)


def render_detached_dot(dot: str, fileformat: str, image_root: str | None = None) -> bytes:
    """Render a drawing's DOT source to bytes the way ``draw`` renders it.

    A drawing's producer injects this (bound to the trace's image root) as the
    renderer its notebook display and ``save`` to another format use: the
    same bounded Graphviz child as ``draw``, with the bundled font, so the
    key's table, sized to its words in that face, fits them; relative node
    images resolved against ``image_root`` while it exists; Graphviz stderr
    surfaced as the usual layout warning; and an SVG's local image references
    embedded as data URIs, so the picture never links a file under the temp
    folder. Failures propagate; ``Drawing`` maps them to its typed refusal.
    """

    root = Path(image_root) if image_root and Path(image_root).is_dir() else None
    completed = _render_utils.run_bounded_subprocess(
        ["dot", f"-T{fileformat}"],
        timeout=_DETACHED_RENDER_TIMEOUT_S,
        check=True,
        capture_output=True,
        input=dot.encode("utf-8"),
        cwd=str(root) if root else None,
    )
    surface_layout_stderr(completed.stderr, engine="dot")
    payload = bytes(completed.stdout)
    if fileformat != "svg":
        return payload
    return _inline_svg_local_images(payload.decode("utf-8"), root).encode("utf-8")


def _inline_svg_file_local_images(svg_path: str, image_root: Path | None = None) -> None:
    """Inline local image hrefs in a saved SVG file.

    Parameters
    ----------
    svg_path:
        Path to the rendered SVG file to update in place.
    image_root:
        Directory relative image hrefs are resolved against (the DOT
        ``imagepath`` root).
    """

    with open(svg_path, encoding="utf-8") as svg_file:
        svg_text = svg_file.read()
    inlined_svg = _inline_svg_local_images(svg_text, image_root)
    if inlined_svg != svg_text:
        atomic_write_text(svg_path, inlined_svg)


def _normalize_svg_root_viewbox(svg_text: str) -> str:
    """Shift a root SVG with a negative origin onto a positive page.

    Parameters
    ----------
    svg_text:
        SVG text to normalize.

    Returns
    -------
    str
        SVG text whose root viewBox starts at ``0 0``. The drawn region is
        translated by the opposite offset so PDF/PNG rasterizers do not place
        large Graphviz layouts outside the page.
    """

    root_match = _SVG_ROOT_RE.search(svg_text)
    if root_match is None:
        return svg_text
    attrs = root_match.group("attrs")
    viewbox_match = _SVG_VIEWBOX_RE.search(attrs)
    if viewbox_match is None:
        return svg_text
    try:
        min_x, min_y, width, height = (
            float(part) for part in viewbox_match.group("value").replace(",", " ").split()
        )
    except ValueError:
        return svg_text
    if min_x == 0.0 and min_y == 0.0:
        return svg_text

    normalized_viewbox = f'viewBox="0 0 {width:.2f} {height:.2f}"'
    new_attrs = attrs[: viewbox_match.start()] + normalized_viewbox + attrs[viewbox_match.end() :]
    inner_start = root_match.end()
    inner_end = svg_text.rfind("</svg>")
    if inner_end == -1:
        return svg_text
    translate_x = -min_x
    translate_y = -min_y
    return (
        svg_text[: root_match.start()]
        + f"<svg{new_attrs}>"
        + f'<g transform="translate({translate_x:.2f} {translate_y:.2f})">'
        + svg_text[inner_start:inner_end]
        + "</g>"
        + svg_text[inner_end:]
    )


def _inline_svg_local_images(svg_text: str, image_root: Path | None = None) -> str:
    """Replace local SVG image references with embedded data URIs.

    Parameters
    ----------
    svg_text:
        SVG text produced by Graphviz.
    image_root:
        Directory relative hrefs are resolved against — the graph-level
        ``imagepath`` root the DOT source carried (r-b6 R19-6 relativizes
        node image attrs, and Graphviz copies them into the SVG verbatim).

    Returns
    -------
    str
        SVG text with existing local image hrefs inlined. Non-file hrefs are
        left untouched. Missing or unreadable local files are replaced with a
        short placeholder so the node is not an empty image box.
    """

    def replace_image_tag(match: re.Match[str]) -> str:
        """Return an updated SVG image tag or placeholder text."""

        tag = match.group(0)
        attrs = _svg_attrs_to_dict(match.group("attrs"))
        href_attr = "xlink:href" if "xlink:href" in attrs else "href"
        href = attrs.get(href_attr)
        if href is None or _is_non_file_svg_href(href):
            return tag
        image_path = _resolve_svg_image_path(href, image_root)
        try:
            payload = image_path.read_bytes()
        except OSError:
            return _svg_image_placeholder(attrs)
        mime_type = _svg_image_mime_type(image_path)
        data_uri = f"data:{mime_type};base64,{base64.b64encode(payload).decode('ascii')}"
        return _replace_svg_attr_value(tag, href_attr, data_uri)

    return _SVG_IMAGE_TAG_RE.sub(replace_image_tag, svg_text)


def _write_composed_code_panel(
    engine: str,
    source_path: str,
    source_text: str,
    rendered_path: str,
    file_format: str,
    timeout: int,
    image_root: Path | None = None,
    *,
    highlighter: HighlighterStyle | None = None,
) -> None:
    """Render the graph and code panel separately and write the joined output.

    The graph is rendered to SVG without any code subgraph, composed beside a
    standalone code-panel SVG, then written in ``file_format``. SVG is written
    directly; PDF and PNG are converted from the composed SVG with ``cairosvg``
    so vectors are preserved. With ``highlighter`` the downstream highlighter
    is laid into the graph SVG before composition.
    """

    graph_svg = _render_graph_only_svg(engine, source_path, timeout, image_root)
    if highlighter is not None:
        graph_svg = highlight_svg(graph_svg, highlighter)
    graph_svg = _normalize_svg_root_viewbox(graph_svg)
    combined_svg = _normalize_svg_root_viewbox(
        compose_graph_with_code_panel(graph_svg, source_text)
    )
    if file_format == "svg":
        atomic_write_text(rendered_path, combined_svg)
        return
    import cairosvg

    svg_bytes = combined_svg.encode("utf-8")
    if file_format == "pdf":
        cairosvg.svg2pdf(bytestring=svg_bytes, write_to=rendered_path)
    else:  # png, rendered at 2x for crispness
        cairosvg.svg2png(bytestring=svg_bytes, write_to=rendered_path, scale=2.0)


def _svg_attrs_to_dict(attrs_text: str) -> dict[str, str]:
    """Parse simple SVG tag attributes into a dictionary.

    Parameters
    ----------
    attrs_text:
        Raw attribute text from an SVG tag.

    Returns
    -------
    dict[str, str]
        Attribute values keyed by attribute name.
    """

    return {
        match.group("name"): html.unescape(match.group("value"))
        for match in _SVG_ATTR_RE.finditer(attrs_text)
    }


def _is_non_file_svg_href(href: str) -> bool:
    """Return whether an SVG href should not be treated as a local file.

    Parameters
    ----------
    href:
        SVG image href value.

    Returns
    -------
    bool
        True for data URIs, fragments, and network URLs.
    """

    lowered = href.lower()
    return (
        lowered.startswith("data:")
        or lowered.startswith("#")
        or lowered.startswith("http://")
        or lowered.startswith("https://")
        or lowered.startswith("file:")
    )


def _replace_svg_attr_value(tag: str, attr_name: str, value: str) -> str:
    """Replace an attribute value in an SVG tag.

    Parameters
    ----------
    tag:
        SVG tag text.
    attr_name:
        Attribute name to replace.
    value:
        Replacement attribute value.

    Returns
    -------
    str
        Updated SVG tag.
    """

    escaped_value = html.escape(value, quote=True)
    pattern = re.compile(
        rf"""(?P<prefix>\b{re.escape(attr_name)}\s*=\s*)(?P<quote>["']).*?(?P=quote)"""
    )
    return pattern.sub(rf"\g<prefix>\g<quote>{escaped_value}\g<quote>", tag, count=1)


def _svg_image_mime_type(path: Path) -> str:
    """Return the data-URI MIME type for an SVG image file.

    Parameters
    ----------
    path:
        Local image path.

    Returns
    -------
    str
        MIME type for the data URI.
    """

    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".gif":
        return "image/gif"
    if suffix == ".svg":
        return "image/svg+xml"
    return "image/png"


def _svg_image_placeholder(attrs: Mapping[str, str]) -> str:
    """Return SVG text for an unreadable image placeholder.

    Parameters
    ----------
    attrs:
        Parsed attributes from the original SVG image tag.

    Returns
    -------
    str
        Replacement SVG text element.
    """

    x = _svg_numeric_attr(attrs, "x")
    y = _svg_numeric_attr(attrs, "y")
    width = _svg_numeric_attr(attrs, "width")
    height = _svg_numeric_attr(attrs, "height")
    text_x = x + (width / 2.0)
    text_y = y + (height / 2.0)
    return (
        f'<text text-anchor="middle" x="{text_x:.2f}" y="{text_y:.2f}" '
        'font-family="Times,serif" font-size="10.00">preview unavailable</text>'
    )


def _svg_numeric_attr(attrs: Mapping[str, str], name: str) -> float:
    """Read a numeric SVG attribute, defaulting to zero.

    Parameters
    ----------
    attrs:
        Parsed SVG attributes.
    name:
        Attribute name to parse.

    Returns
    -------
    float
        Parsed number, or zero when absent/unparseable.
    """

    value = attrs.get(name, "0")
    match = re.match(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value)
    if match is None:
        return 0.0
    return float(match.group(0))


__all__ = [
    "_inline_svg_file_local_images",
    "_inline_svg_local_images",
    "_normalize_svg_root_viewbox",
    "_render_graph_only_svg",
    "_write_composed_code_panel",
    "render_detached_dot",
]
