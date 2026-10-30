"""SVG post-processing and code-panel composition for rendered graphs.

Extracted verbatim from ``_render_dot.py`` (renderer-thinning pass): these
helpers run the bounded Graphviz subprocess for SVG-only renders, inline
local image references as data URIs, normalize negative-origin viewBoxes,
and compose the optional code panel beside the rendered graph.
"""

from __future__ import annotations

import base64
import os
import re
from pathlib import Path

from ..utils.display import atomic_write_text
from . import _render_utils
from ._edit_highlighter import HighlighterStyle, highlight_svg
from ._render_common import (
    _SVG_IMAGE_TAG_RE,
    _SVG_ROOT_RE,
    _SVG_VIEWBOX_RE,
)
from ._render_flow import (
    _is_non_file_svg_href,
    _replace_svg_attr_value,
    _resolve_svg_image_path,
    _svg_attrs_to_dict,
    _svg_image_mime_type,
    _svg_image_placeholder,
)
from .code_panel import compose_graph_with_code_panel


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


__all__ = [
    "_inline_svg_file_local_images",
    "_inline_svg_local_images",
    "_normalize_svg_root_viewbox",
    "_render_graph_only_svg",
    "_write_composed_code_panel",
]
