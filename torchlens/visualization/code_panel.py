"""Source-code panel helpers for Graphviz visualizations."""

from __future__ import annotations

import html
import inspect
import math
import re
import textwrap
import weakref
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, Literal, TypeAlias, cast

import graphviz
from torch import nn

from .._source_links import file_line_text

CodePanelMode: TypeAlias = Literal["forward", "class", "init+forward"]
CodePanelOption: TypeAlias = bool | CodePanelMode | Callable[[nn.Module], str]
CodePanelSide: TypeAlias = Literal["right", "left"]

MAX_CODE_PANEL_LINES = 120
MIN_CODE_PANEL_DISPLAY_LINES = 36
# Display lines longer than this wrap (hanging indent) instead of widening the
# panel further. Chosen from the data: the widest demo source line is 79 chars,
# so 120 leaves generous headroom while bounding the panel at a readable width.
MAX_CODE_PANEL_LINE_CHARS = 120
# Graphviz's default label font size; the panel label never overrides it.
CODE_PANEL_FONT_SIZE_PT = 14.0
# Per-character advance used to size the panel box. Graphviz under-measures
# Courier (~0.48 em/char on systems without real Courier metrics) while SVG
# rasterizers resolve "Courier" to true monospace fonts at 0.600-0.602 em/char
# (Courier/Nimbus Mono/Liberation Mono/Noto Sans Mono/DejaVu Sans Mono), so the
# box must be sized from monospace math, with ~3% safety margin, not trusted to
# Graphviz's estimate.
CODE_PANEL_CHAR_ADVANCE_EM = 0.62
_CODE_PANEL_WRAP_INDENT = "    "
# Minimum content characters a wrapped continuation line must be able to hold;
# pathologically deep indents fall back to a shallow hanging indent instead.
_CODE_PANEL_MIN_WRAP_CONTENT_CHARS = 16


class SourceText(str):
    """Source text with optional file-line metadata for clickable renderers."""

    file_path: str | None
    line_number: int | None

    def __new__(
        cls,
        value: str,
        *,
        file_path: str | None = None,
        line_number: int | None = None,
    ) -> "SourceText":
        """Create a source string carrying optional file-line metadata.

        Parameters
        ----------
        value:
            Source text.
        file_path:
            Source file path.
        line_number:
            First source line number.

        Returns
        -------
        SourceText
            String subclass with source-location attributes.
        """

        obj = str.__new__(cls, value)
        obj.file_path = file_path
        obj.line_number = line_number
        return obj


def capture_model_source_code(model: nn.Module) -> dict[str, str]:
    """Capture source text for built-in code-panel modes.

    Parameters
    ----------
    model:
        Model whose class and forward source should be captured.

    Returns
    -------
    dict[str, str]
        Source snippets keyed by ``"forward"``, ``"class"``, and
        ``"init+forward"`` when introspection succeeds.
    """

    source_blob: dict[str, str] = {}
    forward_source = _get_source_or_empty(model.forward)
    class_source = _get_source_or_empty(model.__class__)
    init_source = _get_source_or_empty(model.__class__.__init__)
    if forward_source:
        source_blob["forward"] = forward_source
    if class_source:
        source_blob["class"] = class_source
    if init_source or forward_source:
        source_blob["init+forward"] = SourceText(
            "\n\n".join(source for source in (init_source, forward_source) if source),
            file_path=getattr(init_source or forward_source, "file_path", None),
            line_number=getattr(init_source or forward_source, "line_number", None),
        )
    return source_blob


def make_weak_model_ref(model: nn.Module) -> weakref.ReferenceType[nn.Module] | None:
    """Return a weak reference to a model when the object supports weakrefs.

    Parameters
    ----------
    model:
        Model to reference without extending its lifetime.

    Returns
    -------
    weakref.ReferenceType[nn.Module] | None
        Weak reference, or ``None`` if the model type cannot be weakly referenced.
    """

    try:
        return weakref.ref(model)
    except TypeError:
        return None


def resolve_code_panel_source(
    code_panel: CodePanelOption,
    source_code_blob: dict[str, str],
    model_ref: weakref.ReferenceType[nn.Module] | None,
) -> str | None:
    """Resolve a code-panel option to displayable source text.

    Parameters
    ----------
    code_panel:
        User-facing code-panel option.
    source_code_blob:
        Captured source snippets stored on the Trace.
    model_ref:
        Optional weak reference to the live model for callable options.

    Returns
    -------
    str | None
        Source text to render, or ``None`` when the panel is disabled.

    Raises
    ------
    ValueError
        If a requested built-in mode was not captured or the option is invalid.
    RuntimeError
        If a callable option is used after the live model is unavailable.
    TypeError
        If a callable option does not return a string.
    """

    if code_panel is False:
        return None
    if callable(code_panel):
        model = model_ref() if model_ref is not None else None
        if model is None:
            raise RuntimeError(
                "Callable code_panel options require the original model object to "
                "still be alive. Use a built-in code_panel mode for saved Trace "
                "rendering."
            )
        source_text = code_panel(model)
        if not isinstance(source_text, str):
            raise TypeError("Callable code_panel options must return a string.")
        return source_text
    mode: CodePanelMode
    if code_panel is True:
        mode = "forward"
    elif code_panel in {"forward", "class", "init+forward"}:
        mode = code_panel
    else:
        raise ValueError(
            "code_panel must be False, True, 'forward', 'class', 'init+forward', or a callable."
        )
    captured_source = source_code_blob.get(mode)
    if captured_source is None:
        raise ValueError(f"Source code for code_panel={mode!r} was not captured.")
    return captured_source


def render_code_panel_subgraph(
    dot: graphviz.Digraph,
    source_text: str,
    *,
    side: CodePanelSide = "right",
) -> None:
    """Add a source-code panel cluster to a Graphviz digraph.

    Parameters
    ----------
    dot:
        Graphviz digraph to mutate.
    source_text:
        Source code to render in the panel.
    side:
        Requested side relative to the computational graph.

    Returns
    -------
    None
        The input ``dot`` is mutated in place.
    """

    if side not in {"right", "left"}:
        raise ValueError("side must be either 'right' or 'left'.")

    label = _code_panel_label(source_text)
    # Pure Graphviz keeps graph and code in one output file. The invisible edge
    # gives the otherwise detached code cluster a directional relationship to the
    # main graph without adding a rendering dependency or second composition pass.
    dot.node("__tl_graph_panel_anchor", label="", shape="point", style="invis", width="0")
    with dot.subgraph(name="cluster_torchlens_code_panel") as panel:
        panel.attr(
            label="",
            style="filled,rounded",
            fillcolor="#FAFAFA",
            color="#A8A8A8",
            margin="12",
        )
        panel.node(
            "__tl_code_panel_node",
            label=label,
            shape="plaintext",
            fontname="Courier",
            margin="0",
        )
    if side == "right":
        dot.edge("__tl_graph_panel_anchor", "__tl_code_panel_node", style="invis", weight="10")
    else:
        dot.edge("__tl_code_panel_node", "__tl_graph_panel_anchor", style="invis", weight="10")


def _code_panel_label(source_text: str) -> str:
    """Build the HTML-like Graphviz label for a code panel.

    The header cell carries an explicit minimum ``WIDTH`` computed from true
    monospace metrics so the single table column -- and therefore the panel
    box -- is always wide enough for the longest displayed line, regardless of
    how Graphviz estimates Courier text widths.

    Parameters
    ----------
    source_text:
        Source code to display.

    Returns
    -------
    str
        HTML-like label string for the panel node.
    """

    displayed_lines = _displayed_source_lines(source_text)
    rows = _source_text_to_html_rows(source_text, displayed_lines)
    min_width = _code_panel_min_width_points(displayed_lines)
    header_row = f"<TR><TD ALIGN='LEFT' WIDTH='{min_width}'><B>Source code</B></TD></TR>"
    return (
        "<<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='0' CELLPADDING='2'>"
        f"{header_row}{''.join(rows)}"
        "</TABLE>>"
    )


def render_code_panel_svg(source_text: str) -> str:
    """Render a standalone code panel to its own SVG string.

    The panel is rendered as an independent Graphviz document so it never
    participates in the computational graph's layout. Callers compose the two
    SVGs side by side, which keeps the graph's proportions untouched.

    Parameters
    ----------
    source_text:
        Source code to display.

    Returns
    -------
    str
        SVG document for the code panel.
    """

    panel = graphviz.Digraph()
    panel.attr("graph", bgcolor="transparent", margin="0")
    with panel.subgraph(name="cluster_torchlens_code_panel") as cluster:
        cluster.attr(
            label="",
            style="filled,rounded",
            fillcolor="#FAFAFA",
            color="#A8A8A8",
            margin="12",
        )
        cluster.node(
            "__tl_code_panel_node",
            label=_code_panel_label(source_text),
            shape="plaintext",
            fontname="Courier",
            margin="0",
        )
    return panel.pipe(format="svg").decode("utf-8")


def _parse_svg_geometry(svg: str) -> SimpleNamespace:
    """Extract a Graphviz SVG's point dimensions, viewBox, and inner markup.

    Parameters
    ----------
    svg:
        Graphviz-emitted SVG document.

    Returns
    -------
    SimpleNamespace
        ``width``/``height`` (points), ``viewbox`` string, and ``inner`` markup.
    """

    root = re.search(r"<svg\b([^>]*)>", svg, re.S)
    if root is None:
        raise ValueError("Could not locate an <svg> root element.")
    attrs = root.group(1)
    width_match = re.search(r'width="([\d.]+)pt"', attrs)
    height_match = re.search(r'height="([\d.]+)pt"', attrs)
    viewbox_match = re.search(r'viewBox="([^"]+)"', attrs)
    if not (width_match and height_match and viewbox_match):
        raise ValueError("SVG root is missing width/height/viewBox.")
    inner = svg[root.end() : svg.rindex("</svg>")]
    return SimpleNamespace(
        width=float(width_match.group(1)),
        height=float(height_match.group(1)),
        viewbox=viewbox_match.group(1),
        inner=inner,
    )


def compose_svgs_horizontally(left_svg: str, right_svg: str, *, gap: float = 24.0) -> str:
    """Place two SVGs side by side in one wrapper SVG, preserving vectors.

    Each source SVG is nested unchanged via its own ``viewBox`` so neither one's
    layout is reflowed; they are only translated. Both are vertically centered in
    a white canvas.

    Parameters
    ----------
    left_svg:
        SVG rendered on the left.
    right_svg:
        SVG rendered on the right.
    gap:
        Horizontal gap in points between the two panels.

    Returns
    -------
    str
        Combined SVG document.
    """

    left = _parse_svg_geometry(left_svg)
    right = _parse_svg_geometry(right_svg)
    total_width = left.width + gap + right.width
    total_height = max(left.height, right.height)
    left_y = (total_height - left.height) / 2
    right_y = (total_height - right.height) / 2
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{total_width:.2f}pt" height="{total_height:.2f}pt" '
        f'viewBox="0 0 {total_width:.2f} {total_height:.2f}">'
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>'
        f'<svg x="0" y="{left_y:.2f}" width="{left.width:.2f}" '
        f'height="{left.height:.2f}" viewBox="{left.viewbox}" overflow="visible">'
        f"{left.inner}</svg>"
        f'<svg x="{left.width + gap:.2f}" y="{right_y:.2f}" '
        f'width="{right.width:.2f}" height="{right.height:.2f}" '
        f'viewBox="{right.viewbox}" overflow="visible">{right.inner}</svg>'
        "</svg>"
    )


def compose_graph_with_code_panel(
    graph_svg: str, source_text: str, *, side: CodePanelSide = "right"
) -> str:
    """Return ``graph_svg`` composed beside a separately-rendered code panel.

    Parameters
    ----------
    graph_svg:
        SVG of the computational graph (rendered without any code panel).
    source_text:
        Source code to render in the side panel.
    side:
        Which side the code panel goes on relative to the graph.

    Returns
    -------
    str
        Combined SVG document.
    """

    if side not in {"right", "left"}:
        raise ValueError("side must be either 'right' or 'left'.")
    code_svg = render_code_panel_svg(source_text)
    if side == "right":
        return compose_svgs_horizontally(graph_svg, code_svg)
    return compose_svgs_horizontally(code_svg, graph_svg)


def _get_source_or_empty(obj: object) -> str:
    """Return inspect source text for an object or an empty string.

    Parameters
    ----------
    obj:
        Object passed to ``inspect.getsource``.

    Returns
    -------
    str
        Dedented source text, or an empty string if introspection fails.
    """

    try:
        source_lines, line_number = inspect.getsourcelines(cast(Any, obj))
        source = textwrap.dedent("".join(source_lines)).rstrip()
        return SourceText(
            source,
            file_path=inspect.getsourcefile(cast(Any, obj)),
            line_number=line_number,
        )
    except (OSError, TypeError):
        return ""


def _wrap_source_line(line: str, max_chars: int = MAX_CODE_PANEL_LINE_CHARS) -> list[str]:
    """Wrap one display line to a character cap with a hanging indent.

    Wrapping breaks at word boundaries when possible (hard-splitting only
    tokens longer than the cap) and indents continuation lines four spaces
    past the original indent so wrapped code stays visually attached to its
    source line.

    Parameters
    ----------
    line:
        Tab-expanded source line.
    max_chars:
        Maximum characters per displayed line.

    Returns
    -------
    list[str]
        One or more display lines, each at most ``max_chars`` characters.
    """

    if len(line) <= max_chars:
        return [line]
    stripped = line.lstrip(" ")
    indent = line[: len(line) - len(stripped)]
    # Clamp pathologically deep indents so every wrapped line keeps room for
    # real content; otherwise textwrap would be forced into mid-word splits.
    max_indent_chars = max(0, max_chars - _CODE_PANEL_MIN_WRAP_CONTENT_CHARS)
    indent = indent[:max_indent_chars]
    continuation_indent = (indent + _CODE_PANEL_WRAP_INDENT)[:max_indent_chars]
    wrapped = textwrap.wrap(
        stripped,
        width=max_chars,
        initial_indent=indent,
        subsequent_indent=continuation_indent,
        break_long_words=True,
        break_on_hyphens=False,
        replace_whitespace=False,
        drop_whitespace=True,
    )
    return wrapped or [""]


def _displayed_source_lines(source_text: str) -> list[str]:
    """Expand, wrap, and truncate source text into panel display lines.

    Parameters
    ----------
    source_text:
        Source code to display.

    Returns
    -------
    list[str]
        Display lines, each at most ``MAX_CODE_PANEL_LINE_CHARS`` characters,
        capped at ``MAX_CODE_PANEL_LINES`` rows plus a truncation marker.
    """

    source_lines = source_text.expandtabs(4).splitlines() or [""]
    wrapped_lines: list[str] = []
    for source_line in source_lines:
        wrapped_lines.extend(_wrap_source_line(source_line))
    extra_line_count = max(0, len(wrapped_lines) - MAX_CODE_PANEL_LINES)
    displayed_lines = wrapped_lines[:MAX_CODE_PANEL_LINES]
    if extra_line_count:
        displayed_lines.append(f"... {extra_line_count} more lines")
    return displayed_lines


def _code_panel_min_width_points(displayed_lines: list[str]) -> int:
    """Compute the minimum panel column width for true monospace metrics.

    Parameters
    ----------
    displayed_lines:
        Display lines produced by ``_displayed_source_lines``.

    Returns
    -------
    int
        Minimum table-column width in points that fits the longest displayed
        line at ``CODE_PANEL_CHAR_ADVANCE_EM`` per character.
    """

    longest_chars = max((len(line) for line in displayed_lines), default=0)
    longest_chars = max(longest_chars, len("Source code"), len("Open source"))
    return math.ceil(longest_chars * CODE_PANEL_CHAR_ADVANCE_EM * CODE_PANEL_FONT_SIZE_PT)


def _source_text_to_html_rows(source_text: str, displayed_lines: list[str]) -> list[str]:
    """Convert source text into escaped Graphviz HTML table rows.

    Parameters
    ----------
    source_text:
        Source code to display (carries optional file-line metadata).
    displayed_lines:
        Display lines produced by ``_displayed_source_lines``.

    Returns
    -------
    list[str]
        HTML table-row strings suitable for an HTML-like Graphviz label.
    """

    rows = []
    file_path = getattr(source_text, "file_path", None)
    line_number = getattr(source_text, "line_number", None)
    if file_path is not None:
        link_label = html.escape("Open source", quote=False)
        href = html.escape(
            f"vscode://file/{file_path}:{line_number}"
            if line_number is not None
            else f"vscode://file/{file_path}",
            quote=True,
        )
        tooltip = html.escape(file_line_text(str(file_path), line_number), quote=True)
        rows.append(
            f"<TR><TD ALIGN='LEFT' HREF='{href}' TOOLTIP='{tooltip}'>"
            f"<FONT FACE='Courier' COLOR='#0366D6'>{link_label}</FONT></TD></TR>"
        )
    for line in displayed_lines:
        escaped_line = html.escape(line, quote=False) if line else "&#160;"
        rows.append(f"<TR><TD ALIGN='LEFT'><FONT FACE='Courier'>{escaped_line}</FONT></TD></TR>")
    missing_lines = max(0, MIN_CODE_PANEL_DISPLAY_LINES - len(displayed_lines))
    if missing_lines:
        spacer_height = missing_lines * 14
        rows.append(
            f"<TR><TD HEIGHT='{spacer_height}' ALIGN='LEFT'>"
            "<FONT FACE='Courier' COLOR='#FAFAFA'>&#160;</FONT></TD></TR>"
        )
    return rows
