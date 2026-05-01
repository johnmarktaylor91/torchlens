"""Source-code panel helpers for Graphviz visualizations."""

from __future__ import annotations

import html
import inspect
import textwrap
import weakref
from collections.abc import Callable
from typing import Any, Literal, TypeAlias, cast

import graphviz
from torch import nn

from .._source_links import file_line_text

CodePanelMode: TypeAlias = Literal["forward", "class", "init+forward"]
CodePanelOption: TypeAlias = bool | CodePanelMode | Callable[[nn.Module], str]
CodePanelSide: TypeAlias = Literal["right", "left"]

MAX_CODE_PANEL_LINES = 120
MIN_CODE_PANEL_DISPLAY_LINES = 36


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
        Captured source snippets stored on the ModelLog.
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
                "still be alive. Use a built-in code_panel mode for saved ModelLog "
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

    rows = _source_text_to_html_rows(source_text)
    header_row = "<TR><TD ALIGN='LEFT'><B>Source code</B></TD></TR>"
    label = (
        "<<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='0' CELLPADDING='2'>"
        f"{header_row}{''.join(rows)}"
        "</TABLE>>"
    )
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


def _source_text_to_html_rows(source_text: str) -> list[str]:
    """Convert source text into escaped Graphviz HTML table rows.

    Parameters
    ----------
    source_text:
        Source code to display.

    Returns
    -------
    list[str]
        HTML table-row strings suitable for an HTML-like Graphviz label.
    """

    source_lines = source_text.expandtabs(4).splitlines() or [""]
    extra_line_count = max(0, len(source_lines) - MAX_CODE_PANEL_LINES)
    displayed_lines = source_lines[:MAX_CODE_PANEL_LINES]
    if extra_line_count:
        displayed_lines.append(f"... {extra_line_count} more lines")
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
