"""Internal Graphviz rendering helpers shared across rendering paths.

Private module: not part of the public API. Provides rendering primitives
shared by single-trace graph rendering and any internal bundle renderers, so
we have one canonical implementation of file-format dispatch, direction
translation, module cluster styling, and HTML label escaping.

Keep this module narrow on purpose -- only primitives that take no
ModelLog/Bundle context and can be reasoned about as pure utilities.
The orchestration that knows WHICH nodes / edges / module paths to use
lives in the per-input-shape callers, such as ``rendering.render_graph`` for
ModelLog.
"""

from __future__ import annotations

import os
import subprocess
import warnings
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Iterable

import graphviz

if TYPE_CHECKING:  # pragma: no cover - typing-only
    pass


# Recognised file extensions that callers may include on ``vis_outpath``.
# Mirrors the legacy list in ``rendering.render_graph`` (kept as a tuple
# so it stays cheap and immutable).
_KNOWN_EXTS = ("pdf", "png", "jpg", "svg", "jpeg", "bmp", "pic", "tif", "tiff")

# Default subprocess timeout for the dot/sfdp render call. Mirrors the
# legacy literal that lived inside ``rendering.render_graph``.
RENDER_TIMEOUT_SECONDS = 120

# -- Module subgraph border widths (shared between ModelLog and bundle paths)
# Outermost modules get the thickest border; deeper modules thin out by depth
# fraction so visual hierarchy reads at a glance.  These constants are the
# canonical source for both ``rendering.py`` and the bundle renderer.
MAX_MODULE_PENWIDTH = 5
MIN_MODULE_PENWIDTH = 2
PENWIDTH_RANGE = MAX_MODULE_PENWIDTH - MIN_MODULE_PENWIDTH


def strip_known_extension(outpath: str) -> str:
    """Strip a recognised image extension off ``outpath`` if present.

    The Graphviz Python binding wants the basename without an extension --
    it adds the extension itself based on the requested ``format``. Users
    who pass ``"out.pdf"`` should still get ``"out.pdf"`` rather than
    ``"out.pdf.pdf"``, so we trim a trailing recognised extension.
    """

    parts = outpath.split(".")
    if len(parts) > 1 and parts[-1].lower() in _KNOWN_EXTS:
        return ".".join(parts[:-1])
    return outpath


def direction_to_rankdir(direction: str) -> str:
    """Translate a TorchLens vis-direction literal into a Graphviz ``rankdir``.

    Accepted inputs: ``'bottomup'``, ``'topdown'``, ``'leftright'``.  Raises
    ``ValueError`` on anything else so callers don't silently render with the
    Graphviz default when a typo'd direction sneaks through.
    """

    if direction == "bottomup":
        return "BT"
    if direction == "leftright":
        return "LR"
    if direction == "topdown":
        return "TB"
    raise ValueError(
        f"direction must be one of 'bottomup', 'topdown', or 'leftright'; got {direction!r}"
    )


def compute_module_penwidth(nesting_depth: int, max_nesting_depth: int) -> float:
    """Return the cluster border width for a module at ``nesting_depth``.

    ``nesting_depth`` is 0-based (outermost is depth 0).  Outermost modules
    get the maximum penwidth; deepest modules get the minimum.  When the
    overall hierarchy has only one level (``max_nesting_depth == 0`` or
    ``1``) we still return a sensible value so callers don't have to
    special-case shallow models.
    """

    if max_nesting_depth <= 0:
        return float(MIN_MODULE_PENWIDTH + PENWIDTH_RANGE)
    nesting_fraction = (max_nesting_depth - nesting_depth) / max_nesting_depth
    nesting_fraction = max(0.0, min(1.0, nesting_fraction))
    return MIN_MODULE_PENWIDTH + nesting_fraction * PENWIDTH_RANGE


def html_escape(value: str) -> str:
    """Escape the three Graphviz HTML-label specials.

    Graphviz HTML-like labels reserve ``<``, ``>``, and ``&``; embedding any
    of those raw breaks the parser.  Mirrors ``html.escape`` minus the
    ``quote`` argument because Graphviz attribute values are themselves
    already inside double quotes (no need to escape ``"``).
    """

    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def format_node_html(lines: Iterable[str]) -> str:
    """Wrap pre-escaped label lines into a Graphviz HTML-like label string.

    The caller is responsible for escaping content (use :func:`html_escape`)
    -- a deliberate choice so callers can embed ``<B>``/``<I>``/``<FONT>``
    tags where they want emphasis without us double-escaping them.
    """

    return "<" + "<BR/>".join(lines) + ">"


def make_module_cluster_label(
    title: str,
    module_type: str | None = None,
    *,
    title_already_escaped: bool = False,
) -> str:
    """Return the HTML-style label string for a module cluster.

    Mirrors the legacy format used by ``rendering._setup_subgraphs_recurse``:
    ``<<B>@title</B><br align='left'/>(type)<br align='left'/>>``.  The
    ``module_type`` line is omitted when no type information is available
    (which is the case for bundle clusters because the supergraph stores
    the module path string but not the underlying module class).

    ``title_already_escaped`` lets the ModelLog path keep its existing
    raw-title behaviour (where the title may itself contain ``:`` and is
    fed verbatim) while the bundle path can opt-in to escaping arbitrary
    user-provided strings.
    """

    title_str = title if title_already_escaped else html_escape(title)
    if module_type:
        return (
            f"<<B>@{title_str}</B><br align='left'/>({html_escape(module_type)})<br align='left'/>>"
        )
    return f"<<B>@{title_str}</B><br align='left'/>>"


def make_module_cluster_attrs(
    *,
    title: str,
    module_type: str | None,
    line_style: str,
    penwidth: float,
    fillcolor: str = "white",
    title_already_escaped: bool = False,
) -> dict[str, str]:
    """Return the standard cluster attribute dict used by both renderers.

    Centralises the Graphviz attrs that ModelLog and bundle clusters share:
    HTML label, bottom labelloc, ``filled,<line_style>`` style, fill colour,
    and depth-aware penwidth.  Module-type information is optional: bundle
    clusters omit it because the supergraph doesn't preserve the module
    class, while ModelLog clusters always pass it through.
    """

    return {
        "label": make_module_cluster_label(
            title, module_type, title_already_escaped=title_already_escaped
        ),
        "labelloc": "b",
        "style": f"filled,{line_style}",
        "fillcolor": fillcolor,
        "penwidth": str(penwidth),
    }


StyleOverride = Mapping[str, str] | Callable[[Any], Mapping[str, str] | None]


def resolve_style_override(
    override: StyleOverride | None,
    context: Any,
) -> dict[str, str]:
    """Resolve a per-node or per-edge style override.

    Parameters
    ----------
    override:
        Static attribute mapping or callable receiving ``context``.
    context:
        Node or edge context passed to callable overrides.

    Returns
    -------
    dict[str, str]
        Graphviz attributes with string values.
    """

    if override is None:
        return {}
    resolved = override(context) if callable(override) else override
    if resolved is None:
        return {}
    return {str(key): str(value) for key, value in resolved.items()}


def merge_node_style(
    base_style: Mapping[str, str],
    node_overrides: Mapping[str, StyleOverride] | None,
    node_name: str,
    context: Any,
) -> dict[str, str]:
    """Return merged Graphviz node style attributes.

    Parameters
    ----------
    base_style:
        Default node attributes.
    node_overrides:
        Optional mapping from node names to static or callable overrides.
    node_name:
        Node key to look up in ``node_overrides``.
    context:
        Node context passed to callable overrides.

    Returns
    -------
    dict[str, str]
        Merged node attributes.
    """

    merged = {str(key): str(value) for key, value in base_style.items()}
    if node_overrides is not None:
        merged.update(resolve_style_override(node_overrides.get(node_name), context))
    return merged


def merge_edge_style(
    base_style: Mapping[str, str],
    edge_overrides: Mapping[tuple[str, str], StyleOverride] | None,
    edge_key: tuple[str, str],
    context: Any,
) -> dict[str, str]:
    """Return merged Graphviz edge style attributes.

    Parameters
    ----------
    base_style:
        Default edge attributes.
    edge_overrides:
        Optional mapping from ``(source, target)`` keys to overrides.
    edge_key:
        Edge key to look up.
    context:
        Edge context passed to callable overrides.

    Returns
    -------
    dict[str, str]
        Merged edge attributes.
    """

    merged = {str(key): str(value) for key, value in base_style.items()}
    if edge_overrides is not None:
        merged.update(resolve_style_override(edge_overrides.get(edge_key), context))
    return merged


def render_dot_to_file(
    dot: "graphviz.Digraph",
    outpath: str,
    file_format: str,
    save_only: bool,
    *,
    timeout_seconds: int = RENDER_TIMEOUT_SECONDS,
    timeout_warning: str | None = None,
) -> str:
    """Render ``dot`` to ``outpath.<file_format>``, optionally previewing it.

    Mirrors the dot/save/subprocess/view flow used internally by
    ``rendering.render_graph`` and ``rendering.render_backward_graph``,
    factored out so the multi-trace renderer can share the same plumbing.

    Returns the DOT source string (``dot.source``) regardless of whether
    the subprocess render succeeded -- failures are surfaced via
    ``warnings.warn`` to match existing behaviour.
    """

    parent = os.path.dirname(os.path.abspath(outpath))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    source_path = dot.save(outpath)
    try:
        rendered_path = f"{outpath}.{file_format}"
        cmd = [dot.engine, f"-T{file_format}", "-o", rendered_path, source_path]
        subprocess.run(
            cmd,
            timeout=timeout_seconds,
            check=True,
            capture_output=True,
        )
        if not save_only:
            graphviz.backend.viewing.view(rendered_path)
    except subprocess.TimeoutExpired:
        warnings.warn(
            timeout_warning
            or (
                f"Graphviz render timed out ({timeout_seconds}s). "
                f"DOT source saved to '{source_path}'."
            )
        )
    except subprocess.CalledProcessError as exc:
        warnings.warn(f"Graphviz render failed: {exc.stderr.decode()}")
    finally:
        if os.path.exists(source_path):
            os.remove(source_path)
    return dot.source
