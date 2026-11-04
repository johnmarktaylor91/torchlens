"""The Graphviz graph shell of a forward draw: caption, graph, node and edge defaults."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import graphviz

from ._focus_scope import focus_address
from ._render_utils import direction_to_rankdir
from .request import RenderTarget, ResolvedRenderRequest
from .themes import VisualizationTheme, theme_edge_attrs, theme_graph_attrs, theme_node_attrs

if TYPE_CHECKING:
    from .._data_classes.trace import Trace
    from ._render_common import VisualizationOverrides

__all__ = ["FOCUS_RANKSEP", "build_graphviz_shell"]

#: Rank separation (inches) of a ``module=`` focused draw. Graphviz's default
#: is 0.5; a focused picture is one module's short chain of ops, and the
#: tighter ranks keep its input and output on one screen.
FOCUS_RANKSEP = "0.3"


def build_graphviz_shell(
    trace: Trace,
    request: ResolvedRenderRequest,
    target: RenderTarget,
    theme: VisualizationTheme,
    graph_caption: str,
) -> graphviz.Digraph:
    """Return the configured, empty Graphviz graph carrying ``graph_caption``."""

    dot = graphviz.Digraph(
        name=trace._source_class_name,
        comment="Computational graph for the feedforward sweep",
        format=target.fileformat,
    )
    dot.graph_attr.update(_graph_attrs(trace, request, target, theme, graph_caption))
    dot.node_attr.update(
        {"ordering": "out", **theme_node_attrs(theme, font_size=request.font_size)}
    )
    dot.edge_attr.update(theme_edge_attrs(theme, font_size=request.font_size))
    return dot


def _graph_attrs(
    trace: Trace,
    request: ResolvedRenderRequest,
    target: RenderTarget,
    theme: VisualizationTheme,
    graph_caption: str,
) -> dict[str, str]:
    """Return the graph-level attributes: direction, caption, rank spacing, theme, overrides."""

    graph_args = {
        "rankdir": direction_to_rankdir(request.direction),
        "label": graph_caption,
        "labelloc": "t",
        "labeljust": "left",
        "ordering": "out",
    }
    if request.collapse_fn is not None:
        graph_args["newrank"] = "true"
    if focus_address(request.module) is not None:
        # One module's ops read as a short chain: closer ranks keep the input
        # and the output of a focused draw on one screen.
        graph_args["ranksep"] = FOCUS_RANKSEP
    # r-b6 R19-6 / T9: node images are RELATIVE to the trace scratch root, which
    # Graphviz gets out of band (subprocess cwd), never as an in-source
    # ``imagepath``: a baked temp path broke saved DOT once the dir was GC'd.
    graph_args.update(
        theme_graph_attrs(
            theme,
            font_size=request.font_size,
            dpi=request.dpi,
            fileformat=target.fileformat,
        )
    )
    overrides = cast("VisualizationOverrides", request.overrides)
    for arg_name, arg_val in overrides.graph.items():  # type: ignore[union-attr]
        if callable(arg_val):
            graph_args[arg_name] = str(arg_val(trace))
        else:
            graph_args[arg_name] = str(arg_val)
    return graph_args
