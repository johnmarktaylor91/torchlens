"""Experimental Dagua rendering bridge exports.

Import this module before selecting ``vis_renderer="dagua"`` from the core
visualization dispatcher.
"""

from ._bridge import (
    build_render_audit,
    model_log_to_dagua_graph,
    render_model_log_with_dagua,
)
from ...visualization.node_spec import NodeSpec, render_lines_to_html

__torchlens_dagua_opted_in__ = True

__all__ = [
    "NodeSpec",
    "build_render_audit",
    "model_log_to_dagua_graph",
    "render_lines_to_html",
    "render_model_log_with_dagua",
]
