"""Computational graph visualization via Graphviz and Dagua."""

from .dagua_bridge import build_render_audit, model_log_to_dagua_graph, render_model_log_with_dagua
from .node_spec import NodeSpec, render_lines_to_html

__all__ = [
    "NodeSpec",
    "build_render_audit",
    "model_log_to_dagua_graph",
    "render_lines_to_html",
    "render_model_log_with_dagua",
]
