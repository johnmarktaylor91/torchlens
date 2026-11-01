"""Translate grouped ``VisualizationOptions`` into ``Trace.draw`` keyword arguments.

Lives beside the drawing surface it feeds; ``torchlens.options`` re-exports it at
its public path (``tl.options.visualization_to_render_kwargs``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..options import VisualizationOptions


def visualization_to_render_kwargs(visualization: VisualizationOptions) -> dict[str, Any]:
    """Translate grouped visualization options into ``Trace.draw`` kwargs.

    Parameters
    ----------
    visualization:
        Resolved grouped visualization options.

    Returns
    -------
    dict[str, Any]
        Keyword arguments expected by ``Trace.draw``: the field names, with
        ``container_path`` as ``path``; ``fold_repeats`` only when set or
        explicit; the tri-state and optional channels only when set.
    """

    kwargs: dict[str, Any] = {
        "path": visualization.container_path,
        "graph": visualization.graph,
        "rolled": visualization.rolled,
        "depth": visualization.depth,
        "graph_overrides": visualization.graph_overrides,
        "node_style": visualization.node_style,
        "node_spec_fn": visualization.node_spec_fn,
        "collapsed_node_spec_fn": visualization.collapsed_node_spec_fn,
        "collapse_fn": visualization.collapse_fn,
        "collapse": visualization.collapse,
        "skip_fn": visualization.skip_fn,
        "edge_overrides": visualization.edge_overrides,
        "grad_edge_overrides": visualization.grad_edge_overrides,
        "module_overrides": visualization.module_overrides,
        "open_viewer": visualization.open_viewer,
        "file_format": visualization.file_format,
        "show_buffers": visualization.show_buffers,
        "direction": visualization.direction,
        "layout": visualization.layout,
        "renderer": visualization.renderer,
        "theme": visualization.theme,
        "intervention_mode": visualization.intervention_mode,
        "show_cone": visualization.show_cone,
    }
    if visualization.fold_repeats is not None or visualization.is_field_explicit("fold_repeats"):
        kwargs["fold_repeats"] = visualization.fold_repeats
    optional_kwargs = {
        "node_overlay": visualization.node_overlay,
        "node_label_fields": visualization.node_label_fields,
        "show_legend": visualization.show_legend,
        "color_by": visualization.color_by,
        "size_by": visualization.size_by,
        "size_scale": visualization.size_scale,
        "stack_by": visualization.stack_by,
        "animate": visualization.animate,
        "show_redundant_args": visualization.show_redundant_args,
        "font_size": visualization.font_size,
        "dpi": visualization.dpi,
        "order_siblings": visualization.order_siblings,
    }
    for field_name, value in optional_kwargs.items():
        if value is not None and value is not False or visualization.is_field_explicit(field_name):
            kwargs[field_name] = value
    return kwargs
