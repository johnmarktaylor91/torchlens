"""Forward-graph keyword translation for the Trace drawing door."""

from collections.abc import Callable
from typing import Any


def _forward_render_arguments(
    options: dict[str, Any], resolve_color_source: Callable[..., Any]
) -> dict[str, Any]:
    """Translate public drawing options to forward-renderer keyword names.

    Parameters
    ----------
    options : dict
        Drawing arguments and resolved graph-local values.
    resolve_color_source : callable
        Color-source resolver from the drawing implementation.
    """
    return {
        "vis_mode": options["vis_mode"],
        "vis_call_depth": options["depth"],
        "vis_outpath": options["outpath"],
        "announce_path": options["path"] is None,
        "vis_graph_overrides": options["graph_overrides"],
        "module": options["module"],
        "node_mode": options["node_style"],
        "node_spec_fn": options["node_spec_fn"],
        "collapsed_node_spec_fn": options["collapsed_node_spec_fn"],
        "collapse_fn": options["collapse_fn"],
        "collapse": options["collapse"],
        "fold_repeats": options["fold_repeats"],
        "fold_patterns": options["fold_patterns"],
        "skip_fn": options["skip_fn"],
        "vis_edge_overrides": options["edge_overrides"],
        "vis_grad_edge_overrides": options["grad_edge_overrides"],
        "vis_module_overrides": options["module_overrides"],
        "vis_save_only": not options["open_viewer"],
        "vis_fileformat": options["file_format"],
        "show_buffer_layers": options["show_buffers"],
        "direction": "bottomup" if options["direction"] is None else options["direction"],
        "vis_node_placement": options["layout"],
        "vis_renderer": options["renderer"],
        "vis_theme": options["theme"],
        "vis_intervention_mode": options["intervention_mode"],
        "vis_show_cone": options["show_cone"],
        "code_panel": options["code_panel"],
        "node_overlay": options["node_overlay"],
        "node_label_fields": options["node_label_fields"],
        "show_legend": options["show_legend"],
        "font_size": options["font_size"],
        "dpi": options["dpi"],
        "order_siblings": options["order_siblings"],
        "show_containers": options["show_containers"],
        "container_max_inline": options["container_max_inline"],
        "show_input_transform_summary": options["show_input_transform_summary"],
        "show_orphans": options["show_orphans"],
        "color_by": resolve_color_source(options["trace"], options["color_by"]),
        "size_by": options["size_by"],
        "scale": options["size_scale"],
        "stack_by": options["stack_by"],
        "show_redundant_args": options["show_redundant_args"],
        "show_saved_for_backward": options["show_saved_for_backward"],
    }
