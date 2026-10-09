"""Committed tables of the visual language coverage check.

``ROWS`` is the closed inventory of what TorchLens draws, keyed by stable row ids
(``VN`` nodes, ``VL`` label rows, ``VE`` edges, ``VR`` regions, ``VC`` caption, ``VV``
views, ``VK`` encoding channels, overlays and lenses, ``VI`` legends, ``VT`` skins and
type, ``VY`` layout, ``VO`` output, ``VS`` side pictures). The other tables classify
every item the renderer can emit to a row id or to ``"exempt: <reason>"``. Regenerate the
suggestions with ``python -m scripts.visual_language.coverage discover`` and classify each
new entry by hand; nothing is accepted automatically.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Row:
    """One inventory row: what it is, where the deck teaches it and how it is proven.

    Attributes
    ----------
    slug:
        Dotted descriptive name, unique across rows.
    slide:
        Primary slide id (``None`` only for inactive rows, which draw nothing).
    witness:
        ``"picture"`` when the slide's render shows the mark, ``"caption"`` when only the
        slide's words state it.
    locator:
        ``module:symbol`` near the code that draws it; short modules are relative to
        ``torchlens.visualization``. The check imports it, so a stale locator fails.
    status:
        ``"active"``, ``"experimental"`` or ``"inactive"``.
    gate:
        For non-active rows, an expression over :func:`coverage.gate_namespace` that is
        true while the status holds.
    variants:
        Finite variants that need a witness (vocabulary values the row teaches).
    caption_variants:
        Variants the deck states only in words (no small model draws them).
    """

    slug: str
    slide: str | None
    witness: str
    locator: str
    status: str = "active"
    gate: str | None = None
    variants: tuple[str, ...] = ()
    caption_variants: tuple[str, ...] = ()


P, C = "picture", "caption"
_R = Row

ROWS: Mapping[str, Row] = {
    # Nodes
    "VN01": _R("node.shape.oval.op", "ovals-boxes", P, "_render_nodes:_build_layer_node"),
    "VN02": _R(
        "node.shape.box.leaf_module", "ovals-boxes", P, "_render_nodes:_atomic_module_split_range"
    ),
    "VN03": _R("node.fill.input_output", "start-here", P, "_render_common:INPUT_COLOR"),
    "VN04": _R("node.fill.boolean", "branches", P, "_render_common:BOOL_NODE_COLOR"),
    "VN05": _R(
        "node.fill.parameters",
        "grey-params",
        P,
        "_render_common:TRAINABLE_PARAMS_BG_COLOR",
        caption_variants=("#E6E6E6",),
    ),
    "VN06": _R(
        "node.border.dashed_not_from_input", "dashed", P, "_render_nodes:_get_node_bg_color"
    ),
    "VN07": _R(
        "node.shape.cylinder.buffer",
        "buffers",
        P,
        "_render_nodes:_buffer_versions_for_layer",
        variants=("never", "meaningful", "always"),
    ),
    "VN08": _R(
        "node.peripheries.hidden_buffers",
        "buffers",
        P,
        "_render_nodes:_get_hidden_parent_buffer_addresses",
    ),
    "VN09": _R(
        "node.shape.cylinder.mutated_parameter",
        "mutated-param",
        P,
        "_mutated_params:add_mutated_parameter_nodes",
    ),
    "VN10": _R(
        "node.shape.box3d.collapsed_module",
        "depth",
        P,
        "_render_nodes:_build_collapsed_module_node",
    ),
    "VN11": _R("node.chip.segment", "collapse-max", P, "_render_nodes:_queue_segment_node"),
    "VN12": _R(
        "node.chip.pattern",
        "fold-patterns",
        P,
        "collapse_patterns:match_patterns",
        variants=("ConvBnRelu", "ConvBn", "ConvRelu", "LinearRelu"),
    ),
    "VN13": _R(
        "node.fold.representative_ellipsis",
        "fold-repeats",
        P,
        "_render_edges:_run_fold_ellipsis_label",
    ),
    "VN14": _R("node.orphan", "orphans", P, "_render_dot:_add_orphan_island_nodes"),
    "VN15": _R("node.focus_boundary", "focus", P, "_render_common:FocusNode"),
    "VN16": _R(
        "node.border.intervention_site",
        "interventions",
        P,
        "node_spec:make_intervention_node_spec_fn",
        variants=("node_mark",),
    ),
    "VN17": _R(
        "node.border.intervention_cone",
        "interventions",
        P,
        "node_spec:intervention_site_and_cone_labels",
    ),
    "VN18": _R(
        "node.shape.diamond.hook",
        "interventions",
        P,
        "_render_edges:_add_intervention_hook_nodes",
        variants=("as_node",),
    ),
    "VN19": _R(
        "node.border.surgery_mark",
        "surgery",
        P,
        "surgery_visuals:make_surgery_mark_spec_fn",
        variants=(
            "fire",
            "region_member",
            "region_exit",
            "injection_host",
            "declared_target",
            "fact",
            "heuristic",
        ),
    ),
    "VN20": _R("node.border.overlay_flag", "overlays", P, "overlays:overlay_border_attrs"),
    "VN21": _R("node.image", "data-previews", P, "_render_nodes:_render_raw_input_image_batch"),
    "VN22": _R("node.raw_input_preview", "data-previews", P, "_render_nodes:_render_raw_input"),
    "VN23": _R("node.raw_output_decode", "data-outputs", P, "_render_nodes:_render_raw_output"),
    "VN24": _R(
        "node.container.summary", "container-lists", P, "_render_leaf:_add_collapsed_container_node"
    ),
    "VN25": _R(
        "node.container.record", "containers", P, "_render_flow:_container_record_node_args"
    ),
    "VN26": _R("node.grad_fn", "backward", P, "_render_leaf:_add_backward_node_to_graphviz"),
    "VN27": _R("node.legend_table", "builtin-legend", P, "_legend:add_legend_table_to_graphviz"),
    "VN28": _R("node.spec_surface", "custom", P, "torchlens._vocab.node_spec:NodeSpec"),
    "VN29": _R(
        "node.nonfinite_state",
        "nonfinite",
        P,
        "lenses._nonfinite:nonfinite_spec_fn",
        variants=("finite", "nan", "pos_inf", "neg_inf", "mixed", "not_checked"),
    ),
    # Label rows
    "VL01": _R("label.title", "node-rows", P, "_render_nodes:compute_default_node_lines"),
    "VL02": _R("label.shape_memory", "node-rows", P, "_label_format:format_shape"),
    "VL03": _R(
        "label.across_pass_shape", "reuse-shapes", P, "_render_nodes:_rolled_multicall_shape_note"
    ),
    "VL04": _R("label.saved_for_backward", "more-rows", P, "_label_format:saved_for_backward_line"),
    "VL05": _R(
        "label.constructor_args", "node-rows", P, "_arg_suppression:compute_suppressed_arg_keys"
    ),
    "VL06": _R("label.parameters", "node-rows", P, "_label_format:format_param_list"),
    "VL07": _R("label.module_path", "node-rows", P, "_label_format:format_module_path"),
    "VL08": _R("label.overlay_row", "overlays", P, "overlays:overlay_line"),
    "VL09": _R(
        "label.profiling_rows", "more-rows", P, "modes:profiling_node_mode", variants=("profiling",)
    ),
    "VL10": _R(
        "label.vision_attention_rows",
        "more-rows",
        C,
        "modes:vision_node_mode",
        status="experimental",
        gate="'vision' not in modes.MODE_REGISTRY and 'attention' not in modes.MODE_REGISTRY",
    ),
    "VL11": _R(
        "label.selected_fields", "more-rows", P, "_label_format:compute_selected_node_lines"
    ),
    "VL12": _R(
        "label.input_transform_summary",
        "data-previews",
        P,
        "_render_nodes:_input_transform_summary_attrs",
    ),
    # Edges
    "VE01": _R("edge.data", "start-here", P, "_render_edges:_add_edges_for_node"),
    "VE02": _R(
        "edge.label.argument", "arrow-words", P, "_edge_multiplicity:_set_argument_edge_label"
    ),
    "VE03": _R(
        "edge.label.multiplicity",
        "arrows-as-one",
        P,
        "_edge_multiplicity:_html_multiplicity_edge_label",
    ),
    "VE04": _R("edge.label.rolled_passes", "loop-rolled", P, "_render_edges:_RolledEdgeMaps"),
    "VE05": _R(
        "edge.self_loop", "loop-rolled", C, "_render_edges:_is_rolled_loop_carried_self_edge"
    ),
    "VE06": _R(
        "edge.label.conditional", "branches", P, "_render_leaf:_format_branch_edge_label_html"
    ),
    "VE07": _R("edge.skip_bridge", "skip", P, "_render_flow:_build_skip_filtered_edge_map"),
    "VE08": _R("edge.forward_gradient", "combined", P, "_render_leaf:_add_grad_edge"),
    "VE09": _R("edge.backward_accumulate", "backward", P, "_render_leaf:_backward_edge_attrs"),
    "VE10": _R(
        "edge.correspondence", "combined", P, "_render_leaf:_add_combined_correspondence_edges"
    ),
    "VE11": _R(
        "edge.mutated_parameter", "mutated-param", P, "_mutated_params:add_mutated_parameter_nodes"
    ),
    "VE12": _R("edge.container", "containers", P, "_render_flow:_container_member_edge_attrs"),
    "VE13": _R(
        "edge.antiparallel_fold",
        "collapse-max",
        C,
        "_render_edges:_projected_antiparallel_edge_attrs",
    ),
    "VE14": _R(
        "edge.invisible_ordering",
        "direction-layout",
        C,
        "_render_ordering:_queue_sibling_rank_group",
    ),
    "VE15": _R(
        "edge.surgery_diff_join",
        "surgery-diff",
        P,
        "_surgery_diff:surgery_diff",
        variants=("site_key", "label", "site_key_positional"),
    ),
    "VE16": _R("edge.fold_run", "fold-repeats", P, "_render_edges:_edge_touches_run_fold"),
    "VE17": _R("edge.overrides", "custom", P, "_render_utils:merge_edge_style"),
    # Regions
    "VR01": _R(
        "region.module_cluster", "module-boxes", P, "_render_utils:make_module_cluster_attrs"
    ),
    "VR02": _R("region.depth_penwidth", "module-boxes", P, "_render_utils:compute_module_penwidth"),
    "VR03": _R(
        "region.nesting_pruning", "module-boxes", P, "_render_regions:_module_subtree_payload_empty"
    ),
    "VR04": _R(
        "region.cluster_title_by_view", "module-boxes", P, "_render_utils:make_module_cluster_label"
    ),
    "VR05": _R("region.depth_collapse_fn", "depth", P, "_render_leaf:_should_collapse_module"),
    "VR06": _R(
        "region.collapse_ladder",
        "collapse-max",
        P,
        "collapse_ladder:build_event_ladder",
        variants=("none", "auto", "max", "None", "True", "False"),
    ),
    "VR07": _R(
        "region.collapse_disclosures",
        "collapse-max",
        C,
        "_collapse_disclosures:NEAR_UNCOLLAPSED_FRACTION",
    ),
    "VR08": _R("region.module_focus", "focus", P, "_render_flow:_build_module_focus_entries"),
    "VR09": _R(
        "region.show_containers",
        "containers",
        P,
        "_draw_validation:_SHOW_CONTAINERS_MODES",
        variants=("nodes", "collapsed"),
        caption_variants=("False", "labels", "cluster", "auto"),
    ),
    "VR10": _R(
        "region.sibling_ordering", "direction-layout", C, "_render_ordering:_should_order_siblings"
    ),
    "VR11": _R(
        "region.backward_pass_cluster",
        "grad-of-grad",
        P,
        "_render_nodes:_add_unrolled_backward_pass_clusters",
    ),
    "VR12": _R(
        "region.intervening_cluster",
        "combined",
        C,
        "_render_regions:_setup_combined_special_clusters",
        caption_variants=("upstream", "outside", "downstream", "own"),
    ),
    "VR13": _R("region.orphan_cluster", "orphans", P, "_render_dot:_add_orphan_island_nodes"),
    "VR14": _R("region.stack_groups", "stack-by", P, "_stacking:compute_stack_groups"),
    # Caption, views
    "VC01": _R("caption.graph", "grey-params", P, "_render_dot:_graph_caption_body"),
    "VC02": _R("caption.direct_writes", "surgery", C, "_render_dot:_graph_caption_body"),
    "VC03": _R(
        "caption.honesty_banner",
        "start-here",
        C,
        "torchlens._capture_honesty:honesty_preamble_lines",
    ),
    "VC04": _R(
        "caption.backward_combined", "backward", P, "_render_entrypoints:render_backward_graph"
    ),
    "VC05": _R("caption.surgery_census", "surgery", C, "surgery_visuals:surgery_census"),
    "VV01": _R(
        "view.unrolled_rolled",
        "loop-unrolled",
        P,
        "torchlens._literals:VisModeLiteral",
        variants=("unrolled", "rolled"),
        caption_variants=("none",),
    ),
    "VV02": _R(
        "view.entry_point_defaults", "backward", P, "_render_entrypoints:render_combined_graph"
    ),
    # Encoding channels, overlays, lenses
    "VK01": _R(
        "channel.color_by",
        "color-by",
        P,
        "_encoding:resolve_color_by",
        variants=("flops", "time", "bytes", "magnitude", "grad_norm", "device_time"),
    ),
    "VK02": _R(
        "channel.color_transforms",
        "color-transforms",
        P,
        "_encoding:COLOR_TRANSFORM_VOCABULARY",
        variants=("linear", "rank", "log"),
    ),
    "VK03": _R(
        "channel.unencoded_reasons",
        "unencoded",
        C,
        "_encoding:_varying_marker",
        caption_variants=(
            "reconciled_aggregate",
            "summed_aggregate",
            "mirrored_per_call_numeric",
            "structurally_uniform",
            "derived_composite",
            "per_pass",
            "shape",
            "non_numeric",
        ),
    ),
    "VK04": _R(
        "channel.size_by_scale",
        "size-by",
        P,
        "_encoding:resolve_size_by",
        variants=("sqrt", "linear", "dims"),
    ),
    "VK05": _R("channel.stack_by", "stack-by", P, "_stacking:resolve_stack_by"),
    "VK06": _R("legend.encoding_section", "color-by", P, "_legend:encoding_sections"),
    "VK07": _R(
        "overlay.node_overlay",
        "overlays",
        P,
        "overlays:resolve_overlay_request",
        variants=("nan",),
        caption_variants=(
            "flops",
            "time",
            "bytes",
            "magnitude",
            "grad_norm",
            "grad-norm",
            "intervention",
            "bundle_delta",
            "bundle delta",
        ),
    ),
    "VK08": _R(
        "label.node_style", "more-rows", P, "modes:MODE_REGISTRY", variants=("default", "profiling")
    ),
    "VK09": _R(
        "lens.roster",
        "lenses",
        P,
        "lenses._roster:ROSTER",
        variants=("overview", "speed", "dims", "debug"),
        caption_variants=(
            "blueprint",
            "transformer",
            "memory",
            "sequence",
            "compute",
            "surgery",
            "runtime_storage",
            "vision",
            "debug_edge_shapes",
        ),
    ),
    "VK10": _R(
        "lens.display_filter",
        "skip",
        P,
        "lenses._filter:compile_display_filter",
        variants=("reshapes",),
        caption_variants=("constants", "non_module_ops"),
    ),
    # Legend, skins, typography, layout, output
    "VI01": _R("legend.role_section", "builtin-legend", P, "_legend:theme_role_sections"),
    "VI02": _R(
        "legend.show_legend_tristate", "builtin-legend", P, "_render_dot:_emit_and_finish_forward"
    ),
    "VI03": _R("legend.backward_key", "backward-key", P, "_legend:backward_key_sections"),
    "VI04": _R(
        "legend.rank_engine", "layout-engine", P, "_legend:legend_table_lines_for_rank_path"
    ),
    "VI05": _R("legend.bundle_diff", "other-pictures", C, "bundle_diff:bundle_diff"),
    "VT01": _R(
        "skin.presets",
        "skins",
        P,
        "themes:THEME_PRESETS",
        variants=("torchlens", "paper", "dark", "colorblind", "high_contrast"),
    ),
    "VT02": _R("skin.application", "skins", P, "themes:apply_theme_to_spec"),
    "VT03": _R("type.roles", "text-size", P, "_typography:DEFAULT_TYPOGRAPHY"),
    "VY01": _R(
        "layout.direction",
        "direction-layout",
        P,
        "_render_utils:direction_to_rankdir",
        variants=("bottomup", "topdown", "leftright"),
    ),
    "VY02": _R(
        "layout.engine",
        "layout-engine",
        P,
        "torchlens.visualization._rank_layout_internal.layout:RANK_LAYOUT_COST_THRESHOLD",
        variants=("auto", "dot", "rank"),
    ),
    "VY03": _R(
        "layout.renderer",
        "other-pictures",
        C,
        "torchlens.experimental.dagua._bridge:render_trace_with_dagua",
        status="experimental",
        gate="'dagua' in _literals.VisRendererLiteral.__args__",
        caption_variants=("graphviz", "dagua"),
    ),
    "VO01": _R("output.format", "export", P, "_render_utils:render_dot_to_file"),
    "VO02": _R("output.save_return", "export", P, "_render_utils:strip_known_extension"),
    "VO03": _R("output.graph_module_overrides", "custom", P, "node_spec:graphviz_graph_overrides"),
    "VO04": _R(
        "output.code_panel",
        "code-panel",
        C,
        "code_panel:render_code_panel_subgraph",
        variants=("forward",),
        caption_variants=("class", "init+forward"),
    ),
    "VO05": _R("output.draw_validation", "export", C, "_draw_validation:_validate_draw_options"),
    "VO06": _R(
        "output.all_draw_parameters",
        "cheat-sheet",
        C,
        "torchlens.data_classes._trace_viz:TraceVisualizationMixin.draw",
    ),
    # Side pictures
    "VS01": _R("side.bundle_graph", "other-pictures", C, "_bundle_graph:_add_bundle_forward_nodes"),
    "VS02": _R("side.bundle_diff", "other-pictures", C, "bundle_diff:bundle_diff"),
    "VS03": _R(
        "side.fastlog_preview",
        "other-pictures",
        C,
        "fastlog_preview:preview_fastlog",
        caption_variants=("kept", "rejected", "unreachable", "exception"),
    ),
    "VS04": _R(
        "side.other_surfaces", "other-pictures", C, "_summary_internal._builder:format_model_repr"
    ),
    # Dead surface (mystery: planned feature or dead code)
    "VT04": _R(
        "skin.collapse_kind_tokens",
        None,
        C,
        "themes:CollapseTokens",
        status="inactive",
        gate="themes.COLLAPSE_KIND_TOKENS_DEFAULT is False",
    ),
}

#: Grouped ``VisualizationOptions`` field to the ``Trace.draw`` kwarg it becomes
#: (``torchlens.options.visualization_to_render_kwargs``). Every field is listed, identity
#: included, so a new field fails until someone maps it.
VIS_OPTION_ALIASES: Mapping[str, str] = {
    "view": "vis_mode",
    "depth": "vis_call_depth",
    "container_path": "vis_outpath",
    "save_only": "vis_save_only",
    "file_format": "vis_fileformat",
    "show_buffers": "show_buffer_layers",
    "direction": "direction",
    "graph_overrides": "vis_graph_overrides",
    "node_style": "node_mode",
    "node_spec_fn": "node_spec_fn",
    "collapsed_node_spec_fn": "collapsed_node_spec_fn",
    "collapse_fn": "collapse_fn",
    "collapse": "collapse",
    "fold_repeats": "fold_repeats",
    "skip_fn": "skip_fn",
    "edge_overrides": "vis_edge_overrides",
    "grad_edge_overrides": "vis_grad_edge_overrides",
    "module_overrides": "vis_module_overrides",
    "layout": "vis_node_placement",
    "renderer": "vis_renderer",
    "theme": "vis_theme",
    "intervention_mode": "vis_intervention_mode",
    "show_cone": "vis_show_cone",
    "node_overlay": "node_overlay",
    "node_label_fields": "node_label_fields",
    "show_legend": "show_legend",
    "color_by": "color_by",
    "size_by": "size_by",
    "scale": "scale",
    "stack_by": "stack_by",
    "show_redundant_args": "show_redundant_args",
    "font_size": "font_size",
    "dpi": "dpi",
    "for_paper": "for_paper",
    "return_graph": "return_graph",
    "order_siblings": "order_siblings",
}

#: Cheat-sheet lines for the backward and combined entry points (``Trace.draw`` uses
#: ``slides.CHEAT_GROUPS``): the parameters no panel passes explicitly.
ENTRY_CHEATS: Mapping[str, tuple[str, ...]] = {
    "Trace.draw_backward": (
        "vis_outpath",
        "node_spec_fn",
        "collapsed_node_spec_fn",
        "vis_node_mode",
        "vis_edge_overrides",
        "vis_direction",
        "code_panel",
        "bwd",
    ),
    "Trace.draw_combined": (
        "vis_outpath",
        "node_spec_fn",
        "backward_node_spec_fn",
        "vis_edge_overrides",
        "vis_direction",
        "vis_mode",
        "intervening_cluster",
        "show_buffer_layers",
        "bwd",
    ),
}

#: ``"<entry>.<param>"`` to the reason it changes nothing visible.
NONVISUAL_DRAW_PARAMS: Mapping[str, str] = {}

#: Closed vocabularies that are not drawing choices.
VOCABULARY_EXEMPT: Mapping[str, str] = {
    "_literals.OutputDeviceLiteral": "capture output device, not a drawing choice",
}

#: Legend section title to the row that teaches it.
LEGEND_SECTION_ROWS: Mapping[str, str] = {
    "TorchLens legend": "VI01",
    "backward key": "VI03",
    "TorchLens encoding": "VK06",
}

#: Legend row text to the slide whose render draws it (confirmed against the real DOT by
#: the slide test). Rows the deck states in words need no entry.
LEGEND_ROW_WITNESS: Mapping[str, str] = {
    "TorchLens legend": "builtin-legend",
    "parameterized": "builtin-legend",
    "intervention/cone": "builtin-legend",
    "mutated parameter": "mutated-param",
    "backward op (grad_fn)": "backward-key",
    "order 2+: grad-of-grad (double backprop)": "backward-key",
    "[i] = intervening grad_fn (no forward op)": "backward-key",
    "[custom] = custom autograd function": "backward-key",
    "accum = gradient accumulation into a leaf": "backward-key",
    "bwd N = backward pass N; order N = derivative order": "backward-key",
    "order N = derivative order": "backward-key",
    "TorchLens encoding": "color-by",
    "color_by:": "color-by",
    "eligible nodes": "color-by",
    "linear min-max": "color-by",
    "n/a = unencoded": "color-by",
    "size_by:": "size-by",
    "size ~": "size-by",
    "min": "size-by",
    "stack_by:": "stack-by",
    "same rank = same annotation value": "stack-by",
}

__all__ = [
    "ENTRY_CHEATS",
    "LEGEND_ROW_WITNESS",
    "LEGEND_SECTION_ROWS",
    "NONVISUAL_DRAW_PARAMS",
    "ROWS",
    "VIS_OPTION_ALIASES",
    "VOCABULARY_EXEMPT",
    "Row",
]
