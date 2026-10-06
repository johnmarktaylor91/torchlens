"""DOT orchestration and SVG composition helpers for Graphviz rendering."""

# ruff: noqa: F403, F405

import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace

from .._errors import CaptureContextError, InvalidArgumentError, PayloadUnavailableError
from ..errors._base import TorchLensWarning
from ..utils.display import user_stacklevel
from . import _render_utils
from ._draw_validation import _validate_draw_flag_options, _validate_draw_options
from ._render_common import *
from ._render_edges import *
from ._render_flow import *
from ._render_leaf import *
from ._render_nodes import *
from ._render_ordering import (
    _layout_dot_plain,
    _queue_sibling_rank_group,
    _should_order_siblings,
    _sibling_chain_key,
    _sibling_chain_stretch_ratio,
    _sibling_order_decision,
    _strict_sibling_order_checks_enabled,
    _strip_sibling_rank_groups,
    _verify_and_apply_sibling_ordering,
    _warn_sibling_order_fallback_once,
)
from ._render_regions import (
    _queue_container_clusters,
    _setup_combined_special_clusters,
    _setup_subgraphs,
    _setup_subgraphs_recurse,
)
from ._render_utils import html_escape
from ._svg_compose import (
    _inline_svg_file_local_images,
    _inline_svg_local_images,
    _normalize_svg_root_viewbox,
    _render_graph_only_svg,
    _write_composed_code_panel,
)
from ._typography import DEFAULT_TYPOGRAPHY
from .overlays import resolve_overlay_request
from .render_execution import (
    atomic_render_target,
    build_render_geometry_record,
    surface_layout_stderr,
)
from .renderers.graphviz import GraphvizRenderer
from .request import RenderTarget, ResolvedRenderRequest
from .source_graph import _resolve_focus_module, build_source_graph


def _view_rendered_file(filepath: str) -> None:
    """Open a rendered visualization file when a local viewer is available.

    Parameters
    ----------
    filepath:
        Rendered artifact path.
    """

    _open_file_quietly(filepath, announce_headless=True)


if TYPE_CHECKING:
    from ..data_classes.module import Module
    from ..data_classes.trace import Trace
    from .auto_collapse import ModuleRepeatFold
    from .node_universe import NodeUniverse
    from .render_ir import RenderIR
    from .source_graph import SourceGraph


@dataclass(frozen=True)
class _ForwardRenderContext:
    """Resolved state shared by forward-IR construction and emission."""

    request: ResolvedRenderRequest
    target: RenderTarget
    theme: VisualizationTheme
    repeat_folds: dict[str, "ModuleRepeatFold"]
    segments: dict[str, SegmentDescriptor]
    segment_lookup: Any
    source_graph: "SourceGraph"
    node_universe: "NodeUniverse"
    rankdir: str
    source_text: str | None
    num_nodes: int
    layout_cost: int
    engine: str
    graph_caption: str
    dot: graphviz.Digraph
    site_labels: set[str]


@dataclass
class _ForwardIRWork:
    """Mutable forward-IR assembly state passed into region finalization."""

    render_ir: "RenderIR"
    builder: _RenderIRDecisionBuilder
    module_clusters: dict[str, Any]
    top_level_sibling_rank_groups: list[SiblingOrderChain]
    captured_edges: list[CapturedForwardEdge]
    container_regions: list[ContainerClusterSpec]
    container_overlay_edges: list[ContainerOverlayEdge]
    mutated_parameter_nodes: tuple[str, ...] = ()


def _strip_render_extension(vis_outpath: str) -> str:
    """Return output path without a Graphviz-rendered file extension.

    Parameters
    ----------
    vis_outpath:
        User-provided render output path.

    Returns
    -------
    str
        Path without a recognized render extension.
    """

    from ._render_utils import strip_known_extension

    return strip_known_extension(vis_outpath)


def _validate_rendered_output(rendered_path: str, source_path: str, graph_kind: str) -> None:
    """Raise if Graphviz did not produce a non-empty rendered artifact.

    Parameters
    ----------
    rendered_path:
        Expected rendered artifact path.
    source_path:
        DOT source path that was passed to Graphviz.
    graph_kind:
        Human-readable graph type for the error message.

    Raises
    ------
    GraphvizRenderError
        If the output path is missing or zero bytes.
    """

    if not os.path.exists(rendered_path):
        raise GraphvizRenderError(
            f"Graphviz reported success for {graph_kind} rendering but did not create "
            f"'{rendered_path}'. DOT source was saved to '{source_path}'. {_GRAPHVIZ_ESCAPE_HINT}"
        )
    if os.path.getsize(rendered_path) == 0:
        raise GraphvizRenderError(
            f"Graphviz reported success for {graph_kind} rendering but produced a zero-byte "
            f"output file at '{rendered_path}'. DOT source was saved to '{source_path}'. "
            f"{_GRAPHVIZ_ESCAPE_HINT}"
        )


def _raise_graphviz_timeout(
    graph_kind: str,
    node_description: str,
    source_path: str,
    timeout: int,
    error: subprocess.TimeoutExpired,
) -> None:
    """Raise a typed Graphviz timeout error with mitigation guidance.

    Parameters
    ----------
    graph_kind:
        Human-readable graph type.
    node_description:
        Description of the rendered graph size.
    source_path:
        DOT source path that was passed to Graphviz.
    timeout:
        Render timeout in seconds.
    error:
        Original timeout exception.

    Raises
    ------
    GraphvizRenderError
        Always raised with actionable rendering guidance.
    """

    raise GraphvizRenderError(
        f"Graphviz render timed out after {timeout}s for {graph_kind} with "
        f"{node_description}. DOT source was saved to '{source_path}'. {_GRAPHVIZ_ESCAPE_HINT}"
    ) from error


def _raise_graphviz_failure(
    graph_kind: str,
    source_path: str,
    error: subprocess.CalledProcessError,
) -> None:
    """Raise a typed Graphviz process failure with stderr and mitigation guidance.

    Parameters
    ----------
    graph_kind:
        Human-readable graph type.
    source_path:
        DOT source path that was passed to Graphviz.
    error:
        Original process failure.

    Raises
    ------
    GraphvizRenderError
        Always raised with Graphviz stderr and actionable rendering guidance.
    """

    stderr = _decode_graphviz_stderr(error)
    raise GraphvizRenderError(
        f"Graphviz failed while rendering {graph_kind}. DOT source was saved to "
        f"'{source_path}'. Graphviz stderr: {stderr} {_GRAPHVIZ_ESCAPE_HINT}"
    ) from error


def _resolve_draw_request(
    trace: "Trace",
    request: ResolvedRenderRequest,
) -> tuple[ResolvedRenderRequest, VisualizationTheme, set[str]]:
    """Resolve buffer, theme, override, overlay, and intervention options.

    Returns
    -------
    tuple
        Updated request, resolved theme, and intervention site labels.
    """

    show_buffer_layers = _normalize_buffer_visibility(request.show_buffer_layers)
    theme = resolve_theme(request.theme, for_paper=request.for_paper)
    node_overlay = request.node_overlay
    if (
        node_overlay is None
        or isinstance(node_overlay, str)
        and node_overlay == getattr(trace, "_node_overlay_name", None)
    ):
        node_overlay = getattr(trace, "_node_overlay_scores", None)
    node_overlay = resolve_overlay_request(trace, node_overlay)
    overrides = VisualizationOverrides(
        graph=graphviz_graph_overrides(cast(Optional[Dict[str, Any]], request.graph_overrides)),
        edge=cast(Optional[Dict[str, Any]], request.edge_overrides) or {},
        grad_edge=cast(Optional[Dict[str, Any]], request.grad_edge_overrides) or {},
        module=cast(Optional[Dict[str, Any]], request.module_overrides) or {},
    )
    request = replace(
        request,
        show_buffer_layers=show_buffer_layers,
        node_overlay=cast("str | OverlayScores | None", node_overlay),
        overrides=overrides,
    )
    site_labels, _ = intervention_site_and_cone_labels(trace, show_cone=request.show_cone)
    intervention_node_spec_fn = make_intervention_node_spec_fn(
        trace,
        show_cone=request.show_cone,
        graph_overrides=cast(Optional[Dict[str, Any]], request.graph_overrides),
        user_node_spec_fn=request.node_spec_fn,
    )
    return replace(request, node_spec_fn=intervention_node_spec_fn), theme, site_labels


def _resolve_collapse_request(
    trace: "Trace",
    request: ResolvedRenderRequest,
) -> tuple[
    ResolvedRenderRequest,
    dict[str, "ModuleRepeatFold"],
    dict[str, SegmentDescriptor],
    Any,
]:
    """Resolve smart-collapse predicates, repeat folds, and segment lookup data.

    Returns
    -------
    tuple
        Updated request, repeat folds, segments, and segment lookup.

    Raises
    ------
    ValueError
        If the trace no longer retains the complete layer graph.
    """

    render_context = request
    collapse_fn = request.collapse_fn
    if request.collapse != "none" and collapse_fn is None:
        from .auto_collapse import resolve_collapse_fn

        collapse_fn = resolve_collapse_fn(
            trace,
            request.collapse,
            request.vis_mode,
            context=render_context,
        )
    request = request.with_resolved_collapse(collapse_fn)
    render_context = request
    repeat_folds: dict[str, ModuleRepeatFold] = {}
    collapse_uses_default_folds = request.collapse in {"auto", "max"} or (
        isinstance(request.collapse, float) and request.collapse > 0.0
    )
    if request.fold_repeats is not False and (
        request.fold_repeats is True or collapse_uses_default_folds
    ):
        from .auto_collapse import resolve_repeat_folds

        repeat_folds = resolve_repeat_folds(
            trace,
            collapse_fn,
            context=render_context,
            fold_repeats=request.fold_repeats,
        )
    segments: dict[str, SegmentDescriptor] = {}
    if collapse_fn is not None:
        segments = dict(getattr(collapse_fn, "_torchlens_v2_segments", {}) or {})
    if request.fold_patterns:
        # Declarative pattern folding (F11, collapse memo D11): matched on
        # this trace, realized as K4-family chips through the segment
        # channel. v1 supports the pattern-only view (collapse="none");
        # chip atomicity across the automatic planner (band/score/ceiling/
        # schedule) is the named follow-on, so the combination refuses
        # typed rather than silently double-counting chip members.
        if request.collapse != "none":
            raise InvalidArgumentError(
                "fold_patterns cannot combine with automatic collapse in v1; "
                f"received collapse={request.collapse!r}",
                code="pattern_collapse_combination_unsupported",
                remedy=(
                    "draw with collapse='none' for the pattern-only view, or drop fold_patterns"
                ),
            )
        if request.engine == "rank":
            # The rank backend cannot position segment-family nodes yet (the
            # pre-existing __segment__ gap, owned by the typed-units item);
            # chips fail closed rather than crash mid-layout (memo D12: no
            # rank arm until the fix lands).
            raise InvalidArgumentError(
                "fold_patterns is not supported under vis_node_placement="
                "'rank' in v1: the rank backend cannot position "
                "segment-family chip nodes yet",
                code="pattern_rank_layout_unsupported",
                remedy="render with vis_node_placement='dot' (or 'auto')",
            )
        from .collapse_patterns import match_patterns

        pattern_segments, _pattern_report = match_patterns(trace, request, request.fold_patterns)
        segments.update(pattern_segments)
    segment_lookup = _build_segment_lookup(segments)
    if not trace._layers_logged:
        raise PayloadUnavailableError(
            "Must have all layers logged in order to render the graph",
            code="layers_not_logged",
            remedy="re-capture with tl.trace(model, x) (default exhaustive capture) before drawing",
        )
    return request, repeat_folds, segments, segment_lookup


def _resolve_fold_patterns(fold_patterns: object) -> tuple:
    """Resolve the raw ``fold_patterns=`` draw value to hashable specs."""

    if not fold_patterns:
        return ()
    from .collapse_patterns import resolve_pattern_request

    return resolve_pattern_request(fold_patterns)


def _params_caption_detail(trace: "Trace") -> str:
    """Return the caption's parameter-count line for one trace.

    Returns
    -------
    str
        Human-readable parameter summary (count, trainable split, memory).
    """

    if trace.num_params == 0:
        return "0 params"
    if trace.num_params_frozen == 0:
        return f"{trace.num_params} params (all trainable, {trace.total_param_memory})"
    if trace.num_params_trainable == 0:
        return f"{trace.num_params} params (all frozen, {trace.total_param_memory})"
    return (
        f"{trace.num_params} params "
        f"({trace.num_params_trainable}/{trace.num_params} trainable, "
        f"{trace.total_param_memory})"
    )


def _graph_caption_body(trace: "Trace", request: ResolvedRenderRequest) -> str:
    """Assemble the caption body: stats, honesty banner, and disclosures.

    Returns
    -------
    str
        HTML-like caption body without the outer FONT wrapper.
    """

    caption_body = (
        f"<B>{html_escape(trace.model_class_name)}</B>"
        f"<br align='left'/>{trace.num_tensors} tensors total ({trace.total_activation_memory})"
        f"<br align='left'/>{_params_caption_detail(trace)}<br align='left'/>"
    )
    if getattr(trace, "_has_direct_writes", False):
        caption_body += "Direct writes detected - recipe propagation will overlay<br align='left'/>"
    # Capture-honesty banner (WT1 A-V row 23): a poisoned/diverged sparse run,
    # an unverified capture, or an episode capture must not RENDER clean --
    # the graph is a publishable artifact, so the caption carries the facts.
    from .._capture_honesty import honesty_banner_lines

    for banner_line in honesty_banner_lines(trace):
        caption_body += f"<B>{html_escape(banner_line)}</B><br align='left'/>"
    encoding_state = getattr(request, "encoding", None)
    if encoding_state is not None and getattr(encoding_state, "stack_spec", None) is not None:
        # Stacking disclosure is part of the contract (memo 4.1): the
        # rendered output captions which annotation produced the columns.
        caption_body += (
            f"stacked by: {html_escape(encoding_state.stack_spec.display_name)}<br align='left'/>"
        )
    # No silent floor (collapse memo D4): a floor-fallback plan renders with a
    # VISIBLE notice, never as if the optimizer had chosen it.
    collapse_result = getattr(request.collapse_fn, "_torchlens_v2_result", None)
    if collapse_result is not None and getattr(collapse_result, "planner", "") == "floor_fallback":
        floor_note = (
            f"collapse fell back to the floor plan ({collapse_result.visible_count} nodes visible)"
        )
        if getattr(collapse_result, "k_cap_exhausted", False):
            floor_note += f" -- root owns {collapse_result.root_own_units} units, above K_CAP"
        caption_body += f"<B>{html_escape(floor_note)}</B><br align='left'/>"
    return caption_body


def _build_graphviz_shell(
    trace: "Trace",
    request: ResolvedRenderRequest,
    target: RenderTarget,
    theme: VisualizationTheme,
    rankdir: str,
) -> tuple[str, graphviz.Digraph]:
    """Build the caption and configured Graphviz graph shell.

    Returns
    -------
    tuple
        Graph caption and configured empty Graphviz graph.
    """

    caption_body = _graph_caption_body(trace, request)
    graph_caption = f"<<FONT COLOR='{theme.default_font}'>{caption_body}</FONT>>"

    dot = graphviz.Digraph(
        name=trace.model_class_name,
        comment="Computational graph for the feedforward sweep",
        format=target.fileformat,
    )
    graph_args = {
        "rankdir": rankdir,
        "label": graph_caption,
        "labelloc": "t",
        "labeljust": "left",
        "ordering": "out",
    }
    if request.collapse_fn is not None:
        graph_args["newrank"] = "true"
    # r-b6 R19-6 / T9 (grind-p3): node image attributes are emitted RELATIVE
    # to the trace visualizer scratch root, and the root is supplied to
    # Graphviz OUT OF BAND (subprocess cwd at render time), never as an
    # in-source ``imagepath`` — a baked per-run mkdtemp path made every
    # user-saved DOT unrenderable once the trace's scratch dir was GC'd.
    graph_args.update(
        theme_graph_attrs(
            theme,
            font_size=request.font_size,
            dpi=request.dpi,
            fileformat=target.fileformat,
        )
    )
    overrides = cast(VisualizationOverrides, request.overrides)
    for arg_name, arg_val in overrides.graph.items():  # type: ignore[union-attr]
        if callable(arg_val):
            graph_args[arg_name] = str(arg_val(trace))
        else:
            graph_args[arg_name] = str(arg_val)
    dot.graph_attr.update(graph_args)
    dot.node_attr.update(
        {"ordering": "out", **theme_node_attrs(theme, font_size=request.font_size)}
    )
    dot.edge_attr.update(theme_edge_attrs(theme, font_size=request.font_size))
    return graph_caption, dot


def _resolve_forward_context(
    trace: "Trace",
    request: ResolvedRenderRequest,
    target: RenderTarget,
    theme: VisualizationTheme,
    repeat_folds: dict[str, "ModuleRepeatFold"],
    segments: dict[str, SegmentDescriptor],
    segment_lookup: Any,
    site_labels: set[str],
) -> _ForwardRenderContext:
    """Resolve the source graph, layout, caption, and Graphviz shell.

    Returns
    -------
    _ForwardRenderContext
        State shared by IR construction and final renderer dispatch.
    """

    source_graph = build_source_graph(trace, request)
    from .node_universe import build_node_universe

    node_universe = build_node_universe(
        source_graph,
        request.collapse_fn,
        repeat_folds,
        segments,
        request.show_containers,
    )
    entries_to_plot = source_graph.entries_to_plot
    rankdir = direction_to_rankdir(request.direction)

    from ._rank_layout_internal.layout import (
        RANK_LAYOUT_COST_THRESHOLD,
        RANK_LAYOUT_NOTICE,
        estimate_rank_layout_cost,
        get_node_placement_engine,
    )

    source_text = resolve_code_panel_source(
        request.code_panel,
        getattr(trace, "_source_code_blob", {}),
        getattr(trace, "_source_model_ref", None),
    )
    num_nodes = len(entries_to_plot) - len(source_graph.skipped_labels)
    cost_node_labels, cost_edges = _rank_layout_cost_inputs(
        trace,
        entries_to_plot,
        source_graph.edge_map,
        vis_mode=request.vis_mode,
        vis_call_depth=request.vis_call_depth,
        collapse_fn=request.collapse_fn,
    )
    layout_cost = estimate_rank_layout_cost(cost_node_labels, cost_edges)
    engine = get_node_placement_engine(request.engine, layout_cost)
    if request.show_containers:
        engine = "dot"
    if request.encoding is not None:
        # Engine-resolution fence (L5): explicit rank refuses typed; AUTO
        # forces dot (show_containers precedent), noticed on cost override.
        from ._encoding import resolve_encoding_engine

        engine = resolve_encoding_engine(
            request.engine, engine, layout_cost, request.encoding.active_channels()
        )
    # Session diagnostic (never persisted; scrub-declared like the sibling
    # decision): the last draw's encoding state, for tests and debugging.
    trace._last_encoding_state = request.encoding
    trace._last_sibling_ordering_decision = SiblingOrderDecision(0, 0, {}, ())
    if request.engine == "auto" and engine == "rank":
        warnings.warn(
            RANK_LAYOUT_NOTICE.format(
                cost=layout_cost,
                threshold=RANK_LAYOUT_COST_THRESHOLD,
            ),
            stacklevel=user_stacklevel(),
        )
    _vprint(
        trace,
        f"Rendering {request.vis_mode} graph ({num_nodes} nodes, format={target.fileformat})",
    )
    _vprint(trace, f"Layout engine: {engine} (estimated cost={layout_cost})")
    graph_caption, dot = _build_graphviz_shell(trace, request, target, theme, rankdir)
    return _ForwardRenderContext(
        request=request,
        target=target,
        theme=theme,
        repeat_folds=repeat_folds,
        segments=segments,
        segment_lookup=segment_lookup,
        source_graph=source_graph,
        node_universe=node_universe,
        rankdir=rankdir,
        source_text=source_text,
        num_nodes=num_nodes,
        layout_cost=layout_cost,
        engine=engine,
        graph_caption=graph_caption,
        dot=dot,
        site_labels=site_labels,
    )


def _populate_forward_ir(trace: "Trace", context: _ForwardRenderContext) -> _ForwardIRWork:
    """Materialize forward nodes, edge decisions, and container overlays.

    Returns
    -------
    _ForwardIRWork
        Mutable IR assembly state ready for region finalization.
    """

    request = context.request
    overrides = cast(VisualizationOverrides, request.overrides)
    forward_ir_builder = _RenderIRDecisionBuilder()
    module_cluster_dict: Dict[str, Any] = defaultdict(
        lambda: {
            "edges": [],
            "has_input_ancestor": False,
            "rank_groups": [],
            "container_clusters": [],
        }
    )
    top_level_sibling_rank_groups: list[SiblingOrderChain] = []
    collapsed_modules: Set[str] = set()
    edges_used: Set[tuple[str, str, tuple[Any, ...]]] = set()
    deduped_edge_registry: dict[tuple[Any, ...], dict[str, Any]] = {}
    run_fold_ellipsis_nodes: set[str] = set()
    emitted_segment_nodes: set[str] = set()
    captured_forward_edges: list[CapturedForwardEdge] = []
    pending_container_collapse_nodes: list[dict[str, Any]] = []
    container_clusters: list[ContainerClusterSpec] = []
    collapsed_container_nodes = _collapsed_container_leaf_nodes(
        trace,
        context.source_graph.entries_to_plot,
        vis_mode=request.vis_mode,
        show_containers=request.show_containers,
        container_max_inline=request.container_max_inline,
        pending_nodes=pending_container_collapse_nodes,
    )
    # Checked suppression (L5 M4, DEFAULT-ON): the trace-bearing prepass
    # proves which constructor-arg rows duplicate captured shapes on THIS
    # trace; unprovable or mismatching args stay visible (self-honest).
    # Computed ONCE and shared by the IR decision pass and node emission.
    suppressed_args: dict[int, frozenset[str]] = {}
    if not request.show_redundant_args:
        from ._arg_suppression import compute_suppressed_arg_keys

        suppressed_args = compute_suppressed_arg_keys(trace, context.node_universe)
    forward_render_ir = build_render_ir(
        trace,
        collapse_fn=request.collapse_fn,
        repeat_folds=context.repeat_folds,
        context=request,
        universe=context.node_universe,
        segments=context.segments,
        segment_lookup=context.segment_lookup,
        suppressed_args=suppressed_args,
    )
    antiparallel_projected_edges = projected_antiparallel_endpoint_pairs(forward_render_ir)
    decisions_by_name = {node.name: node for node in forward_render_ir.nodes}
    rolled_maps = _RolledEdgeMaps() if request.vis_mode == "rolled" else None
    node_label_fields = (
        list(request.node_label_fields) if request.node_label_fields is not None else None
    )
    for unit in context.node_universe.units:
        node_record = decisions_by_name[unit.unit_id]
        for source_index, node in enumerate(unit.source_nodes):
            _add_node_to_graphviz(
                trace,
                node,
                cast(graphviz.Digraph, forward_ir_builder),
                module_cluster_dict,
                edges_used,
                request.vis_mode,
                collapsed_modules,
                request.vis_call_depth,
                request.show_buffer_layers,
                overrides,
                request.node_mode,
                request.node_spec_fn,
                request.collapsed_node_spec_fn,
                request.collapse_fn,
                context.source_graph.edge_map,
                request.intervention_mode,
                context.site_labels,
                context.theme,
                cast("str | OverlayScores | None", request.node_overlay),
                node_label_fields,
                captured_forward_edges,
                context.rankdir,
                request.show_containers,
                collapsed_container_nodes,
                request.show_input_transform_summary,
                context.repeat_folds,
                run_fold_ellipsis_nodes,
                context.segment_lookup,
                emitted_segment_nodes,
                antiparallel_projected_edges,
                node_record
                if source_index == 0
                else replace(node_record, node_calls=(), owned_node_args=()),
                rolled_maps,
                deduped_edge_registry,
                encoding=request.encoding,
                suppressed_args=suppressed_args,
                show_saved_for_backward=request.show_saved_for_backward,
            )
    for node_args in pending_container_collapse_nodes:
        forward_ir_builder.node(**node_args)
    from ._mutated_params import add_mutated_parameter_nodes

    mutated_parameter_nodes = add_mutated_parameter_nodes(
        trace,
        universe=context.node_universe,
        builder=forward_ir_builder,
        module_clusters=module_cluster_dict,
        vis_mode=request.vis_mode,
        vis_call_depth=request.vis_call_depth,
        theme=context.theme,
    )

    container_overlay_edges: list[ContainerOverlayEdge] = []
    if request.show_containers == "nodes" and request.vis_mode == "unrolled":
        context.dot.graph_attr.update({"pad": "0.20"})
        container_overlay_nodes, container_overlay_edges = _container_nodes_and_overlay_edges(
            trace,
            collapsed_container_nodes,
            vis_call_depth=request.vis_call_depth,
            collapse_fn=request.collapse_fn,
        )
        for overlay_node in container_overlay_nodes:
            if overlay_node.owner_key is None:
                forward_ir_builder.node(**overlay_node.args)
            else:
                module_cluster_dict[overlay_node.owner_key].setdefault("nodes", []).append(
                    overlay_node.args
                )
    if request.show_containers in {"cluster", "nodes"}:
        container_clusters = _container_clusters_for_graphviz(
            trace,
            context.source_graph.entries_to_plot,
            vis_mode=request.vis_mode,
            vis_call_depth=request.vis_call_depth,
            collapse_fn=request.collapse_fn,
            collapsed_container_nodes=collapsed_container_nodes,
        )
        _queue_container_clusters(module_cluster_dict, container_clusters)
    if request.intervention_mode == "as_node":
        _add_intervention_hook_nodes(
            cast(graphviz.Digraph, forward_ir_builder),
            context.site_labels,
            cast(Optional[Dict[str, Any]], request.graph_overrides),
        )
    return _ForwardIRWork(
        render_ir=forward_render_ir,
        builder=forward_ir_builder,
        module_clusters=module_cluster_dict,
        top_level_sibling_rank_groups=top_level_sibling_rank_groups,
        captured_edges=captured_forward_edges,
        container_regions=container_clusters,
        container_overlay_edges=container_overlay_edges,
        mutated_parameter_nodes=mutated_parameter_nodes,
    )


def _finalize_forward_ir(
    trace: "Trace",
    context: _ForwardRenderContext,
    work: _ForwardIRWork,
) -> "RenderIR":
    """Resolve sibling constraints, nested regions, and ordered statements.

    Returns
    -------
    RenderIR
        Decision-complete forward IR ready for renderer dispatch.
    """

    from ._rank_layout_internal.layout import RANK_LAYOUT_COST_THRESHOLD

    request = context.request
    sibling_order_chains: tuple[SiblingOrderChain, ...] = ()
    # Stacking fence (L5 M3, conservative no-op): stack_by pins ranks, and
    # two independent constraint systems fighting over dot's layout is how
    # oscillation starts (visualization/CLAUDE.md no-op pattern). Stacking
    # is strictly opt-in, so this never fires on plain draw().
    stack_channel_active = (
        request.encoding is not None and getattr(request.encoding, "stack_spec", None) is not None
    )
    if not stack_channel_active and _should_order_siblings(
        order_siblings=request.order_siblings,
        engine=context.engine,
        vis_mode=request.vis_mode,
        num_nodes=context.num_nodes,
        module=request.module,
        vis_intervention_mode=request.intervention_mode,
        collapse_fn=request.collapse_fn,
        vis_call_depth=request.vis_call_depth,
    ):
        sibling_order_chains = _build_sibling_order_chains(work.captured_edges)
        if sibling_order_chains and (
            context.layout_cost * SIBLING_ORDER_VERIFY_LAYOUT_BUDGET > RANK_LAYOUT_COST_THRESHOLD
        ):
            warnings.warn(
                SIBLING_ORDER_COST_NOTICE.format(
                    cost=context.layout_cost,
                    budget=SIBLING_ORDER_VERIFY_LAYOUT_BUDGET,
                    threshold=RANK_LAYOUT_COST_THRESHOLD,
                ),
                stacklevel=user_stacklevel(),
            )
            sibling_order_chains = ()
        if sibling_order_chains:
            for chain in sibling_order_chains:
                _queue_sibling_rank_group(
                    work.module_clusters,
                    work.top_level_sibling_rank_groups,
                    chain,
                )
    forward_render_ir = replace(
        work.render_ir,
        ordering_constraints=tuple(
            RenderIROrderingConstraint(
                kind="sibling_order",
                source_label=chain.source_label,
                source_name=chain.source_name,
                targets=chain.targets,
                target_labels=chain.target_labels,
                lca_key=chain.lca_key,
            )
            for chain in sibling_order_chains
        ),
    )
    overrides = cast(VisualizationOverrides, request.overrides)
    forward_render_ir = finalize_forward_regions(
        forward_render_ir,
        trace,
        vis_mode=request.vis_mode,
        module_payloads=work.module_clusters,
        container_regions=tuple(work.container_regions),
        captured_edges=tuple(work.captured_edges),
        overrides=overrides,
        rankdir=context.rankdir,
    )
    _setup_subgraphs(
        trace,
        cast(graphviz.Digraph, work.builder),
        request.vis_mode,
        work.module_clusters,
        overrides,
        list(forward_render_ir.ordering_constraints),
        forward_render_ir.regions,
    )
    return replace(forward_render_ir, dot_statements=tuple(work.builder.calls))


def _emit_and_finish_forward(
    trace: "Trace",
    context: _ForwardRenderContext,
    work: _ForwardIRWork,
    forward_render_ir: "RenderIR",
) -> Any:
    """Dispatch completed forward IR and finish output handling.

    Returns
    -------
    Any
        Rank-renderer result, Graphviz graph, or final DOT source.
    """

    request = context.request
    target = context.target
    overrides = cast(VisualizationOverrides, request.overrides)
    if context.engine == "rank":
        from ._rank_layout_internal.layout import render_rank_layout

        resolved_graph_overrides = {
            key: str(val(trace)) if callable(val) else str(val)
            for key, val in overrides.graph.items()  # type: ignore[union-attr]
        }
        rank_visualizer_dir = getattr(trace, "_visualizer_dir", None)
        if rank_visualizer_dir and "imagepath" not in resolved_graph_overrides:
            # r-b6 R19-6: one-attribute image root for the rank engine. T9
            # (grind-p3) removed the in-source root on the dot path (cwd
            # replaces it); the rank renderer runs neato internally and its
            # source file is deleted after a successful render, so the
            # remaining temp-path exposure here is the returned source
            # string and a failure-path leftover — a disclosed residual
            # until the rank internals grow a cwd-based root.
            resolved_graph_overrides["imagepath"] = str(rank_visualizer_dir)
        rank_execution: dict[str, str] = {}
        with _timed_phase(trace, "render:graphviz:forward"):
            result = render_rank_layout(
                forward_render_ir,
                request.vis_mode,
                target.outpath,
                target.fileformat,
                target.save_only,
                context.graph_caption,
                context.rankdir,
                context.source_text,
                # Tri-state None = AUTO resolves to no legend on the rank
                # backend (channels are never active here -- the 2.1 fence
                # forces dot or refuses before this point).
                show_legend=bool(request.show_legend),
                theme=context.theme,
                dpi=request.dpi,
                graph_overrides=resolved_graph_overrides,
                execution_record=rank_execution,
            )
        trace._last_render_geometry = build_render_geometry_record(
            engine=rank_execution.get("engine", "neato"),
            layout_path="rank",
            fileformat=target.fileformat,
            output_path=f"{target.outpath}.{target.fileformat}",
            stderr_text=rank_execution.get("stderr_text", ""),
        )
        _vprint(trace, f"Graph saved to {target.outpath}.{target.fileformat}")
        return result

    dot = context.dot
    GraphvizRenderer().emit(forward_render_ir, dot)
    if forward_render_ir.stack_rank_groups:
        # Stacking channel (L5 M3): rank=same groups span module clusters,
        # so newrank=true opts dot into global rank constraints (without it
        # cross-cluster rank=same is silently ignored -- a dishonest no-op).
        dot.graph_attr.update({"newrank": "true"})
        for rank_group in forward_render_ir.stack_rank_groups:
            with dot.subgraph() as rank_subgraph:
                rank_subgraph.attr(rank="same")
                for member in rank_group.members:
                    rank_subgraph.node(member)
    for overlay_edge in work.container_overlay_edges:
        dot.edge(
            tail_name=overlay_edge.tail_name,
            head_name=overlay_edge.head_name,
            **overlay_edge.attrs,
        )
    if request.show_orphans:
        _add_orphan_island_nodes(trace, dot, request.vis_mode, context.theme)
    # Legend visibility rule (L5 channel core): show_legend is tri-state
    # (None = AUTO: channel-only disclosure legend iff a channel is active).
    # Both legends render as SECTIONS of one compact HTML table in a
    # dedicated rank (vizmech item 13, D28): the historical six disconnected
    # nodes were packed BESIDE the model by dot and drove a 366-pt page to
    # 1048 pt while the collision audit scored the render clean.
    from ._legend import (
        LegendSection,
        add_legend_table_to_graphviz,
        encoding_sections,
        theme_role_sections,
    )

    legend_sections: tuple[LegendSection, ...] = ()
    if request.show_legend is True:
        legend_sections += theme_role_sections(
            context.theme, mutated_parameter=bool(work.mutated_parameter_nodes)
        )
    if request.encoding is not None and request.show_legend is not False:
        # None (AUTO) or True with an active channel -> disclosure section;
        # explicit False is honored (a deliberate act; the docs state the
        # encoding is then undisclosed).
        legend_sections += encoding_sections(request.encoding)
    legend_anchor = forward_render_ir.nodes[-1].name if forward_render_ir.nodes else None
    add_legend_table_to_graphviz(dot, context.theme, legend_sections, anchor=legend_anchor)
    compose_code_panel = context.source_text is not None and _code_panel_composition_available(
        target.fileformat, context.engine
    )
    if context.source_text is not None and not compose_code_panel:
        render_code_panel_subgraph(dot, context.source_text)

    render_timeout = 120
    if in_notebook() and not target.save_only:
        try:
            from IPython.display import SVG, display  # #72: lazy import
        except ImportError as error:
            raise ImportError(
                "IPython is required for this feature. Install with "
                "`pip install torchlens[notebook]`."
            ) from error

        display_fn = cast(Any, display)
        # r-b6 R40-2: render through the BOUNDED subprocess runner instead of
        # ``dot.pipe()`` / ``display(dot)`` — graphviz 0.21 exposes no pipe
        # timeout, so a wedged ``dot`` hung the kernel indefinitely while
        # every CLI path was already bounded at ``render_timeout`` with a
        # typed error. Timeout/failure map to the same typed raises.
        with tempfile.NamedTemporaryFile("w", suffix=".dot", delete=False) as notebook_source_file:
            notebook_source_file.write(dot.source)
            notebook_source_path = notebook_source_file.name
        try:
            notebook_image_root = getattr(trace, "_visualizer_dir", None)
            graph_svg = _render_graph_only_svg(
                dot.engine,
                notebook_source_path,
                render_timeout,
                Path(notebook_image_root) if notebook_image_root else None,
            )
        except subprocess.TimeoutExpired as error:
            # The typed raise names the saved source path, so keep the file.
            _raise_graphviz_timeout(
                "forward graph (notebook display)",
                f"{trace.num_tensors} nodes",
                notebook_source_path,
                render_timeout,
                error,
            )
        except subprocess.CalledProcessError as error:
            _raise_graphviz_failure("forward graph (notebook display)", notebook_source_path, error)
        if os.path.exists(notebook_source_path):
            os.remove(notebook_source_path)
        if compose_code_panel:
            graph_svg = compose_graph_with_code_panel(
                graph_svg,
                cast(str, context.source_text),
            )
        display_fn(SVG(graph_svg))

    source_override = None
    trace._last_sibling_ordering_decision = SiblingOrderDecision(0, 0, {}, ())
    if forward_render_ir.ordering_constraints:
        try:
            source_override, decision = _verify_and_apply_sibling_ordering(
                dot.source,
                forward_render_ir.ordering_constraints,
                work.captured_edges,
                context.rankdir,
            )
            trace._last_sibling_ordering_decision = decision
        except (subprocess.SubprocessError, OSError) as exc:
            if _strict_sibling_order_checks_enabled():
                raise
            _warn_sibling_order_fallback_once(exc)

    # r-b6 R19-6 / T9 (grind-p3): the visualizer scratch dir is created
    # LAZILY while nodes render (raw-input montages, feature maps). The root
    # is passed to the Graphviz subprocess as its working directory below,
    # never written into the source: user-saved DOT keeps only stable
    # relative image refs instead of a per-run mkdtemp path that dies with
    # the trace.
    late_visualizer_dir = getattr(trace, "_visualizer_dir", None)
    final_source = source_override if source_override is not None else dot.source
    source_path = dot.save(target.outpath)
    with open(source_path, "w", encoding="utf-8") as source_file:
        source_file.write(final_source)
    with _timed_phase(trace, "render:graphviz:forward"):
        try:
            rendered_path = f"{target.outpath}.{target.fileformat}"
            render_image_root = Path(late_visualizer_dir) if late_visualizer_dir else None
            layout_stderr = ""
            if compose_code_panel:
                _write_composed_code_panel(
                    dot.engine,
                    source_path,
                    cast(str, context.source_text),
                    rendered_path,
                    target.fileformat,
                    render_timeout,
                    render_image_root,
                )
            else:
                # Atomic publish (vizmech D20/D24): render into a sibling temp
                # file, validate it, and only then rename it over the user's
                # path -- a failed/timed-out layout never leaves a stub -- and
                # surface stderr even on exit 0 (the cairo "too large ...
                # scaling by N" clamp warning is the signal that raster
                # dimensions are fiction; it was silently discarded).
                with atomic_render_target(rendered_path) as temp_rendered_path:
                    cmd = [
                        dot.engine,
                        f"-T{target.fileformat}",
                        "-o",
                        os.path.abspath(temp_rendered_path),
                        os.path.abspath(source_path),
                    ]
                    completed = _render_utils.run_bounded_subprocess(
                        cmd,
                        timeout=render_timeout,
                        # T9 (grind-p3): relative node image refs resolve against
                        # the scratch root via cwd, keeping the per-run temp path
                        # out of the saved DOT source.
                        cwd=str(render_image_root) if render_image_root else None,
                    )
                    layout_stderr = surface_layout_stderr(completed.stderr, engine=dot.engine)
                    if target.fileformat == "svg":
                        _inline_svg_file_local_images(temp_rendered_path, render_image_root)
                    _validate_rendered_output(temp_rendered_path, source_path, "forward graph")
            _validate_rendered_output(rendered_path, source_path, "forward graph")
            trace._last_render_geometry = build_render_geometry_record(
                engine=dot.engine,
                layout_path="dot",
                fileformat=target.fileformat,
                output_path=rendered_path,
                stderr_text=layout_stderr,
            )
            if not target.save_only:
                _view_rendered_file(rendered_path)
            _vprint(trace, f"Graph saved to {target.outpath}.{target.fileformat}")
            if os.path.exists(source_path):
                os.remove(source_path)
        except subprocess.TimeoutExpired as error:
            _raise_graphviz_timeout(
                "forward graph",
                f"{trace.num_tensors} nodes",
                source_path,
                render_timeout,
                error,
            )
        except subprocess.CalledProcessError as error:
            _raise_graphviz_failure("forward graph", source_path, error)
    if request.return_graph:
        return dot
    return final_source


@_with_per_draw_collapse_cache
def draw(
    self: "Trace",
    vis_mode: VisModeLiteral = "unrolled",
    vis_call_depth: int = 1000,
    vis_outpath: str = "modelgraph",
    vis_graph_overrides: Optional[Dict[str, Any]] = None,
    module: "Module | str | None" = None,
    node_mode: VisNodeModeLiteral = "default",
    node_spec_fn: NodeSpecFn | None = None,
    collapsed_node_spec_fn: CollapsedNodeSpecFn | None = None,
    collapse_fn: CollapseFn | None = None,
    collapse: CollapseLiteral = "none",
    fold_repeats: FoldRepeatsLiteral = None,
    fold_patterns: object = None,
    skip_fn: SkipFn | None = None,
    vis_edge_overrides: Optional[Dict[str, Any]] = None,
    vis_grad_edge_overrides: Optional[Dict[str, Any]] = None,
    vis_module_overrides: Optional[Dict[str, Any]] = None,
    vis_save_only: bool = False,
    vis_fileformat: str = "pdf",
    show_buffer_layers: BufferVisibilityLiteral = "meaningful",
    direction: VisDirectionLiteral = "bottomup",
    vis_node_placement: VisNodePlacementLiteral = "auto",
    vis_renderer: VisRendererLiteral = "graphviz",
    vis_theme: str = "torchlens",
    vis_intervention_mode: VisInterventionModeLiteral = "node_mark",
    vis_show_cone: bool = True,
    code_panel: CodePanelOption = False,
    node_overlay: "str | OverlayScores | Callable[[Any], Any] | None" = None,
    node_label_fields: list[str] | None = None,
    show_legend: bool | None = None,
    font_size: int | None = None,
    dpi: int | None = None,
    for_paper: bool = False,
    return_graph: bool = False,
    order_siblings: bool = True,
    show_containers: ShowContainersLiteral = False,
    container_max_inline: int = 12,
    show_input_transform_summary: bool = False,
    show_orphans: bool = False,
    *,
    color_by: "str | Callable[[Any], Any] | None" = None,
    size_by: "str | Callable[[Any], Any] | None" = None,
    scale: "str | None" = None,
    stack_by: "str | bool | Callable[[Any], Any] | None" = None,
    show_redundant_args: bool = False,
    show_saved_for_backward: bool = False,
) -> Any:
    """Render the computational graph through the resolved forward IR pipeline.

    Notes
    -----
    ``Trace.draw`` owns the public parameter documentation. This internal
    implementation validates and resolves one request before renderer dispatch.

    Returns
    -------
    Any
        Renderer-specific result, Graphviz graph, or final DOT source.
    """
    _validate_draw_options(node_mode, vis_intervention_mode, collapse, fold_repeats)
    _validate_draw_flag_options(
        show_containers,
        vis_save_only=vis_save_only,
        vis_show_cone=vis_show_cone,
        show_legend=show_legend,
        for_paper=for_paper,
        return_graph=return_graph,
        order_siblings=order_siblings,
        show_input_transform_summary=show_input_transform_summary,
        show_orphans=show_orphans,
    )
    from ._encoding import resolve_color_by, resolve_size_by, resolve_size_scale

    # Option validation for the encoding channels: an unknown source refuses
    # HERE, before any render work (encoding_source_invalid), and scale=
    # without size_by refuses scale_requires_size_by.
    encoding_channel_spec = resolve_color_by(color_by)
    size_channel_spec = resolve_size_by(size_by)
    size_scale = resolve_size_scale(scale, size_by_active=size_channel_spec is not None)
    from ._stacking import resolve_stack_by

    stack_channel_spec = resolve_stack_by(stack_by, vis_mode)
    request = ResolvedRenderRequest(
        vis_mode=vis_mode,
        show_buffer_layers=cast(BufferVisibilityLiteral, show_buffer_layers),
        show_containers=show_containers,
        engine=vis_node_placement,
        skip_fn=skip_fn,
        vis_call_depth=vis_call_depth,
        module=module,
        node_mode=node_mode,
        node_spec_fn=node_spec_fn,
        collapsed_node_spec_fn=collapsed_node_spec_fn,
        collapse_fn=collapse_fn,
        collapse=collapse,
        fold_repeats=fold_repeats,
        fold_patterns=_resolve_fold_patterns(fold_patterns),
        graph_overrides=vis_graph_overrides,
        edge_overrides=vis_edge_overrides,
        grad_edge_overrides=vis_grad_edge_overrides,
        module_overrides=vis_module_overrides,
        overrides=None,
        theme=vis_theme,
        intervention_mode=vis_intervention_mode,
        show_cone=vis_show_cone,
        code_panel=code_panel,
        node_overlay=node_overlay,
        node_label_fields=tuple(node_label_fields) if node_label_fields is not None else None,
        show_legend=show_legend,
        font_size=font_size,
        dpi=dpi,
        for_paper=for_paper,
        return_graph=return_graph,
        order_siblings=order_siblings,
        container_max_inline=container_max_inline,
        show_input_transform_summary=show_input_transform_summary,
        show_orphans=show_orphans,
        direction=direction,
        color_by=color_by,
        size_by=size_by,
        scale=scale,
        stack_by=stack_by,
        show_redundant_args=show_redundant_args,
        show_saved_for_backward=show_saved_for_backward,
    )
    request, theme, site_labels = _resolve_draw_request(self, request)
    if (
        encoding_channel_spec is not None
        or size_channel_spec is not None
        or stack_channel_spec is not None
    ):
        from ._encoding import attach_encoding_state

        request = attach_encoding_state(
            request,
            theme,
            channel_specs=(encoding_channel_spec, size_channel_spec, stack_channel_spec),
            size_scale=size_scale,
        )
    show_buffer_layers = request.show_buffer_layers

    from .renderer_registry import renderer_info, renderer_names

    if vis_renderer not in renderer_names():
        raise InvalidArgumentError(
            f"vis_renderer must be one of {sorted(renderer_names())}; received "
            f"{vis_renderer!r}. Out-of-tree renderers register through "
            "torchlens.visualization.renderer_registry.register_renderer.",
            code="visualization_renderer_invalid",
            remedy="pass a registered renderer name",
            argument="vis_renderer",
        )
    renderer_capability_rows = renderer_info(vis_renderer).capabilities
    if request.encoding is not None and not bool(
        renderer_capability_rows.get("encoding_channels", False)
    ):
        from ._encoding import raise_encoding_dagua_refusal

        raise_encoding_dagua_refusal(request.encoding.active_channels())
    if vis_renderer == "dagua":
        opted_in_module = sys.modules.get("torchlens.experimental.dagua")
        if not getattr(opted_in_module, "__torchlens_dagua_opted_in__", False):
            raise CaptureContextError(
                "dagua renderer is experimental and has not been opted into",
                code="dagua_renderer_not_opted_in",
                remedy="opt in via `from torchlens.experimental import dagua` first",
            )
        from ..experimental.dagua import render_trace_with_dagua

        return render_trace_with_dagua(
            self,
            vis_mode=vis_mode,
            vis_call_depth=vis_call_depth,
            vis_outpath=vis_outpath,
            vis_save_only=vis_save_only,
            vis_fileformat=vis_fileformat,
            vis_buffers=show_buffer_layers == "always",
            vis_direction=direction,
            vis_theme=vis_theme,
        )
    request, repeat_folds, segments, segment_lookup = _resolve_collapse_request(self, request)

    target = RenderTarget(
        outpath=_strip_render_extension(vis_outpath),
        fileformat=vis_fileformat,
        save_only=vis_save_only,
        viewer=not vis_save_only,
        renderer_name=vis_renderer,
    )
    context = _resolve_forward_context(
        self,
        request,
        target,
        theme,
        repeat_folds,
        segments,
        segment_lookup,
        site_labels,
    )
    work = _populate_forward_ir(self, context)

    forward_render_ir = _finalize_forward_ir(self, context, work)
    return _emit_and_finish_forward(self, context, work, forward_render_ir)


def _add_orphan_island_nodes(
    self: "Trace",
    dot: graphviz.Digraph,
    vis_mode: str,
    theme: Any,
) -> None:
    """Render orphan (island) ops as a dashed cluster of disconnected nodes.

    Orphans are ops unreachable from both the model inputs and outputs; they are pruned
    from the main graph and live only on ``trace.orphans`` when
    ``keep_orphans=True``. ``draw(show_orphans=True)`` surfaces them as a
    labelled, dashed cluster of edgeless nodes so a user can SEE the dead-end computation
    without it polluting the connected graph. This is a prototype rendering: nodes carry the
    op label, function, and tensor shape, and are intentionally styled distinctly (dashed,
    greyed) to read as "captured but unreachable".

    Parameters
    ----------
    self:
        Trace whose ``_orphan_logs`` are rendered.
    dot:
        Graphviz graph receiving the orphan cluster.
    vis_mode:
        ``"unrolled"`` or ``"rolled"`` (orphans render identically -- they are raw ops with
        no layer aggregation, having been pruned before labelling).
    theme:
        Active visualization theme (currently unused; reserved for themed orphan styling).
    """
    orphan_logs = tuple(
        op
        for op in getattr(self, "_orphan_logs", ())
        if bool(getattr(op, "is_orphan", False))
        and bool(getattr(op, "label", "") or getattr(op, "_label_raw", ""))
    )
    if not orphan_logs:
        if getattr(self, "_orphan_logs", ()):
            warnings.warn(
                "orphans were dropped from this capture; re-trace with keep_orphans=True",
                TorchLensWarning,
                stacklevel=user_stacklevel(),
            )
        return

    with dot.subgraph(name="cluster_orphans") as orphan_cluster:
        orphan_cluster.attr(
            label="orphans (unreachable from inputs & outputs)",
            style="dashed",
            color="gray70",
            fontcolor="gray50",
            fontsize=DEFAULT_TYPOGRAPHY.secondary_pt,
        )
        for op in orphan_logs:
            base_label = str(getattr(op, "label", "") or getattr(op, "_label_raw", "")).split(
                ":", 1
            )[0]
            if not base_label:
                continue
            shape = getattr(op, "tensor_shape", None)
            shape_text = f"\n{tuple(shape)}" if shape is not None else ""
            orphan_cluster.node(
                f"orphan__{base_label}",
                label=f"{base_label}{shape_text}",
                shape="box",
                style="dashed,filled",
                color="gray60",
                fillcolor="gray95",
                fontcolor="gray40",
            )


def _normalize_backward_pass_filter(bwd: int | Iterable[int] | None) -> BackwardPassFilter:
    """Normalize a backward-pass render filter.

    Parameters
    ----------
    bwd:
        Optional one-based backward pass number or iterable of pass numbers.

    Returns
    -------
    set[int] | None
        Pass numbers to render, or ``None`` for all passes.
    """

    if bwd is None:
        return None
    if isinstance(bwd, int):
        pass_indices = {bwd}
    else:
        pass_indices = {int(pass_index) for pass_index in bwd}
    if any(pass_index < 1 for pass_index in pass_indices):
        raise InvalidArgumentError(
            "bwd pass filters use one-based positive backward pass numbers",
            code="backward_pass_filter_invalid",
            remedy="pass a positive one-based backward pass number",
            argument="bwd",
        )
    return pass_indices


def _rank_layout_cost_inputs(
    trace: "Trace",
    entries_to_plot: Mapping[str, GraphNode],
    edge_map: Mapping[str, Sequence[RenderEdge]],
    *,
    vis_mode: str,
    vis_call_depth: int,
    collapse_fn: CollapseFn | None,
) -> tuple[set[str], list[tuple[str, str]]]:
    """Convert rendered edges into node and edge labels for rank-cost estimation.

    Parameters
    ----------
    trace:
        Owning Trace.
    entries_to_plot:
        Candidate nodes for the current visualization mode.
    edge_map:
        Skip-filtered render edge map.
    vis_mode:
        ``"unrolled"`` or ``"rolled"``.
    vis_call_depth:
        Module nesting depth for collapsed modules.
    collapse_fn:
        Optional user collapse predicate.

    Returns
    -------
    tuple[set[str], list[tuple[str, str]]]
        Render-node labels and directed render edges.
    """

    nodes_by_render_label = {
        _render_node_label(node, vis_mode): node for node in entries_to_plot.values()
    }
    node_labels: set[str] = set()
    edges: list[tuple[str, str]] = []
    for source_label, render_edges in edge_map.items():
        source_node = nodes_by_render_label.get(source_label)
        if source_node is None:
            continue
        source_name = _rank_cost_node_name(
            trace,
            source_node,
            vis_mode=vis_mode,
            vis_call_depth=vis_call_depth,
            collapse_fn=collapse_fn,
        )
        node_labels.add(source_name)
        for render_edge in render_edges:
            target_name = _rank_cost_node_name(
                trace,
                render_edge.target,
                vis_mode=vis_mode,
                vis_call_depth=vis_call_depth,
                collapse_fn=collapse_fn,
            )
            node_labels.add(target_name)
            if source_name != target_name:
                edges.append((source_name, target_name))
    return node_labels, edges


def _rank_cost_node_name(
    trace: "Trace",
    node: GraphNode,
    *,
    vis_mode: str,
    vis_call_depth: int,
    collapse_fn: CollapseFn | None,
) -> str:
    """Return the render node name used for rank-cost estimation.

    Parameters
    ----------
    trace:
        Owning Trace.
    node:
        Render node.
    vis_mode:
        ``"unrolled"`` or ``"rolled"``.
    vis_call_depth:
        Module nesting depth for collapsed modules.
    collapse_fn:
        Optional user collapse predicate.

    Returns
    -------
    str
        Name matching the final rendered node after collapse decisions.
    """

    collapse_address = _collapse_address_for_node(
        trace,
        node,
        collapse_fn=collapse_fn,
        max_module_depth=vis_call_depth,
    )
    if collapse_address is None:
        return node.layer_label.replace(":", "pass")
    parts = collapse_address.rsplit(":", 1)
    return "pass".join(parts) if vis_mode == "unrolled" else parts[0]


def _is_collapsed_module(
    node: GraphNode,
    vis_call_depth: int,
    trace: Optional["Trace"] = None,
    collapse_fn: CollapseFn | None = None,
) -> bool:
    """THE IndexError guard for collapsed module rendering.

    Returns True if the node is nested deep enough to be rendered as a
    collapsed ``box3d`` module summary node instead of an individual layer.

    This function is the single decision point that determines whether a node
    gets its own graphviz node or is absorbed into a module box.  Getting this
    wrong causes IndexError when ``_build_collapsed_module_node`` tries to
    access ``modules[vis_call_depth - 1]``.

    Special cases:
    - ``vis_call_depth == 0``: show all layers, never collapse (#94).
    - ``is_atomic_module``: the node represents the output of
      its innermost module, so its effective nesting depth is one less (it
      visually "belongs" to the parent scope).

    Args:
        node: The Op or Layer node to check.
        vis_call_depth: Maximum nesting depth before collapsing into a module box.
    """
    if trace is not None:
        return (
            _collapse_address_for_node(
                trace,
                node,
                vis_mode="unrolled",
                collapse_fn=collapse_fn,
                max_module_depth=vis_call_depth,
            )
            is not None
        )
    if vis_call_depth == 0:
        return False  # #94: depth 0 means show all layers, never collapse

    node_call_depth = len(node.modules)
    # Bottom-level submodule outputs are rendered at the parent nesting level,
    # not their own. Top-level atomic leaves have no module parent to bubble
    # up to, so they remain eligible for top-level collapse.
    if getattr(node, "is_atomic_module", False) and node_call_depth > 1:
        node_call_depth -= 1

    return node_call_depth >= vis_call_depth


def _run_fold_for_graph_node_name(
    graph_node_name: str,
    repeat_folds: Mapping[str, "ModuleRepeatFold"] | None,
    vis_mode: str,
) -> "ModuleRepeatFold | None":
    """Return the fold represented by ``graph_node_name``.

    Parameters
    ----------
    graph_node_name:
        Rendered Graphviz node identifier.
    repeat_folds:
        Fold descriptors keyed by pass-free module address.
    vis_mode:
        ``"unrolled"`` or ``"rolled"`` visualization mode.

    Returns
    -------
    ModuleRepeatFold | None
        Matching fold, or ``None`` when ``graph_node_name`` is not a folded representative.
    """

    if not repeat_folds:
        return None
    for fold in _unique_repeat_folds(repeat_folds):
        representative_name = _run_fold_graph_node_name(
            f"{fold.representative}:1",
            vis_mode,
            {fold.representative: fold},
        )
        if representative_name == graph_node_name:
            return fold
    return None


__all__ = [
    "_inline_svg_file_local_images",
    "_inline_svg_local_images",
    "_is_collapsed_module",
    "_layout_dot_plain",
    "_normalize_backward_pass_filter",
    "_normalize_svg_root_viewbox",
    "_queue_container_clusters",
    "_queue_sibling_rank_group",
    "_raise_graphviz_failure",
    "_raise_graphviz_timeout",
    "_rank_cost_node_name",
    "_rank_layout_cost_inputs",
    "_render_graph_only_svg",
    "_resolve_focus_module",
    "_run_fold_for_graph_node_name",
    "_setup_combined_special_clusters",
    "_setup_subgraphs",
    "_setup_subgraphs_recurse",
    "_should_order_siblings",
    "_sibling_chain_key",
    "_sibling_chain_stretch_ratio",
    "_sibling_order_decision",
    "_strict_sibling_order_checks_enabled",
    "_strip_render_extension",
    "_strip_sibling_rank_groups",
    "_validate_rendered_output",
    "_verify_and_apply_sibling_ordering",
    "_view_rendered_file",
    "_warn_sibling_order_fallback_once",
    "_write_composed_code_panel",
    "draw",
]
