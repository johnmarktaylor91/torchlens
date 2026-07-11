"""DOT orchestration and SVG composition helpers for Graphviz rendering."""

# ruff: noqa: F403, F405

import warnings
from dataclasses import replace

from ._render_common import *
from ._render_leaf import *
from ._render_edges import *
from ._render_nodes import *
from ._render_flow import *
from ._render_utils import html_escape
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
    skip_fn: SkipFn | None = None,
    vis_edge_overrides: Optional[Dict[str, Any]] = None,
    vis_grad_edge_overrides: Optional[Dict[str, Any]] = None,
    vis_module_overrides: Optional[Dict[str, Any]] = None,
    vis_save_only: bool = False,
    vis_fileformat: str = "pdf",
    show_buffer_layers: BufferVisibilityLiteral | bool = "meaningful",
    direction: VisDirectionLiteral = "bottomup",
    vis_node_placement: VisNodePlacementLiteral = "auto",
    vis_renderer: VisRendererLiteral = "graphviz",
    vis_theme: str = "torchlens",
    vis_intervention_mode: VisInterventionModeLiteral = "node_mark",
    vis_show_cone: bool = True,
    code_panel: CodePanelOption = False,
    node_overlay: "str | OverlayScores | Callable[[Any], Any] | None" = None,
    node_label_fields: list[str] | None = None,
    show_legend: bool = False,
    font_size: int | None = None,
    dpi: int | None = None,
    for_paper: bool = False,
    return_graph: bool = False,
    order_siblings: bool = True,
    show_containers: ShowContainersLiteral = False,
    container_max_inline: int = 12,
    show_input_transform_summary: bool = False,
    show_orphans: bool = False,
) -> Any:
    """Render the computational graph as a Graphviz Digraph.

    Orchestrates the full rendering pipeline:
    1. Validates that all layers are logged (``_layers_logged`` guard).
    2. Iterates over entries_to_plot, building nodes and edges.
    3. Groups edges into module subgraph clusters.
    4. Renders to file and optionally displays.

    Args:
        vis_mode: ``'unrolled'`` (each pass is a separate node) or ``'rolled'``
            (multi-pass layers collapsed into one node with pass annotations).
        vis_call_depth: Maximum module nesting levels to show before
            collapsing deeper layers into ``box3d`` module summary nodes.
            Use 0 to show all layers without collapsing.
        vis_outpath: Output file path (extension auto-stripped).
        vis_graph_overrides: Graphviz graph-level attribute overrides.
        module: Optional module focus. A Module focuses that module; a string
            is interpreted as a module address.
        node_mode: Preset applied to default ``NodeSpec`` objects before
            user callbacks run.
        node_spec_fn: Optional callback receiving ``(layer_log, default_spec)``.
            In unrolled mode, ``layer_log`` is the parent aggregate Layer for
            the rendered Op.
        collapsed_node_spec_fn: Optional callback receiving
            ``(module_log, default_spec)`` for collapsed module nodes.
        collapse_fn: Optional predicate receiving a Module. When provided,
            it replaces ``vis_call_depth`` collapse decisions.
        collapse: Smart module-collapse mode. ``"none"`` preserves existing
            rendering, ``"auto"`` targets a readable overview, and ``"max"``
            aggressively collapses eligible modules. The v2 engine supports
            rolled and unrolled rendering, may emit segment boxes in ``"max"``,
            and uses honest labels for ``(xN)`` collapsed calls, ellipsis
            repeat-folds, and segment summaries. A float in ``[0.0, 1.0]`` selects
            the public monotone collapse schedule: ``0.0`` is equivalent to
            ``"none"``, ``1.0`` is equivalent to ``"max"``, and larger values
            never increase the visible node count or uncollapse a collapsed
            unit. ``"auto"`` is the schedule point where the visible count first
            enters the readable band; the existing ``"auto"`` implementation is
            unchanged for compatibility.
        fold_repeats: Repeat-fold policy. ``None`` preserves the collapse mode
            default: off for ``collapse="none"`` and band-pressure two-pass
            folding for ``"auto"``/``"max"``. ``True`` folds every eligible
            repeated run, including standalone run folding when
            ``collapse="none"``. ``False`` disables run folding.
        skip_fn: Optional predicate receiving a Layer. Skipped nodes are
            elided and edges are chained through them.
        vis_edge_overrides: Overrides for forward edges.
        vis_grad_edge_overrides: Overrides for backward (grad) edges.
        vis_module_overrides: Overrides for module subgraph boxes.
        vis_save_only: If True, save without opening a viewer.
        vis_fileformat: Output format (pdf, png, svg, etc.).
        show_buffer_layers: Buffer visibility mode. ``"never"`` hides all
            buffers, ``"meaningful"`` hides hardcoded BatchNorm running-stat
            noise buffers, and ``"always"`` shows all buffers. Legacy bools are
            deprecated but supported: ``True`` maps to ``"always"`` and
            ``False`` maps to ``"never"``.
        direction: Layout direction: ``'bottomup'``, ``'topdown'``, or ``'leftright'``.
        vis_node_placement: Layout engine: ``'auto'`` (default), ``'dot'``,
            or ``'rank'``.
        vis_renderer: Renderer backend: ``'graphviz'`` or experimental
            ``'dagua'``. Import ``torchlens.experimental.dagua`` before using
            the Dagua renderer.
        vis_theme: Renderer theme name for backends that support themes.
        vis_intervention_mode: Intervention overlay mode. ``"node_mark"``
            marks sites and cones; ``"as_node"`` inserts hook nodes after
            intervention sites.
        vis_show_cone: Whether ``"node_mark"`` mode marks downstream cone
            members.
        code_panel: Optional source-code panel. ``True`` is equivalent to
            ``"forward"``; callable values receive the live model object when
            it is still available.
        node_overlay: Built-in overlay name or external mapping from node label
            to score. Supported built-ins include ``"flops"``, ``"time"``,
            ``"bytes"``, ``"magnitude"``, ``"grad_norm"``, ``"nan"``,
            ``"intervention"``, and ``"bundle_delta"``.
        node_label_fields: Optional label field picker. When omitted, the
            default TorchLens label rows are used.
        show_legend: Whether to render a compact colorblind-safe legend with
            the graph.
        font_size: Optional Graphviz font size.
        dpi: Optional Graphviz output DPI.
        for_paper: Whether to force the paper theme preset.
        return_graph: If True, return the underlying ``graphviz.Digraph`` on
            the Graphviz path or DOT text for direct text renderers.
        order_siblings: Whether Graphviz ``dot`` renders should add verified invisible
            rank constraints so true parallel sibling fanouts follow execution order.
        show_containers: Optional output-container overlay. ``False`` preserves
            the default render. ``"labels"`` adds midpoint key/index labels on
            container leaf edges. ``"cluster"`` also clusters single-owner
            containers. ``"collapsed"`` and ``"auto"`` collapse large
            homogeneous containers to one summary node. ``"nodes"`` adds
            collapsed labeled container nodes for source/sink boundaries and
            dashed member-of ties for mid-graph output containers.
        container_max_inline: Maximum homogeneous container leaves to inline in
            ``"collapsed"``/``"auto"`` modes.
        show_input_transform_summary: Whether to show input preprocessing
            provenance next to the raw input node when available. Defaults to
            ``False`` to preserve existing raw-input node rendering.

    Returns:
        The Graphviz DOT source string.

    Raises:
        ValueError: If ``_layers_logged`` is False (layers were discarded
            by missing final lookup containers).
    """
    if node_mode not in MODE_REGISTRY:
        raise ValueError(
            "Visualization node_style/node_mode must be one of 'default', "
            "'profiling', 'vision', or 'attention'."
        )
    if node_mode in DOMAIN_NODE_MODES:
        warnings.warn(
            f"node_style={node_mode!r} is moving out of core; use the equivalent "
            f"recipe at examples/recipes/{node_mode}.py or wait for the "
            f"torchlens.{node_mode} plugin",
            DeprecationWarning,
            stacklevel=2,
        )
    if vis_intervention_mode not in {"node_mark", "as_node"}:
        raise ValueError("vis_intervention_mode must be either 'node_mark' or 'as_node'.")
    if isinstance(collapse, float):
        if not 0.0 <= collapse <= 1.0:
            raise ValueError("collapse float level must be in [0.0, 1.0].")
    elif collapse not in {"none", "auto", "max"}:
        raise ValueError("collapse must be 'none', 'auto', 'max', or a float in [0.0, 1.0].")
    if fold_repeats not in {None, True, False}:
        raise ValueError("fold_repeats must be None, True, or False.")
    show_buffer_layers = _normalize_buffer_visibility(show_buffer_layers)
    theme = resolve_theme(vis_theme, for_paper=for_paper)
    if node_overlay is None:
        node_overlay = getattr(self, "_node_overlay_scores", None)
    elif isinstance(node_overlay, str) and node_overlay == getattr(
        self, "_node_overlay_name", None
    ):
        node_overlay = getattr(self, "_node_overlay_scores", None)
    resolved_node_overlay = cast("str | OverlayScores | None", node_overlay)
    overrides = VisualizationOverrides(
        graph=graphviz_graph_overrides(vis_graph_overrides),
        edge=vis_edge_overrides or {},
        grad_edge=vis_grad_edge_overrides or {},
        module=vis_module_overrides or {},
    )
    request = ResolvedRenderRequest(
        vis_mode=vis_mode,
        show_buffer_layers=show_buffer_layers,
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
        graph_overrides=vis_graph_overrides,
        edge_overrides=vis_edge_overrides,
        grad_edge_overrides=vis_grad_edge_overrides,
        module_overrides=vis_module_overrides,
        overrides=overrides,
        theme=vis_theme,
        intervention_mode=vis_intervention_mode,
        show_cone=vis_show_cone,
        code_panel=code_panel,
        node_overlay=resolved_node_overlay,
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
    )
    site_labels, _ = intervention_site_and_cone_labels(self, show_cone=vis_show_cone)
    intervention_node_spec_fn = make_intervention_node_spec_fn(
        self,
        show_cone=vis_show_cone,
        graph_overrides=vis_graph_overrides,
        user_node_spec_fn=node_spec_fn,
    )
    request = replace(request, node_spec_fn=intervention_node_spec_fn)

    if vis_renderer == "dagua":
        opted_in_module = sys.modules.get("torchlens.experimental.dagua")
        if not getattr(opted_in_module, "__torchlens_dagua_opted_in__", False):
            raise RuntimeError(
                "dagua renderer is experimental; opt in via "
                "`from torchlens.experimental import dagua` first"
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
    if vis_renderer not in {"graphviz", "dagua"}:
        raise ValueError("vis_renderer must be 'graphviz' or 'dagua'")
    render_context = request
    if collapse != "none" and collapse_fn is None:
        from .auto_collapse import resolve_collapse_fn

        collapse_fn = resolve_collapse_fn(self, collapse, vis_mode, context=render_context)
    request = request.with_resolved_collapse(collapse_fn)
    render_context = request
    repeat_folds: dict[str, ModuleRepeatFold] = {}
    collapse_uses_default_folds = collapse in {"auto", "max"} or (
        isinstance(collapse, float) and collapse > 0.0
    )
    if fold_repeats is not False and (fold_repeats is True or collapse_uses_default_folds):
        from .auto_collapse import resolve_repeat_folds

        repeat_folds = resolve_repeat_folds(
            self,
            collapse_fn,
            context=render_context,
            fold_repeats=fold_repeats,
        )
    segments: dict[str, SegmentDescriptor] = {}
    if collapse_fn is not None:
        segments = dict(getattr(collapse_fn, "_torchlens_v2_segments", {}) or {})
    # THE _layers_logged guard: protects all downstream rendering code from missing-layer lookups.
    if not self._layers_logged:
        raise ValueError(
            "Must have all layers logged in order to render the graph; use show_model_graph."
        )

    target = RenderTarget(
        outpath=_strip_render_extension(vis_outpath),
        fileformat=vis_fileformat,
        save_only=vis_save_only,
        viewer=not vis_save_only,
        renderer_name=vis_renderer,
    )
    vis_outpath = target.outpath
    vis_fileformat = target.fileformat
    vis_save_only = target.save_only
    vis_renderer = target.renderer_name
    source_graph = build_source_graph(self, request)
    from .node_universe import build_node_universe

    node_universe = build_node_universe(
        source_graph,
        collapse_fn,
        repeat_folds,
        segments,
        show_containers,
    )
    entries_to_plot = source_graph.entries_to_plot

    rankdir = direction_to_rankdir(direction)

    # Resolve the layout engine early to potentially skip graphviz.Digraph construction.
    from ._rank_layout_internal.layout import (
        RANK_LAYOUT_COST_THRESHOLD,
        RANK_LAYOUT_NOTICE,
        estimate_rank_layout_cost,
        get_node_placement_engine,
    )

    edge_map = source_graph.edge_map
    skipped_labels = source_graph.skipped_labels
    source_text = resolve_code_panel_source(
        code_panel,
        getattr(self, "_source_code_blob", {}),
        getattr(self, "_source_model_ref", None),
    )
    num_nodes = len(entries_to_plot) - len(skipped_labels)
    cost_node_labels, cost_edges = _rank_layout_cost_inputs(
        self,
        entries_to_plot,
        edge_map,
        vis_mode=vis_mode,
        vis_call_depth=vis_call_depth,
        collapse_fn=collapse_fn,
    )
    layout_cost = estimate_rank_layout_cost(cost_node_labels, cost_edges)
    engine = get_node_placement_engine(vis_node_placement, layout_cost)
    if show_containers:
        engine = "dot"
    # The sibling-ordering post-pass only runs on the dot engine; set the
    # trivial decision up front so the attribute exists on every path.
    self._last_sibling_ordering_decision = SiblingOrderDecision(0, 0, {}, ())
    if vis_node_placement == "auto" and engine == "rank":
        warnings.warn(
            RANK_LAYOUT_NOTICE.format(
                cost=layout_cost,
                threshold=RANK_LAYOUT_COST_THRESHOLD,
            )
        )
    _vprint(self, f"Rendering {vis_mode} graph ({num_nodes} nodes, format={vis_fileformat})")
    _vprint(self, f"Layout engine: {engine} (estimated cost={layout_cost})")

    if self.num_params == 0:
        params_detail = "0 params"
    elif self.num_params_frozen == 0:
        params_detail = f"{self.num_params} params (all trainable, {self.total_param_memory})"
    elif self.num_params_trainable == 0:
        params_detail = f"{self.num_params} params (all frozen, {self.total_param_memory})"
    else:
        params_detail = (
            f"{self.num_params} params "
            f"({self.num_params_trainable}/{self.num_params} trainable, "
            f"{self.total_param_memory})"
        )

    graph_caption = (
        f"<<FONT COLOR='{theme.default_font}'><B>{html_escape(self.model_class_name)}</B>"
        f"<br align='left'/>{self.num_tensors} tensors total ({self.total_activation_memory})"
        f"<br align='left'/>{params_detail}<br align='left'/></FONT>>"
    )
    if getattr(self, "_has_direct_writes", False):
        graph_caption = graph_caption[:-2] + (
            "Direct writes detected - recipe propagation will overlay<br align='left'/>>"
        )

    # Rank fast path: skip graphviz.Digraph construction entirely.
    # Generates DOT directly with topological-rank positions and cluster boxes.
    if engine == "rank":
        from ._rank_layout_internal.layout import render_rank_layout

        with _timed_phase(self, "render:graphviz:forward"):
            result = render_rank_layout(
                self,
                entries_to_plot,
                vis_mode,
                vis_call_depth,
                show_buffer_layers,
                overrides,
                node_mode,
                intervention_node_spec_fn,
                collapsed_node_spec_fn,
                collapse_fn,
                skip_fn,
                edge_map,
                skipped_labels,
                vis_outpath,
                vis_fileformat,
                vis_save_only,
                graph_caption,
                rankdir,
                source_text,
            )
        _vprint(self, f"Graph saved to {vis_outpath}.{vis_fileformat}")
        return result

    dot = graphviz.Digraph(
        name=self.model_class_name,
        comment="Computational graph for the feedforward sweep",
        format=vis_fileformat,
    )

    graph_args = {
        "rankdir": rankdir,
        "label": graph_caption,
        "labelloc": "t",
        "labeljust": "left",
        "ordering": "out",
    }
    if collapse_fn is not None:
        graph_args["newrank"] = "true"
    graph_args.update(theme_graph_attrs(theme, font_size=font_size, dpi=dpi))

    # Override system: callers can pass dicts of Graphviz attributes to
    # customize rendering.  Values can be static (str) or dynamic (callable
    # receiving the Trace, evaluated at render time).
    for arg_name, arg_val in overrides.graph.items():  # type: ignore[union-attr]
        if callable(arg_val):
            graph_args[arg_name] = str(arg_val(self))
        else:
            graph_args[arg_name] = str(arg_val)

    dot.graph_attr.update(graph_args)
    dot.node_attr.update({"ordering": "out", **theme_node_attrs(theme, font_size=font_size)})
    dot.edge_attr.update(theme_edge_attrs(theme, font_size=font_size))
    forward_dot_recorder = _ForwardDotRecorder()

    # Accumulate edges per module cluster; actual Graphviz subgraphs are
    # created at the end in _setup_subgraphs to ensure proper nesting.
    module_cluster_dict: Dict[str, Any] = defaultdict(
        lambda: {
            "edges": [],
            "has_input_ancestor": False,
            "rank_groups": [],
            "container_clusters": [],
        }
    )
    top_level_sibling_rank_groups: list[SiblingOrderChain] = []
    # Track which collapsed module nodes have been added to avoid duplicates
    # (multiple layers in the same collapsed module would otherwise each try
    # to create the same box3d node).
    collapsed_modules: Set[str] = set()
    # Edge deduplication: (tail_name, head_name) pairs already added.
    # Critical when collapsed modules cause many layers to map to the same
    # node name -- without this, we'd get duplicate edges.
    edges_used: Set[tuple[str, str, tuple[Any, ...]]] = set()
    run_fold_ellipsis_nodes: set[str] = set()
    emitted_segment_nodes: set[str] = set()
    captured_forward_edges: list[CapturedForwardEdge] = []
    pending_container_collapse_nodes: list[dict[str, Any]] = []
    container_clusters: list[ContainerClusterSpec] = []
    collapsed_container_nodes = _collapsed_container_leaf_nodes(
        self,
        entries_to_plot,
        vis_mode=vis_mode,
        show_containers=show_containers,
        container_max_inline=container_max_inline,
        pending_nodes=pending_container_collapse_nodes,
    )
    forward_render_ir = build_render_ir(
        self,
        collapse_fn=collapse_fn,
        repeat_folds=repeat_folds,
        context=request,
        universe=node_universe,
        segments=segments,
    )
    antiparallel_projected_edges = projected_antiparallel_endpoint_pairs(forward_render_ir)

    decisions_by_name = {node.name: node for node in forward_render_ir.nodes}
    for unit in node_universe.units:
        node_record = decisions_by_name[unit.unit_id]
        for source_index, node in enumerate(unit.source_nodes):
            _add_node_to_graphviz(
                self,
                node,
                cast(graphviz.Digraph, forward_dot_recorder),
                module_cluster_dict,
                edges_used,
                vis_mode,
                collapsed_modules,
                vis_call_depth,
                show_buffer_layers,
                overrides,
                node_mode,
                intervention_node_spec_fn,
                collapsed_node_spec_fn,
                collapse_fn,
                edge_map,
                vis_intervention_mode,
                site_labels,
                theme,
                resolved_node_overlay,
                node_label_fields,
                captured_forward_edges,
                rankdir,
                show_containers,
                collapsed_container_nodes,
                show_input_transform_summary,
                repeat_folds,
                run_fold_ellipsis_nodes,
                segments,
                emitted_segment_nodes,
                antiparallel_projected_edges,
                node_record
                if source_index == 0
                else replace(node_record, node_calls=(), owned_node_args=()),
            )

    for node_args in pending_container_collapse_nodes:
        forward_dot_recorder.node(**node_args)

    container_overlay_edges: list[ContainerOverlayEdge] = []
    if show_containers == "nodes" and vis_mode == "unrolled":
        dot.graph_attr.update({"pad": "0.20"})
        container_overlay_nodes, container_overlay_edges = _container_nodes_and_overlay_edges(
            self,
            collapsed_container_nodes,
            vis_call_depth=vis_call_depth,
            collapse_fn=collapse_fn,
        )
        for overlay_node in container_overlay_nodes:
            if overlay_node.owner_key is None:
                forward_dot_recorder.node(**overlay_node.args)
            else:
                module_cluster_dict[overlay_node.owner_key].setdefault("nodes", []).append(
                    overlay_node.args
                )

    if show_containers in {"cluster", "nodes"}:
        container_clusters = _container_clusters_for_graphviz(
            self,
            entries_to_plot,
            vis_mode=vis_mode,
            vis_call_depth=vis_call_depth,
            collapse_fn=collapse_fn,
            collapsed_container_nodes=collapsed_container_nodes,
        )
        _queue_container_clusters(module_cluster_dict, container_clusters)

    if vis_intervention_mode == "as_node":
        _add_intervention_hook_nodes(
            cast(graphviz.Digraph, forward_dot_recorder),
            site_labels,
            vis_graph_overrides,
        )

    sibling_order_chains: tuple[SiblingOrderChain, ...] = ()
    if _should_order_siblings(
        order_siblings=order_siblings,
        engine=engine,
        vis_mode=vis_mode,
        num_nodes=num_nodes,
        module=module,
        vis_intervention_mode=vis_intervention_mode,
        collapse_fn=collapse_fn,
        vis_call_depth=vis_call_depth,
    ):
        sibling_order_chains = _build_sibling_order_chains(captured_forward_edges)
        if sibling_order_chains:
            for chain in sibling_order_chains:
                _queue_sibling_rank_group(
                    module_cluster_dict,
                    top_level_sibling_rank_groups,
                    chain,
                )

    forward_dot_ir = ForwardDotIR(
        render_ir=forward_render_ir,
        calls=tuple(forward_dot_recorder.calls),
        module_cluster_dict=module_cluster_dict,
        top_level_sibling_rank_groups=tuple(top_level_sibling_rank_groups),
        captured_forward_edges=tuple(captured_forward_edges),
        container_overlay_edges=tuple(container_overlay_edges),
    )
    _replay_forward_dot_calls(dot, forward_dot_ir.calls)

    # Finally, set up the subgraphs.
    _setup_subgraphs(
        self,
        dot,
        vis_mode,
        forward_dot_ir.module_cluster_dict,
        overrides,
        list(forward_dot_ir.top_level_sibling_rank_groups),
    )
    for overlay_edge in forward_dot_ir.container_overlay_edges:
        dot.edge(
            tail_name=overlay_edge.tail_name,
            head_name=overlay_edge.head_name,
            **overlay_edge.attrs,
        )
    if show_orphans:
        _add_orphan_island_nodes(self, dot, vis_mode, theme)
    if show_legend:
        _add_legend_to_graphviz(dot, theme)
    # A code panel is composed side by side (separate render) when the output
    # format supports it, so the code never distorts the graph's layout. Only
    # fall back to an in-graph subgraph for formats we cannot compose.
    compose_code_panel = source_text is not None and _code_panel_composition_available(
        vis_fileformat, engine
    )
    if source_text is not None and not compose_code_panel:
        render_code_panel_subgraph(dot, source_text)

    if in_notebook() and not vis_save_only:
        try:
            from IPython.display import SVG, display  # #72: lazy import
        except ImportError as e:
            raise ImportError(
                "IPython is required for this feature. Install with "
                "`pip install torchlens[notebook]`."
            ) from e

        display_fn = cast(Any, display)
        if compose_code_panel:
            # Compose the graph SVG beside a standalone code panel so the inline
            # preview matches the saved output and leaves the graph undistorted.
            graph_svg = _inline_svg_local_images(dot.pipe(format="svg").decode("utf-8"))
            combined_svg = compose_graph_with_code_panel(graph_svg, cast(str, source_text))
            display_fn(SVG(combined_svg))
        else:
            display_fn(dot)

    # Rank was already handled above (early return). Only dot reaches here.
    _RENDER_TIMEOUT = 120  # seconds
    source_override = None
    self._last_sibling_ordering_decision = SiblingOrderDecision(0, 0, {}, ())
    if sibling_order_chains:
        try:
            source_override, decision = _verify_and_apply_sibling_ordering(
                dot.source,
                sibling_order_chains,
                captured_forward_edges,
                rankdir,
            )
            self._last_sibling_ordering_decision = decision
        except (subprocess.SubprocessError, OSError) as exc:
            if _strict_sibling_order_checks_enabled():
                raise
            _warn_sibling_order_fallback_once(exc)

    final_source = source_override if source_override is not None else dot.source
    source_path = dot.save(vis_outpath)
    with open(source_path, "w", encoding="utf-8") as source_file:
        source_file.write(final_source)
    with _timed_phase(self, "render:graphviz:forward"):
        try:
            # dot engine (default for local-topology graphs)
            rendered_path = f"{vis_outpath}.{vis_fileformat}"
            if compose_code_panel:
                _write_composed_code_panel(
                    dot.engine,
                    source_path,
                    cast(str, source_text),
                    rendered_path,
                    vis_fileformat,
                    _RENDER_TIMEOUT,
                )
            else:
                cmd = [dot.engine, f"-T{vis_fileformat}", "-o", rendered_path, source_path]
                subprocess.run(cmd, timeout=_RENDER_TIMEOUT, check=True, capture_output=True)
                if vis_fileformat == "svg":
                    _inline_svg_file_local_images(rendered_path)
            _validate_rendered_output(rendered_path, source_path, "forward graph")
            if not vis_save_only:
                _view_rendered_file(rendered_path)
            _vprint(self, f"Graph saved to {vis_outpath}.{vis_fileformat}")
            # Success: remove the intermediate DOT source. On FAILURE we keep it so
            # the error's "DOT source was saved to ..." hint points to a real file.
            if os.path.exists(source_path):
                os.remove(source_path)
        except subprocess.TimeoutExpired as e:
            _raise_graphviz_timeout(
                "forward graph",
                f"{self.num_tensors} nodes",
                source_path,
                _RENDER_TIMEOUT,
                e,
            )
        except subprocess.CalledProcessError as e:
            _raise_graphviz_failure("forward graph", source_path, e)
    if return_graph:
        return dot
    return final_source


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
                UserWarning,
                stacklevel=2,
            )
        return

    with dot.subgraph(name="cluster_orphans") as orphan_cluster:
        orphan_cluster.attr(
            label="orphans (unreachable from inputs & outputs)",
            style="dashed",
            color="gray70",
            fontcolor="gray50",
            fontsize="10",
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


def _replay_forward_dot_calls(dot: graphviz.Digraph, calls: Sequence[ForwardDotCall]) -> None:
    """Replay recorded forward DOT calls into a Graphviz graph.

    Parameters
    ----------
    dot:
        Graphviz graph receiving the recorded calls.
    calls:
        Forward DOT calls recorded by the render IR builder.
    """

    for call in calls:
        if call.kind == "node":
            dot.node(*call.args, **call.kwargs)
        elif call.kind == "edge":
            dot.edge(*call.args, **call.kwargs)
        elif call.kind == "attr":
            dot.attr(*call.args, **call.kwargs)
        elif call.kind == "subgraph":
            with dot.subgraph(*call.args, **call.kwargs) as subgraph:
                _replay_forward_dot_calls(subgraph, call.children)
        else:
            raise ValueError(f"Unknown forward DOT call kind: {call.kind}")


def _render_graph_only_svg(engine: str, source_path: str, timeout: int) -> str:
    """Render a saved DOT source to an SVG string (no code panel)."""

    completed = subprocess.run(
        [engine, "-Tsvg", source_path],
        timeout=timeout,
        check=True,
        capture_output=True,
    )
    return _inline_svg_local_images(completed.stdout.decode("utf-8"))


def _inline_svg_file_local_images(svg_path: str) -> None:
    """Inline local image hrefs in a saved SVG file.

    Parameters
    ----------
    svg_path:
        Path to the rendered SVG file to update in place.
    """

    with open(svg_path, encoding="utf-8") as svg_file:
        svg_text = svg_file.read()
    inlined_svg = _inline_svg_local_images(svg_text)
    if inlined_svg != svg_text:
        with open(svg_path, "w", encoding="utf-8") as svg_file:
            svg_file.write(inlined_svg)


def _normalize_svg_root_viewbox(svg_text: str) -> str:
    """Shift a root SVG with a negative origin onto a positive page.

    Parameters
    ----------
    svg_text:
        SVG text to normalize.

    Returns
    -------
    str
        SVG text whose root viewBox starts at ``0 0``. The drawn region is
        translated by the opposite offset so PDF/PNG rasterizers do not place
        large Graphviz layouts outside the page.
    """

    root_match = _SVG_ROOT_RE.search(svg_text)
    if root_match is None:
        return svg_text
    attrs = root_match.group("attrs")
    viewbox_match = _SVG_VIEWBOX_RE.search(attrs)
    if viewbox_match is None:
        return svg_text
    try:
        min_x, min_y, width, height = (
            float(part) for part in viewbox_match.group("value").replace(",", " ").split()
        )
    except ValueError:
        return svg_text
    if min_x == 0.0 and min_y == 0.0:
        return svg_text

    normalized_viewbox = f'viewBox="0 0 {width:.2f} {height:.2f}"'
    new_attrs = attrs[: viewbox_match.start()] + normalized_viewbox + attrs[viewbox_match.end() :]
    inner_start = root_match.end()
    inner_end = svg_text.rfind("</svg>")
    if inner_end == -1:
        return svg_text
    translate_x = -min_x
    translate_y = -min_y
    return (
        svg_text[: root_match.start()]
        + f"<svg{new_attrs}>"
        + f'<g transform="translate({translate_x:.2f} {translate_y:.2f})">'
        + svg_text[inner_start:inner_end]
        + "</g>"
        + svg_text[inner_end:]
    )


def _inline_svg_local_images(svg_text: str) -> str:
    """Replace local SVG image references with embedded data URIs.

    Parameters
    ----------
    svg_text:
        SVG text produced by Graphviz.

    Returns
    -------
    str
        SVG text with existing local image hrefs inlined. Non-file hrefs are
        left untouched. Missing or unreadable local files are replaced with a
        short placeholder so the node is not an empty image box.
    """

    def replace_image_tag(match: re.Match[str]) -> str:
        """Return an updated SVG image tag or placeholder text."""

        tag = match.group(0)
        attrs = _svg_attrs_to_dict(match.group("attrs"))
        href_attr = "xlink:href" if "xlink:href" in attrs else "href"
        href = attrs.get(href_attr)
        if href is None or _is_non_file_svg_href(href):
            return tag
        image_path = _resolve_svg_image_path(href)
        try:
            payload = image_path.read_bytes()
        except OSError:
            return _svg_image_placeholder(attrs)
        mime_type = _svg_image_mime_type(image_path)
        data_uri = f"data:{mime_type};base64,{base64.b64encode(payload).decode('ascii')}"
        return _replace_svg_attr_value(tag, href_attr, data_uri)

    return _SVG_IMAGE_TAG_RE.sub(replace_image_tag, svg_text)


def _write_composed_code_panel(
    engine: str,
    source_path: str,
    source_text: str,
    rendered_path: str,
    file_format: str,
    timeout: int,
) -> None:
    """Render the graph and code panel separately and write the joined output.

    The graph is rendered to SVG without any code subgraph, composed beside a
    standalone code-panel SVG, then written in ``file_format``. SVG is written
    directly; PDF and PNG are converted from the composed SVG with ``cairosvg``
    so vectors are preserved.
    """

    graph_svg = _normalize_svg_root_viewbox(_render_graph_only_svg(engine, source_path, timeout))
    combined_svg = _normalize_svg_root_viewbox(
        compose_graph_with_code_panel(graph_svg, source_text)
    )
    if file_format == "svg":
        with open(rendered_path, "w", encoding="utf-8") as svg_file:
            svg_file.write(combined_svg)
        return
    import cairosvg

    svg_bytes = combined_svg.encode("utf-8")
    if file_format == "pdf":
        cairosvg.svg2pdf(bytestring=svg_bytes, write_to=rendered_path)
    else:  # png, rendered at 2x for crispness
        cairosvg.svg2png(bytestring=svg_bytes, write_to=rendered_path, scale=2.0)


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
        raise ValueError("bwd pass filters use one-based positive backward pass numbers.")
    return pass_indices


def _setup_combined_special_clusters(
    graphviz_graph: graphviz.Digraph,
    module_cluster_dict: Dict[str, Any],
) -> None:
    """Render non-module combined graph clusters.

    Parameters
    ----------
    graphviz_graph:
        Graphviz graph being rendered.
    module_cluster_dict:
        Shared module cluster accumulator.
    """

    cluster_data = module_cluster_dict.get("__intervening__")
    if cluster_data is None:
        return
    with graphviz_graph.subgraph(name="cluster___intervening__") as subgraph:
        subgraph.attr(
            label="intervening grad_fns",
            color=GRADIENT_ARROW_COLOR,
            fontcolor="black",
            style="dashed",
        )
        for node_args in cluster_data.get("nodes", []):
            subgraph.node(**node_args)
        for edge_dict in cluster_data.get("edges", []):
            subgraph.edge(**edge_dict)


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


def _queue_container_clusters(
    module_cluster_dict: Dict[str, Any],
    clusters: Sequence[ContainerClusterSpec],
) -> None:
    """Queue container clusters into their owning module cluster payload."""

    for cluster in clusters:
        if cluster.owner_key == -1:
            continue
        module_cluster_dict[cast(str, cluster.owner_key)]["container_clusters"].append(cluster)


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

    if node_call_depth >= vis_call_depth:
        return True
    else:
        return False


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


def _should_order_siblings(
    *,
    order_siblings: bool,
    engine: str,
    vis_mode: str,
    num_nodes: int,
    module: "Module | str | None",
    vis_intervention_mode: VisInterventionModeLiteral,
    collapse_fn: CollapseFn | None,
    vis_call_depth: int,
) -> bool:
    """Return whether sibling ordering is in scope for this render."""

    return (
        order_siblings
        and engine == "dot"
        and vis_mode == "unrolled"
        and num_nodes <= SIBLING_ORDER_NODE_CAP
        and module is None
        and vis_intervention_mode == "node_mark"
        and vis_call_depth >= 1000
    )


def _queue_sibling_rank_group(
    module_edge_dict: Dict[str, Any],
    top_level_rank_groups: list[SiblingOrderChain],
    chain: SiblingOrderChain,
) -> None:
    """Queue a sibling rank group in the cluster dictionary."""

    if chain.lca_key == -1:
        top_level_rank_groups.append(chain)
    else:
        module_edge_dict[cast(str, chain.lca_key)]["rank_groups"].append(chain)


def _verify_and_apply_sibling_ordering(
    source: str,
    chains: tuple[SiblingOrderChain, ...],
    captured_edges: list[CapturedForwardEdge],
    rankdir: str,
) -> tuple[str, SiblingOrderDecision]:
    """Verify sibling rank chains and return final DOT source."""

    baseline_source = _strip_sibling_rank_groups(source)
    baseline = _layout_dot_plain(baseline_source, rankdir, captured_edges)
    chains = _filter_sibling_chains_to_rendered_nodes(chains, baseline.nodes)
    if not chains:
        return baseline_source, _sibling_order_decision((), (), {})
    injected = _layout_dot_plain(source, rankdir, captured_edges)
    _assert_sibling_backstops(baseline, injected, chains, captured_edges)

    ratios = {
        _sibling_chain_key(chain): _sibling_chain_stretch_ratio(
            chain, captured_edges, baseline, injected
        )
        for chain in chains
    }
    survivors = tuple(
        chain for chain in chains if ratios[_sibling_chain_key(chain)] <= SIBLING_ORDER_STRETCH_CAP
    )
    current_source = (
        source if survivors == chains else _inject_sibling_rank_groups(baseline_source, survivors)
    )
    current_layout = (
        injected
        if survivors == chains
        else _layout_dot_plain(current_source, rankdir, captured_edges)
    )

    for _ in range(2):
        bad_chains = tuple(
            chain
            for chain in survivors
            if _sibling_chain_stretch_ratio(chain, captured_edges, baseline, current_layout)
            > SIBLING_ORDER_STRETCH_CAP
        )
        if not bad_chains:
            return current_source, _sibling_order_decision(chains, survivors, ratios)
        survivors = tuple(chain for chain in survivors if chain not in bad_chains)
        current_source = _inject_sibling_rank_groups(baseline_source, survivors)
        current_layout = _layout_dot_plain(current_source, rankdir, captured_edges)
    return current_source, _sibling_order_decision(chains, survivors, ratios)


def _strict_sibling_order_checks_enabled() -> bool:
    """Return whether sibling-order verification failures should raise.

    Returns
    -------
    bool
        True under pytest or when ``TORCHLENS_COLLAPSE_STRICT=1`` is set.
    """

    return os.environ.get("TORCHLENS_COLLAPSE_STRICT") == "1" or "PYTEST_CURRENT_TEST" in os.environ


def _warn_sibling_order_fallback_once(exc: BaseException) -> None:
    """Warn once when sibling-order verification is skipped in production.

    Parameters
    ----------
    exc:
        Verification failure that triggered the fallback.
    """

    global _SIBLING_ORDER_WARNING_EMITTED
    if _SIBLING_ORDER_WARNING_EMITTED:
        return
    _SIBLING_ORDER_WARNING_EMITTED = True
    warnings.warn(
        "Sibling-order verification failed; rendering without the optional sibling-order "
        f"post-pass. ({type(exc).__name__}: {exc})",
        RuntimeWarning,
        stacklevel=3,
    )


def _sibling_chain_key(chain: SiblingOrderChain) -> tuple[str, tuple[str, ...]]:
    """Return a stable key for decision reporting."""

    return chain.source_name, chain.targets


def _sibling_order_decision(
    chains: tuple[SiblingOrderChain, ...],
    survivors: tuple[SiblingOrderChain, ...],
    ratios: dict[tuple[str, tuple[str, ...]], float],
) -> SiblingOrderDecision:
    """Build a sibling-order decision record."""

    return SiblingOrderDecision(
        candidate_count=len(chains),
        survivor_count=len(survivors),
        ratios=ratios,
        surviving_keys=tuple(_sibling_chain_key(chain) for chain in survivors),
    )


def _layout_dot_plain(
    source: str,
    rankdir: str,
    captured_edges: list[CapturedForwardEdge],
) -> PlainLayout:
    """Run ``dot -Tplain`` and parse coordinates and real-edge spans."""

    real_edges = {(edge.tail_name, edge.head_name) for edge in captured_edges}
    with tempfile.NamedTemporaryFile("w", suffix=".dot", delete=False) as source_file:
        source_file.write(source)
        source_path = source_file.name
    try:
        proc = subprocess.run(
            ["dot", "-Tplain", source_path],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    finally:
        os.remove(source_path)

    nodes: dict[str, tuple[float, float]] = {}
    pending_edges: list[tuple[str, str]] = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "node" and len(parts) >= 4:
            nodes[parts[1]] = (float(parts[2]), float(parts[3]))
        elif parts[0] == "edge" and len(parts) >= 4:
            edge_key = (parts[1], parts[2])
            if edge_key in real_edges:
                pending_edges.append(edge_key)

    edge_spans: dict[tuple[str, str], float] = {}
    for edge_key in pending_edges:
        if edge_key[0] in nodes and edge_key[1] in nodes:
            edge_spans[edge_key] = _flow_span(nodes[edge_key[0]], nodes[edge_key[1]], rankdir)
    return PlainLayout(nodes=nodes, edge_spans=edge_spans)


def _sibling_chain_stretch_ratio(
    chain: SiblingOrderChain,
    captured_edges: list[CapturedForwardEdge],
    baseline: PlainLayout,
    candidate: PlainLayout,
) -> float:
    """Return the local incident-edge stretch ratio for ``chain``."""

    local_nodes = {chain.source_name, *chain.targets}
    ratios: list[float] = []
    for edge in captured_edges:
        edge_key = (edge.tail_name, edge.head_name)
        if edge.tail_name not in local_nodes and edge.head_name not in local_nodes:
            continue
        if edge_key not in baseline.edge_spans or edge_key not in candidate.edge_spans:
            continue
        ratios.append(
            candidate.edge_spans[edge_key]
            / max(SIBLING_ORDER_EPSILON, baseline.edge_spans[edge_key])
        )
    return max(ratios, default=1.0)


def _strip_sibling_rank_groups(source: str) -> str:
    """Remove TorchLens sibling-order rank-group blocks from DOT source."""

    lines = source.splitlines()
    stripped: list[str] = []
    skipping = False
    for line in lines:
        if "tl:sibling-order:start" in line:
            skipping = True
            continue
        if "tl:sibling-order:end" in line:
            skipping = False
            continue
        if not skipping:
            stripped.append(line)
    return "\n".join(stripped) + "\n"


def _setup_subgraphs(
    self: "Trace",
    graphviz_graph: graphviz.Digraph,
    vis_mode: str,
    module_edge_dict: Dict[str, Any],
    overrides: Optional[VisualizationOverrides] = None,
    top_level_rank_groups: Sequence[SiblingOrderChain] = (),
) -> None:
    """Build nested Graphviz subgraphs for module clusters.

    Creates the module hierarchy as nested Graphviz subgraphs (clusters),
    placing edges into the appropriate depth level.  Uses a BFS-like
    approach: starts from top-level modules, builds each subgraph via
    ``_setup_subgraphs_recurse``, and pushes child modules onto a stack.

    In **unrolled** mode, each module pass is a separate subgraph (keyed by
    ``"module_addr:call_index"``).  In **rolled** mode, all ops share one
    subgraph (keyed by ``"module_addr"``).

    Subgraph names are prefixed with ``"cluster_"`` (Graphviz convention to
    draw a border box around them).

    Args:
        graphviz_graph: The top-level Graphviz Digraph.
        vis_mode: ``'rolled'`` or ``'unrolled'``.
        module_edge_dict: Dict mapping each module cluster name to
            ``{"edges": [...], "has_input_ancestor": bool}``.
        overrides: Graphviz attribute overrides for module subgraphs.
    """
    if "self" not in self.modules:
        return
    if vis_mode == "unrolled":
        module_submodule_dict = defaultdict(list)
        for call_label, mpl in self.modules._pass_dict.items():
            module_submodule_dict[call_label] = list(mpl.call_children)
        subgraphs = list(self.modules["self"].ops[0].call_children)  # type: ignore[union-attr]
    else:
        module_submodule_dict = defaultdict(list)
        for ml in self.modules:
            if ml.address != "self":
                module_submodule_dict[ml.address] = list(ml.call_children)
        subgraphs = list(self.modules["self"].call_children)

    # Get the max module nesting depth:

    max_call_depth = _get_max_call_depth(subgraphs, module_edge_dict, module_submodule_dict)

    subgraph_stack = [[subgraph] for subgraph in subgraphs]
    call_depth = 0
    emitted_rank_groups = 0
    while len(subgraph_stack) > 0:
        parent_graph_list = subgraph_stack.pop(0)
        emitted_rank_groups += _setup_subgraphs_recurse(
            self,
            graphviz_graph,
            parent_graph_list,
            module_edge_dict,
            module_submodule_dict,
            subgraph_stack,
            call_depth,
            max_call_depth,
            vis_mode,
            overrides,  # type: ignore[arg-type]
        )
    for chain in top_level_rank_groups:
        _emit_sibling_rank_group(graphviz_graph, chain)
        emitted_rank_groups += 1
    queued_rank_groups = len(top_level_rank_groups) + sum(
        len(data.get("rank_groups", [])) for data in module_edge_dict.values()
    )
    assert queued_rank_groups == emitted_rank_groups


def _setup_subgraphs_recurse(
    self: "Trace",
    starting_subgraph: graphviz.Digraph,
    parent_graph_list: List[str],
    module_edge_dict: Dict[str, Any],
    module_submodule_dict: Dict[str, list[str]],
    subgraph_stack: list[list[str]],
    call_depth: int,
    max_call_depth: int,
    vis_mode: str,
    overrides: VisualizationOverrides,
) -> int:
    """Recursively build a single branch of the module subgraph hierarchy.

    Walks down ``parent_graph_list`` (a path from root to leaf module),
    creating nested Graphviz context managers at each level.  When the
    leaf is reached, adds all accumulated edges and pushes child modules
    onto ``subgraph_stack`` for later processing.

    Module border width scales inversely with nesting depth (deeper modules
    get thinner borders) to provide visual hierarchy.

    Args:
        starting_subgraph: The parent Graphviz subgraph to nest into.
        parent_graph_list: Path of module names from root to current target.
        module_edge_dict: Dict mapping each cluster to its edges.
        module_submodule_dict: Dict mapping each cluster to its subclusters.
        subgraph_stack: BFS work queue for remaining branches.
        call_depth: Current position in ``parent_graph_list``.
        max_call_depth: Maximum depth across all branches (for penwidth scaling).
        vis_mode: ``'rolled'`` or ``'unrolled'``.
        overrides: Graphviz attribute overrides.
    """
    subgraph_name_w_pass = parent_graph_list[call_depth]
    subgraph_module = subgraph_name_w_pass.split(":")[0]
    if vis_mode == "unrolled":
        cluster_name = f"cluster_{subgraph_name_w_pass.replace(':', '_pass')}"
        subgraph_name = subgraph_name_w_pass
    elif vis_mode == "rolled":
        cluster_name = f"cluster_{subgraph_module}"
        subgraph_name = subgraph_module
    else:
        raise ValueError("vis_mode must be 'rolled' or 'unrolled'")
    sg_ml = self.modules[subgraph_module]
    module_type = sg_ml.class_name  # type: ignore[union-attr]
    if (sg_ml.num_calls > 1) and (vis_mode == "unrolled"):  # type: ignore[union-attr]
        subgraph_title = subgraph_name_w_pass
    elif (sg_ml.num_calls > 1) and (vis_mode == "rolled"):  # type: ignore[union-attr]
        subgraph_title = (
            f"{subgraph_module} (x{sg_ml.num_calls}"
            f"{_collapsed_module_rolling_suffix(self, subgraph_module)})"
        )
    else:
        subgraph_title = subgraph_module

    if call_depth < len(parent_graph_list) - 1:  # we haven't gotten to the bottom yet, keep going.
        with starting_subgraph.subgraph(name=cluster_name) as s:
            return _setup_subgraphs_recurse(
                self,
                s,
                parent_graph_list,
                module_edge_dict,
                module_submodule_dict,
                subgraph_stack,
                call_depth + 1,
                max_call_depth,
                vis_mode,
                overrides,
            )

    else:  # Leaf of this branch: create the subgraph and add all edges.
        emitted_rank_groups = 0
        cluster_payload = module_edge_dict[subgraph_name]
        if (
            sg_ml.num_layers <= 1  # type: ignore[union-attr]
            and not module_submodule_dict[subgraph_name_w_pass]
            and not cluster_payload.get("nodes")
            and not cluster_payload.get("edges")
            and not cluster_payload.get("rank_groups")
        ):
            return emitted_rank_groups
        with starting_subgraph.subgraph(name=cluster_name) as s:
            # Penwidth + cluster attrs come from ``_render_utils`` so the
            # bundle renderer in ``multi_trace/visualization.py`` can build
            # equivalent clusters with the same formula and label format.
            pen_width = compute_module_penwidth(call_depth, max_call_depth)
            if cluster_payload["has_input_ancestor"]:
                line_style = "solid"
            else:
                line_style = "dashed"

            # Module-address-derived titles can contain arbitrary user text
            # (e.g. an ``nn.ModuleDict`` key like ``"score & rank"``), so the
            # title must go through ``html_escape`` like any other
            # user-provided string -- do not assume it is HTML-safe.
            module_args = make_module_cluster_attrs(
                title=subgraph_title,
                module_type=module_type,
                line_style=line_style,
                penwidth=pen_width,
            )

            for arg_name, arg_val in overrides.module.items():  # type: ignore[union-attr]
                if callable(arg_val):
                    module_args[arg_name] = str(arg_val(self, subgraph_name))
                else:
                    module_args[arg_name] = str(arg_val)
            s.attr(**module_args)
            for chain in cluster_payload.get("rank_groups", []):
                _emit_sibling_rank_group(s, chain)
                emitted_rank_groups += 1
            subgraph_nodes = cluster_payload.get("nodes", [])
            for node_args in subgraph_nodes:
                s.node(**node_args)
            for container_cluster in cluster_payload.get("container_clusters", []):
                _emit_container_cluster(s, cast(ContainerClusterSpec, container_cluster))
            subgraph_edges = cluster_payload["edges"]
            for edge_dict in subgraph_edges:
                s.edge(**edge_dict)
            subgraph_children = module_submodule_dict[subgraph_name_w_pass]
            for subgraph_child in subgraph_children:  # it's weird but have to go in reverse order.
                subgraph_stack.append(parent_graph_list[:] + [subgraph_child])
        return emitted_rank_groups


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
    "_replay_forward_dot_calls",
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
