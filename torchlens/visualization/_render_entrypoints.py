"""Public render entrypoints and rendered-node universe adapters."""

# ruff: noqa: F403, F405

from ._render_common import *
from ._render_leaf import *
from ._render_edges import *
from ._render_nodes import *
from ._render_flow import *
from ._render_dot import *


def render_backward_graph(
    self: "Trace",
    vis_outpath: str = "backward_modelgraph",
    vis_graph_overrides: Optional[Dict[str, Any]] = None,
    node_spec_fn: BackwardNodeSpecFn | None = None,
    collapsed_node_spec_fn: CollapsedNodeSpecFn | None = None,
    vis_node_mode: VisNodeModeLiteral = "default",
    vis_edge_overrides: Optional[Dict[str, Any]] = None,
    vis_save_only: bool = False,
    vis_fileformat: str = "pdf",
    direction: VisDirectionLiteral = "topdown",
    code_panel: CodePanelOption = False,
    vis_mode: VisModeLiteral = "rolled",
    bwd: int | Iterable[int] | None = None,
) -> str:
    """Render the captured backward grad_fn_handle DAG as a Graphviz graph.

    Intervening grad_fns use a ``[i]`` label prefix. Custom autograd grad_fns
    use a ``[custom]`` label suffix so the two cues compose on the same node.

    Parameters
    ----------
    self:
        Trace containing captured backward metadata.
    vis_outpath:
        Output path for the rendered graph.
    vis_graph_overrides:
        Graphviz graph-level overrides.
    node_spec_fn:
        Optional callback receiving ``(grad_fn_handle, default_spec)``.
    collapsed_node_spec_fn:
        Accepted for API symmetry with forward visualization. Not applied
        because backward graphs do not render collapsed module nodes.
    vis_node_mode:
        Accepted for API symmetry with forward visualization. Not applied to
        grad_fn_handle nodes.
    vis_edge_overrides:
        Graphviz edge-level overrides.
    vis_save_only:
        If True, save without opening a viewer.
    vis_fileformat:
        Output format.
    direction:
        Layout direction: ``'bottomup'``, ``'topdown'``, or ``'leftright'``.
    code_panel:
        Optional source-code panel mode.
    vis_mode:
        ``"rolled"`` renders one node per GradFn. ``"unrolled"`` renders one
        node per GradFnCall, grouped into backward-pass clusters.
    bwd:
        Optional one-based backward pass number or numbers to render.

    Returns
    -------
    str
        Graphviz DOT source.

    Raises
    ------
    ValueError
        If no explicit backward graph has been captured.
    """

    if not self.has_backward_pass or not self.grad_fn_logs:
        raise ValueError("No backward graph is available; call log_backward(loss) first.")
    _ = collapsed_node_spec_fn, vis_node_mode
    if vis_mode not in {"rolled", "unrolled"}:
        raise ValueError("vis_mode must be either 'rolled' or 'unrolled'")
    pass_filter = _normalize_backward_pass_filter(bwd)

    rankdir = direction_to_rankdir(direction)

    split_outpath = vis_outpath.split(".")
    if split_outpath[-1] in [
        "pdf",
        "png",
        "jpg",
        "svg",
        "jpeg",
        "bmp",
        "pic",
        "tif",
        "tiff",
    ]:
        vis_outpath = ".".join(split_outpath[:-1])

    graph_caption = (
        f"<<B>{self.model_class_name} backward graph</B><br align='left'/>"
        f"{self.num_grad_fns} grad_fn_handle nodes"
        f"<br align='left'/>{self.num_backward_passes} backward pass(es)"
        f"{_format_backward_filter_caption(pass_filter)}"
        f"<br align='left'/>mode: {vis_mode}<br align='left'/>>"
    )
    dot = graphviz.Digraph(
        name=f"{self.model_class_name}_backward",
        comment="Backward grad_fn_handle graph",
        format=vis_fileformat,
    )
    graph_args = {
        "rankdir": rankdir,
        "label": graph_caption,
        "labelloc": "t",
        "labeljust": "left",
        "ordering": "out",
        "compound": "true",
    }
    for arg_name, arg_val in (vis_graph_overrides or {}).items():
        if callable(arg_val):
            graph_args[arg_name] = str(arg_val(self))
        else:
            graph_args[arg_name] = str(arg_val)

    edge_args = {"color": GRADIENT_ARROW_COLOR, "fontcolor": GRADIENT_ARROW_COLOR}
    for arg_name, arg_val in (vis_edge_overrides or {}).items():
        if callable(arg_val):
            edge_args[arg_name] = str(arg_val(self))
        else:
            edge_args[arg_name] = str(arg_val)

    dot.graph_attr.update(graph_args)
    dot.node_attr.update({"ordering": "out"})
    dot.edge_attr.update(edge_args)

    if vis_mode == "rolled":
        visible_ids = {
            grad_fn_handle.grad_fn_object_id
            for grad_fn_handle in self.grad_fns
            if _grad_fn_matches_backward_filter(grad_fn_handle, pass_filter)
        }
        for grad_fn_handle in self.grad_fns:
            if grad_fn_handle.grad_fn_object_id in visible_ids:
                _add_backward_node_to_graphviz(
                    grad_fn_handle,
                    dot,
                    node_spec_fn,
                    pass_filter=pass_filter,
                )

        for grad_fn_handle in self.grad_fns:
            if grad_fn_handle.grad_fn_object_id not in visible_ids:
                continue
            tail_name = _backward_dot_node_name(grad_fn_handle)
            for next_grad_fn_id in grad_fn_handle.next_grad_fn_ids:
                if next_grad_fn_id not in visible_ids:
                    continue
                head_name = _backward_dot_node_name(self.grad_fn_logs[next_grad_fn_id])
                dot.edge(
                    tail_name,
                    head_name,
                    **_backward_edge_attrs(grad_fn_handle, self.grad_fn_logs[next_grad_fn_id]),
                )
    else:
        _add_unrolled_backward_pass_clusters(self, dot, node_spec_fn, pass_filter)

    source_text = resolve_code_panel_source(
        code_panel,
        getattr(self, "_source_code_blob", {}),
        getattr(self, "_source_model_ref", None),
    )
    # Compose the code panel side by side when the format supports it so it never
    # distorts the backward graph; otherwise fall back to an in-graph subgraph.
    compose_code_panel = source_text is not None and _code_panel_composition_available(
        vis_fileformat, dot.engine
    )
    if source_text is not None and not compose_code_panel:
        render_code_panel_subgraph(dot, source_text)

    if in_notebook() and not vis_save_only:
        try:
            from IPython.display import SVG, display
        except ImportError as e:
            raise ImportError(
                "IPython is required for this feature. Install with "
                "`pip install torchlens[notebook]`."
            ) from e

        display_fn = cast(Any, display)
        if compose_code_panel:
            graph_svg = dot.pipe(format="svg").decode("utf-8")
            combined_svg = compose_graph_with_code_panel(graph_svg, cast(str, source_text))
            display_fn(SVG(combined_svg))
        else:
            display_fn(dot)

    _RENDER_TIMEOUT = 120
    source_path = dot.save(vis_outpath)
    with _timed_phase(self, "render:graphviz:backward"):
        try:
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
            _validate_rendered_output(rendered_path, source_path, "backward graph")
            if not vis_save_only:
                _view_rendered_file(rendered_path)
            _vprint(self, f"Backward graph saved to {vis_outpath}.{vis_fileformat}")
            # Success: remove the intermediate DOT source. On FAILURE we keep it so
            # the error's "DOT source was saved to ..." hint points to a real file.
            if os.path.exists(source_path):
                os.remove(source_path)
        except subprocess.TimeoutExpired as e:
            _raise_graphviz_timeout(
                "backward graph",
                f"{self.num_grad_fns} grad_fn_handle nodes",
                source_path,
                _RENDER_TIMEOUT,
                e,
            )
        except subprocess.CalledProcessError as e:
            _raise_graphviz_failure("backward graph", source_path, e)
    return cast(str, dot.source)


if TYPE_CHECKING:
    from ..data_classes.grad_fn import GradFn
    from ..data_classes.module import Module
    from ..data_classes.trace import Trace
    from .auto_collapse import ModuleRunFold


def render_combined_graph(
    self: "Trace",
    vis_outpath: str = "combined_modelgraph",
    vis_graph_overrides: Optional[Dict[str, Any]] = None,
    node_spec_fn: NodeSpecFn | None = None,
    backward_node_spec_fn: BackwardNodeSpecFn | None = None,
    vis_edge_overrides: Optional[Dict[str, Any]] = None,
    vis_save_only: bool = False,
    vis_fileformat: str = "pdf",
    direction: VisDirectionLiteral = "leftright",
    vis_mode: VisModeLiteral = "unrolled",
    intervening_cluster: InterveningClusterMode = "upstream",
    show_buffer_layers: BufferVisibilityLiteral | bool = "meaningful",
    bwd: int | Iterable[int] | None = None,
) -> str:
    """Render one Graphviz graph containing forward ops and backward grad_fns.

    Parameters
    ----------
    self:
        Trace containing forward and explicit backward metadata.
    vis_outpath:
        Output path for the rendered graph.
    vis_graph_overrides:
        Graphviz graph-level overrides.
    node_spec_fn:
        Optional callback receiving ``(layer_log, default_spec)`` for forward nodes.
    backward_node_spec_fn:
        Optional callback receiving ``(grad_fn_handle, default_spec)`` for backward nodes.
    vis_edge_overrides:
        Graphviz edge-level overrides applied to forward edges.
    vis_save_only:
        If True, save without opening a viewer.
    vis_fileformat:
        Output format.
    direction:
        Layout direction.
    vis_mode:
        Combined rendering currently supports only ``"unrolled"``.
    intervening_cluster:
        Placement mode for grad_fns that have no corresponding forward op.
    show_buffer_layers:
        Buffer visibility mode for the forward side.
    bwd:
        Optional one-based backward pass number or numbers to render.

    Returns
    -------
    str
        Graphviz DOT source.

    Raises
    ------
    NotImplementedError
        If ``vis_mode="rolled"`` is requested.
    ValueError
        If no explicit backward graph has been captured.
    """

    if vis_mode == "rolled":
        raise NotImplementedError("draw_combined does not support vis_mode='rolled' yet.")
    if vis_mode != "unrolled":
        raise ValueError("vis_mode must be either 'unrolled' or 'rolled'")
    if not self.has_backward_pass or not self.grad_fn_logs:
        raise ValueError("No backward graph is available; call log_backward(loss) first.")
    if not self._layers_logged:
        raise ValueError(
            "Must have all layers logged in order to render the graph; use show_model_graph."
        )
    pass_filter = _normalize_backward_pass_filter(bwd)

    show_buffer_layers = _normalize_buffer_visibility(show_buffer_layers)
    vis_outpath = _strip_render_extension(vis_outpath)
    rankdir = direction_to_rankdir(direction)
    overrides = VisualizationOverrides(
        graph=graphviz_graph_overrides(vis_graph_overrides),
        edge=vis_edge_overrides or {},
        grad_edge={},
        module={},
    )

    graph_caption = (
        f"<<B>{self.model_class_name} combined forward/backward graph</B><br align='left'/>"
        f"{self.num_tensors} forward nodes, {self.num_grad_fns} grad_fn_handle nodes"
        f"<br align='left'/>{self.num_backward_passes} backward pass(es)"
        f"{_format_backward_filter_caption(pass_filter)}<br align='left'/>>"
    )
    dot = graphviz.Digraph(
        name=f"{self.model_class_name}_combined",
        comment="Combined forward and backward graph",
        format=vis_fileformat,
    )
    graph_args = {
        "rankdir": rankdir,
        "label": graph_caption,
        "labelloc": "t",
        "labeljust": "left",
        "ordering": "out",
    }
    for arg_name, arg_val in overrides.graph.items():  # type: ignore[union-attr]
        graph_args[arg_name] = str(arg_val(self) if callable(arg_val) else arg_val)
    dot.graph_attr.update(graph_args)
    dot.node_attr.update({"ordering": "out"})
    dot.edge_attr.update({"ordering": "out"})

    module_cluster_dict: Dict[str, Any] = defaultdict(
        lambda: {"edges": [], "nodes": [], "has_input_ancestor": False}
    )
    entries_to_plot: dict[str, GraphNode] = dict(self.layer_dict_main_keys)
    edge_map, skipped_labels = _build_skip_filtered_edge_map(
        self,
        entries_to_plot,
        vis_mode="unrolled",
        show_buffer_layers=show_buffer_layers,
        skip_fn=None,
    )
    edges_used: Set[tuple[str, str, tuple[Any, ...]]] = set()
    collapsed_modules: Set[str] = set()
    for node in entries_to_plot.values():
        if node.layer_label in skipped_labels:
            continue
        if node.is_buffer and not _is_buffer_visible(node, show_buffer_layers):
            continue
        _add_node_to_graphviz(
            self,
            node,
            dot,
            module_cluster_dict,
            edges_used,
            "unrolled",
            collapsed_modules,
            show_buffer_layers=show_buffer_layers,
            overrides=overrides,
            node_spec_fn=node_spec_fn,
            edge_map=edge_map,
        )

    _add_combined_backward_nodes(
        self,
        module_cluster_dict,
        dot,
        backward_node_spec_fn,
        intervening_cluster,
        pass_filter,
    )
    _add_combined_backward_edges(self, dot, pass_filter)
    _add_combined_correspondence_edges(self, dot, intervening_cluster, pass_filter)
    _setup_subgraphs(self, dot, "unrolled", module_cluster_dict, overrides)
    _setup_combined_special_clusters(dot, module_cluster_dict)

    _RENDER_TIMEOUT = 120
    source_path = dot.save(vis_outpath)
    with _timed_phase(self, "render:graphviz:combined"):
        try:
            rendered_path = f"{vis_outpath}.{vis_fileformat}"
            cmd = [dot.engine, f"-T{vis_fileformat}", "-o", rendered_path, source_path]
            subprocess.run(cmd, timeout=_RENDER_TIMEOUT, check=True, capture_output=True)
            _validate_rendered_output(rendered_path, source_path, "combined graph")
            if not vis_save_only:
                _view_rendered_file(rendered_path)
            _vprint(self, f"Combined graph saved to {vis_outpath}.{vis_fileformat}")
        except subprocess.TimeoutExpired as e:
            _raise_graphviz_timeout(
                "combined graph",
                f"{self.num_tensors + self.num_grad_fns} nodes",
                source_path,
                _RENDER_TIMEOUT,
                e,
            )
        except subprocess.CalledProcessError as e:
            _raise_graphviz_failure("combined graph", source_path, e)
        else:
            if os.path.exists(source_path):
                os.remove(source_path)
    return cast(str, dot.source)


def rendered_node_universe_from_v1(
    trace: "Trace",
    *,
    collapse_fn: CollapseFn | None,
    run_folds: Mapping[str, "ModuleRunFold"] | None,
    context: RenderContext | None = None,
    vis_call_depth: int = 1000,
) -> tuple[RenderedNodeEmission, ...]:
    """Return the forward renderer's visible node universe for v1 collapse.

    Parameters
    ----------
    trace:
        Trace whose forward graph is being inspected.
    collapse_fn:
        Active collapse predicate.
    run_folds:
        Active run-fold descriptors keyed by module address.
    context:
        Render context. Defaults to the S7 parity matrix.
    vis_call_depth:
        Legacy call-depth threshold used when ``collapse_fn`` is ``None``.

    Returns
    -------
    tuple[RenderedNodeEmission, ...]
        Visible rendered nodes in deterministic emission order.
    """

    resolved_context = RenderContext() if context is None else context
    show_buffer_layers = _normalize_buffer_visibility(resolved_context.show_buffer_layers)
    entries_to_plot = _entries_to_plot_for_context(trace, resolved_context.vis_mode)
    skipped_labels: set[str] = set()
    edge_map: dict[str, list[RenderEdge]] = {}
    if run_folds:
        edge_map, skipped_labels = _build_skip_filtered_edge_map(
            trace,
            entries_to_plot,
            vis_mode=resolved_context.vis_mode,
            show_buffer_layers=show_buffer_layers,
            skip_fn=None,
        )
    collapsed_container_nodes = _collapsed_container_leaf_nodes(
        trace,
        entries_to_plot,
        vis_mode=resolved_context.vis_mode,
        show_containers=resolved_context.show_containers,
        container_max_inline=12,
        pending_nodes=[],
    )
    emissions = _enumerate_base_rendered_node_emissions(
        trace,
        entries_to_plot,
        skipped_labels=skipped_labels,
        vis_mode=resolved_context.vis_mode,
        vis_call_depth=vis_call_depth,
        show_buffer_layers=show_buffer_layers,
        collapse_fn=collapse_fn,
        run_folds=run_folds,
        show_containers=resolved_context.show_containers,
        collapsed_container_nodes=collapsed_container_nodes,
    )
    ellipsis_emissions = _enumerate_run_fold_ellipsis_emissions(
        trace,
        entries_to_plot,
        edge_map=edge_map,
        skipped_labels=skipped_labels,
        vis_mode=resolved_context.vis_mode,
        vis_call_depth=vis_call_depth,
        show_buffer_layers=show_buffer_layers,
        collapse_fn=collapse_fn,
        run_folds=run_folds,
        collapsed_container_nodes=collapsed_container_nodes,
    )
    return (*emissions, *ellipsis_emissions)


__all__ = [
    "render_backward_graph",
    "render_combined_graph",
    "rendered_node_universe_from_v1",
]
