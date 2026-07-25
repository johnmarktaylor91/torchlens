"""Public backward and combined-graph rendering entrypoints."""

# ruff: noqa: F403, F405

from dataclasses import replace

from ._render_common import *
from ._render_leaf import *
from ._render_edges import *
from ._render_nodes import *
from ._render_flow import *
from ._render_dot import *
from ._render_utils import html_escape
from .render_ir import build_backward_render_ir, build_combined_render_ir
from .renderers.graphviz import GraphvizRenderer
from .request import RenderContext, RenderTarget
from .source_graph import build_source_graph


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

    vis_outpath = _strip_render_extension(vis_outpath)

    graph_caption = (
        f"<<B>{html_escape(self.model_class_name)} backward graph</B><br align='left'/>"
        f"{self.num_grad_fns} grad_fn_handle nodes"
        f"<br align='left'/>{self.num_backward_passes} backward pass(es)"
        f"{_format_backward_filter_caption(pass_filter)}"
        f"<br align='left'/>mode: {vis_mode}<br align='left'/>>"
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

    ir_builder = _RenderIRDecisionBuilder()
    ir_builder.attr("graph", **graph_args)
    ir_builder.attr("node", ordering="out")
    ir_builder.attr("edge", **edge_args)

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
                    cast(graphviz.Digraph, ir_builder),
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
                ir_builder.edge(
                    tail_name,
                    head_name,
                    **_backward_edge_attrs(grad_fn_handle, self.grad_fn_logs[next_grad_fn_id]),
                )
    else:
        _add_unrolled_backward_pass_clusters(
            self, cast(graphviz.Digraph, ir_builder), node_spec_fn, pass_filter
        )

    source_text = resolve_code_panel_source(
        code_panel,
        getattr(self, "_source_code_blob", {}),
        getattr(self, "_source_model_ref", None),
    )
    # Compose the code panel side by side when the format supports it so it never
    # distorts the backward graph; otherwise fall back to an in-graph subgraph.
    compose_code_panel = source_text is not None and _code_panel_composition_available(
        vis_fileformat, "dot"
    )
    if source_text is not None and not compose_code_panel:
        render_code_panel_subgraph(cast(graphviz.Digraph, ir_builder), source_text)

    _RENDER_TIMEOUT = 120
    render_ir = build_backward_render_ir(
        self,
        vis_mode=cast(Literal["rolled", "unrolled"], vis_mode),
        pass_filter=pass_filter,
        dot_statements=tuple(ir_builder.calls),
    )
    target = RenderTarget(
        outpath=vis_outpath,
        fileformat=vis_fileformat,
        save_only=vis_save_only,
        viewer=not vis_save_only,
        graph_name=f"{self.model_class_name}_backward",
        graph_comment="Backward grad_fn_handle graph",
        timeout=_RENDER_TIMEOUT,
    )
    source_path = f"{vis_outpath}"
    with _timed_phase(self, "render:graphviz:backward"):
        try:
            rendered_path = f"{vis_outpath}.{vis_fileformat}"
            report = GraphvizRenderer().render(render_ir, target)
            source_path = str(report.source_path)
            if compose_code_panel:
                _write_composed_code_panel(
                    "dot",
                    source_path,
                    cast(str, source_text),
                    rendered_path,
                    vis_fileformat,
                    _RENDER_TIMEOUT,
                )
            _validate_rendered_output(rendered_path, source_path, "backward graph")
            if in_notebook() and not vis_save_only:
                try:
                    from IPython.display import SVG, display
                except ImportError as e:
                    raise ImportError(
                        "IPython is required for this feature. Install with "
                        "`pip install torchlens[notebook]`."
                    ) from e
                display_fn = cast(Any, display)
                display_fn(SVG(filename=rendered_path))
            elif not vis_save_only:
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
    return report.source


if TYPE_CHECKING:
    from ..data_classes.trace import Trace


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
        f"<<B>{html_escape(self.model_class_name)} combined forward/backward graph</B><br align='left'/>"
        f"{self.num_tensors} forward nodes, {self.num_grad_fns} grad_fn_handle nodes"
        f"<br align='left'/>{self.num_backward_passes} backward pass(es)"
        f"{_format_backward_filter_caption(pass_filter)}<br align='left'/>>"
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
    ir_builder = _RenderIRDecisionBuilder()
    ir_builder.attr("graph", **graph_args)
    ir_builder.attr("node", ordering="out")
    ir_builder.attr("edge", ordering="out")

    module_cluster_dict: Dict[str, Any] = defaultdict(
        lambda: {"edges": [], "nodes": [], "has_input_ancestor": False}
    )
    context = RenderContext(
        vis_mode="unrolled",
        show_buffer_layers=show_buffer_layers,
        node_spec_fn=node_spec_fn,
        overrides=overrides,
        direction=direction,
    )
    source_graph = build_source_graph(self, context)
    from .node_universe import build_node_universe

    universe = build_node_universe(source_graph, None, {})
    forward_ir = build_render_ir(
        self,
        collapse_fn=None,
        repeat_folds={},
        context=context,
        universe=universe,
    )
    edge_map = source_graph.edge_map
    edges_used: Set[tuple[str, str, tuple[Any, ...]]] = set()
    collapsed_modules: Set[str] = set()
    captured_forward_edges: list[CapturedForwardEdge] = []
    decisions_by_name = {node.name: node for node in forward_ir.nodes}
    for unit in universe.units:
        node_record = decisions_by_name[unit.unit_id]
        for source_index, node in enumerate(unit.source_nodes):
            _add_node_to_graphviz(
                self,
                node,
                cast(graphviz.Digraph, ir_builder),
                module_cluster_dict,
                edges_used,
                "unrolled",
                collapsed_modules,
                show_buffer_layers=show_buffer_layers,
                overrides=overrides,
                node_spec_fn=node_spec_fn,
                edge_map=edge_map,
                captured_forward_edges=captured_forward_edges,
                node_decision=(
                    node_record
                    if source_index == 0
                    else replace(node_record, node_calls=(), owned_node_args=())
                ),
            )

    _add_combined_backward_nodes(
        self,
        module_cluster_dict,
        cast(graphviz.Digraph, ir_builder),
        backward_node_spec_fn,
        intervening_cluster,
        pass_filter,
    )
    _add_combined_backward_edges(self, cast(graphviz.Digraph, ir_builder), pass_filter)
    _add_combined_correspondence_edges(
        self, cast(graphviz.Digraph, ir_builder), intervening_cluster, pass_filter
    )
    forward_ir = finalize_forward_regions(
        forward_ir,
        self,
        vis_mode="unrolled",
        module_payloads={
            key: payload for key, payload in module_cluster_dict.items() if key != "__intervening__"
        },
        container_regions=(),
        captured_edges=tuple(captured_forward_edges),
        overrides=overrides,
    )
    _setup_subgraphs(
        self,
        cast(graphviz.Digraph, ir_builder),
        "unrolled",
        module_cluster_dict,
        overrides,
        regions=forward_ir.regions,
    )
    _setup_combined_special_clusters(cast(graphviz.Digraph, ir_builder), module_cluster_dict)

    _RENDER_TIMEOUT = 120
    intervening_names = tuple(
        str(node_args["name"])
        for node_args in module_cluster_dict.get("__intervening__", {}).get("nodes", ())
    )
    render_ir = build_combined_render_ir(
        self,
        forward_ir,
        pass_filter=pass_filter,
        intervening_node_names=intervening_names,
        dot_statements=tuple(ir_builder.calls),
    )
    target = RenderTarget(
        outpath=vis_outpath,
        fileformat=vis_fileformat,
        save_only=vis_save_only,
        viewer=not vis_save_only,
        graph_name=f"{self.model_class_name}_combined",
        graph_comment="Combined forward and backward graph",
        timeout=_RENDER_TIMEOUT,
    )
    source_path = f"{vis_outpath}"
    with _timed_phase(self, "render:graphviz:combined"):
        try:
            rendered_path = f"{vis_outpath}.{vis_fileformat}"
            report = GraphvizRenderer().render(render_ir, target)
            source_path = str(report.source_path)
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
    return report.source


__all__ = [
    "render_backward_graph",
    "render_combined_graph",
]
