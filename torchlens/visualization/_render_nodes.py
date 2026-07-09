"""Node construction and raw value helper functions for Graphviz rendering."""

# ruff: noqa: F403, F405

from ._render_common import *
from ._render_leaf import *
from ._render_edges import *


def _normalize_buffer_visibility(
    show_buffer_layers: BufferVisibilityLiteral | bool,
) -> BufferVisibilityLiteral:
    """Normalize buffer visibility values accepted by the render path.

    Parameters
    ----------
    show_buffer_layers:
        Tri-state buffer visibility mode or legacy bool.

    Returns
    -------
    BufferVisibilityLiteral
        Canonical tri-state mode.

    Raises
    ------
    ValueError
        If ``show_buffer_layers`` is not supported.
    """

    if show_buffer_layers is True:
        return "always"
    if show_buffer_layers is False:
        return "never"
    if show_buffer_layers in {"never", "meaningful", "always"}:
        return show_buffer_layers
    raise ValueError("show_buffer_layers must be 'never', 'meaningful', 'always', or a bool.")


if TYPE_CHECKING:
    from ..data_classes.grad_fn import GradFn
    from ..data_classes.trace import Trace
    from .auto_collapse import ModuleRepeatFold


def _get_hidden_parent_buffer_addresses(
    trace: "Trace",
    node: GraphNode,
    show_buffer_layers: BufferVisibilityLiteral,
) -> list[str]:
    """Return hidden buffer addresses attached as parents of ``node``.

    Parameters
    ----------
    trace:
        Owning Trace.
    node:
        Non-buffer node to inspect.
    show_buffer_layers:
        Canonical tri-state visibility mode.

    Returns
    -------
    list[str]
        Hidden buffer addresses in parent order, de-duplicated.
    """

    if show_buffer_layers == "always" or node.is_buffer:
        return []

    hidden_addresses: list[str] = []
    seen_addresses: set[str] = set()
    source_node = _unwrap_focus_node(node)
    for parent_label in node.parents:
        if parent_label.startswith("__module_focus_"):
            continue
        parent_node: BaseGraphNode
        if isinstance(source_node, Op):
            parent_node = trace[parent_label]
        else:
            parent_node = trace.layer_logs[parent_label]
        if not parent_node.is_buffer or _is_buffer_visible(parent_node, show_buffer_layers):
            continue
        address = parent_node.address
        if address is None or address in seen_addresses:
            continue
        hidden_addresses.append(address)
        seen_addresses.add(address)
    return hidden_addresses


def _add_unrolled_backward_pass_clusters(
    trace: "Trace",
    graphviz_graph: graphviz.Digraph,
    node_spec_fn: BackwardNodeSpecFn | None,
    pass_filter: BackwardPassFilter,
) -> None:
    """Render GradFnCall nodes grouped by backward pass.

    Parameters
    ----------
    trace:
        Trace containing backward projections.
    graphviz_graph:
        Graphviz graph being rendered.
    node_spec_fn:
        Optional callback receiving ``(grad_fn_handle, default_spec)``.
    pass_filter:
        Normalized backward-pass filter.
    """

    calls_by_pass = _visible_backward_calls_by_pass(trace, pass_filter)
    for pass_index, grad_fn_calls in calls_by_pass.items():
        with graphviz_graph.subgraph(name=f"cluster_backward_pass_{pass_index}") as subgraph:
            subgraph.attr(
                label=f"backward pass {pass_index}",
                color=GRADIENT_ARROW_COLOR,
                fontcolor="black",
                style="rounded,dashed",
            )
            for grad_fn_handle, call in grad_fn_calls:
                node_args = _backward_node_graphviz_args(grad_fn_handle, node_spec_fn, call=call)
                subgraph.node(**node_args)

    for pass_index, grad_fn_calls in calls_by_pass.items():
        calls_for_grad_fn: dict[int, list[tuple["GradFn", Any]]] = defaultdict(list)
        for grad_fn_handle, call in grad_fn_calls:
            calls_for_grad_fn[grad_fn_handle.grad_fn_object_id].append((grad_fn_handle, call))
        for grad_fn_handle, call in grad_fn_calls:
            tail_name = _backward_dot_call_node_name(grad_fn_handle, call)
            for next_grad_fn_id in grad_fn_handle.next_grad_fn_ids:
                for head_grad_fn, head_call in calls_for_grad_fn.get(next_grad_fn_id, []):
                    head_name = _backward_dot_call_node_name(head_grad_fn, head_call)
                    graphviz_graph.edge(
                        tail_name,
                        head_name,
                        **_backward_edge_attrs(grad_fn_handle, head_grad_fn),
                    )


def _visible_backward_calls_by_pass(
    trace: "Trace",
    pass_filter: BackwardPassFilter,
) -> dict[int, list[tuple["GradFn", Any]]]:
    """Group visible GradFnCalls by backward pass.

    Parameters
    ----------
    trace:
        Trace containing backward projections.
    pass_filter:
        Normalized backward-pass filter.

    Returns
    -------
    dict[int, list[tuple[GradFn, Any]]]
        Visible calls keyed by one-based backward pass number.
    """

    calls_by_pass: dict[int, list[tuple["GradFn", Any]]] = defaultdict(list)
    for grad_fn_handle in trace.grad_fns:
        for call in grad_fn_handle.calls.values():
            if not _grad_fn_call_matches_backward_filter(call, pass_filter):
                continue
            pass_index = getattr(call, "backward_pass_index", None)
            if pass_index is None:
                continue
            calls_by_pass[int(pass_index)].append((grad_fn_handle, call))
    return dict(sorted(calls_by_pass.items()))


def _append_container_overlay_edge(
    edge_keys: set[tuple[str, str, str, str]],
    edges: list[ContainerOverlayEdge],
    *,
    namespace: str,
    tail_name: str,
    head_name: str,
    attrs: dict[str, str],
) -> None:
    """Append one deduplicated overlay edge in an isolated namespace.

    Parameters
    ----------
    edge_keys:
        Overlay-only dedupe keys.
    edges:
        Mutable overlay edge sink.
    namespace:
        Logical edge type. This keeps overlay dedupe separate from real
        dataflow ``edges_used`` keys.
    tail_name:
        Graphviz tail node name.
    head_name:
        Graphviz head node name.
    attrs:
        Graphviz edge attributes.
    """

    key = (namespace, tail_name, head_name, attrs.get("label", ""))
    if key in edge_keys:
        return
    edge_keys.add(key)
    edges.append(ContainerOverlayEdge(tail_name=tail_name, head_name=head_name, attrs=attrs))


def _container_boundary_edge_attrs(label: str | None) -> dict[str, str]:
    """Return Graphviz attrs for source/sink container-node edges.

    Parameters
    ----------
    label:
        Optional key/index label for the pulled-out field.

    Returns
    -------
    dict[str, str]
        Edge attributes for a boundary container overlay edge.
    """

    attrs = {
        "color": "#777777",
        "fontcolor": "#555555",
        "style": "solid",
        "arrowsize": ".6",
        "labelfontsize": "8",
        "constraint": "false",
    }
    if label is not None:
        attrs["label"] = _html_container_edge_label(label)
    return attrs


def _render_node_name_for_occurrence(trace: "Trace", label: str | None) -> str | None:
    """Return the rendered Graphviz node name for a container leaf occurrence.

    Parameters
    ----------
    trace:
        Trace whose ops should be searched.
    label:
        Final or raw producer op label.

    Returns
    -------
    str | None
        Rendered unrolled Graphviz node name, or ``None`` if unresolved.
    """

    if label is None:
        return None
    op = _op_for_occurrence_label(trace, label)
    if op is not None:
        return op.label.replace(":", "pass")
    return None


def _op_for_occurrence_label(trace: "Trace", label: str | None) -> Op | None:
    """Return the Op identified by a container occurrence producer label.

    Parameters
    ----------
    trace:
        Trace whose ops should be searched.
    label:
        Final or raw producer op label.

    Returns
    -------
    Op | None
        Matching operation, or ``None`` when unresolved.
    """

    if label is None:
        return None
    if label in trace.ops:
        return trace.ops[label]
    for op in trace.ops:
        if (
            getattr(op, "layer_label_raw", None) == label
            or getattr(op, "_layer_label_raw", None) == label
        ):
            return op
    return None


def _render_output_node_name_for_path(
    trace: "Trace",
    path: tuple[OutputPathComponent, ...],
) -> str | None:
    """Return the rendered final-output node matching a container path.

    Parameters
    ----------
    trace:
        Trace whose output layers should be searched.
    path:
        Captured container leaf path.

    Returns
    -------
    str | None
        Rendered output node name, or ``None`` if no output leaf matches.
    """

    output_labels = set(getattr(trace, "output_layers", ()) or ())
    for op in trace.ops:
        if getattr(op, "layer_label", None) not in output_labels:
            continue
        if tuple(getattr(op, "container_path", ()) or ()) == path:
            return op.label.replace(":", "pass")
    for output_label in getattr(trace, "output_layers", ()) or ():
        output_op = trace.layer_dict_all_keys.get(output_label)
        if output_op is None:
            continue
        if tuple(getattr(output_op, "container_path", ()) or ()) == path:
            return output_op.label.replace(":", "pass")
    return None


def _container_path_leaf_label(path: tuple[OutputPathComponent, ...]) -> str | None:
    """Return the visible label for the final component of a container path.

    Parameters
    ----------
    path:
        Captured container path.

    Returns
    -------
    str | None
        Final key/index/field label, or ``None`` for root leaves.
    """

    if not path:
        return None
    return _container_component_role(path[-1])


def _add_node_to_graphviz(
    self: "Trace",
    node: GraphNode,
    graphviz_graph: graphviz.Digraph,
    module_edge_dict: Dict[str, Any],
    edges_used: Set[tuple[str, str, tuple[Any, ...]]],
    vis_mode: str,
    collapsed_modules: Set[str],
    vis_call_depth: int = 1000,
    show_buffer_layers: BufferVisibilityLiteral = "meaningful",
    overrides: Optional[VisualizationOverrides] = None,
    node_mode: VisNodeModeLiteral = "default",
    node_spec_fn: NodeSpecFn | None = None,
    collapsed_node_spec_fn: CollapsedNodeSpecFn | None = None,
    collapse_fn: CollapseFn | None = None,
    edge_map: Optional[dict[str, list[RenderEdge]]] = None,
    vis_intervention_mode: VisInterventionModeLiteral = "node_mark",
    intervention_site_labels: set[str] | None = None,
    theme: VisualizationTheme | None = None,
    node_overlay: str | OverlayScores | None = None,
    node_label_fields: list[str] | None = None,
    captured_forward_edges: list[CapturedForwardEdge] | None = None,
    rankdir: str = "BT",
    show_containers: ShowContainersLiteral = False,
    collapsed_container_nodes: Mapping[str, str] | None = None,
    show_input_transform_summary: bool = False,
    repeat_folds: Mapping[str, "ModuleRepeatFold"] | None = None,
    run_fold_ellipsis_nodes: set[str] | None = None,
    segments: Mapping[str, SegmentDescriptor] | None = None,
    emitted_segment_nodes: set[str] | None = None,
    antiparallel_projected_edges: frozenset[tuple[str, str]] = frozenset(),
) -> None:
    """Adds a node and its relevant edges to the graphviz figure.

    Args:
        node: node to add
        graphviz_graph: The graphviz object to add the node to.
        module_edge_dict: Dictionary of the module clusters.
        vis_mode: Whether to roll the graph or not
        vis_call_depth: How many levels of nested modules to show
        collapsed_modules: Labels of collapsed module nodes that have been made so far.
        show_buffer_layers: Buffer visibility mode.
        overrides: Graphviz attribute overrides for nodes, edges, etc.
    """
    collapse_address = _collapse_address_for_node(
        self,
        node,
        vis_mode=vis_mode,
        collapse_fn=collapse_fn,
        max_module_depth=vis_call_depth,
    )
    fold_ancestor_address = _run_fold_ancestor_for_node(node, repeat_folds)
    if fold_ancestor_address is not None:
        collapse_address = fold_ancestor_address
    segment = _segment_for_node(node, segments)
    if segment is not None:
        _queue_segment_node(
            graphviz_graph,
            module_edge_dict,
            emitted_segment_nodes,
            segment,
        )
    is_collapsed_module = collapse_address is not None
    is_hidden_run_member = (
        collapse_address is not None
        and repeat_folds is not None
        and not _is_run_fold_representative(collapse_address, repeat_folds)
    )

    if segment is not None:
        node_color = "black"
    elif is_collapsed_module and not is_hidden_run_member:
        _build_collapsed_module_node(
            self,
            node,
            graphviz_graph,
            module_edge_dict,
            collapsed_modules,
            vis_mode,
            vis_call_depth,
            collapse_address,
            overrides,  # type: ignore[arg-type]
            node_mode,
            collapsed_node_spec_fn,
            theme,
            repeat_folds,
            collapse_fn,
        )
        node_color = "black"
    elif is_hidden_run_member:
        node_color = "black"
    else:
        node_color = _build_layer_node(
            self,
            node,
            graphviz_graph,
            show_buffer_layers,
            vis_mode,
            overrides,  # type: ignore[arg-type]
            node_mode,
            node_spec_fn,
            theme,
            node_overlay,
            node_label_fields,
            show_containers,
            collapsed_container_nodes,
            show_input_transform_summary,
        )

    _add_edges_for_node(
        self,
        node,
        is_collapsed_module,
        vis_call_depth,
        node_color,
        module_edge_dict,
        edges_used,
        graphviz_graph,
        vis_mode,
        show_buffer_layers,
        overrides,
        collapse_fn,
        edge_map,
        vis_intervention_mode,
        intervention_site_labels,
        captured_forward_edges,
        rankdir,
        show_containers,
        collapsed_container_nodes,
        repeat_folds,
        run_fold_ellipsis_nodes,
        segments,
        segment,
        antiparallel_projected_edges,
    )


def _build_layer_node(
    self: "Trace",
    node: GraphNode,
    graphviz_graph: graphviz.Digraph,
    show_buffer_layers: BufferVisibilityLiteral,
    vis_mode: str,
    overrides: VisualizationOverrides,
    node_mode: VisNodeModeLiteral,
    node_spec_fn: NodeSpecFn | None = None,
    theme: VisualizationTheme | None = None,
    node_overlay: str | OverlayScores | None = None,
    node_label_fields: list[str] | None = None,
    show_containers: ShowContainersLiteral = False,
    collapsed_container_nodes: Mapping[str, str] | None = None,
    show_input_transform_summary: bool = False,
) -> str:
    """Builds and adds a standard (non-collapsed) layer node to the graphviz graph.

    Args:
        node: The Op or Layer node to render.
        graphviz_graph: The graphviz Digraph object to add the node to.
        show_buffer_layers: Buffer visibility mode.
        vis_mode: 'unrolled' or 'rolled'.
        overrides: Graphviz attribute overrides.

    Returns:
        The node color string used for this node.
    """
    if isinstance(node, BoundaryNode):
        fillcolor = INPUT_COLOR if node.boundary_kind == "input" else OUTPUT_COLOR
        spec = NodeSpec(
            lines=[node.display_label],
            shape="oval",
            fillcolor=fillcolor,
            fontcolor="black",
            color="black",
            style="filled,solid",
            extra_attrs={"ordering": "out"},
        )
        if theme is not None:
            spec = apply_theme_to_spec(spec, theme)
        node_args = _node_spec_to_graphviz_args(spec)
        node_args["name"] = node.layer_label
        graphviz_graph.node(**node_args)
        return "black"

    # Get the address, shape, color, and line style:

    node_address, node_shape, node_color = _get_node_address_shape_color(
        self, node, show_buffer_layers
    )
    node_bg_color = _get_node_bg_color(self, node)

    if node.has_input_ancestor:
        line_style = "solid"
    else:
        line_style = "dashed"

    default_spec = NodeSpec(
        lines=compute_default_node_lines(
            node,
            node_address,
            vis_mode,
            node_label_fields=node_label_fields,
            node_overlay=node_overlay,
        ),
        shape=node_shape,
        fillcolor=node_bg_color,
        fontcolor=node_color,
        style=f"filled,{line_style}",
        color=node_color,
        extra_attrs={"ordering": "out"},
    )
    visualizer_path = getattr(node, "visualizer_path", None)
    if isinstance(visualizer_path, str) and visualizer_path.lower().endswith(".png"):
        default_spec = default_spec.replace(
            image=visualizer_path,
            shape="none",
            style="",
            fillcolor=None,
            color=None,
            fontcolor=node_color,
            extra_attrs={
                **default_spec.extra_attrs,
                "imagescale": "true",
                "labelloc": "b",
                "fixedsize": "false",
            },
        )
    annotation_image = _annotation_image_path_for_node(self, node)
    if annotation_image is not None:
        default_spec = default_spec.replace(
            image=annotation_image,
            shape="none",
            style="",
            fillcolor=None,
            color=None,
            fontcolor=node_color,
            extra_attrs={
                **default_spec.extra_attrs,
                "imagescale": "true",
                "labelloc": "b",
                "fixedsize": "false",
            },
        )
    if theme is not None:
        default_spec = apply_theme_to_spec(default_spec, theme)
    spec = _apply_node_spec_fn(self, node, default_spec, node_mode, node_spec_fn)

    # Graphviz node names can't contain colons (used for port syntax), so
    # replace ":" with "pass" in pass-qualified labels (e.g., "relu_1:2" -> "relu_1pass2").
    node_args = _node_spec_to_graphviz_args(spec)
    if node.is_input:
        raw_input_attrs = _render_raw_input(
            self,
            getattr(self, "raw_input", None),
            batch_render=getattr(self, "batch_render", "auto"),
        )
        if raw_input_attrs is not None:
            node_args.update(raw_input_attrs)
        if show_input_transform_summary:
            node_args.update(_input_transform_summary_attrs(self, node_args))
    elif node.is_output:
        raw_output_attrs = _render_raw_output(getattr(self, "decoded_output", None))
        if raw_output_attrs is None:
            raw_output_attrs = _render_raw_output(getattr(self, "raw_output", None))
        if raw_output_attrs is not None:
            node_args.update(raw_output_attrs)
    node_args["name"] = _render_node_label(node, vis_mode).replace(":", "pass")
    if (
        show_containers in {"collapsed", "auto"}
        and collapsed_container_nodes is not None
        and node_args["name"] in collapsed_container_nodes
    ):
        return node_color
    hidden_buffer_addresses = _get_hidden_parent_buffer_addresses(self, node, show_buffer_layers)
    if hidden_buffer_addresses and not (node.is_input or node.is_output or node.is_buffer):
        node_args["peripheries"] = "2"
        hidden_tooltip = f"Hidden buffers: {', '.join(hidden_buffer_addresses)}"
        if "tooltip" in node_args:
            node_args["tooltip"] = f"{node_args['tooltip']}; {hidden_tooltip}"
        else:
            node_args["tooltip"] = hidden_tooltip
    # Colon in bg_color means it's a grad fill (e.g.,
    # "#D9D9D9:#B0B0B0" for mixed trainable/frozen params).
    # Graphviz requires gradangle to render grads.
    if spec.fillcolor is not None and ":" in spec.fillcolor:
        node_args["gradangle"] = "0"
    node_args.update(overlay_border_attrs(node, node_overlay))

    graphviz_graph.node(**node_args)

    if node.is_final_output:
        with graphviz_graph.subgraph() as s:
            s.attr(rank="sink")
            s.node(_render_node_label(node, vis_mode).replace(":", "pass"))

    return node_color


def _queue_segment_node(
    graphviz_graph: graphviz.Digraph,
    module_edge_dict: Dict[str, Any],
    emitted_segment_nodes: set[str] | None,
    segment: SegmentDescriptor,
) -> None:
    """Queue one dashed segment node if it has not already been emitted.

    Parameters
    ----------
    graphviz_graph:
        Top-level Graphviz graph.
    module_edge_dict:
        Module-cluster accumulator.
    emitted_segment_nodes:
        Mutable set of emitted segment node names.
    segment:
        Segment descriptor to render.
    """

    if emitted_segment_nodes is None:
        emitted_segment_nodes = set()
    if segment.name in emitted_segment_nodes:
        return
    emitted_segment_nodes.add(segment.name)
    node_args = {
        "name": segment.name,
        "label": render_lines_to_html([segment.label]),
        "shape": "box",
        "style": "rounded,dashed,filled",
        "fillcolor": "#f7f7f7",
        "color": "#666666",
        "fontcolor": "#222222",
        "ordering": "out",
    }
    if segment.owner is None:
        graphviz_graph.node(**node_args)
    else:
        module_edge_dict[segment.owner].setdefault("nodes", []).append(node_args)


def _render_raw_input(
    trace: "Trace",
    value: Any,
    *,
    batch_render: str = "auto",
) -> dict[str, str] | None:
    """Return Graphviz attributes for a renderable raw input value.

    Parameters
    ----------
    trace:
        Trace that owns the rendered input node.
    value:
        Raw user input stored on the owning ``Trace``.
    batch_render:
        Batch rendering policy.

    Returns
    -------
    dict[str, str] | None
        Node attributes to merge into an input-node spec, or ``None`` when the
        default tensor-shape rendering should be used.
    """

    max_items = _batch_render_limit(batch_render)
    if max_items == 0:
        return None
    if isinstance(value, str):
        text = _truncate_raw_input_text(value, limit=80)
        return {
            "label": render_lines_to_html(["input", text]),
            "tooltip": value,
        }
    include_more = batch_render != "first"
    if isinstance(value, torch.Tensor):
        return _render_raw_input_tensor_batch(
            trace,
            value,
            max_items=max_items,
            include_more=include_more,
        )
    sequence = _raw_input_sequence(value)
    if sequence is None:
        return None
    if len(sequence) == 1:
        return _render_raw_input(trace, sequence[0], batch_render="first")
    if all(isinstance(item, str) for item in sequence):
        strings = cast(Sequence[str], sequence)
        if not include_more:
            strings = strings[:max_items]
        return {
            "label": batch_summary.text_table(strings, max_items),
            "tooltip": repr(strings),
        }
    if all(isinstance(item, Image.Image) for item in sequence):
        images = cast(Sequence[Image.Image], sequence)
        total = len(images) if include_more else min(len(images), max_items)
        return _render_raw_input_image_batch(trace, images, max_items=max_items, total=total)
    return None


def _input_transform_summary_attrs(trace: "Trace", node_args: dict[str, str]) -> dict[str, str]:
    """Return opt-in input preprocessing Graphviz attributes.

    Parameters
    ----------
    trace:
        Trace that owns the rendered input node.
    node_args:
        Current input-node attributes, used to preserve and extend tooltips.

    Returns
    -------
    dict[str, str]
        Attributes to merge into the input node, or an empty dict when no
        preprocessing record is present.
    """

    record = getattr(trace, "input_preprocessor", None)
    if record is None:
        return {}
    verified = bool(getattr(record, "verified", False))
    status = "verified" if verified else "UNVERIFIED"
    source = str(getattr(record, "source", "unknown"))
    description = _truncate_raw_input_text(str(getattr(record, "description", "")), limit=72)
    attrs = {"xlabel": render_lines_to_html(["preprocess", description, f"{status}: {source}"])}
    summary = f"Input preprocessing {status}: {description}"
    existing_tooltip = node_args.get("tooltip")
    attrs["tooltip"] = f"{existing_tooltip}\n{summary}" if existing_tooltip else summary
    return attrs


def _batch_render_limit(batch_render: str) -> int:
    """Return the maximum number of raw-input batch items to render.

    Parameters
    ----------
    batch_render:
        Batch rendering policy string.

    Returns
    -------
    int
        Maximum number of items to render; zero means shape-only fallback.

    Raises
    ------
    ValueError
        If ``batch_render`` is unsupported.
    """

    if batch_render == "auto":
        return 4
    if batch_render == "all":
        return 16
    if batch_render == "first":
        return 1
    if batch_render == "shape_only":
        return 0
    if batch_render.startswith("first_n:"):
        raw_n = batch_render.removeprefix("first_n:")
        try:
            n_items = int(raw_n)
        except ValueError as exc:
            raise ValueError("batch_render first_n value must be an integer.") from exc
        if n_items < 1:
            raise ValueError("batch_render first_n value must be at least 1.")
        return min(n_items, 16)
    raise ValueError("batch_render must be 'auto', 'all', 'first', 'first_n:<N>', or 'shape_only'.")


def _raw_input_sequence(value: Any) -> Sequence[Any] | None:
    """Return a concrete raw-input sequence when ``value`` is a batch container.

    Parameters
    ----------
    value:
        Candidate raw input.

    Returns
    -------
    Sequence[Any] | None
        Concrete sequence for batch rendering, or ``None`` for fallback.
    """

    if isinstance(value, str | bytes | bytearray | Mapping):
        return None
    if isinstance(value, Sequence):
        return value
    if isinstance(value, Iterable) and hasattr(value, "__len__"):
        return tuple(value)
    return None


def _render_raw_input_tensor_batch(
    trace: "Trace",
    tensor: torch.Tensor,
    *,
    max_items: int,
    include_more: bool,
) -> dict[str, str] | None:
    """Return render attributes for a batched raw-input tensor.

    Parameters
    ----------
    trace:
        Trace that owns the rendered input node.
    tensor:
        Candidate raw-input tensor.
    max_items:
        Maximum number of batch items to render.
    include_more:
        Whether to annotate hidden batch items.

    Returns
    -------
    dict[str, str] | None
        Graphviz attributes or ``None`` for shape fallback.
    """

    if tensor.dim() < 2 or int(tensor.shape[0]) <= 1:
        return None
    images = _tensor_batch_to_images(tensor, max_items=max_items)
    if images is None:
        return None
    return _render_raw_input_image_batch(
        trace,
        images,
        max_items=max_items,
        total=int(tensor.shape[0]) if include_more else len(images),
    )


def _render_raw_input_image_batch(
    trace: "Trace",
    images: Sequence[Image.Image],
    *,
    max_items: int,
    total: int,
) -> dict[str, str] | None:
    """Return Graphviz attributes for a PIL image batch.

    Parameters
    ----------
    trace:
        Trace that owns the rendered input node.
    images:
        PIL images to summarize.
    max_items:
        Maximum number of images to render.
    total:
        Total batch size before sampling.

    Returns
    -------
    dict[str, str] | None
        Graphviz attributes or ``None`` for shape fallback.
    """

    if not images:
        return None
    image_dir = _raw_input_visualizer_dir(trace)
    image_path = image_dir / "input_batch_montage.png"
    label_lines = ["input"]
    more_count = total - min(total, max_items)
    if more_count > 0:
        label_lines.append(f"+{more_count} more")
    try:
        montage = batch_summary.montage(images, max_items)
        montage.save(image_path)
    except OSError:
        return {
            "label": render_lines_to_html([*label_lines, "preview unavailable"]),
            "tooltip": f"{total} input images (preview unavailable)",
        }
    width_px, height_px = montage.size
    width_in = max(width_px / 96.0, 0.1)
    height_in = max((height_px + 24 * len(label_lines)) / 96.0, 0.1)
    return {
        "image": str(image_path),
        "imagescale": "true",
        "fixedsize": "true",
        "width": f"{width_in:.3f}",
        "height": f"{height_in:.3f}",
        "label": render_lines_to_html(label_lines),
        "labelloc": "b",
        "margin": "0",
        "shape": "none",
        "tooltip": f"{total} input images",
    }


def _raw_input_visualizer_dir(trace: "Trace") -> Path:
    """Return a directory for raw-input visualization artifacts.

    Parameters
    ----------
    trace:
        Trace that owns the rendered input node.

    Returns
    -------
    Path
        Directory where image artifacts can be written.
    """

    output_dir = getattr(trace, "_visualizer_dir", None)
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="torchlens_visualizers_")
        trace._visualizer_dir = str(output_dir)
    input_dir = Path(output_dir) / "raw_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    return input_dir


def _tensor_batch_to_images(tensor: torch.Tensor, *, max_items: int) -> list[Image.Image] | None:
    """Convert a 4D image tensor batch into PIL images.

    Parameters
    ----------
    tensor:
        Candidate tensor with shape ``(B, C, H, W)``.
    max_items:
        Maximum number of batch items to convert.

    Returns
    -------
    list[Image.Image] | None
        Converted images, or ``None`` for non-image tensors.
    """

    if tensor.dim() != 4 or int(tensor.shape[1]) not in {1, 3}:
        return None
    shown = tensor.detach().cpu()[:max_items].float()
    images = []
    for item in shown:
        item = _normalize_image_tensor(item)
        if item.shape[0] == 1:
            array = (item.squeeze(0).numpy() * 255).astype("uint8")
            images.append(Image.fromarray(array, mode="L").convert("RGB"))
        else:
            array = (item.permute(1, 2, 0).numpy() * 255).astype("uint8")
            images.append(Image.fromarray(array, mode="RGB"))
    return images


def _normalize_image_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize an image tensor into the ``[0, 1]`` display range.

    Parameters
    ----------
    tensor:
        Image tensor with shape ``(C, H, W)``.

    Returns
    -------
    torch.Tensor
        Float tensor clipped or min-max normalized to ``[0, 1]``.
    """

    if float(tensor.min()) >= 0.0 and float(tensor.max()) <= 1.0:
        return tensor.clamp(0.0, 1.0)
    min_value = tensor.min()
    max_value = tensor.max()
    if bool(torch.isclose(max_value, min_value)):
        return torch.zeros_like(tensor)
    return ((tensor - min_value) / (max_value - min_value)).clamp(0.0, 1.0)


def _render_raw_output(value: Any) -> dict[str, str] | None:
    """Return Graphviz attributes for a renderable raw output value.

    Parameters
    ----------
    value:
        Human-readable output metadata stored on the owning ``Trace``.

    Returns
    -------
    dict[str, str] | None
        Node attributes to merge into an output-node spec, or ``None`` when the
        default tensor-shape rendering should be used.
    """

    if value is None:
        return None
    if isinstance(value, str):
        text = _truncate_raw_input_text(value, limit=80)
        return {
            "label": render_lines_to_html(["output", text]),
            "tooltip": value,
        }
    if _is_label_score_sequence(value):
        lines = ["output", *[_format_label_score_row(label, score) for label, score in value]]
        return {
            "label": render_lines_to_html(lines),
            "tooltip": repr(value),
        }
    if _is_batch_topk_output(value):
        lines = _format_batch_topk_output_lines(value)
        return {
            "label": render_lines_to_html(lines),
            "tooltip": repr(value),
        }
    if isinstance(value, Mapping):
        rows = list(value.items())[:5]
        lines = [
            "output",
            *[f"{key}: {_truncate_raw_input_text(str(item), limit=60)}" for key, item in rows],
        ]
        return {
            "label": render_lines_to_html(lines),
            "tooltip": repr(value),
        }
    return None


def _is_batch_topk_output(value: Any) -> bool:
    """Return whether ``value`` is a typed batch top-k decoded output.

    Parameters
    ----------
    value:
        Candidate decoded output value.

    Returns
    -------
    bool
        Whether the value has a renderable ``batch_topk`` row payload.
    """

    return (
        isinstance(value, Mapping)
        and value.get("kind") == "batch_topk"
        and isinstance(value.get("rows"), list)
    )


def _format_batch_topk_output_lines(value: Mapping[str, Any]) -> list[str]:
    """Format a compact batch top-k table for an output node.

    Parameters
    ----------
    value:
        Typed decoded output representation.

    Returns
    -------
    list[str]
        Plain text lines suitable for ``render_lines_to_html``.
    """

    rows = [
        row
        for row in value.get("rows", [])
        if isinstance(row, Mapping) and {"batch_item", "rank", "label", "prob"} <= set(row)
    ]
    lines = ["output"]
    for batch_item in sorted({int(row.get("batch_item", 0)) for row in rows})[:3]:
        item_rows = [row for row in rows if int(row.get("batch_item", -1)) == batch_item][:5]
        lines.append(f"item {batch_item}")
        lines.extend(
            [
                f"{int(row.get('rank', 0))}. "
                f"{_truncate_raw_input_text(str(row.get('label')), limit=36)} "
                f"{float(row.get('prob', 0.0)):.0%}"
                for row in item_rows
            ]
        )
    if not rows:
        lines.append("no decoded rows")
    return lines


def _is_label_score_sequence(value: Any) -> bool:
    """Return whether ``value`` is a flat list of label-score pairs.

    Parameters
    ----------
    value:
        Candidate raw output value.

    Returns
    -------
    bool
        Whether ``value`` can be rendered as prediction rows.
    """

    return isinstance(value, list) and all(
        isinstance(item, tuple)
        and len(item) == 2
        and isinstance(item[0], str | int | float)
        and isinstance(item[1], int | float)
        for item in value
    )


def _format_label_score_row(label: Any, score: int | float) -> str:
    """Format one label-score prediction row.

    Parameters
    ----------
    label:
        Prediction label.
    score:
        Prediction confidence or score.

    Returns
    -------
    str
        Display row for the output node.
    """

    label_text = _truncate_raw_input_text(str(label), limit=48)
    if 0 <= float(score) <= 1:
        score_text = f"{float(score):.0%}"
    else:
        score_text = f"{float(score):.3g}"
    return f"{label_text} {score_text}"


def _truncate_raw_input_text(text: str, *, limit: int) -> str:
    """Return ``text`` truncated to a display-safe length.

    Parameters
    ----------
    text:
        Text to truncate.
    limit:
        Maximum displayed character count including the ellipsis.

    Returns
    -------
    str
        Original text or a shortened form ending in ``...``.
    """

    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _build_collapsed_module_node(
    self: "Trace",
    node: GraphNode,
    graphviz_graph: graphviz.Digraph,
    module_edge_dict: Dict[str, Any],
    collapsed_modules: set[str],
    vis_mode: str,
    vis_call_depth: int,
    collapse_address: str | None,
    overrides: VisualizationOverrides,
    node_mode: VisNodeModeLiteral,
    collapsed_node_spec_fn: CollapsedNodeSpecFn | None = None,
    theme: VisualizationTheme | None = None,
    repeat_folds: Mapping[str, "ModuleRepeatFold"] | None = None,
    collapse_fn: CollapseFn | None = None,
) -> None:
    """Builds and adds a collapsed module box node to the graphviz graph.

    Args:
        node: The Op or Layer node triggering the collapse.
        graphviz_graph: The graphviz Digraph object to add the node to.
        module_edge_dict: Dict mapping each cluster to its queued nodes and edges.
        collapsed_modules: Set of collapsed module names already added; updated in place.
        vis_mode: 'unrolled' or 'rolled'.
        vis_call_depth: Maximum nesting depth; nodes at this depth are collapsed.
        overrides: Graphviz attribute overrides.
    """
    # Access the module at the collapse threshold depth.  This index is safe
    # because _is_collapsed_module already verified the node is deep enough.
    address_w_pass = (
        collapse_address if collapse_address is not None else node.modules[vis_call_depth - 1]
    )
    # rsplit with maxsplit=1 handles module names containing colons (#104).
    module_tuple = address_w_pass.rsplit(":", 1)
    try:
        module_call = self.module_calls[address_w_pass]
        module_output_layer = self.ops[module_call.output_ops[-1]]
    except (KeyError, IndexError):
        module_output_layer = node
    module_output_shape = getattr(module_output_layer, "shape", None)
    if module_output_shape is None:
        module_output_shape = getattr(module_output_layer, "out_shape", None)
    module_output_shape = module_output_shape or ()
    module_output_fsize = getattr(module_output_layer, "activation_memory", None)
    if module_output_fsize is None:
        module_output_fsize = "0 B"
    address, call_index = module_tuple
    fold = _run_fold_for_address(address, repeat_folds)
    if fold is not None:
        address = fold.representative
        address_w_pass = f"{address}:{call_index}" if vis_mode == "unrolled" else address
        module_tuple = address_w_pass.rsplit(":", 1)
    ml = self.modules[address]
    module_type = ml.class_name  # type: ignore[union-attr]
    module_num_calls = ml.num_calls  # type: ignore[union-attr]
    module_nparams = ml.num_params  # type: ignore[union-attr]
    module_nparams_trainable = ml.num_params_trainable  # type: ignore[union-attr]
    module_nparams_frozen = ml.num_params_frozen  # type: ignore[union-attr]

    # In unrolled mode, each pass of a module is a separate collapsed node
    # (e.g., "encoder.layer.0pass1").  In rolled mode, all ops share one
    # node (e.g., "encoder.layer.0").
    if vis_mode == "unrolled":
        graph_node_label = "pass".join(module_tuple)
        module_call = ml.ops[int(call_index) - 1]  # type: ignore[index]
        module_num_tensors = module_call.num_layers
        module_has_input_ancestor = any(self[layer].has_input_ancestor for layer in module_call.ops)
        if (
            _collapsed_module_should_show_remainder(self, address, module_call.ops, collapse_fn)
            and fold is None
        ):
            remainder_stats = _collapsed_module_remainder_stats(self, address, module_call.ops)
            module_num_tensors = remainder_stats["num_layers"]
            module_nparams = remainder_stats["num_params"]
            module_nparams_trainable = remainder_stats["num_params_trainable"]
            module_nparams_frozen = remainder_stats["num_params_frozen"]
    else:
        graph_node_label = module_tuple[0]
        module_num_tensors = ml.num_layers
        module_has_input_ancestor = any(self[layer].has_input_ancestor for layer in ml.layer_labels)  # type: ignore[union-attr]

    # Deduplicate: multiple layers in the same collapsed module will each
    # trigger this function, but the node should only be added once.
    if graph_node_label in collapsed_modules:
        return

    module_suffix = _collapsed_module_rolling_suffix(self, address)
    if module_num_calls == 1:
        node_title = f"<b>@{address}</b>"
    elif vis_mode == "unrolled" and (module_num_calls > 1):
        node_title = f"<b>@{address}:{call_index}</b>"
    elif module_suffix:
        node_title = f"<b>@{address}{module_suffix}</b>"
    else:
        node_title = f"<b>@{address} (x{module_num_calls})</b>"

    shape_str = format_shape(module_output_shape)

    if module_nparams > 0:
        if module_nparams_frozen == 0:
            bg_color = TRAINABLE_PARAMS_BG_COLOR
        elif module_nparams_trainable == 0:
            bg_color = FROZEN_PARAMS_BG_COLOR
        else:
            bg_color = TRAINABLE_PARAMS_BG_COLOR + ":" + FROZEN_PARAMS_BG_COLOR
    else:
        bg_color = DEFAULT_BG_COLOR

    if module_has_input_ancestor:
        line_style = "solid"
    else:
        line_style = "dashed"

    # Build param detail string for collapsed module
    if module_nparams == 0:
        param_detail = "0 parameters"
    elif module_nparams_frozen == 0:
        param_detail = f"{module_nparams} params (all trainable)"
    elif module_nparams_trainable == 0:
        param_detail = f"{module_nparams} params (all frozen)"
    else:
        param_detail = (
            f"{module_nparams} params ({module_nparams_trainable} trainable, "
            f"{module_nparams_frozen} frozen)"
        )

    lines = [
        node_title.replace("<b>", "").replace("</b>", ""),
        module_type,
        f"{shape_str}, {format_memory(module_output_fsize)}",
    ]
    if fold is not None and fold.shape_summary is not None:
        lines.append(f"shapes {fold.shape_summary}")
    lines.extend([f"{module_num_tensors} layers total", param_detail])
    default_spec = NodeSpec(
        lines=lines,
        shape="box3d",
        fillcolor=bg_color,
        fontcolor="black",
        color="black",
        style=f"filled,{line_style}",
        extra_attrs={"ordering": "out"},
    )
    if theme is not None:
        default_spec = apply_theme_to_spec(default_spec, theme)
    mode_fn = COLLAPSED_MODE_REGISTRY[node_mode]
    mode_result = mode_fn(ml, default_spec)  # type: ignore[arg-type]
    mode_spec = default_spec if mode_result is None else mode_result
    if collapsed_node_spec_fn is not None:
        result = collapsed_node_spec_fn(ml, mode_spec)  # type: ignore[arg-type]
        spec = mode_spec if result is None else result
    else:
        spec = mode_spec

    node_args = _node_spec_to_graphviz_args(spec)
    node_args["name"] = graph_node_label
    if spec.fillcolor is not None and ":" in spec.fillcolor:
        node_args["gradangle"] = "0"

    owner_key = _collapsed_module_owner_key(self, address, call_index, vis_mode)
    if owner_key is None:
        graphviz_graph.node(**node_args)
    else:
        module_edge_dict[owner_key].setdefault("nodes", []).append(node_args)
    collapsed_modules.add(graph_node_label)


def _atomic_module_split_range(trace: "Trace", layer_log: GraphNode, address: str) -> str:
    """Return the call-range an atomic module rectangle should mark, or ``""``.

    An atomic (single-op) module renders as a rectangle per call site. When the
    module is reused across split sites the rectangle's ``@module`` marking carries
    the call range that distinguishes it, e.g. ``@shared:1-2,3-5`` for a single
    rolled node spanning two loops, or ``@relu:1`` / ``@relu:2-4`` for two separate
    rectangles. A module used at a single contiguous site needs no range (the
    title's ``(xN)`` count suffices), so this returns ``""``.

    Parameters
    ----------
    trace:
        Owning trace.
    layer_log:
        Atomic module layer being rendered.
    address:
        The atomic module's address.

    Returns
    -------
    str
        Compact call range (no leading colon), or ``""`` when not split.
    """

    if not isinstance(layer_log, Layer):
        return ""
    groups = _call_groups_for_layer(layer_log)
    if groups:
        return _format_call_groups(groups)
    sibling_atomic_layers = sum(
        1
        for other in trace.layer_logs.values()
        if isinstance(other, Layer)
        and getattr(other, "is_atomic_module", False)
        and other.modules
        and other.modules[-1].rsplit(":", 1)[0] == address
    )
    if sibling_atomic_layers > 1:
        calls = _common_module_call_indices(layer_log).get(address, [])
        if calls:
            return _compact_int_ranges(calls)
    return ""


def _buffer_versions_for_layer(layer_log: "Layer") -> tuple[int, ...]:
    """Return flat buffer versions represented by a layer.

    Parameters
    ----------
    layer_log:
        Layer to inspect.

    Returns
    -------
    tuple[int, ...]
        Sorted buffer pass indices, excluding ``None``.
    """

    versions = {
        int(op.buffer_pass)
        for op in layer_log.ops.values()
        if getattr(op, "buffer_pass", None) is not None
    }
    return tuple(sorted(versions))


def _rolling_annotation(layer_log: GraphNode, vis_mode: str) -> RollingAnnotation | None:
    """Build a rolled-view annotation for a layer node.

    Parameters
    ----------
    layer_log:
        Layer or Op being rendered.
    vis_mode:
        Active visualization mode.

    Returns
    -------
    RollingAnnotation | None
        Annotation for rolled multi-pass layers, otherwise ``None``.
    """

    layer_log = _unwrap_focus_node(layer_log)
    if vis_mode != "rolled" or not isinstance(layer_log, Layer) or layer_log.num_passes <= 1:
        return None

    call_groups = _call_groups_for_layer(layer_log)
    buffer_versions = _buffer_versions_for_layer(layer_log) if layer_log.is_buffer else ()
    if not call_groups and not buffer_versions:
        return None
    return RollingAnnotation(call_groups=call_groups, buffer_versions=buffer_versions)


def _format_rolling_suffix(annotation: RollingAnnotation | None) -> str:
    """Return the face suffix for a rolled annotation.

    Parameters
    ----------
    annotation:
        Annotation to render.

    Returns
    -------
    str
        Suffix beginning with a colon, or an empty string.
    """

    if annotation is None or not annotation.call_groups:
        return ""
    return f":{_format_call_groups(annotation.call_groups)}"


def _get_node_address_shape_color(
    self: "Trace",
    node: GraphNode,
    show_buffer_layers: BufferVisibilityLiteral | bool,
) -> Tuple[str, str, str]:
    """Gets the node shape, address, and color for the graphviz figure.

    Args:
        node: node to add

    Returns:
        node_address: address of the node
        node_shape: shape of the node
        node_color: color of the node
    """
    source_node = _unwrap_focus_node(node)
    if isinstance(source_node, BoundaryNode):
        raise ValueError("Boundary nodes are rendered by the boundary path.")
    show_buffer_layers = _normalize_buffer_visibility(show_buffer_layers)
    if show_buffer_layers != "always":
        only_non_buffer_layer = _is_only_non_buffer_in_module(self, node, show_buffer_layers)
    else:
        only_non_buffer_layer = False

    if (node.is_atomic_module or only_non_buffer_layer) and (len(node.modules) > 0):
        if isinstance(source_node, Op):
            module_pass_exited = node.modules[-1]
            module, _ = module_pass_exited.split(":")
            if self.modules[module].num_calls == 1:  # type: ignore[union-attr]
                node_address = module
            else:
                node_address = module_pass_exited
        else:
            sample_module_pass = node.modules[-1]
            module = sample_module_pass.split(":")[0]
            split_range = _atomic_module_split_range(self, source_node, module)
            node_address = f"{module}:{split_range}" if split_range else module

        node_address = "<br/>@" + node_address
        node_shape = "box"
        node_color = "black"
    elif node.is_buffer:
        annotation = (
            _rolling_annotation(source_node, "rolled") if isinstance(source_node, Layer) else None
        )
        if annotation is not None and annotation.buffer_versions:
            address = f"{source_node.address}:{_compact_int_ranges(annotation.buffer_versions)}"
        elif self.buffer_num_calls[source_node.address] == 1:
            address = source_node.address
        else:
            address = f"{source_node.address}:{source_node.buffer_pass}"
        node_address = "<br/>@" + address
        node_shape = "cylinder"
        node_color = "black"
    elif node.is_output or node.is_input:
        node_address = "<br/>@" + node.io_role
        node_shape = "oval"
        node_color = "black"
    else:
        node_address = ""
        node_shape = "oval"
        node_color = "black"

    return node_address, node_shape, node_color


def _is_only_non_buffer_in_module(
    self: "Trace", node: GraphNode, show_buffer_layers: BufferVisibilityLiteral
) -> bool:
    """Returns True if a layer is the only non-buffer layer in a leaf module.

    Leaf modules are those with no child submodules. Container modules with
    functional ops at the end should NOT match — those ops are rendered as
    ovals, not boxes (issue #48).

    Args:
        node: The Op or Layer node to check.
        show_buffer_layers: Buffer visibility mode.
    """
    # Check whether it leaves its module:
    if not (
        (len(node.output_of_modules) > 0)
        and (len(node.modules) > 0)
        and (node.modules[-1].split(":")[0] in node.output_of_modules)
    ):
        return False

    # Only apply box rendering for leaf modules (no child submodules).
    exited_module = node.modules[-1].split(":")[0]
    if exited_module in self.modules and len(self.modules[exited_module].call_children) > 0:
        return False

    # Now check whether all of its parents are either buffers, or are outside the module.
    # If any aren't, return False.

    for parent_layer_label in node.parents:
        if parent_layer_label.startswith("__module_focus_"):
            continue
        source_node = _unwrap_focus_node(node)
        if isinstance(source_node, Op):
            parent_layer = self[parent_layer_label]
        else:
            parent_layer = self.layer_logs[parent_layer_label]
        if (
            (not parent_layer.is_buffer) or _is_buffer_visible(parent_layer, show_buffer_layers)
        ) and ((len(parent_layer.modules) > 0) and parent_layer.modules[-1] == node.modules[-1]):
            return False

    return True


def _get_node_bg_color(self: "Trace", node: GraphNode) -> str:
    """Returns the background color hex string for a graph node based on its type.

    Maps node types to colors: input=green, output=red, boolean=orange,
    parameterized layers=blue (trainable) or gray (frozen), default=white.

    Args:
        node: node to add

    Returns:
        node_bg_color: background color of the node
    """
    if node.is_input:
        bg_color = INPUT_COLOR
    elif node.is_output:
        bg_color = OUTPUT_COLOR
    elif node.is_terminal_bool:
        bg_color = BOOL_NODE_COLOR
    elif node.uses_params:
        param_logs = getattr(node, "_param_logs", [])
        if param_logs:
            trainable_flags = [pl.is_trainable for pl in param_logs]
            all_trainable = all(trainable_flags)
            all_frozen = not any(trainable_flags)
            if all_trainable:
                bg_color = TRAINABLE_PARAMS_BG_COLOR
            elif all_frozen:
                bg_color = FROZEN_PARAMS_BG_COLOR
            else:
                bg_color = TRAINABLE_PARAMS_BG_COLOR + ":" + FROZEN_PARAMS_BG_COLOR
        else:
            bg_color = PARAMS_NODE_BG_COLOR
    else:
        bg_color = DEFAULT_BG_COLOR
    return bg_color


def _apply_node_spec_fn(
    trace: "Trace",
    node: GraphNode,
    default_spec: NodeSpec,
    node_mode: VisNodeModeLiteral,
    node_spec_fn: NodeSpecFn | None,
) -> NodeSpec:
    """Apply a layer node callback to a default spec.

    Parameters
    ----------
    trace:
        Owning Trace.
    node:
        Rendered Op or Layer.
    default_spec:
        Default node spec.
    node_mode:
        Preset to apply before the optional user callback.
    node_spec_fn:
        Optional user callback. Unrolled nodes are represented to the callback
        by their parent Layer.

    Returns
    -------
    NodeSpec
        Callback result, or ``default_spec`` when the callback returns ``None``.
    """

    layer_log = _layer_log_for_node(trace, node)
    mode_fn = MODE_REGISTRY[node_mode]
    mode_result = mode_fn(layer_log, default_spec)
    mode_spec = default_spec if mode_result is None else mode_result
    if node_spec_fn is None:
        return mode_spec
    result = node_spec_fn(layer_log, mode_spec)
    return mode_spec if result is None else result


def _annotation_image_path_for_node(trace: "Trace", node: GraphNode) -> str | None:
    """Return a user annotation image path for a rendered node.

    Parameters
    ----------
    trace:
        Owning Trace.
    node:
        Rendered Op or Layer.

    Returns
    -------
    str | None
        Image path stored in ``annotations["user"]["image"]``, if present.
    """

    if isinstance(node, BoundaryNode):
        return None
    candidates: list[Any] = [node]
    try:
        candidates.append(_layer_log_for_node(trace, node))
    except ValueError:
        pass
    for candidate in candidates:
        annotations = getattr(candidate, "annotations", None)
        if not isinstance(annotations, dict):
            continue
        user_annotations = annotations.get("user")
        if not isinstance(user_annotations, dict):
            continue
        image = user_annotations.get("image")
        if isinstance(image, str) and image:
            return image
    return None


def _layer_log_for_node(trace: "Trace", node: GraphNode) -> "Layer":
    """Return the aggregate Layer for ``node``.

    Parameters
    ----------
    trace:
        Owning Trace.
    node:
        Op or Layer.

    Returns
    -------
    Layer
        Aggregate layer log for callbacks.
    """

    node = _unwrap_focus_node(node)
    if isinstance(node, BoundaryNode):
        raise ValueError("Synthetic boundary nodes do not have Layer metadata.")
    if isinstance(node, Layer):
        return node
    return trace.layer_logs[node.layer_label]


def compute_default_node_lines(
    layer_log: GraphNode,
    node_address: str = "",
    vis_mode: str = "unrolled",
    *,
    node_label_fields: list[str] | None = None,
    node_overlay: str | OverlayScores | None = None,
) -> list[str]:
    """Build default plain-text rows for a layer node.

    Parameters
    ----------
    layer_log:
        Op or Layer to render.
    node_address:
        Existing address suffix from TorchLens node address logic.
    vis_mode:
        ``"unrolled"`` or ``"rolled"``.
    node_label_fields:
        Optional label fields to render instead of the default field set.
    node_overlay:
        Optional overlay to append as an additional label row.

    Returns
    -------
    list[str]
        Plain-text rows for ``NodeSpec.lines``.
    """

    layer_log = _unwrap_focus_node(layer_log)
    if isinstance(layer_log, BoundaryNode):
        return [layer_log.display_label]

    if node_label_fields is not None:
        selected_lines = _compute_selected_node_lines(
            layer_log, node_address, vis_mode, node_label_fields
        )
        overlay = overlay_line(layer_log, node_overlay)
        if overlay is not None:
            selected_lines.append(overlay)
        return selected_lines

    annotation = _rolling_annotation(layer_log, vis_mode)
    # An atomic module rectangle carries its split call range on the ``@module``
    # marking (e.g. ``@shared:1-2,3-5``); keep the title's count clean and
    # non-redundant in that case, and fall back to a plain ``(xN)`` count.
    atomic_marking_has_range = getattr(layer_log, "is_atomic_module", False) and (
        ":" in node_address
    )
    if (layer_log.num_passes > 1) and (vis_mode == "unrolled"):
        call_label = f":{layer_log.pass_index}"
    elif (layer_log.num_passes > 1) and (vis_mode == "rolled"):
        if atomic_marking_has_range:
            call_label = ""
        else:
            rolling_suffix = _format_rolling_suffix(annotation)
            if rolling_suffix:
                call_label = rolling_suffix
            else:
                call_label = f" (x{_rolled_visual_num_passes(layer_log)})"
    else:
        call_label = ""

    if layer_log.layer_type in ["input", "output", "buffer"]:
        title = f"{layer_log.layer_type}_{layer_log.type_index}{call_label}"
    else:
        title = f"{layer_log.layer_type}_{layer_log.type_index}_{layer_log.step_index}{call_label}"

    lines: list[str] = []
    if layer_log.is_terminal_bool:
        lines.append(str(layer_log.bool_value).upper())
    lines.append(title)
    lines.append(f"{format_shape(layer_log.shape)}, {format_memory(layer_log.activation_memory)}")

    module_kwargs = format_module_kwargs(layer_log)
    if module_kwargs is not None:
        lines.append(module_kwargs)

    param_line = format_param_list(layer_log)
    if param_line is not None:
        lines.append(param_line)

    address_line = format_module_path(node_address)
    if address_line is not None:
        lines.append(address_line)
    overlay = overlay_line(layer_log, node_overlay)
    if overlay is not None:
        lines.append(overlay)
    return lines


def _compute_selected_node_lines(
    layer_log: GraphNode,
    node_address: str,
    vis_mode: str,
    node_label_fields: list[str],
) -> list[str]:
    """Build node-label rows from an explicit field picker.

    Parameters
    ----------
    layer_log:
        Op or Layer to render.
    node_address:
        Existing address suffix from TorchLens node address logic.
    vis_mode:
        ``"unrolled"`` or ``"rolled"``.
    node_label_fields:
        Requested field names.

    Returns
    -------
    list[str]
        Selected label rows.

    Raises
    ------
    ValueError
        If an unknown field is requested.
    """

    rows: list[str] = []
    for field_name in node_label_fields:
        if field_name in {"label", "name"}:
            rows.append(str(getattr(layer_log, "layer_label", "")))
        elif field_name in {"type", "op", "operation"}:
            rows.append(str(getattr(layer_log, "func_name", None) or layer_log.layer_type))
        elif field_name == "shape":
            rows.append(format_shape(layer_log.shape))
        elif field_name in {"memory", "bytes"}:
            rows.append(str(getattr(layer_log, "activation_memory", "")))
        elif field_name == "module":
            rows.append(format_module_path(node_address) or "@root")
        elif field_name == "params":
            param_line = format_param_list(layer_log)
            if param_line is not None:
                rows.append(param_line)
        elif field_name == "pass":
            rows.append(
                str(
                    getattr(layer_log, "call_index", 1)
                    if vis_mode == "unrolled"
                    else getattr(layer_log, "num_passes", 1)
                )
            )
        elif field_name == "flops":
            rows.append(str(getattr(layer_log, "flops_forward", 0) or 0))
        elif field_name == "time":
            rows.append(str(Duration(float(getattr(layer_log, "func_duration", 0.0) or 0.0))))
        else:
            raise ValueError(f"Unsupported node label field: {field_name!r}.")
    return rows or compute_default_node_lines(layer_log, node_address, vis_mode)


__all__ = [
    "_add_node_to_graphviz",
    "_add_unrolled_backward_pass_clusters",
    "_annotation_image_path_for_node",
    "_append_container_overlay_edge",
    "_apply_node_spec_fn",
    "_atomic_module_split_range",
    "_batch_render_limit",
    "_buffer_versions_for_layer",
    "_build_collapsed_module_node",
    "_build_layer_node",
    "_compute_selected_node_lines",
    "_container_boundary_edge_attrs",
    "_container_path_leaf_label",
    "_format_batch_topk_output_lines",
    "_format_label_score_row",
    "_format_rolling_suffix",
    "_get_hidden_parent_buffer_addresses",
    "_get_node_address_shape_color",
    "_get_node_bg_color",
    "_input_transform_summary_attrs",
    "_is_batch_topk_output",
    "_is_label_score_sequence",
    "_is_only_non_buffer_in_module",
    "_layer_log_for_node",
    "_normalize_buffer_visibility",
    "_normalize_image_tensor",
    "_op_for_occurrence_label",
    "_queue_segment_node",
    "_raw_input_sequence",
    "_raw_input_visualizer_dir",
    "_render_node_name_for_occurrence",
    "_render_output_node_name_for_path",
    "_render_raw_input",
    "_render_raw_input_image_batch",
    "_render_raw_input_tensor_batch",
    "_render_raw_output",
    "_rolling_annotation",
    "_tensor_batch_to_images",
    "_truncate_raw_input_text",
    "_visible_backward_calls_by_pass",
    "compute_default_node_lines",
]
