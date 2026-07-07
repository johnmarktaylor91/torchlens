"""Leaf helper functions for Graphviz rendering."""

# ruff: noqa: F403, F405

from ._render_common import *


def _backward_dot_node_name(grad_fn_handle: "GradFn") -> str:
    """Return a DOT-safe node name for a grad_fn_handle log.

    Parameters
    ----------
    grad_fn_handle:
        GradFn to name.

    Returns
    -------
    str
        DOT-safe node identifier.
    """

    return f"grad_fn_{grad_fn_handle.grad_fn_object_id}"


if TYPE_CHECKING:
    from ..data_classes.grad_fn import GradFn
    from ..data_classes.module import Module
    from ..data_classes.trace import Trace
    from .auto_collapse import ModuleRunFold


def _backward_dot_call_node_name(grad_fn_handle: "GradFn", call: Any) -> str:
    """Return a DOT-safe node name for one GradFnCall.

    Parameters
    ----------
    grad_fn_handle:
        GradFn owning the call.
    call:
        GradFnCall-like record.

    Returns
    -------
    str
        DOT-safe node identifier.
    """

    return (
        f"grad_fn_{grad_fn_handle.grad_fn_object_id}_"
        f"bwd{getattr(call, 'backward_pass_index', 0)}_call{getattr(call, 'call_index', 0)}"
    )


def _grad_fn_call_matches_backward_filter(call: Any, pass_filter: BackwardPassFilter) -> bool:
    """Return whether a GradFnCall should be visible for a pass filter.

    Parameters
    ----------
    call:
        GradFnCall-like record.
    pass_filter:
        Normalized backward-pass filter.

    Returns
    -------
    bool
        ``True`` when the call participates in a requested pass.
    """

    if pass_filter is None:
        return True
    return getattr(call, "backward_pass_index", None) in pass_filter


def _grad_fn_matches_backward_filter(
    grad_fn_handle: "GradFn",
    pass_filter: BackwardPassFilter,
) -> bool:
    """Return whether a GradFn has at least one visible call.

    Parameters
    ----------
    grad_fn_handle:
        GradFn to inspect.
    pass_filter:
        Normalized backward-pass filter.

    Returns
    -------
    bool
        ``True`` when any call participates in the selected passes.
    """

    return any(
        _grad_fn_call_matches_backward_filter(call, pass_filter)
        for call in grad_fn_handle.calls.values()
    )


def _add_backward_node_to_graphviz(
    grad_fn_handle: "GradFn",
    graphviz_graph: graphviz.Digraph,
    node_spec_fn: BackwardNodeSpecFn | None,
    pass_filter: BackwardPassFilter = None,
) -> None:
    """Add one backward grad_fn_handle node to a Graphviz graph.

    Parameters
    ----------
    grad_fn_handle:
        GradFn to render.
    graphviz_graph:
        Graphviz Digraph object.
    node_spec_fn:
        Optional callback receiving ``(grad_fn_handle, default_spec)``.
    pass_filter:
        Normalized backward-pass filter.
    """

    node_args = _backward_node_graphviz_args(
        grad_fn_handle,
        node_spec_fn,
        pass_filter=pass_filter,
    )
    graphviz_graph.node(**node_args)


def _backward_node_graphviz_args(
    grad_fn_handle: "GradFn",
    node_spec_fn: BackwardNodeSpecFn | None,
    call: Any | None = None,
    pass_filter: BackwardPassFilter = None,
) -> dict[str, Any]:
    """Build Graphviz node arguments for one backward grad_fn_handle.

    Parameters
    ----------
    grad_fn_handle:
        GradFn to render.
    node_spec_fn:
        Optional callback receiving ``(grad_fn_handle, default_spec)``.
    call:
        Optional GradFnCall when rendering in unrolled mode.
    pass_filter:
        Normalized backward-pass filter.

    Returns
    -------
    dict[str, Any]
        Keyword arguments accepted by ``graphviz.Digraph.node``.
    """

    default_spec = NodeSpec(
        lines=_compute_backward_node_lines(
            grad_fn_handle,
            call=call,
            pass_filter=pass_filter,
        ),
        shape="oval",
        fillcolor=_backward_node_fillcolor(grad_fn_handle),
        fontcolor="black",
        color=BACKWARD_NODE_BORDER_COLOR,
        style="filled,solid",
        penwidth=1.8,
        extra_attrs={"ordering": "out"},
    )
    if node_spec_fn is not None:
        result = node_spec_fn(grad_fn_handle, default_spec)
        spec = default_spec if result is None else result
    else:
        spec = default_spec
    node_args = _node_spec_to_graphviz_args(spec)
    node_args["name"] = (
        _backward_dot_node_name(grad_fn_handle)
        if call is None
        else _backward_dot_call_node_name(grad_fn_handle, call)
    )
    return node_args


def _backward_node_fillcolor(grad_fn_handle: "GradFn") -> str:
    """Return the fill color for a backward node.

    Parameters
    ----------
    grad_fn_handle:
        GradFn to style.

    Returns
    -------
    str
        Graphviz fill color.
    """

    order = getattr(grad_fn_handle, "order", None)
    if order is not None and order > 1:
        return BACKWARD_HIGHER_ORDER_COLOR
    return BACKWARD_NODE_COLOR


def _backward_edge_attrs(tail: "GradFn", head: "GradFn") -> dict[str, str]:
    """Return Graphviz attributes for a backward GradFn edge.

    Parameters
    ----------
    tail:
        Edge tail GradFn.
    head:
        Edge head GradFn.

    Returns
    -------
    dict[str, str]
        Graphviz edge attributes.
    """

    edge_attrs = {"color": GRADIENT_ARROW_COLOR, "fontcolor": GRADIENT_ARROW_COLOR}
    if tail.type == "accumulategrad" or head.type == "accumulategrad":
        edge_attrs["style"] = BACKWARD_ACCUMULATION_EDGE_STYLE
        edge_attrs["label"] = "accum"
        edge_attrs["labelfontsize"] = "8"
    return edge_attrs


def _add_combined_backward_nodes(
    trace: "Trace",
    module_cluster_dict: Dict[str, Any],
    graphviz_graph: graphviz.Digraph,
    node_spec_fn: BackwardNodeSpecFn | None,
    intervening_cluster: InterveningClusterMode,
    pass_filter: BackwardPassFilter,
) -> None:
    """Add backward nodes to the combined graph and module clusters.

    Parameters
    ----------
    trace:
        Trace containing grad_fn_handle metadata.
    module_cluster_dict:
        Shared module cluster accumulator.
    graphviz_graph:
        Graphviz graph being rendered.
    node_spec_fn:
        Optional backward node callback.
    intervening_cluster:
        Placement mode for intervening grad_fns.
    pass_filter:
        Normalized backward-pass filter.
    """

    for grad_fn_handle in trace.grad_fns:
        if not _grad_fn_matches_backward_filter(grad_fn_handle, pass_filter):
            continue
        node_args = _backward_node_graphviz_args(
            grad_fn_handle,
            node_spec_fn,
            pass_filter=pass_filter,
        )
        module_key = _module_key_for_grad_fn(trace, grad_fn_handle, intervening_cluster)
        if module_key is None:
            graphviz_graph.node(**node_args)
            continue
        module_cluster_dict[module_key]["nodes"].append(node_args)
        module_cluster_dict[module_key]["has_input_ancestor"] = True


def _add_combined_backward_edges(
    trace: "Trace",
    graphviz_graph: graphviz.Digraph,
    pass_filter: BackwardPassFilter,
) -> None:
    """Add backward grad_fn_handle edges to a combined graph.

    Parameters
    ----------
    trace:
        Trace containing grad_fn_handle metadata.
    graphviz_graph:
        Graphviz graph being rendered.
    pass_filter:
        Normalized backward-pass filter.
    """

    visible_ids = {
        grad_fn_handle.grad_fn_object_id
        for grad_fn_handle in trace.grad_fns
        if _grad_fn_matches_backward_filter(grad_fn_handle, pass_filter)
    }
    for grad_fn_handle in trace.grad_fns:
        if grad_fn_handle.grad_fn_object_id not in visible_ids:
            continue
        tail_name = _backward_dot_node_name(grad_fn_handle)
        for next_grad_fn_id in grad_fn_handle.next_grad_fn_ids:
            if next_grad_fn_id not in visible_ids:
                continue
            head_name = _backward_dot_node_name(trace.grad_fn_logs[next_grad_fn_id])
            graphviz_graph.edge(
                tail_name,
                head_name,
                **_backward_edge_attrs(grad_fn_handle, trace.grad_fn_logs[next_grad_fn_id]),
            )


def _add_combined_correspondence_edges(
    trace: "Trace",
    graphviz_graph: graphviz.Digraph,
    intervening_cluster: InterveningClusterMode,
    pass_filter: BackwardPassFilter,
) -> None:
    """Add dashed forward-to-backward correspondence edges.

    Parameters
    ----------
    trace:
        Trace containing paired forward and grad_fn_handle metadata.
    graphviz_graph:
        Graphviz graph being rendered.
    intervening_cluster:
        Placement mode used to infer optional cluster boundary attributes.
    pass_filter:
        Normalized backward-pass filter.
    """

    for grad_fn_handle in trace.grad_fns:
        if not grad_fn_handle.has_op:
            continue
        if not _grad_fn_matches_backward_filter(grad_fn_handle, pass_filter):
            continue
        edge_attrs = {
            "color": GRADIENT_ARROW_COLOR,
            "fontcolor": GRADIENT_ARROW_COLOR,
            "style": "dashed",
            "constraint": "false",
            "arrowsize": ".6",
        }
        module_key = _module_key_for_grad_fn(trace, grad_fn_handle, intervening_cluster)
        if module_key is not None:
            cluster_name = f"cluster_{module_key.replace(':', '_pass')}"
            edge_attrs["ltail"] = cluster_name
            edge_attrs["lhead"] = cluster_name
        op = grad_fn_handle.op
        if op is not None:
            graphviz_graph.edge(
                op.layer_label,
                _backward_dot_node_name(grad_fn_handle),
                **edge_attrs,
            )


def _module_key_for_grad_fn(
    trace: "Trace",
    grad_fn_handle: "GradFn",
    mode: InterveningClusterMode,
) -> str | None:
    """Return the module cluster key for a grad_fn_handle in combined rendering.

    Parameters
    ----------
    trace:
        Trace containing forward, backward, and parameter metadata.
    grad_fn_handle:
        GradFn to place.
    mode:
        Placement mode for intervening grad_fns.

    Returns
    -------
    str | None
        Unrolled module-call key, special cluster key, or None for top level.
    """

    op = grad_fn_handle.op
    if op is not None:
        return _module_key_for_forward_op(op)
    if grad_fn_handle.type == "accumulategrad":
        param_key = _param_module_for_accumulate_grad(trace, grad_fn_handle)
        if param_key is not None:
            return param_key
    if mode == "outside":
        return None
    if mode == "own":
        return "__intervening__"
    if mode == "upstream":
        return _infer_intervening_module_upstream(trace, grad_fn_handle)
    if mode == "downstream":
        return _infer_intervening_module_downstream(trace, grad_fn_handle)
    raise ValueError("intervening_cluster must be 'upstream', 'outside', 'downstream', or 'own'.")


def _module_key_for_forward_op(op: "Layer") -> str | None:
    """Return the unrolled module cluster key for a forward op.

    Parameters
    ----------
    op:
        Forward operation or layer log associated with a grad_fn_handle.

    Returns
    -------
    str | None
        Module-call key or None for top-level ops.
    """

    output_modules = list(getattr(op, "output_of_modules", []) or [])
    if getattr(op, "is_module_output", False) and output_modules:
        output_module = str(output_modules[0])
        output_calls = list(getattr(op, "output_of_module_calls", []) or [])
        for output_call in output_calls:
            if str(output_call).split(":", 1)[0] == output_module:
                return str(output_call)
        return f"{output_module}:1"
    modules = list(getattr(op, "modules", []) or [])
    if not modules:
        return None
    return str(modules[-1])


def _param_module_for_accumulate_grad(trace: "Trace", grad_fn_handle: "GradFn") -> str | None:
    """Return an unambiguous owning module for an AccumulateGrad node.

    Parameters
    ----------
    trace:
        Trace containing parameter metadata and grad_fn_handle parameter refs.
    grad_fn_handle:
        AccumulateGrad log.

    Returns
    -------
    str | None
        Owning module-call key, or None when attribution is missing or ambiguous.
    """

    param_address = trace._grad_fn_param_refs.get(grad_fn_handle.label)
    if param_address is None:
        return None
    param_log = trace.params[param_address]
    if param_log.co_parent_params:
        return None
    module_address = param_log.module_address
    if module_address is None:
        return None
    return f"{module_address}:1"


def _infer_intervening_module_upstream(trace: "Trace", grad_fn_handle: "GradFn") -> str | None:
    """Infer an intervening grad_fn_handle module from downstream autograd edges.

    Parameters
    ----------
    trace:
        Trace containing grad_fn_handle metadata.
    grad_fn_handle:
        Intervening GradFn to place.

    Returns
    -------
    str | None
        Inherited module key, if a paired grad_fn_handle is reachable.
    """

    return _infer_intervening_module_bfs(trace, [grad_fn_handle.grad_fn_object_id], reverse=False)


def _infer_intervening_module_downstream(trace: "Trace", grad_fn_handle: "GradFn") -> str | None:
    """Infer an intervening grad_fn_handle module from reverse autograd edges.

    Parameters
    ----------
    trace:
        Trace containing grad_fn_handle metadata.
    grad_fn_handle:
        Intervening GradFn to place.

    Returns
    -------
    str | None
        Inherited module key, if a paired grad_fn_handle is reachable.
    """

    reverse_edges: dict[int, list[int]] = defaultdict(list)
    for candidate in trace.grad_fns:
        for next_grad_fn_id in candidate.next_grad_fn_ids:
            reverse_edges[next_grad_fn_id].append(candidate.grad_fn_object_id)
    return _infer_intervening_module_bfs(
        trace,
        reverse_edges.get(grad_fn_handle.grad_fn_object_id, []),
        reverse=True,
    )


def _infer_intervening_module_bfs(
    trace: "Trace",
    start_ids: Iterable[int],
    *,
    reverse: bool,
) -> str | None:
    """Find the nearest module-anchored grad_fn_handle by breadth-first search.

    Parameters
    ----------
    trace:
        Trace containing grad_fn_handle metadata.
    start_ids:
        Initial grad_fn_handle ids to inspect.
    reverse:
        Whether traversal uses reverse edges.

    Returns
    -------
    str | None
        Module key for the nearest paired grad_fn_handle, if found.
    """

    queue = list(start_ids)
    seen: set[int] = set()
    reverse_edges: dict[int, list[int]] = defaultdict(list)
    if reverse:
        for candidate in trace.grad_fns:
            for next_grad_fn_id in candidate.next_grad_fn_ids:
                reverse_edges[next_grad_fn_id].append(candidate.grad_fn_object_id)
    while queue:
        grad_fn_object_id = queue.pop(0)
        if grad_fn_object_id in seen or grad_fn_object_id not in trace.grad_fn_logs:
            continue
        seen.add(grad_fn_object_id)
        candidate = trace.grad_fn_logs[grad_fn_object_id]
        candidate_op = candidate.op
        if candidate_op is not None:
            module_key = _module_key_for_forward_op(candidate_op)
            if module_key is not None:
                return module_key
        if reverse:
            queue.extend(reverse_edges.get(grad_fn_object_id, []))
        else:
            queue.extend(candidate.next_grad_fn_ids)
    return None


def _compute_backward_node_lines(
    grad_fn_handle: "GradFn",
    call: Any | None = None,
    pass_filter: BackwardPassFilter = None,
) -> list[str]:
    """Build default label rows for a backward grad_fn_handle node.

    Parameters
    ----------
    grad_fn_handle:
        GradFn to render.
    call:
        Optional GradFnCall when rendering an unrolled backward graph.
    pass_filter:
        Normalized backward-pass filter.

    Returns
    -------
    list[str]
        Plain-text rows for ``NodeSpec.lines``.
    """

    title = grad_fn_handle.label
    if call is not None:
        call_index = getattr(call, "call_index", getattr(call, "ordinal", 0))
        title = getattr(call, "call_label", f"{grad_fn_handle.label}:{call_index}")
    if not grad_fn_handle.has_op:
        title = f"[i] {title}"
    if grad_fn_handle.is_custom:
        title = f"{title} [custom]"

    lines = [title]
    order = getattr(grad_fn_handle, "order", None)
    if order is not None:
        lines.append(f"order {order}")
    if call is None:
        pass_indices = sorted(
            {
                int(pass_index)
                for pass_index in (
                    getattr(grad_fn_call, "backward_pass_index", None)
                    for grad_fn_call in grad_fn_handle.calls.values()
                )
                if pass_index is not None
            }
        )
        if pass_filter is not None:
            pass_indices = [pass_index for pass_index in pass_indices if pass_index in pass_filter]
        if pass_indices:
            lines.append(f"bwd {int_list_to_compact_str(pass_indices)}")
    else:
        pass_index = getattr(call, "backward_pass_index", None)
        if pass_index is not None:
            lines.append(f"bwd {pass_index}")
    if grad_fn_handle.op is not None:
        lines.append(f"@{grad_fn_handle.op.layer_label}")
    lines.append(f"grad {_format_backward_output_shape(grad_fn_handle)}")
    return lines


def _format_backward_output_shape(grad_fn_handle: "GradFn") -> str:
    """Return the first captured output-grad shape for a grad_fn_handle.

    Parameters
    ----------
    grad_fn_handle:
        GradFn to inspect.

    Returns
    -------
    str
        Compact shape string, or ``"N/A"`` when no tensor was captured
        (typical for intervening grad_fns that have no forward counterpart).
    """

    for grad_fn_pass in reversed(list(grad_fn_handle.calls.values())):
        tensor = _first_tensor_in_obj(grad_fn_pass.grad_outputs)
        if tensor is not None:
            return _format_shape_str(tuple(tensor.shape))
    return "N/A"


def _first_tensor_in_obj(value: Any) -> torch.Tensor | None:
    """Return the first tensor found in a nested value.

    Parameters
    ----------
    value:
        Arbitrarily nested hook payload.

    Returns
    -------
    torch.Tensor | None
        First tensor in traversal order, if present.
    """

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor_in_obj(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor_in_obj(item)
            if tensor is not None:
                return tensor
    return None


def _container_group_id(node: BaseGraphNode) -> str | None:
    """Return a stable semantic group id for a container leaf.

    Parameters
    ----------
    node:
        Layer or Op metadata.

    Returns
    -------
    str | None
        Container group id, or ``None`` when the node has no container.
    """

    spec = getattr(node, "container_spec", None)
    path = tuple(getattr(node, "container_path", ()) or ())
    if spec is None or not path:
        return None
    func_call_id = getattr(node, "func_call_id", None)
    if bool(getattr(node, "is_output", False)):
        root = "final_output:0"
    elif func_call_id is not None:
        root = f"call:{func_call_id}"
    else:
        root = f"path:{_container_path_label(path[:-1])}"
    return f"{root}:{getattr(spec, 'kind', 'container')}"


def _container_path_label(path: Sequence[OutputPathComponent]) -> str:
    """Return a compact label for a typed container path.

    Parameters
    ----------
    path:
        Typed path components.

    Returns
    -------
    str
        Dot-safe-ish display fragment.
    """

    if not path:
        return "root"
    return ".".join(_container_component_role(component) for component in path)


def _container_kind(node: BaseGraphNode) -> str | None:
    """Return the node's container kind, if present."""

    spec = getattr(node, "container_spec", None)
    if spec is None:
        return None
    return str(getattr(spec, "kind", "container"))


def _add_collapsed_container_node(
    pending_nodes: list[dict[str, Any]],
    leaves: Sequence[GraphNode],
    *,
    vis_mode: str,
) -> None:
    """Record a collapsed container summary node for later emission."""

    first = leaves[0]
    group_id = cast(str, _container_group_id(cast(BaseGraphNode, first)))
    kind = _container_kind(cast(BaseGraphNode, first)) or "container"
    shape = "x".join(str(dim) for dim in (getattr(first, "shape", ()) or ())) or "scalar"
    node_name = _collapsed_container_node_name(group_id)
    pending_nodes.append(
        {
            "name": node_name,
            "label": render_lines_to_html([f"{kind} x{len(leaves)}", shape]),
            "shape": "box",
            "style": "filled,dashed",
            "fillcolor": "white",
            "color": "black",
            "fontcolor": "black",
            "ordering": "out",
        }
    )


def _collapsed_container_node_name(group_id: str) -> str:
    """Return a stable Graphviz node name for a collapsed container."""

    safe = "".join(char if char.isalnum() else "_" for char in group_id)
    return f"container_{safe}"


def _unwrap_focus_node(node: GraphNode) -> GraphNode:
    """Return the source node behind a focus proxy."""

    if isinstance(node, FocusNode):
        return node.original
    return node


def _base_node_for_metadata(node: GraphNode) -> BaseGraphNode:
    """Return a non-boundary graph node for metadata helpers."""

    unwrapped = _unwrap_focus_node(node)
    if isinstance(unwrapped, BoundaryNode):
        raise ValueError("Boundary nodes do not carry edge metadata.")
    return cast(BaseGraphNode, unwrapped)


def _should_collapse_module(
    module_log: "Module",
    *,
    collapse_fn: CollapseFn | None,
    max_module_depth: int,
) -> bool:
    """Return whether ``module_log`` should render as a collapsed module node.

    Parameters
    ----------
    module_log:
        Module metadata to check.
    collapse_fn:
        Optional user predicate. When supplied, it overrides depth logic.
    max_module_depth:
        Legacy nesting-depth threshold.

    Returns
    -------
    bool
        True if the module should be collapsed.
    """

    if collapse_fn is not None:
        return bool(collapse_fn(module_log))
    if max_module_depth == 0:
        return False
    return module_log.address_depth >= max_module_depth


def _module_has_single_rendered_op(module_log: "Module") -> bool:
    """Return whether ``module_log`` contains exactly one rendered op.

    Parameters
    ----------
    module_log:
        Module metadata to inspect.

    Returns
    -------
    bool
        True when the module contains one op and should keep op rendering.
    """

    return int(getattr(module_log, "num_layers", 0) or 0) == 1


def _single_op_module_should_keep_op_render(trace: "Trace", address: str) -> bool:
    """Return whether a one-op module should render as its op rather than collapse.

    Parameters
    ----------
    trace:
        Owning trace.
    address:
        Module address without call suffix.

    Returns
    -------
    bool
        True when the module has one op and no split call ranges to show.
    """

    module_log = cast("Module", trace.modules[address])
    return _module_has_single_rendered_op(module_log) and not _collapsed_module_rolling_suffix(
        trace, address
    )


def _collapse_address_for_node(
    trace: "Trace",
    node: GraphNode,
    *,
    vis_mode: str = "unrolled",
    collapse_fn: CollapseFn | None,
    max_module_depth: int,
) -> Optional[str]:
    """Return the module-pass address that should absorb ``node``, if any.

    Parameters
    ----------
    trace:
        Owning Trace.
    node:
        Layer node being rendered.
    vis_mode:
        ``"unrolled"`` or ``"rolled"`` visualization mode.
    collapse_fn:
        Optional user collapse predicate.
    max_module_depth:
        Legacy nesting-depth threshold.

    Returns
    -------
    Optional[str]
        Pass-qualified module address for unrolled lookup, or ``None``.
    """

    if isinstance(node, BoundaryNode):
        return None

    modules = list(node.modules)
    # An atomic (single-op) module is already maximally collapsed: it renders as
    # its own rectangle and is never absorbed into a box3d collapse on its own
    # account, even at the top level or when reused across split call sites. Drop
    # its innermost (own) module address so only genuinely-collapsible ancestor
    # modules remain eligible to absorb it.
    if getattr(node, "is_atomic_module", False) and modules:
        modules = modules[:-1]
    if not modules:
        return None

    if collapse_fn is None:
        if max_module_depth == 0 or len(modules) < max_module_depth:
            return None
        address_w_pass = cast(str, modules[max_module_depth - 1])
        address = address_w_pass.rsplit(":", 1)[0]
        if vis_mode == "rolled" and _single_op_module_should_keep_op_render(trace, address):
            return None
        return address_w_pass

    for address_w_pass in modules:
        address = address_w_pass.rsplit(":", 1)[0]
        if vis_mode == "rolled" and _single_op_module_should_keep_op_render(trace, address):
            continue
        if _should_collapse_module(
            cast("Module", trace.modules[address]),
            collapse_fn=collapse_fn,
            max_module_depth=max_module_depth,
        ):
            return str(address_w_pass)
    return None


def _run_fold_for_address(
    address_w_pass: str,
    run_folds: Mapping[str, "ModuleRunFold"] | None,
) -> "ModuleRunFold | None":
    """Return the fold descriptor for a module address.

    Parameters
    ----------
    address_w_pass:
        Pass-qualified or pass-free module address.
    run_folds:
        Fold descriptors keyed by pass-free module address.

    Returns
    -------
    ModuleRunFold | None
        Matching fold descriptor, or ``None``.
    """

    if run_folds is None:
        return None
    return run_folds.get(address_w_pass.rsplit(":", 1)[0])


def _run_fold_graph_node_name(
    address_w_pass: str,
    vis_mode: str,
    run_folds: Mapping[str, "ModuleRunFold"] | None,
) -> str:
    """Return the Graphviz node name after run-fold remapping.

    Parameters
    ----------
    address_w_pass:
        Pass-qualified or pass-free module address.
    vis_mode:
        ``"unrolled"`` or ``"rolled"`` visualization mode.
    run_folds:
        Fold descriptors keyed by pass-free module address.

    Returns
    -------
    str
        Graphviz node identifier for the folded representative or original module.
    """

    fold = _run_fold_for_address(address_w_pass, run_folds)
    if fold is None:
        module_tuple = address_w_pass.split(":")
    else:
        suffix = address_w_pass.rsplit(":", 1)[1] if ":" in address_w_pass else "1"
        module_tuple = [fold.representative, suffix]
    if vis_mode == "unrolled":
        return "pass".join(module_tuple)
    return module_tuple[0]


def _unique_run_folds(run_folds: Mapping[str, "ModuleRunFold"]) -> tuple["ModuleRunFold", ...]:
    """Return unique fold descriptors in deterministic representative order.

    Parameters
    ----------
    run_folds:
        Fold descriptors keyed by pass-free module address.

    Returns
    -------
    tuple[ModuleRunFold, ...]
        Unique folds sorted by representative address.
    """

    seen: set[str] = set()
    unique: list[ModuleRunFold] = []
    for address in sorted(run_folds):
        fold = run_folds[address]
        if fold.representative in seen:
            continue
        seen.add(fold.representative)
        unique.append(fold)
    return tuple(unique)


def _run_fold_representative_names(
    run_folds: Mapping[str, "ModuleRunFold"],
    vis_mode: str,
) -> set[str]:
    """Return Graphviz node names for unique folded-run representatives.

    Parameters
    ----------
    run_folds:
        Fold descriptors keyed by pass-free module address.
    vis_mode:
        ``"unrolled"`` or ``"rolled"`` visualization mode.

    Returns
    -------
    set[str]
        Rendered representative node names.
    """

    return {
        _run_fold_graph_node_name(
            f"{fold.representative}:1",
            vis_mode,
            {fold.representative: fold},
        )
        for fold in _unique_run_folds(run_folds)
    }


def _compact_int_ranges(values: Sequence[int]) -> str:
    """Return sorted integers in compact range notation.

    Parameters
    ----------
    values:
        Integer values to format.

    Returns
    -------
    str
        Comma-separated values and ranges, for example ``"1,2-4"``.
    """

    if not values:
        return ""
    sorted_values = sorted(set(values))
    ranges: list[str] = []
    start = sorted_values[0]
    previous = sorted_values[0]
    for value in sorted_values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _module_address_and_call(module_call: str) -> tuple[str, int] | None:
    """Parse a pass-qualified module call label.

    Parameters
    ----------
    module_call:
        Module call label of the form ``"address:call_index"``.

    Returns
    -------
    tuple[str, int] | None
        Parsed address and call index, or ``None`` if the suffix is not an integer.
    """

    address, separator, call_index_text = module_call.rpartition(":")
    if not separator:
        return None
    try:
        return address, int(call_index_text)
    except ValueError:
        return None


def _node_for_label(trace: "Trace", label: str) -> GraphNode | None:
    """Return an op or layer-like graph node for ``label`` when available.

    Parameters
    ----------
    trace:
        Trace containing graph nodes.
    label:
        Layer or op label to resolve.

    Returns
    -------
    GraphNode | None
        Matching node, or ``None`` if the label is not present.
    """

    try:
        return cast(GraphNode, trace.layer_dict_all_keys[label])
    except KeyError:
        try:
            return cast(GraphNode, trace.ops[label])
        except KeyError:
            return None


def _same_layer_reachability(layer_log: "Layer") -> dict[int, set[int]]:
    """Compute transitive same-layer reachability among passes.

    Parameters
    ----------
    layer_log:
        Rolled layer whose same-layer pass reachability is needed.

    Returns
    -------
    dict[int, set[int]]
        Mapping from pass index to reachable same-layer pass indices.
    """

    trace = layer_log.source_trace
    same_layer_labels = {op.label for op in layer_log.ops.values()}
    label_to_pass = {op.label: pass_index for pass_index, op in layer_log.ops.items()}
    reachability: dict[int, set[int]] = {pass_index: set() for pass_index in layer_log.ops}

    for pass_index, op in layer_log.ops.items():
        seen: set[str] = set()
        stack = list(op.children)
        while stack:
            label = stack.pop()
            if label in seen:
                continue
            seen.add(label)
            if label in same_layer_labels:
                reachability[pass_index].add(label_to_pass[label])
            child = _node_for_label(trace, label)
            if child is not None:
                stack.extend(child.children)
    return reachability


def _common_module_call_indices(layer_log: "Layer") -> dict[str, list[int]]:
    """Return module call indices for module addresses present on every pass.

    Parameters
    ----------
    layer_log:
        Layer whose per-pass module stacks should be inspected.

    Returns
    -------
    dict[str, list[int]]
        Address to call indices in pass order, limited to common addresses.
    """

    per_op: list[dict[str, int]] = []
    for op in layer_log.ops.values():
        parsed: dict[str, int] = {}
        for module_call in op.modules:
            parsed_call = _module_address_and_call(module_call)
            if parsed_call is not None:
                address, call_index = parsed_call
                parsed[address] = call_index
        per_op.append(parsed)
    if not per_op:
        return {}
    common_addresses = set(per_op[0])
    for parsed in per_op[1:]:
        common_addresses &= set(parsed)
    return {address: [parsed[address] for parsed in per_op] for address in sorted(common_addresses)}


def _same_layer_dependency_components(layer_log: "Layer") -> tuple[tuple[int, ...], ...]:
    """Return weak components in the same-layer dependency graph.

    Parameters
    ----------
    layer_log:
        Layer whose passes should be partitioned.

    Returns
    -------
    tuple[tuple[int, ...], ...]
        Pass-index components, sorted by first pass.
    """

    reachability = _same_layer_reachability(layer_log)
    adjacency: dict[int, set[int]] = {pass_index: set() for pass_index in layer_log.ops}
    for source, targets in reachability.items():
        for target in targets:
            adjacency[source].add(target)
            adjacency[target].add(source)

    components: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for pass_index in sorted(layer_log.ops):
        if pass_index in seen:
            continue
        stack = [pass_index]
        component: set[int] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.add(current)
            stack.extend(sorted(adjacency[current] - seen, reverse=True))
        components.append(tuple(sorted(component)))
    return tuple(sorted(components, key=lambda values: values[0]))


def _call_groups_for_layer(layer_log: "Layer") -> tuple[tuple[int, ...], ...]:
    """Return grouped module calls for disjoint same-layer regions.

    Parameters
    ----------
    layer_log:
        Layer to inspect.

    Returns
    -------
    tuple[tuple[int, ...], ...]
        Module call-index groups. Empty when there is only one dependency component or
        no single common module address.
    """

    common_calls = _common_module_call_indices(layer_log)
    if len(common_calls) != 1:
        return ()
    pass_to_call_index = {
        pass_index: call_index
        for pass_index, call_index in zip(
            layer_log.ops,
            next(iter(common_calls.values())),
            strict=True,
        )
    }
    components = _same_layer_dependency_components(layer_log)
    if len(components) <= 1:
        return ()
    groups: list[tuple[int, ...]] = []
    for component in components:
        groups.append(tuple(pass_to_call_index[pass_index] for pass_index in component))
    return tuple(groups)


def _format_call_groups(call_groups: Sequence[Sequence[int]]) -> str:
    """Format grouped module call partitions.

    Parameters
    ----------
    call_groups:
        Call-index groups to format.

    Returns
    -------
    str
        Comma-separated compact ranges, preserving group boundaries.
    """

    return ",".join(_compact_int_ranges(group) for group in call_groups)


def _collapsed_module_rolling_suffix(trace: "Trace", address: str) -> str:
    """Return a face suffix for a collapsed module's hidden call partitions.

    Parameters
    ----------
    trace:
        Trace containing the rendered module.
    address:
        Collapsed module address.

    Returns
    -------
    str
        Suffix beginning with ``":"`` or an empty string.
    """

    candidate_groups: tuple[tuple[int, ...], ...] = ()
    for layer_log in trace.layer_logs.values():
        if not isinstance(layer_log, Layer) or layer_log.num_passes <= 1:
            continue
        layer_addresses = {
            parsed[0]
            for op in layer_log.ops.values()
            for module_call in op.modules
            if (parsed := _module_address_and_call(module_call)) is not None
        }
        if address not in layer_addresses:
            continue
        groups = _call_groups_for_layer(layer_log)
        if len(groups) > len(candidate_groups):
            candidate_groups = groups
    if not candidate_groups:
        return ""
    return f":{_format_call_groups(candidate_groups)}"


def _node_spec_to_graphviz_args(spec: NodeSpec) -> dict[str, str]:
    """Convert a ``NodeSpec`` to Graphviz node keyword arguments.

    Parameters
    ----------
    spec:
        Node spec to convert.

    Returns
    -------
    dict[str, str]
        Graphviz keyword arguments except for ``name``.
    """

    node_args: dict[str, str] = {
        "label": render_lines_to_html(spec.lines),
        "shape": spec.shape,
        "style": spec.style,
    }
    optional_attrs: dict[str, object | None] = {
        "fillcolor": spec.fillcolor,
        "fontcolor": spec.fontcolor,
        "color": spec.color,
        "penwidth": spec.penwidth,
        "tooltip": spec.tooltip,
        "image": spec.image,
    }
    for attr_name, attr_value in optional_attrs.items():
        if attr_value is not None:
            node_args[attr_name] = str(attr_value)
    node_args.update(spec.extra_attrs)
    return node_args


def _format_shape_str(shape: tuple[Any, ...]) -> str:
    """Format a shape tuple in Python tuple notation."""

    return format_shape(shape)


def _compute_edge_label(
    parent_node: Union["Op", "Layer"],
    child_node: Union["Op", "Layer"],
    trace: "Trace",
    vis_mode: str,
) -> Optional[str]:
    """Return the highest-priority semantic label for an edge.

    Precedence matches the Phase 7 conditional rendering spec:

    1. Arm-entry labels from ``Trace.conditional_arm_entry_edges`` /
       ``Trace.conditional_edge_call_indices``.
    2. ``IF`` labels from ``Trace.conditional_branch_edges``.
    3. ``None`` when the edge has no branch semantics.

    Args:
        parent_node:
            Source node for the edge.
        child_node:
            Destination node for the edge.
        trace:
            Owning model log containing conditional metadata.
        vis_mode:
            ``"unrolled"`` or ``"rolled"``.

    Returns
    -------
    Optional[str]
        Graphviz HTML label string, or ``None`` if no semantic label applies.
    """
    arm_label = _compute_arm_entry_edge_label(parent_node, child_node, trace, vis_mode)
    if arm_label is not None:
        return _format_branch_edge_label_html(arm_label)

    if _edge_is_conditional_branch(parent_node, child_node, trace, vis_mode):
        return _format_branch_edge_label_html("IF")

    return None


def _compute_arm_entry_edge_label(
    parent_node: Union["Op", "Layer"],
    child_node: Union["Op", "Layer"],
    trace: "Trace",
    vis_mode: str,
) -> Optional[str]:
    """Return the arm-entry text for an edge, without Graphviz HTML wrapping.

    Args:
        parent_node:
            Source node for the edge.
        child_node:
            Destination node for the edge.
        trace:
            Owning model log containing conditional metadata.
        vis_mode:
            ``"unrolled"`` or ``"rolled"``.

    Returns
    -------
    Optional[str]
        Plain-text arm label, or ``None`` if the edge is not an arm-entry edge.
    """
    arm_entries = _get_arm_edge_entries(parent_node, child_node, trace, vis_mode)
    if not arm_entries:
        return None

    if vis_mode == "rolled":
        return _format_rolled_arm_entry_label(arm_entries, trace)

    if len(arm_entries) == 1:
        conditional_id, branch_kind, _ = arm_entries[0]
        return _format_arm_entry_text(conditional_id, branch_kind, trace)

    return " · ".join(
        [
            _format_arm_entry_text(
                conditional_id,
                branch_kind,
                trace,
                include_conditional_reference=True,
            )
            for conditional_id, branch_kind, _ in arm_entries
        ]
    )


def _get_arm_edge_entries(
    parent_node: Union["Op", "Layer"],
    child_node: Union["Op", "Layer"],
    trace: "Trace",
    vis_mode: str,
) -> List[Tuple[int, str, Optional[Tuple[int, ...]]]]:
    """Collect conditional-arm metadata for one rendered edge.

    Args:
        parent_node:
            Source node for the edge.
        child_node:
            Destination node for the edge.
        trace:
            Owning model log containing conditional metadata.
        vis_mode:
            ``"unrolled"`` or ``"rolled"``.

    Returns
    -------
    List[Tuple[int, str, Optional[Tuple[int, ...]]]]
        Sorted ``(conditional_id, branch_kind, call_indexs)`` tuples. Unrolled
        edges use ``call_indexs=None``.
    """
    arm_entries: List[Tuple[int, str, Optional[Tuple[int, ...]]]] = []
    if vis_mode == "unrolled":
        edge_key = (parent_node.layer_label, child_node.layer_label)
        for (conditional_id, branch_kind), edge_list in trace.conditional_arm_entry_edges.items():
            if edge_key in edge_list:
                arm_entries.append((conditional_id, branch_kind, None))
    elif vis_mode == "rolled":
        parent_no_pass = parent_node.layer_label
        child_no_pass = child_node.layer_label
        for (
            edge_parent,
            edge_child,
            conditional_id,
            branch_kind,
        ), call_indexs in trace.conditional_edge_call_indices.items():
            if (edge_parent, edge_child) == (parent_no_pass, child_no_pass):
                arm_entries.append((conditional_id, branch_kind, tuple(call_indexs)))
    else:
        raise ValueError(f"vis_mode must be 'unrolled' or 'rolled', not {vis_mode}")

    return sorted(arm_entries, key=lambda entry: _arm_entry_sort_key(entry[0], entry[1], trace))


def _format_rolled_arm_entry_label(
    arm_entries: List[Tuple[int, str, Optional[Tuple[int, ...]]]],
    trace: "Trace",
) -> str:
    """Format a rolled-mode arm-entry label with pass-awareness.

    Args:
        arm_entries:
            Sorted ``(conditional_id, branch_kind, call_indexs)`` tuples for one
            rolled edge.
        trace:
            Owning model log containing conditional metadata.

    Returns
    -------
    str
        Plain-text arm label for the rolled edge.
    """
    if len(arm_entries) == 1:
        conditional_id, branch_kind, _ = arm_entries[0]
        return _format_arm_entry_text(conditional_id, branch_kind, trace)

    pass_sets = [set(call_indexs or ()) for _, _, call_indexs in arm_entries]
    if pass_sets and len({tuple(sorted(pass_set)) for pass_set in pass_sets}) == 1:
        return " · ".join(
            [
                _format_arm_entry_text(
                    conditional_id,
                    branch_kind,
                    trace,
                    include_conditional_reference=True,
                )
                for conditional_id, branch_kind, _ in arm_entries
            ]
        )

    pass_counts: Dict[int, int] = defaultdict(int)
    for _, _, call_indexs in arm_entries:
        for call_index in call_indexs or ():
            pass_counts[call_index] += 1

    if pass_counts and all(pass_count == 1 for pass_count in pass_counts.values()):
        return " / ".join(
            [
                _format_rolled_pass_arm_text(
                    conditional_id,
                    branch_kind,
                    call_indexs,
                    trace,
                    include_conditional_reference=_rolled_labels_need_disambiguation(arm_entries),
                )
                for conditional_id, branch_kind, call_indexs in arm_entries
            ]
        )

    return "mixed"


def _rolled_labels_need_disambiguation(
    arm_entries: List[Tuple[int, str, Optional[Tuple[int, ...]]]],
) -> bool:
    """Return True when rolled branch labels need conditional disambiguation.

    Args:
        arm_entries:
            Sorted ``(conditional_id, branch_kind, call_indexs)`` tuples for one
            rolled edge.

    Returns
    -------
    bool
        True when multiple entries would otherwise share the same branch label.
    """
    base_labels = [_format_branch_kind_text(branch_kind) for _, branch_kind, _ in arm_entries]
    return len(base_labels) != len(set(base_labels))


def _format_rolled_pass_arm_text(
    conditional_id: int,
    branch_kind: str,
    call_indexs: Optional[Tuple[int, ...]],
    trace: "Trace",
    include_conditional_reference: bool,
) -> str:
    """Format one rolled arm label with its pass list.

    Args:
        conditional_id:
            Dense conditional id.
        branch_kind:
            Branch kind such as ``"then"`` or ``"elif_2"``.
        call_indexs:
            Sorted pass numbers for this rolled edge/arm tuple.
        trace:
            Owning model log containing conditional metadata.
        include_conditional_reference:
            Whether to append a conditional line-number reference.

    Returns
    -------
    str
        Plain-text label like ``"THEN(1,3)"``.
    """
    branch_text = _format_arm_entry_text(
        conditional_id,
        branch_kind,
        trace,
        include_conditional_reference=include_conditional_reference,
    )
    if not call_indexs:
        return branch_text
    return f"{branch_text}({int_list_to_compact_str(list(call_indexs))})"


def _format_arm_entry_text(
    conditional_id: int,
    branch_kind: str,
    trace: "Trace",
    include_conditional_reference: bool = False,
) -> str:
    """Format one arm-entry label as plain text.

    Args:
        conditional_id:
            Dense conditional id.
        branch_kind:
            Branch kind such as ``"then"`` or ``"elif_2"``.
        trace:
            Owning model log containing conditional metadata.
        include_conditional_reference:
            Whether to append ``@L...`` to identify the conditional event.

    Returns
    -------
    str
        Plain-text arm label.
    """
    branch_text = _format_branch_kind_text(branch_kind)
    if not include_conditional_reference:
        return branch_text
    return f"{branch_text}@{_get_conditional_reference_text(conditional_id, trace)}"


def _format_branch_kind_text(branch_kind: str) -> str:
    """Format a branch-kind token as display text.

    Args:
        branch_kind:
            Stored branch kind such as ``"then"``, ``"elif_1"``, or ``"else"``.

    Returns
    -------
    str
        Display label such as ``"THEN"`` or ``"ELIF 1"``.

    Raises
    ------
    ValueError
        If ``branch_kind`` is not recognized.
    """
    if branch_kind == "then":
        return "THEN"
    if branch_kind == "else":
        return "ELSE"
    if branch_kind.startswith("elif_"):
        return f"ELIF {int(branch_kind.split('_', 1)[1])}"
    raise ValueError(f"Unrecognized branch kind: {branch_kind}")


def _get_conditional_reference_text(conditional_id: int, trace: "Trace") -> str:
    """Return a readable conditional identifier for composite edge labels.

    Args:
        conditional_id:
            Dense conditional id.
        trace:
            Owning model log containing conditional metadata.

    Returns
    -------
    str
        Line-based conditional reference when available, otherwise ``"C{id}"``.
    """
    for conditional_event in trace.conditional_records:
        if conditional_event.id == conditional_id:
            return f"L{conditional_event.if_stmt_span[0]}"
    return f"C{conditional_id}"


def _arm_entry_sort_key(
    conditional_id: int,
    branch_kind: str,
    trace: "Trace",
) -> Tuple[int, int, int]:
    """Return a stable sort key for multi-arm edge labels.

    Args:
        conditional_id:
            Dense conditional id.
        branch_kind:
            Branch kind such as ``"then"`` or ``"elif_2"``.
        trace:
            Owning model log containing conditional metadata.

    Returns
    -------
    Tuple[int, int, int]
        Sort key ordered by source line, branch rank, then conditional id.
    """
    source_line = 10**9
    for conditional_event in trace.conditional_records:
        if conditional_event.id == conditional_id:
            source_line = conditional_event.if_stmt_span[0]
            break
    return (source_line, _branch_kind_sort_key(branch_kind), conditional_id)


def _branch_kind_sort_key(branch_kind: str) -> int:
    """Return an ordering key for branch kinds.

    Args:
        branch_kind:
            Stored branch kind such as ``"then"``, ``"elif_1"``, or ``"else"``.

    Returns
    -------
    int
        Sort rank for the branch kind.
    """
    if branch_kind == "then":
        return 0
    if branch_kind.startswith("elif_"):
        return int(branch_kind.split("_", 1)[1])
    if branch_kind == "else":
        return 10**6
    return 10**9


def _edge_is_conditional_branch(
    parent_node: Union["Op", "Layer"],
    child_node: Union["Op", "Layer"],
    trace: "Trace",
    vis_mode: str,
) -> bool:
    """Return True when an edge is an ``IF`` branch-entry edge.

    Args:
        parent_node:
            Source node for the edge.
        child_node:
            Destination node for the edge.
        trace:
            Owning model log containing conditional metadata.
        vis_mode:
            ``"unrolled"`` or ``"rolled"``.

    Returns
    -------
    bool
        True when the edge appears in ``conditional_branch_edges``.
    """
    if vis_mode == "unrolled":
        return (
            parent_node.layer_label,
            child_node.layer_label,
        ) in trace.conditional_branch_edges
    if vis_mode == "rolled":
        edge_key = (parent_node.layer_label, child_node.layer_label)
        return any(
            (branch_parent.split(":")[0], branch_child.split(":")[0]) == edge_key
            for branch_parent, branch_child in trace.conditional_branch_edges
        )
    raise ValueError(f"vis_mode must be 'unrolled' or 'rolled', not {vis_mode}")


def _format_branch_edge_label_html(label_text: str) -> str:
    """Wrap plain branch-label text in the Graphviz HTML used by TorchLens.

    Args:
        label_text:
            Plain text to display on the edge.

    Returns
    -------
    str
        Graphviz HTML edge-label string.
    """
    return f'<<FONT POINT-SIZE="18"><b><u>{label_text}</u></b></FONT>>'


def _container_component_role(component: OutputPathComponent) -> str:
    """Return the visible role label for a typed container path component.

    Parameters
    ----------
    component:
        Typed path component captured on an output leaf.

    Returns
    -------
    str
        User-facing key, index, or field label.
    """

    if isinstance(component, TupleIndex):
        return str(component.index)
    if isinstance(component, (DictKey, HFKey)):
        return str(component.key)
    if isinstance(component, (NamedField, DataclassField)):
        return component.name
    return str(component)


def _container_edge_label(node: BaseGraphNode | None) -> str | None:
    """Return the midpoint container role label for an edge into ``node``.

    Parameters
    ----------
    node:
        Child node metadata, if available.

    Returns
    -------
    str | None
        Last path component label, or ``None`` when the node is not a
        container leaf.
    """

    if node is None:
        return None
    path = tuple(getattr(node, "container_path", ()) or ())
    if not path:
        return None
    return _container_component_role(path[-1])


def _add_grad_edge(
    self: "Trace",
    parent_layer: GraphNode,
    child_layer: GraphNode,
    edge_style: str,
    module: str | int,
    module_edge_dict: Dict[str, Any],
    graphviz_graph: graphviz.Digraph,
    overrides: VisualizationOverrides,
) -> None:
    """Add a backward (grad) edge if both layers have saved grads.

    Gradient edges flow child -> parent (opposite of data flow), drawn in
    ``GRADIENT_ARROW_COLOR`` to distinguish from forward edges.  In rolled
    mode, an aggregate edge is shown when either rolled endpoint has a grad
    on any pass.

    Args:
        parent_layer: The parent Op or Layer (grad destination).
        child_layer: The child Op or Layer (grad source).
        edge_style: ``'solid'`` or ``'dashed'`` (matches the forward edge style).
        module: Module cluster name, or -1 for top-level.
        module_edge_dict: Dict mapping each module cluster to its edges.
        graphviz_graph: The graphviz Digraph object.
        overrides: Graphviz attribute overrides for grad edges.
    """
    if _node_has_grad(parent_layer) and _node_has_grad(child_layer):
        grad_passes = _shared_gradient_passes(parent_layer, child_layer)
        edge_dict = {
            "tail_name": _grad_node_name(child_layer),
            "head_name": _grad_node_name(parent_layer),
            "color": GRADIENT_ARROW_COLOR,
            "fontcolor": GRADIENT_ARROW_COLOR,
            "style": edge_style,
            "arrowsize": ".7",
            "labelfontsize": "8",
        }
        if (
            grad_passes
            and self.num_backward_passes > 1
            and grad_passes != set(range(1, self.num_backward_passes + 1))
        ):
            edge_dict["label"] = f"bwd {int_list_to_compact_str(sorted(grad_passes))}"
        for arg_name, arg_val in overrides.grad_edge.items():  # type: ignore[union-attr]
            if callable(arg_val):
                edge_dict[arg_name] = str(arg_val(self, parent_layer, child_layer))
            else:
                edge_dict[arg_name] = str(arg_val)

        if module != -1:
            module_edge_dict[cast(str, module)]["edges"].append(edge_dict)
        else:
            graphviz_graph.edge(**edge_dict)


def _node_has_grad(layer: Any) -> bool:
    """Return whether a rendered node has any saved grad.

    Parameters
    ----------
    layer:
        ``Op`` or rolled ``Layer``.

    Returns
    -------
    bool
        True if the node has at least one saved grad tensor.
    """

    ops = getattr(layer, "ops", None)
    if ops is not None and hasattr(ops, "values"):
        return any(bool(getattr(pass_log, "has_grad", False)) for pass_log in ops.values())
    return bool(getattr(layer, "has_grad", False))


def _node_gradient_passes(layer: Any) -> set[int]:
    """Return backward pass numbers with saved gradients for a rendered node.

    Parameters
    ----------
    layer:
        ``Op`` or rolled ``Layer``.

    Returns
    -------
    set[int]
        One-based backward pass numbers.
    """

    ops = getattr(layer, "ops", None)
    if ops is not None and hasattr(ops, "values"):
        pass_indices: set[int] = set()
        for pass_log in ops.values():
            pass_indices.update(_node_gradient_passes(pass_log))
        return pass_indices
    grads = getattr(layer, "grads", None)
    if grads is None:
        return set()
    return {
        int(record.backward_pass_index)
        for record in grads
        if getattr(record, "backward_pass_index", None) is not None and record.is_saved
    }


def _shared_gradient_passes(parent_layer: GraphNode, child_layer: GraphNode) -> set[int]:
    """Return backward pass numbers shared by both gradient-edge endpoints.

    Parameters
    ----------
    parent_layer:
        Forward parent node.
    child_layer:
        Forward child node.

    Returns
    -------
    set[int]
        One-based backward pass numbers shared by both endpoints.
    """

    return _node_gradient_passes(parent_layer) & _node_gradient_passes(child_layer)


def _grad_node_name(layer: Any) -> str:
    """Return the Graphviz node name for a grad edge endpoint.

    Parameters
    ----------
    layer:
        Rendered graph node.

    Returns
    -------
    str
        Graphviz-safe node name.
    """

    return str(layer.layer_label).replace(":", "pass")


__all__ = [
    "_add_backward_node_to_graphviz",
    "_add_collapsed_container_node",
    "_add_combined_backward_edges",
    "_add_combined_backward_nodes",
    "_add_combined_correspondence_edges",
    "_add_grad_edge",
    "_arm_entry_sort_key",
    "_backward_dot_call_node_name",
    "_backward_dot_node_name",
    "_backward_edge_attrs",
    "_backward_node_fillcolor",
    "_backward_node_graphviz_args",
    "_base_node_for_metadata",
    "_branch_kind_sort_key",
    "_call_groups_for_layer",
    "_collapse_address_for_node",
    "_collapsed_container_node_name",
    "_collapsed_module_rolling_suffix",
    "_common_module_call_indices",
    "_compact_int_ranges",
    "_compute_arm_entry_edge_label",
    "_compute_backward_node_lines",
    "_compute_edge_label",
    "_container_component_role",
    "_container_edge_label",
    "_container_group_id",
    "_container_kind",
    "_container_path_label",
    "_edge_is_conditional_branch",
    "_first_tensor_in_obj",
    "_format_arm_entry_text",
    "_format_backward_output_shape",
    "_format_branch_edge_label_html",
    "_format_branch_kind_text",
    "_format_call_groups",
    "_format_rolled_arm_entry_label",
    "_format_rolled_pass_arm_text",
    "_format_shape_str",
    "_get_arm_edge_entries",
    "_get_conditional_reference_text",
    "_grad_fn_call_matches_backward_filter",
    "_grad_fn_matches_backward_filter",
    "_grad_node_name",
    "_infer_intervening_module_bfs",
    "_infer_intervening_module_downstream",
    "_infer_intervening_module_upstream",
    "_module_address_and_call",
    "_module_has_single_rendered_op",
    "_module_key_for_forward_op",
    "_module_key_for_grad_fn",
    "_node_for_label",
    "_node_gradient_passes",
    "_node_has_grad",
    "_node_spec_to_graphviz_args",
    "_param_module_for_accumulate_grad",
    "_rolled_labels_need_disambiguation",
    "_run_fold_for_address",
    "_run_fold_graph_node_name",
    "_run_fold_representative_names",
    "_same_layer_dependency_components",
    "_same_layer_reachability",
    "_shared_gradient_passes",
    "_should_collapse_module",
    "_single_op_module_should_keep_op_render",
    "_unique_run_folds",
    "_unwrap_focus_node",
]
