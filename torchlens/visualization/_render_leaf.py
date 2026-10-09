"""Leaf helper functions for Graphviz rendering."""

# ruff: noqa: F403, F405

import functools
from collections import deque
from contextvars import ContextVar

from .._errors import InvalidArgumentError
from ..utils._multipass_access import get_multipass_attr, is_multipass_layer
from ._backward_inventory import (
    BackwardStyleInventory,
    _backward_pass_row,
    compute_backward_style_inventory,
)
from ._render_common import *
from ._typography import DEFAULT_TYPOGRAPHY

# Bound the forward walk that maps a branch-entry edge to its condition bool, so a
# malformed/huge conditional subgraph can never turn edge labelling into a hot loop.
_BRANCH_KIND_SEARCH_LIMIT = 256


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
    from .auto_collapse import ModuleRepeatFold


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
    inventory: "BackwardStyleInventory | None" = None,
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
    inventory:
        Per-render backward style inventory driving uniform-row suppression
        (``None`` keeps every row).
    """

    node_args = _backward_node_graphviz_args(
        grad_fn_handle,
        node_spec_fn,
        pass_filter=pass_filter,
        inventory=inventory,
    )
    graphviz_graph.node(**node_args)


def _backward_node_graphviz_args(
    grad_fn_handle: "GradFn",
    node_spec_fn: BackwardNodeSpecFn | None,
    call: Any | None = None,
    pass_filter: BackwardPassFilter = None,
    inventory: "BackwardStyleInventory | None" = None,
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
    inventory:
        Per-render backward style inventory driving uniform-row suppression
        (``None`` keeps every row).

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
            inventory=inventory,
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


def _backward_edge_attrs(
    tail: "GradFn", head: "GradFn", trace: "Trace | None" = None
) -> dict[str, str]:
    """Return Graphviz attributes for a backward GradFn edge.

    Parameters
    ----------
    tail:
        Edge tail GradFn.
    head:
        Edge head GradFn.
    trace:
        Optional trace for accumulation-target resolution (accum-identity
        groundwork, vizmech item 17/D30: the WGAN-GP render carried twelve
        IDENTICAL bare ``accum`` labels; each edge now carries its target's
        identity in SVG metadata until the annotation plan can place it).

    Returns
    -------
    dict[str, str]
        Graphviz edge attributes.
    """

    edge_attrs = {"color": GRADIENT_ARROW_COLOR, "fontcolor": GRADIENT_ARROW_COLOR}
    if tail.type == "accumulategrad" or head.type == "accumulategrad":
        edge_attrs["style"] = BACKWARD_ACCUMULATION_EDGE_STYLE
        edge_attrs["label"] = "accum"
        edge_attrs["labelfontsize"] = DEFAULT_TYPOGRAPHY.annotation_pt
        accum_node = tail if tail.type == "accumulategrad" else head
        target: str | None = None
        if trace is not None:
            target = _param_module_for_accumulate_grad(trace, accum_node)
        edge_attrs["tooltip"] = f"accum -> {target}" if target else f"accum -> {accum_node.label}"
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

    inventory = compute_backward_style_inventory(trace, pass_filter)
    for grad_fn_handle in trace.grad_fns:
        if not _grad_fn_matches_backward_filter(grad_fn_handle, pass_filter):
            continue
        node_args = _backward_node_graphviz_args(
            grad_fn_handle,
            node_spec_fn,
            pass_filter=pass_filter,
            inventory=inventory,
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
                **_backward_edge_attrs(
                    grad_fn_handle, trace.grad_fn_logs[next_grad_fn_id], trace=trace
                ),
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
        forward_node_name = _forward_correspondence_node_name(grad_fn_handle.op)
        if forward_node_name is not None:
            graphviz_graph.edge(
                forward_node_name,
                _backward_dot_node_name(grad_fn_handle),
                **edge_attrs,
            )


def _forward_correspondence_node_name(op: "Layer | None") -> str | None:
    """Return the forward endpoint for a combined correspondence edge.

    Scoped to the MULTI-PASS case only (r18j gate rework): for a recurrent
    aggregate ``Layer`` the specific forward *pass* a grad_fn maps to is not
    recoverable from current metadata (all passes share ``op_label`` and
    ``backward_pass_index``), so historically every grad_fn attached to ONE
    aggregate node -- return ``None`` and let the caller SKIP the edge rather than
    emit that ambiguous aggregate endpoint. Omitting an unprovable correspondence
    is honest.

    For a NON-recurrent op the endpoint is the declared unrolled forward node
    (the combined view always renders unrolled). A single-pass aggregate
    ``Layer`` resolves to its one ``Op`` first, so the tie names the node the
    forward pass declared instead of a bare ``layer_label`` that Graphviz would
    draw as a stray oval.

    Parameters
    ----------
    op:
        Forward ``Op`` or aggregate ``Layer`` paired with a grad_fn, or ``None``.

    Returns
    -------
    str | None
        Forward endpoint dot name, or ``None`` to skip the edge (recurrent aggregate).
    """

    if op is None:
        return None
    if is_multipass_layer(op):
        return None
    passes = getattr(op, "ops", None)
    if passes is not None and hasattr(passes, "values"):
        op = next(iter(passes.values()), op)
    return _render_node_name(op, "unrolled")


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
    raise InvalidArgumentError(
        f"intervening_cluster must be 'upstream', 'outside', 'downstream', or 'own'; "
        f"received {mode!r}",
        code="intervening_cluster_invalid",
        remedy="pass intervening_cluster='upstream', 'outside', 'downstream', or 'own'",
        argument="intervening_cluster",
    )


def _forward_op_is_module_output(op: "Layer") -> bool:
    """Resolve ``is_module_output`` for a forward op, aggregate-safe on recurrent Layers.

    ``is_module_output`` is a per-pass field whose access raises the multi-pass
    ``ValueError`` tripwire on a recurrent aggregate ``Layer`` (this is what
    detonated ``draw_combined`` on any recurrent model). Module-output status is a
    static containment property, so -- matching how the sibling module fields
    ``output_of_modules`` / ``modules`` are already stored aggregate-as-first-pass
    on the Layer -- resolve it explicitly from the first captured pass instead of
    leaking the tripwire out of the public combined renderer.

    Parameters
    ----------
    op:
        Forward ``Op`` or aggregate ``Layer`` paired with a grad_fn.

    Returns
    -------
    bool
        Whether the forward op is a module output.
    """

    if is_multipass_layer(op):
        ops = getattr(op, "ops", None)
        if ops is not None:
            first_pass = next(iter(ops.values()), None)
            if first_pass is not None:
                return bool(getattr(first_pass, "is_module_output", False))
        return False
    return bool(get_multipass_attr(op, "is_module_output", False, multipass=False))


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
    if _forward_op_is_module_output(op) and output_modules:
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
        reverse_edges=reverse_edges,
    )


def _infer_intervening_module_bfs(
    trace: "Trace",
    start_ids: Iterable[int],
    *,
    reverse: bool,
    reverse_edges: dict[int, list[int]] | None = None,
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
    reverse_edges:
        Prebuilt reverse-edge map for ``reverse=True`` callers. The downstream
        caller already builds this exact map to seed ``start_ids``; rebuilding
        it here doubled the O(E) sweep per intervening grad_fn
        (hunt-6 R52-3).

    Returns
    -------
    str | None
        Module key for the nearest paired grad_fn_handle, if found.
    """

    # deque: list.pop(0) shifted the whole queue per node, Theta(V^2) on
    # wide backward graphs for a linear BFS (R29, b4 sol MED).
    queue = deque(start_ids)
    seen: set[int] = set()
    if reverse and reverse_edges is None:
        reverse_edges = defaultdict(list)
        for candidate in trace.grad_fns:
            for next_grad_fn_id in candidate.next_grad_fn_ids:
                reverse_edges[next_grad_fn_id].append(candidate.grad_fn_object_id)
    if reverse_edges is None:
        reverse_edges = {}
    while queue:
        grad_fn_object_id = queue.popleft()
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
    inventory: "BackwardStyleInventory | None" = None,
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
    inventory:
        Per-render backward style inventory driving uniform-row suppression
        (``None`` keeps every row).

    Returns
    -------
    list[str]
        Plain-text rows for ``NodeSpec.lines``.
    """

    title = grad_fn_handle.label
    if call is not None and len(getattr(grad_fn_handle, "calls", {})) > 1:
        call_index = getattr(call, "call_index", getattr(call, "ordinal", 0))
        title = getattr(call, "call_label", f"{grad_fn_handle.label}:{call_index}")
    if not grad_fn_handle.has_op:
        title = f"[i] {title}"
    if grad_fn_handle.is_custom:
        title = f"{title} [custom]"

    lines = [title]
    order = getattr(grad_fn_handle, "order", None)
    if order is not None and not (inventory is not None and inventory.suppress_order_row):
        # "order 1" on EVERY node of an ordinary backward is a uniform
        # constant row (vizmech item 17): suppressed when no node exceeds 1.
        lines.append(f"order {order}")
    lines.extend(_backward_pass_row(grad_fn_handle, call, pass_filter, inventory))
    if grad_fn_handle.op is not None:
        lines.append(f"@{grad_fn_handle.op.layer_label}")
    if not (inventory is not None and inventory.suppress_grad_row):
        # Suppressed only when EVERY visible node would print "grad N/A"
        # (the memo's x36 uniform row); a mixed render keeps its N/A rows.
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

    # Container metadata is per-pass: a rolled multi-pass Layer has no single
    # honest value (typically only the final pass feeds the output container),
    # so the aggregate node explicitly degrades to "no container decoration" —
    # the same per-pass "n/a" policy the encoding channel uses. A plain
    # getattr here leaked the multi-pass ValueError tripwire out of draw().
    spec = get_multipass_attr(node, "container_spec", None, multipass=None)
    path = tuple(get_multipass_attr(node, "container_path", (), multipass=None) or ())
    if spec is None or not path:
        return None
    func_call_id = get_multipass_attr(node, "func_call_id", None, multipass=None)
    if bool(get_multipass_attr(node, "is_output", False, multipass=False)):
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
    repeat_folds: Mapping[str, "ModuleRepeatFold"] | None,
) -> "ModuleRepeatFold | None":
    """Return the fold descriptor for a module address.

    Parameters
    ----------
    address_w_pass:
        Pass-qualified or pass-free module address.
    repeat_folds:
        Fold descriptors keyed by pass-free module address.

    Returns
    -------
    ModuleRepeatFold | None
        Matching fold descriptor, or ``None``.
    """

    if repeat_folds is None:
        return None
    return repeat_folds.get(address_w_pass.rsplit(":", 1)[0])


def _run_fold_graph_node_name(
    address_w_pass: str,
    vis_mode: str,
    repeat_folds: Mapping[str, "ModuleRepeatFold"] | None,
) -> str:
    """Return the Graphviz node name after repeat-fold remapping.

    Parameters
    ----------
    address_w_pass:
        Pass-qualified or pass-free module address.
    vis_mode:
        ``"unrolled"`` or ``"rolled"`` visualization mode.
    repeat_folds:
        Fold descriptors keyed by pass-free module address.

    Returns
    -------
    str
        Graphviz node identifier for the folded representative or original module.
    """

    fold = _run_fold_for_address(address_w_pass, repeat_folds)
    if fold is None:
        module_tuple = address_w_pass.split(":")
    else:
        suffix = address_w_pass.rsplit(":", 1)[1] if ":" in address_w_pass else "1"
        module_tuple = [fold.representative, suffix]
    if vis_mode == "unrolled":
        return "pass".join(module_tuple)
    return module_tuple[0]


def _unique_repeat_folds(
    repeat_folds: Mapping[str, "ModuleRepeatFold"],
) -> tuple["ModuleRepeatFold", ...]:
    """Return unique fold descriptors in deterministic representative order.

    Parameters
    ----------
    repeat_folds:
        Fold descriptors keyed by pass-free module address.

    Returns
    -------
    tuple[ModuleRepeatFold, ...]
        Unique folds sorted by representative address.
    """

    seen: set[str] = set()
    unique: list[ModuleRepeatFold] = []
    for address in sorted(repeat_folds):
        fold = repeat_folds[address]
        if fold.representative in seen:
            continue
        seen.add(fold.representative)
        unique.append(fold)
    return tuple(unique)


def _run_fold_representative_names(
    repeat_folds: Mapping[str, "ModuleRepeatFold"],
    vis_mode: str,
) -> set[str]:
    """Return Graphviz node names for unique folded-run representatives.

    Parameters
    ----------
    repeat_folds:
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
        for fold in _unique_repeat_folds(repeat_folds)
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
    """Compute direct same-layer reachability among passes.

    Each pass's walk stops at the first same-layer op it reaches instead of
    walking through it. The weak transitive closure of this direct graph equals
    that of full transitive reachability (a path through an intermediate pass
    contributes that pass's own outgoing edges), so the dependency components
    built from it are unchanged while the walk stays near-linear for long
    recurrent chains.

    Parameters
    ----------
    layer_log:
        Rolled layer whose same-layer pass reachability is needed.

    Returns
    -------
    dict[int, set[int]]
        Mapping from pass index to directly reachable same-layer pass indices.
    """

    trace = layer_log.source_trace
    same_layer_labels = {op.label for op in layer_log.ops.values()}
    label_to_pass = {op.label: pass_index for pass_index, op in layer_log.ops.items()}
    reachability: dict[int, set[int]] = {
        pass_index: set()
        # OpAccessor iteration yields Ops (C02); .keys() IS the pass-index
        # view here, not a dict redundancy.
        for pass_index in layer_log.ops.keys()  # noqa: SIM118
    }

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
                continue
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


def _rolled_visual_num_passes(layer_log: GraphNode) -> int:
    """Return the displayed rolled multiplier for a layer.

    Parameters
    ----------
    layer_log:
        Op or Layer to render.

    Returns
    -------
    int
        Visual call count. Multi-output module calls count distinct module
        invocations instead of output tensors.
    """

    if not isinstance(layer_log, Layer):
        return int(getattr(layer_log, "num_passes", 1) or 1)
    common_call_indices = _common_module_call_indices(layer_log)
    if not common_call_indices:
        return int(layer_log.num_passes)
    distinct_counts = {
        len(set(call_indices)) for call_indices in common_call_indices.values() if call_indices
    }
    if len(distinct_counts) == 1:
        return distinct_counts.pop()
    return int(layer_log.num_passes)


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

    trace = layer_log.source_trace
    same_layer_pass = {op.label: pass_index for pass_index, op in layer_log.ops.items()}

    # Build the descendant interior once for the whole layer. Same-layer nodes are
    # boundaries: their outgoing edges are handled from their own seeded traversal,
    # matching the historical per-pass walk's stop-at-first-same-layer rule.
    forward: dict[str, tuple[str, ...]] = {}
    reverse: dict[str, set[str]] = {}
    pending = deque(same_layer_pass)
    expanded: set[str] = set()
    while pending:
        label = pending.popleft()
        if label in expanded:
            continue
        expanded.add(label)
        node = _node_for_label(trace, label)
        children = tuple(node.children) if node is not None else ()
        forward[label] = children
        for child_label in children:
            reverse.setdefault(child_label, set()).add(label)
            if child_label not in same_layer_pass:
                pending.append(child_label)

    # Only interior nodes lying on a path to a same-layer boundary can contribute
    # an edge in the historical reachability graph. Pruning dead descendant tails
    # avoids falsely joining passes that merely converge after their final use.
    productive = set(same_layer_pass)
    pending = deque(same_layer_pass)
    while pending:
        label = pending.popleft()
        for parent_label in reverse.get(label, ()):
            if parent_label in productive:
                continue
            productive.add(parent_label)
            pending.append(parent_label)

    parents = {label: label for label in productive}

    def find(label: str) -> str:
        """Return the canonical union-find root for one productive node."""

        root = label
        while parents[root] != root:
            root = parents[root]
        while parents[label] != label:
            next_label = parents[label]
            parents[label] = root
            label = next_label
        return root

    def union(left: str, right: str) -> None:
        """Join two productive nodes with deterministic lexical-root ownership."""

        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        smaller, larger = sorted((left_root, right_root))
        parents[larger] = smaller

    # Weak components of the productive descendant interior induce exactly the
    # weak transitive closure of the historical direct same-layer reachability.
    for label in productive:
        for child_label in forward.get(label, ()):
            if child_label in productive:
                union(label, child_label)

    components_by_root: dict[str, list[int]] = {}
    for label, pass_index in sorted(same_layer_pass.items(), key=lambda item: item[1]):
        components_by_root.setdefault(find(label), []).append(pass_index)
    components = (tuple(values) for values in components_by_root.values())
    return tuple(sorted(components, key=lambda values: values[0]))


@dataclass
class _PerDrawCollapseCache:
    """Per-draw memo for the pure per-layer collapse-rolling computations.

    ``_call_groups_for_layer`` and ``_collapsed_module_rolling_suffix`` are pure
    functions of the captured graph, but the renderer re-enters them once per
    collapsed-module NODE. Memoizing them for the duration of ONE draw removes
    that ``nodes x layers x passes`` blowup. The cache is per-draw rather than
    per-``Trace`` so a graph mutated between draws is never served stale results.
    """

    #: Memoized ``_call_groups_for_layer`` results. Keyed by ``id`` of the layer
    #: object, with the layer itself retained in the value so the identity key
    #: can never be recycled onto a different object mid-draw.
    call_groups: dict[int, tuple[Any, tuple[tuple[int, ...], ...]]] = field(default_factory=dict)
    #: Memoized ``address -> face suffix`` map, built lazily in one pass.
    rolling_suffixes: dict[str, str] | None = None


_PER_DRAW_COLLAPSE_CACHE: ContextVar["_PerDrawCollapseCache | None"] = ContextVar(
    "torchlens_per_draw_collapse_cache", default=None
)


def _with_per_draw_collapse_cache(render_fn: Any) -> Any:
    """Scope a fresh :class:`_PerDrawCollapseCache` to one render call.

    Parameters
    ----------
    render_fn:
        Render entrypoint to wrap.

    Returns
    -------
    Any
        Wrapper installing (and always tearing down) the per-draw cache, so no
        layer references outlive the draw and no result crosses draw boundaries.
    """

    @functools.wraps(render_fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Install the per-draw collapse cache, always tearing it down afterwards."""

        token = _PER_DRAW_COLLAPSE_CACHE.set(_PerDrawCollapseCache())
        try:
            return render_fn(*args, **kwargs)
        finally:
            _PER_DRAW_COLLAPSE_CACHE.reset(token)

    return wrapper


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

    cache = _PER_DRAW_COLLAPSE_CACHE.get()
    if cache is None:
        return _call_groups_for_layer_uncached(layer_log)
    memo_key = id(layer_log)
    memoized = cache.call_groups.get(memo_key)
    if memoized is not None:
        return memoized[1]
    groups = _call_groups_for_layer_uncached(layer_log)
    cache.call_groups[memo_key] = (layer_log, groups)
    return groups


def _call_groups_for_layer_uncached(layer_log: "Layer") -> tuple[tuple[int, ...], ...]:
    """Compute :func:`_call_groups_for_layer` without consulting the draw memo.

    Parameters
    ----------
    layer_log:
        Layer to inspect.

    Returns
    -------
    tuple[tuple[int, ...], ...]
        Module call-index groups.
    """

    if len(layer_log.ops) <= 1:
        return ()
    common_calls = _common_module_call_indices(layer_log)
    if len(common_calls) != 1:
        return ()
    pass_to_call_index = dict(
        zip(
            layer_log.ops.keys(),
            next(iter(common_calls.values())),
            strict=True,
        )
    )
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


def _collapsed_module_rolling_suffix_map(trace: "Trace") -> dict[str, str]:
    """Build every collapsed module's rolling face suffix in one pass.

    Walks the rolled layers once and keeps, per module address, the first
    partition with the most groups -- exactly the strictly-greater ``candidate``
    rule the per-address scan applied while iterating the same layers in the same
    order. Addresses with no split partition are simply absent, which the caller
    reads back as the empty suffix.

    Parameters
    ----------
    trace:
        Trace containing the rendered modules.

    Returns
    -------
    dict[str, str]
        Module address to face suffix beginning with ``":"``.
    """

    best_groups: dict[str, tuple[tuple[int, ...], ...]] = {}
    for layer_log in trace.layer_logs.values():
        if not isinstance(layer_log, Layer) or layer_log.num_passes <= 1:
            continue
        groups = _call_groups_for_layer(layer_log)
        if not groups:
            # A layer with no split partition could never beat an incumbent
            # (the rule is strictly-greater group count), so skip its addresses.
            continue
        layer_addresses = {
            parsed[0]
            for op in layer_log.ops.values()
            for module_call in op.modules
            if (parsed := _module_address_and_call(module_call)) is not None
        }
        for layer_address in layer_addresses:
            if len(groups) > len(best_groups.get(layer_address, ())):
                best_groups[layer_address] = groups
    return {
        layer_address: f":{_format_call_groups(groups)}"
        for layer_address, groups in best_groups.items()
    }


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

    cache = _PER_DRAW_COLLAPSE_CACHE.get()
    if cache is None:
        return _collapsed_module_rolling_suffix_map(trace).get(address, "")
    if cache.rolling_suffixes is None:
        cache.rolling_suffixes = _collapsed_module_rolling_suffix_map(trace)
    return cache.rolling_suffixes.get(address, "")


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
        # r-b6 R19-6: relative to the visualizer root (graph-level imagepath).
        "image": relativize_visualizer_image(spec.image) if spec.image else spec.image,
        "fixedsize": spec.fixedsize,
    }
    # 2.4(ii) funnel rule (L5): DROP NodeSpec width/height when an image is
    # set -- an image node's size is pixel-derived, and a channel-set width
    # under fixedsize=false would otherwise become a live MINIMUM the image
    # is scaled into. ``extra_attrs`` (merged last, below) stays the
    # power-valve override for a user who genuinely wants a sized image node.
    if spec.image is None:
        optional_attrs["width"] = spec.width
        optional_attrs["height"] = spec.height
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
        return _format_branch_edge_label_html(_conditional_branch_edge_kind(child_node, trace))

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


def _conditional_branch_edge_kind(child_node: Union["Op", "Layer"], trace: "Trace") -> str:
    """Return the branch-test kind (``IF`` or ``ELIF``) for a branch-entry edge.

    ``conditional_branch_edges`` records the edges that enter each condition test but carries
    no kind. Map the edge's child to the nearest downstream branch bool via a bounded forward
    walk, then use that bool's AST-derived ``conditional_context_kind``. This remains honest
    when one source-level ``if`` is evaluated repeatedly in a loop: every evaluation is ``IF``
    even though all evaluations accumulate in one conditional event. Static ``elif`` tests are
    classified directly as ``elif_test`` and continue to render as ``ELIF``.

    Parameters
    ----------
    child_node:
        Destination node of the branch-entry edge (the condition-subgraph entry).
    trace:
        Owning trace with ``conditional_records`` metadata.

    Returns
    -------
    str
        ``"IF"`` or ``"ELIF"`` (falls back to ``"IF"`` if no bool is reachable).
    """
    bool_kind: dict[str, str] = {}
    for event in getattr(trace, "conditional_records", ()) or ():
        for bool_label in getattr(event, "bool_layers", ()) or ():
            bool_layer = trace[bool_label]
            bool_ops = tuple(getattr(bool_layer, "ops", {}).values())
            context_kinds = (
                {getattr(op, "conditional_context_kind", None) for op in bool_ops}
                if bool_ops
                else {getattr(bool_layer, "conditional_context_kind", None)}
            )
            kind = "ELIF" if context_kinds == {"elif_test"} else "IF"
            bool_kind[bool_label.split(":", 1)[0]] = kind
    if not bool_kind:
        return "IF"

    start = child_node.layer_label.split(":", 1)[0]
    seen: set[str] = {start}
    frontier = deque([start])
    steps = 0
    while frontier and steps < _BRANCH_KIND_SEARCH_LIMIT:
        steps += 1
        current = frontier.popleft()
        if current in bool_kind:
            return bool_kind[current]
        # ``label in trace`` disagrees with ``trace[label]`` for some intermediate
        # condition ops, so resolve by indexing and treat a miss as a dead end.
        try:
            node = trace[current]
        except (KeyError, ValueError):
            continue
        for child_label in getattr(node, "children", ()):
            base = child_label.split(":", 1)[0]
            if base not in seen:
                seen.add(base)
                frontier.append(base)
    return "IF"


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
    return (
        f'<<FONT POINT-SIZE="{DEFAULT_TYPOGRAPHY.emphasis_pt}"><b><u>{label_text}</u></b></FONT>>'
    )


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
    # Per-pass field: a rolled multi-pass Layer degrades to no label (see
    # _container_group_id) instead of leaking the multi-pass tripwire.
    path = tuple(get_multipass_attr(node, "container_path", (), multipass=None) or ())
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
    *,
    forward_tail_name: str,
    forward_head_name: str,
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
        forward_tail_name: DOT name the forward edge drew from (the rendered
            parent endpoint, which may be a collapsed or folded box).
        forward_head_name: DOT name the forward edge drew to. The grad edge
            reuses both rendered endpoints reversed, so it always attaches to
            declared nodes.
    """
    if _node_has_grad(parent_layer) and _node_has_grad(child_layer):
        grad_passes = _shared_gradient_passes(parent_layer, child_layer)
        edge_dict = {
            "tail_name": forward_head_name,
            "head_name": forward_tail_name,
            "color": GRADIENT_ARROW_COLOR,
            "fontcolor": GRADIENT_ARROW_COLOR,
            "style": edge_style,
            "arrowsize": ".7",
            "labelfontsize": DEFAULT_TYPOGRAPHY.annotation_pt,
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


__all__ = [
    "BackwardStyleInventory",
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
    "compute_backward_style_inventory",
    "_base_node_for_metadata",
    "_branch_kind_sort_key",
    "_call_groups_for_layer",
    "_call_groups_for_layer_uncached",
    "_collapse_address_for_node",
    "_collapsed_container_node_name",
    "_collapsed_module_rolling_suffix",
    "_collapsed_module_rolling_suffix_map",
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
    "_rolled_visual_num_passes",
    "_run_fold_for_address",
    "_run_fold_graph_node_name",
    "_run_fold_representative_names",
    "_same_layer_dependency_components",
    "_same_layer_reachability",
    "_shared_gradient_passes",
    "_should_collapse_module",
    "_single_op_module_should_keep_op_render",
    "_unique_repeat_folds",
    "_unwrap_focus_node",
    "_with_per_draw_collapse_cache",
]
