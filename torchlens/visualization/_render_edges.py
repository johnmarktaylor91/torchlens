"""Edge and endpoint helper functions for Graphviz rendering."""

# ruff: noqa: F403, F405

from ._render_common import *
from ._render_leaf import *
from ._render_utils import html_escape


def _buffer_name_segment(address: str | None) -> str:
    """Return the last dotted segment of a buffer address.

    Parameters
    ----------
    address:
        Fully qualified buffer address, if available.

    Returns
    -------
    str
        Final dotted address segment, or an empty string for missing addresses.
    """

    if address is None:
        return ""
    return address.split(".")[-1]


if TYPE_CHECKING:
    from ..data_classes.trace import Trace
    from .auto_collapse import ModuleRepeatFold


def _is_noise_buffer(node: GraphNode) -> bool:
    """Return whether ``node`` is a hardcoded noisy buffer.

    Parameters
    ----------
    node:
        Candidate graph node.

    Returns
    -------
    bool
        True when the node is a buffer whose last address segment is filtered in
        ``"meaningful"`` mode.
    """

    source_node = _unwrap_focus_node(node)
    if not source_node.is_buffer:
        return False
    address = getattr(source_node, "address", None)
    return _buffer_name_segment(address) in _NOISE_BUFFER_NAMES


def _is_buffer_visible(node: GraphNode, show_buffer_layers: BufferVisibilityLiteral) -> bool:
    """Return whether a buffer node should be visible in the current mode.

    Parameters
    ----------
    node:
        Candidate graph node.
    show_buffer_layers:
        Canonical tri-state visibility mode.

    Returns
    -------
    bool
        True when the node is visible. Non-buffer nodes are always visible.
    """

    if not node.is_buffer:
        return True
    if show_buffer_layers == "always":
        return True
    if show_buffer_layers == "never":
        return False
    return not _is_noise_buffer(node)


def _add_legend_to_graphviz(dot: graphviz.Digraph, theme: VisualizationTheme) -> None:
    """Add a compact color legend subgraph to a Graphviz graph.

    Parameters
    ----------
    dot:
        Graphviz graph being rendered.
    theme:
        Resolved visualization theme.
    """

    with dot.subgraph(name="cluster_torchlens_legend") as legend:
        legend.attr(
            label="TorchLens legend",
            labelloc="t",
            color=theme.default_border,
            fontcolor=theme.default_font,
            style="rounded",
        )
        legend_specs = (
            NodeSpec(
                ["input"], shape="oval", fillcolor=INPUT_COLOR, fontcolor="black", color="black"
            ),
            NodeSpec(
                ["output"], shape="oval", fillcolor=OUTPUT_COLOR, fontcolor="black", color="black"
            ),
            NodeSpec(
                ["parameterized"],
                shape="oval",
                fillcolor=TRAINABLE_PARAMS_BG_COLOR,
                fontcolor="black",
                color="black",
            ),
            NodeSpec(
                ["buffer"],
                shape="cylinder",
                fillcolor=DEFAULT_BG_COLOR,
                fontcolor="black",
                color="black",
            ),
            NodeSpec(
                ["boolean"],
                shape="oval",
                fillcolor=BOOL_NODE_COLOR,
                fontcolor="black",
                color="black",
            ),
            NodeSpec(
                ["intervention/cone"],
                shape="oval",
                fillcolor=INTERVENTION_CONE_COLOR,
                fontcolor="black",
                color=INTERVENTION_SITE_COLOR,
                penwidth=2.0,
            ),
        )
        for index, spec in enumerate(legend_specs):
            node_args = _node_spec_to_graphviz_args(apply_theme_to_spec(spec, theme))
            node_args["name"] = f"tl_legend_{index}"
            legend.node(
                **node_args,
            )


def _render_node_label(node: GraphNode, vis_mode: str) -> str:
    """Return the graph node label for the active visualization mode.

    Parameters
    ----------
    node:
        Render node.
    vis_mode:
        ``"unrolled"`` renders individual Ops, while ``"rolled"`` renders
        aggregate Layers.

    Returns
    -------
    str
        Stable label used as the DOT node identifier before Graphviz escaping.
    """
    if vis_mode == "unrolled" and isinstance(node, Op):
        return node.label
    return node.layer_label


def _get_node_by_label(trace: "Trace", label: str, vis_mode: str) -> GraphNode:
    """Return a render node by label for the active visualization mode."""

    if vis_mode == "unrolled":
        return trace.layer_dict_main_keys.get(label, trace.layer_dict_all_keys[label])
    if vis_mode == "rolled":
        return trace.layer_logs[label]
    raise ValueError(f"vis_mode must be 'unrolled' or 'rolled', not {vis_mode}")


def _segment_for_node(
    node: GraphNode,
    segments: Mapping[str, SegmentDescriptor] | None,
) -> SegmentDescriptor | None:
    """Return the segment descriptor that absorbs ``node`` if any.

    Parameters
    ----------
    node:
        Render node being emitted.
    segments:
        Segment descriptors keyed by node name.

    Returns
    -------
    SegmentDescriptor | None
        Matching descriptor, or ``None``.
    """

    if not segments or isinstance(node, BoundaryNode):
        return None
    label = str(getattr(node, "layer_label", ""))
    for segment in segments.values():
        if label in segment.ops:
            return segment
        member_set = set(segment.members)
        for address_w_pass in getattr(node, "modules", ()) or ():
            if str(address_w_pass).rsplit(":", 1)[0] in member_set:
                return segment
    return None


def _is_run_fold_representative(
    address_w_pass: str,
    repeat_folds: Mapping[str, "ModuleRepeatFold"],
) -> bool:
    """Return whether ``address_w_pass`` is the representative for its fold.

    Parameters
    ----------
    address_w_pass:
        Pass-qualified or pass-free module address.
    repeat_folds:
        Fold descriptors keyed by pass-free module address.

    Returns
    -------
    bool
        True when the address is not folded or is the first run member.
    """

    address = address_w_pass.rsplit(":", 1)[0]
    fold = repeat_folds.get(address)
    return fold is None or address == fold.representative


def _run_fold_ancestor_for_node(
    node: GraphNode,
    repeat_folds: Mapping[str, "ModuleRepeatFold"] | None,
) -> str | None:
    """Return the pass-qualified folded ancestor that should absorb ``node``.

    Parameters
    ----------
    node:
        Rendered graph node.
    repeat_folds:
        Fold descriptors keyed by pass-free module address.

    Returns
    -------
    str | None
        Pass-qualified folded ancestor address, or ``None``.
    """

    if not repeat_folds:
        return None
    for address_w_pass in getattr(node, "modules", ()) or ():
        if _run_fold_for_address(str(address_w_pass), repeat_folds) is not None:
            return str(address_w_pass)
    return None


def _run_fold_ellipsis_node_name(representative_name: str) -> str:
    """Return the deterministic ellipsis node name for a folded run.

    Parameters
    ----------
    representative_name:
        Graphviz node identifier for the folded representative.

    Returns
    -------
    str
        Collision-resistant ellipsis node name derived from the representative.
    """

    return f"{representative_name}___runfoldellipsis"


def _run_fold_ellipsis_label(fold: "ModuleRepeatFold") -> str:
    """Return the in-flow elision label for ``fold``.

    Parameters
    ----------
    fold:
        Fold descriptor to label.

    Returns
    -------
    str
        Plaintext label describing the number of elided siblings.
    """

    return f"... +{fold.multiplicity - 1} more {fold.class_name}"


def _run_fold_hidden_endpoint(
    address_w_pass: str | None,
    repeat_folds: Mapping[str, "ModuleRepeatFold"] | None,
) -> "ModuleRepeatFold | None":
    """Return the run fold when ``address_w_pass`` is a hidden run member.

    Parameters
    ----------
    address_w_pass:
        Pass-qualified or pass-free module address for one rendered endpoint.
    repeat_folds:
        Fold descriptors keyed by pass-free module address.

    Returns
    -------
    ModuleRepeatFold | None
        Fold descriptor when the address belongs to ``members[1:]``.
    """

    if address_w_pass is None:
        return None
    fold = _run_fold_for_address(address_w_pass, repeat_folds)
    if (
        repeat_folds is None
        or fold is None
        or _is_run_fold_representative(address_w_pass, repeat_folds)
    ):
        return None
    return fold


def _edge_touches_run_fold(
    tail_name: str,
    head_name: str,
    repeat_folds: Mapping[str, "ModuleRepeatFold"] | None,
    vis_mode: str,
) -> bool:
    """Return whether either rendered edge endpoint is a folded-run box.

    Parameters
    ----------
    tail_name:
        Rendered edge tail node name.
    head_name:
        Rendered edge head node name.
    repeat_folds:
        Fold descriptors keyed by pass-free module address.
    vis_mode:
        ``"unrolled"`` or ``"rolled"`` visualization mode.

    Returns
    -------
    bool
        True when either endpoint is a repeat-fold representative.
    """

    if not repeat_folds:
        return False
    representative_names = _run_fold_representative_names(repeat_folds, vis_mode)
    return tail_name in representative_names or head_name in representative_names


def _rendered_edge_signature(edge_dict: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the visual identity used to deduplicate rendered edges.

    Parameters
    ----------
    edge_dict:
        Graphviz edge attributes.

    Returns
    -------
    tuple[Any, ...]
        Signature preserving genuinely distinct labels/styles.
    """

    return (
        "rendered_edge",
        edge_dict.get("style"),
        edge_dict.get("label"),
        edge_dict.get("headlabel"),
        edge_dict.get("taillabel"),
        edge_dict.get("xlabel"),
    )


def _intentional_parallel_edge_key(
    parent_node: GraphNode,
    child_node: GraphNode,
    render_edge: RenderEdge,
    tail_name: str,
    head_name: str,
    vis_mode: str,
) -> tuple[Any, ...] | None:
    """Return an occurrence key for intentional same-op parallel edges.

    Parameters
    ----------
    parent_node:
        Original rendered source node.
    child_node:
        Original rendered target node.
    render_edge:
        Edge occurrence metadata.
    tail_name:
        Final Graphviz tail name after collapse/fold remapping.
    head_name:
        Final Graphviz head name after collapse/fold remapping.
    vis_mode:
        ``"unrolled"`` or ``"rolled"`` visualization mode.

    Returns
    -------
    tuple[Any, ...] | None
        Occurrence key when duplicate rendered edges represent distinct argument
        slots on the same visible op pair.
    """

    occurrence_kind = render_edge.occurrence_key[0] if render_edge.occurrence_key else None
    if occurrence_kind not in {"edge_use", "parent_arg_position"}:
        return None
    original_tail = _render_node_label(parent_node, vis_mode).replace(":", "pass")
    original_head = _render_node_label(child_node, vis_mode).replace(":", "pass")
    if tail_name != original_tail or head_name != original_head:
        return None
    return render_edge.occurrence_key


def _intervention_hook_node_name(site_label: str) -> str:
    """Return the Graphviz node name for an intervention hook node.

    Parameters
    ----------
    site_label:
        Layer-pass label for the intervention site.

    Returns
    -------
    str
        Graphviz-safe hook node identifier.
    """

    return f"intervention_hook_{site_label.replace(':', 'pass')}"


def _add_intervention_hook_nodes(
    graphviz_graph: graphviz.Digraph,
    site_labels: set[str],
    graph_overrides: dict[str, Any] | None,
) -> None:
    """Add standalone hook nodes for ``vis_intervention_mode='as_node'``.

    Parameters
    ----------
    graphviz_graph:
        Graphviz graph being rendered.
    site_labels:
        Layer-pass labels with intervention specs.
    graph_overrides:
        Graph override dictionary, including optional intervention style keys.

    Returns
    -------
    None
        The graph is mutated in place.
    """

    fillcolor = str(
        intervention_graph_override(
            graph_overrides,
            "intervention_hook_fillcolor",
            INTERVENTION_HOOK_FILL_COLOR,
        )
    )
    color = str(
        intervention_graph_override(
            graph_overrides,
            "intervention_hook_color",
            INTERVENTION_HOOK_BORDER_COLOR,
        )
    )
    penwidth = str(intervention_graph_override(graph_overrides, "intervention_hook_penwidth", 2.0))
    for site_label in sorted(site_labels):
        spec = NodeSpec(
            lines=["intervention", site_label],
            shape="diamond",
            fillcolor=fillcolor,
            fontcolor="black",
            style="filled,solid",
            color=color,
            penwidth=float(penwidth),
            tooltip=f"Intervention hook after {site_label}",
            extra_attrs={"ordering": "out", "width": "0.35", "height": "0.25"},
        )
        node_args = _node_spec_to_graphviz_args(spec)
        node_args["name"] = _intervention_hook_node_name(site_label)
        graphviz_graph.node(**node_args)


def _collapsed_module_should_show_remainder(
    trace: "Trace",
    address: str,
    op_labels: Sequence[str],
    collapse_fn: CollapseFn | None,
) -> bool:
    """Return whether a collapsed box has own-output ops surfaced by the plan.

    Parameters
    ----------
    trace:
        Trace that owns the collapse plan and operation logs.
    address:
        Pass-free module address rendered as a collapsed box.
    op_labels:
        Pass-qualified operation labels in this module call.
    collapse_fn:
        Active collapse predicate, optionally carrying v2 plan metadata.

    Returns
    -------
    bool
        ``True`` when this collapsed module's own-output op is drawn as a
        separate visible sibling node in the resolved plan.
    """

    surfaced = _surfaced_own_output_ops(trace, address, op_labels)
    if not surfaced:
        return False
    plan = getattr(collapse_fn, "_torchlens_v2_plan", None)
    if isinstance(plan, CollapsePlan):
        visible_raw_ops = {_raw_op_label(node) for node in plan.nodes if isinstance(node, RawOp)}
        return any(op.layer_label in visible_raw_ops for op in surfaced)
    return getattr(collapse_fn, "_torchlens_v2_mode", None) == "max"


def _raw_op_label(node: RawOp) -> str:
    """Return the pass-qualified operation label represented by a raw plan node.

    Parameters
    ----------
    node:
        Raw operation node from a collapse plan.

    Returns
    -------
    str
        Operation label suitable for comparison with ``Op.layer_label``.
    """

    op = node.op
    if isinstance(op, str):
        return op
    return str(getattr(op, "layer_label", op))


def _collapsed_module_remainder_stats(
    trace: "Trace",
    address: str,
    op_labels: Sequence[str],
) -> dict[str, int]:
    """Return collapsed-box stats excluding separately rendered own-output ops.

    Parameters
    ----------
    trace:
        Trace that owns the module and operation logs.
    address:
        Pass-free module address rendered as a collapsed box.
    op_labels:
        Pass-qualified operation labels in this module call.

    Returns
    -------
    dict[str, int]
        Remainder layer and parameter totals for the inner collapsed box.
    """

    module = trace.modules[address]
    stats = {
        "num_layers": int(getattr(module, "num_layers", 0) or 0),
        "num_params": int(getattr(module, "num_params", 0) or 0),
        "num_params_trainable": int(getattr(module, "num_params_trainable", 0) or 0),
        "num_params_frozen": int(getattr(module, "num_params_frozen", 0) or 0),
    }
    surfaced = _surfaced_own_output_ops(trace, address, op_labels)
    stats["num_layers"] = max(0, stats["num_layers"] - len(surfaced))
    for op in surfaced:
        stats["num_params"] = max(0, stats["num_params"] - int(getattr(op, "num_params", 0) or 0))
        stats["num_params_trainable"] = max(
            0,
            stats["num_params_trainable"] - int(getattr(op, "num_params_trainable", 0) or 0),
        )
        stats["num_params_frozen"] = max(
            0,
            stats["num_params_frozen"] - int(getattr(op, "num_params_frozen", 0) or 0),
        )
    return stats


def _surfaced_own_output_ops(
    trace: "Trace",
    address: str,
    op_labels: Sequence[str],
) -> tuple["Op", ...]:
    """Return atomic own-output ops drawn outside a selected module box.

    Parameters
    ----------
    trace:
        Trace containing operation metadata.
    address:
        Pass-free owner module address.
    op_labels:
        Pass-qualified operation labels in this module call.

    Returns
    -------
    tuple[Op, ...]
        Atomic non-buffer ops whose own module is ``address`` and which are
        rendered separately because atomic collapse drops the innermost module.
    """

    surfaced: list[Op] = []
    for label in op_labels:
        op = trace.ops[label]
        if getattr(op, "is_buffer", False):
            continue
        if not getattr(op, "is_atomic_module", False):
            continue
        modules = list(getattr(op, "modules", ()) or ())
        if not modules:
            continue
        if modules[-1].rsplit(":", 1)[0] != address:
            continue
        surfaced.append(op)
    return tuple(surfaced)


def _collapsed_module_owner_key(
    trace: "Trace",
    address: str,
    call_index: str,
    vis_mode: str,
) -> str | None:
    """Return the parent module cluster key that should own a collapsed node.

    Parameters
    ----------
    trace:
        Trace owning the module hierarchy.
    address:
        Pass-free collapsed module address.
    call_index:
        Unrolled call index for ``address``.
    vis_mode:
        ``"unrolled"`` or ``"rolled"`` visualization mode.

    Returns
    -------
    str | None
        Parent module cluster key, or ``None`` for top-level children.
    """

    if "." not in address:
        return None
    parent_address = address.rsplit(".", 1)[0]
    if parent_address == "self" or parent_address not in trace.modules:
        return None
    if vis_mode == "rolled":
        return parent_address
    parent_key = f"{parent_address}:{call_index}"
    return parent_key if parent_key in trace.modules else None


def _run_fold_ellipsis_owner_key(
    trace: "Trace",
    fold: "ModuleRepeatFold",
    vis_mode: str,
) -> str | None:
    """Return the module cluster that owns a repeat-fold ellipsis node.

    Parameters
    ----------
    trace:
        Trace owning the module hierarchy.
    fold:
        Fold descriptor represented by the ellipsis.
    vis_mode:
        ``"unrolled"`` or ``"rolled"`` visualization mode.

    Returns
    -------
    str | None
        Parent cluster key for the fold representative, or ``None`` for
        top-level emission.
    """

    return _collapsed_module_owner_key(trace, fold.representative, "1", vis_mode)


def _raw_render_node_owner_key(node: GraphNode, vis_mode: str) -> str | None:
    """Return the innermost cluster that owns an unprojected render node.

    Parameters
    ----------
    node:
        Render node whose cluster ownership should be resolved.
    vis_mode:
        ``"unrolled"`` or ``"rolled"`` visualization mode.

    Returns
    -------
    str | None
        Innermost owning cluster, or ``None`` for a top-level node.
    """

    modules = [str(module) for module in getattr(node, "modules", ()) or ()]
    if getattr(node, "is_atomic_module", False) and modules:
        modules = modules[:-1]
    if not modules:
        return None
    owner = modules[-1]
    return owner.rsplit(":", 1)[0] if vis_mode == "rolled" else owner


def _rendered_endpoint_owner_key(
    trace: "Trace",
    node: GraphNode,
    *,
    segment: SegmentDescriptor | None,
    collapsed_address: str | None,
    hidden_fold: "ModuleRepeatFold | None",
    vis_mode: str,
) -> str | None:
    """Return the cluster owning an edge endpoint after render projection.

    Parameters
    ----------
    trace:
        Trace owning the rendered module hierarchy.
    node:
        Original graph endpoint.
    segment:
        Segment descriptor replacing the endpoint, if any.
    collapsed_address:
        Collapsed module address replacing the endpoint, if any.
    hidden_fold:
        Repeat fold whose ellipsis replaces the endpoint, if any.
    vis_mode:
        ``"unrolled"`` or ``"rolled"`` visualization mode.

    Returns
    -------
    str | None
        Owning cluster for the rendered endpoint, or ``None`` at top level.
    """

    if segment is not None:
        return segment.owner
    if hidden_fold is not None:
        return _run_fold_ellipsis_owner_key(trace, hidden_fold, vis_mode)
    if collapsed_address is not None:
        address, _, call_index = collapsed_address.partition(":")
        return _collapsed_module_owner_key(trace, address, call_index or "1", vis_mode)
    return _raw_render_node_owner_key(node, vis_mode)


def _lowest_common_rendered_owner_key(
    owner1: str | None,
    owner2: str | None,
    vis_mode: str,
) -> str | int:
    """Return the lowest common cluster for two rendered endpoint owners.

    Parameters
    ----------
    owner1:
        Owning cluster of the first rendered endpoint.
    owner2:
        Owning cluster of the second rendered endpoint.
    vis_mode:
        ``"unrolled"`` or ``"rolled"`` visualization mode.

    Returns
    -------
    str | int
        Lowest common cluster key, or ``-1`` for top-level emission.
    """

    if owner1 is None or owner2 is None:
        return -1
    address1, separator1, call1 = owner1.rpartition(":")
    address2, separator2, call2 = owner2.rpartition(":")
    if not separator1:
        address1 = owner1
    if not separator2:
        address2 = owner2
    parts1 = address1.split(".")
    parts2 = address2.split(".")
    common_parts: list[str] = []
    for part1, part2 in zip(parts1, parts2, strict=False):
        if part1 != part2:
            break
        common_parts.append(part1)
    if not common_parts:
        return -1
    common_address = ".".join(common_parts)
    if vis_mode == "rolled":
        return common_address
    if call1 != call2:
        return -1
    return f"{common_address}:{call1}" if separator1 and separator2 else common_address


def _queue_run_fold_ellipsis_node(
    graphviz_graph: graphviz.Digraph,
    module_edge_dict: Dict[str, Any],
    emitted_ellipsis_nodes: set[str],
    *,
    representative_name: str,
    fold: "ModuleRepeatFold",
    module_key: str | int,
) -> str:
    """Queue the in-flow ellipsis node for ``fold`` and return its node name.

    Parameters
    ----------
    graphviz_graph:
        Graphviz graph used for top-level node emission.
    module_edge_dict:
        Module-cluster accumulator for nested node placement.
    emitted_ellipsis_nodes:
        Graph-level set of ellipsis node names already emitted.
    representative_name:
        Graphviz node identifier for the folded representative.
    fold:
        Fold descriptor represented by the ellipsis.
    module_key:
        Module cluster key that owns the ellipsis node, or ``-1`` for top-level
        emission.

    Returns
    -------
    str
        Graphviz node name for the ellipsis.
    """

    ellipsis_name = _run_fold_ellipsis_node_name(representative_name)
    if ellipsis_name in emitted_ellipsis_nodes:
        return ellipsis_name
    emitted_ellipsis_nodes.add(ellipsis_name)
    node_args = {
        "name": ellipsis_name,
        "label": _run_fold_ellipsis_label(fold),
        "shape": "plaintext",
        "fontcolor": "#777777",
    }
    if module_key == -1:
        graphviz_graph.node(**node_args)
    else:
        module_edge_dict[cast(str, module_key)].setdefault("nodes", []).append(node_args)
    return ellipsis_name


def _add_edges_for_node(
    self: "Trace",
    parent_node: GraphNode,
    parent_is_collapsed_module: bool,
    vis_call_depth: int,
    node_color: str,
    module_edge_dict: Dict[str, Any],
    edges_used: Set[tuple[str, str, tuple[Any, ...]]],
    graphviz_graph: graphviz.Digraph,
    vis_mode: str = "unrolled",
    show_buffer_layers: BufferVisibilityLiteral = "meaningful",
    overrides: Optional[VisualizationOverrides] = None,
    collapse_fn: CollapseFn | None = None,
    edge_map: Optional[dict[str, list[RenderEdge]]] = None,
    vis_intervention_mode: VisInterventionModeLiteral = "node_mark",
    intervention_site_labels: set[str] | None = None,
    captured_forward_edges: list[CapturedForwardEdge] | None = None,
    rankdir: str = "BT",
    show_containers: ShowContainersLiteral = False,
    collapsed_container_nodes: Mapping[str, str] | None = None,
    repeat_folds: Mapping[str, "ModuleRepeatFold"] | None = None,
    run_fold_ellipsis_nodes: set[str] | None = None,
    segments: Mapping[str, SegmentDescriptor] | None = None,
    parent_segment: SegmentDescriptor | None = None,
    antiparallel_projected_edges: frozenset[tuple[str, str]] = frozenset(),
) -> None:
    """Add forward (and optionally grad) edges from a parent node to all its children.

    Handles several complex cases:

    - **Collapsed module nodes**: when parent or child is collapsed, the edge
      endpoint is the module box name, not the individual layer name.
    - **Intra-module edge skip**: when both parent and child map to the SAME
      collapsed module box AND share the same module nesting prefix up to
      ``vis_call_depth``, the edge is internal to the collapsed module
      and should not be drawn.
    - **Edge deduplication**: ``edges_used`` prevents duplicate edges that
      arise when multiple layers map to the same collapsed module node.
    - **Argument labels**: for non-commutative ops with multiple parents,
      edge labels show which argument position each parent occupies.
      Note: uses substring matching on layer_label for arg_label lookup,
      which has a theoretical false-positive risk if one label is a
      substring of another (extremely rare in practice).
    - **Pass annotations** (rolled mode): ``_label_rolled_call_indexs`` adds
      tail/head labels showing which ops an edge applies to.

    Args:
        parent_node: The node to add edges for.
        parent_is_collapsed_module: Whether the node is a collapsed module node.
        vis_call_depth: How many levels of module nesting to show.
        node_color: Color of the node.
        module_edge_dict: Dict mapping each cluster to its edges.
        edges_used: Set of (tail, head, occurrence) triples already added.
        graphviz_graph: The graphviz graph object.
        vis_mode: ``'unrolled'`` or ``'rolled'``.
        show_buffer_layers: Buffer visibility mode.
        overrides: Graphviz attribute overrides.
    """
    if edge_map is None:
        render_edges = [
            RenderEdge(
                target=_get_node_by_label(self, child_layer_label, vis_mode),
                metadata_child=None,
                occurrence_key=("edge", parent_node.layer_label, child_layer_label),
            )
            for child_layer_label in parent_node.children
        ]
    else:
        render_edges = edge_map.get(_render_node_label(parent_node, vis_mode), [])

    for render_edge in render_edges:
        child_node = render_edge.target
        metadata_child = render_edge.metadata_child
        child_render_name = _render_node_label(child_node, vis_mode).replace(":", "pass")

        if child_node.is_buffer and not _is_buffer_visible(child_node, show_buffer_layers):
            continue
        collapsed_head_name = (
            collapsed_container_nodes.get(child_render_name)
            if collapsed_container_nodes is not None
            else None
        )

        if parent_node.has_input_ancestor:
            edge_style = "solid"
        else:
            edge_style = "dashed"

        parent_module_name_w_pass: str | None = None
        if parent_segment is not None:
            tail_name = parent_segment.name
        elif parent_is_collapsed_module:
            parent_module_name_w_pass = _collapse_address_for_node(
                self,
                parent_node,
                vis_mode=vis_mode,
                collapse_fn=collapse_fn,
                max_module_depth=vis_call_depth,
            )
            parent_fold_ancestor = _run_fold_ancestor_for_node(parent_node, repeat_folds)
            if parent_fold_ancestor is not None:
                parent_module_name_w_pass = parent_fold_ancestor
            if parent_module_name_w_pass is None:
                continue
            tail_name = _run_fold_graph_node_name(parent_module_name_w_pass, vis_mode, repeat_folds)
        else:
            tail_name = _render_node_label(parent_node, vis_mode).replace(":", "pass")

        child_module_name_w_pass = _collapse_address_for_node(
            self,
            child_node,
            vis_mode=vis_mode,
            collapse_fn=collapse_fn,
            max_module_depth=vis_call_depth,
        )
        child_fold_ancestor = _run_fold_ancestor_for_node(child_node, repeat_folds)
        if child_fold_ancestor is not None:
            child_module_name_w_pass = child_fold_ancestor
        child_segment = _segment_for_node(child_node, segments)
        child_is_collapsed_module = child_module_name_w_pass is not None

        if child_segment is not None:
            head_name = child_segment.name
            child_is_collapsed_module = False
        elif child_is_collapsed_module:
            if child_module_name_w_pass is None:
                continue
            head_name = _run_fold_graph_node_name(child_module_name_w_pass, vis_mode, repeat_folds)
        else:
            head_name = _render_node_label(child_node, vis_mode).replace(":", "pass")
        if collapsed_head_name is not None:
            head_name = collapsed_head_name
            child_is_collapsed_module = False

        both_nodes_collapsed_modules = (
            parent_segment is None
            and child_segment is None
            and parent_is_collapsed_module
            and child_is_collapsed_module
        )

        # Collapsed module intra-edge skip: if both nodes are collapsed AND
        # they share the same module path up to vis_call_depth, the edge
        # is internal to the collapsed module box and should not be drawn.
        # The tail_name != head_name check handles the case where they map to
        # different collapsed modules (cross-module edge, should be drawn).
        if both_nodes_collapsed_modules and (tail_name != head_name):
            child_modules = child_node.modules[:]
            parent_modules = parent_node.modules[:]
            # Adjust for bottom-level submodule outputs (they belong to parent scope).
            if child_node.is_atomic_module:
                child_modules = child_modules[:-1]
            if parent_node.is_atomic_module:
                parent_modules = parent_modules[:-1]
            if child_modules[:vis_call_depth] == parent_modules[:vis_call_depth]:
                continue

        # Edge deduplication: multiple layers mapping to the same collapsed
        # module node would produce duplicate edges without this check.
        if (
            vis_intervention_mode == "as_node"
            and intervention_site_labels is not None
            and parent_node.layer_label in intervention_site_labels
        ):
            hook_name = _intervention_hook_node_name(parent_node.layer_label)
            hook_key = ("intervention_hook", parent_node.layer_label)
            if (tail_name, hook_name, hook_key) not in edges_used:
                edges_used.add((tail_name, hook_name, hook_key))
                graphviz_graph.edge(
                    tail_name=tail_name,
                    head_name=hook_name,
                    color=node_color,
                    fontcolor=node_color,
                    style=edge_style,
                    arrowsize=".7",
                    labelfontsize="8",
                )
            tail_name = hook_name

        edge_has_boundary = isinstance(parent_node, BoundaryNode) or isinstance(
            child_node, BoundaryNode
        )

        # Add it to the appropriate module cluster (most nested one containing both nodes)
        if edge_has_boundary:
            module = _get_lowest_module_for_two_render_nodes(
                parent_node,
                child_node,
                both_nodes_collapsed_modules,
                vis_call_depth,
            )
        else:
            module = _get_lowest_module_for_two_nodes(
                _base_node_for_metadata(parent_node),
                _base_node_for_metadata(child_node),
                both_nodes_collapsed_modules,
                vis_call_depth,
            )
        if tail_name == head_name and _self_loop_is_single_op_module(self, parent_node):
            module = -1
        # Preserve the edge's LCA cluster key BEFORE the has_input_ancestor loops
        # below clobber the ``module`` loop variable (they reassign it to each
        # node's own module path). Without this, the captured forward edge would
        # record the wrong cluster key and sibling rank-groups would never resolve
        # to a common cluster (always falling back to top-level emission).
        edge_module_key: str | int = module

        parent_hidden_fold = _run_fold_hidden_endpoint(parent_module_name_w_pass, repeat_folds)
        child_hidden_fold = _run_fold_hidden_endpoint(child_module_name_w_pass, repeat_folds)
        if parent_hidden_fold is not None and child_hidden_fold is parent_hidden_fold:
            continue

        run_fold_ellipsis_edge_key: tuple[Any, ...] | None = None
        ellipsis_fold = parent_hidden_fold or child_hidden_fold
        if ellipsis_fold is not None:
            if run_fold_ellipsis_nodes is None:
                run_fold_ellipsis_nodes = set()
            representative_name = _run_fold_graph_node_name(
                f"{ellipsis_fold.representative}:1",
                vis_mode,
                {ellipsis_fold.representative: ellipsis_fold},
            )
            ellipsis_name = _queue_run_fold_ellipsis_node(
                graphviz_graph,
                module_edge_dict,
                run_fold_ellipsis_nodes,
                representative_name=representative_name,
                fold=ellipsis_fold,
                module_key=_run_fold_ellipsis_owner_key(self, ellipsis_fold, vis_mode) or -1,
            )
            run_fold_ellipsis_edge_key = (
                "run_fold_ellipsis_edge",
                representative_name,
                ellipsis_name,
            )
            if parent_hidden_fold is not None:
                tail_name = ellipsis_name
            if child_hidden_fold is not None:
                head_name = ellipsis_name

        if parent_segment is not None or child_segment is not None or ellipsis_fold is not None:
            parent_owner = _rendered_endpoint_owner_key(
                self,
                parent_node,
                segment=parent_segment,
                collapsed_address=parent_module_name_w_pass,
                hidden_fold=parent_hidden_fold,
                vis_mode=vis_mode,
            )
            child_owner = _rendered_endpoint_owner_key(
                self,
                child_node,
                segment=child_segment,
                collapsed_address=child_module_name_w_pass,
                hidden_fold=child_hidden_fold,
                vis_mode=vis_mode,
            )
            module = _lowest_common_rendered_owner_key(parent_owner, child_owner, vis_mode)
            edge_module_key = module

        edge_is_self_loop = tail_name == head_name
        if edge_is_self_loop and (parent_segment is not None or child_segment is not None):
            continue
        edge_touches_run_fold = _edge_touches_run_fold(tail_name, head_name, repeat_folds, vis_mode)
        if edge_is_self_loop and edge_touches_run_fold and ellipsis_fold is None:
            continue
        if (
            edge_is_self_loop
            and not edge_touches_run_fold
            and not _is_rolled_loop_carried_self_edge(
                parent_node,
                child_node,
                vis_mode,
            )
        ):
            continue

        occurrence_key = render_edge.occurrence_key
        if parent_segment is not None or child_segment is not None:
            occurrence_key = ("segment_edge", tail_name, head_name)
        if run_fold_ellipsis_edge_key is not None and edge_touches_run_fold:
            occurrence_key = run_fold_ellipsis_edge_key
        elif edge_touches_run_fold:
            occurrence_key = ("run_fold_edge", tail_name, head_name)
        dedupe_key = (tail_name, head_name, occurrence_key)
        if dedupe_key in edges_used:
            continue
        edges_used.add(dedupe_key)

        edge_dict = {
            "tail_name": tail_name,
            "head_name": head_name,
            "color": node_color,
            "fontcolor": node_color,
            "style": edge_style,
            "arrowsize": ".7",
            "labelfontsize": "8",
        }
        if (tail_name, head_name) in antiparallel_projected_edges:
            edge_dict.update(_projected_antiparallel_edge_attrs())
        metadata_base = (
            _base_node_for_metadata(metadata_child)
            if metadata_child is not None and not edge_has_boundary
            else None
        )

        edge_label = None
        if not edge_is_self_loop and not child_is_collapsed_module and not edge_has_boundary:
            edge_label = (
                _compute_edge_label(
                    _base_node_for_metadata(parent_node),
                    metadata_base,
                    self,
                    vis_mode,
                )
                if metadata_base is not None
                else None
            )
        if edge_label is not None:
            edge_dict["label"] = edge_label
        if show_containers:
            container_label = _container_edge_label(metadata_base)
            if (
                container_label is None
                and show_containers == "nodes"
                and not _op_is_model_input_container_leaf(parent_node)
                and not _op_is_model_output_container_leaf(parent_node)
            ):
                container_label = _container_edge_label(_base_node_for_metadata(parent_node))
            if container_label is not None and "label" not in edge_dict:
                edge_dict["label"] = _html_container_edge_label(container_label)

        # Annotate ops for rolled node edge if it varies across ops
        if vis_mode == "rolled" and metadata_child is not None and not edge_has_boundary:
            metadata_base_for_pass = _base_node_for_metadata(metadata_child)
            parent_base_for_pass = _base_node_for_metadata(parent_node)
            if isinstance(metadata_base_for_pass, Layer) and isinstance(
                parent_base_for_pass, Layer
            ):
                # A recurrence back-edge may only merge its In/Out annotations
                # into a midpoint ``label`` if no conditional label was set above
                # and the argument labeler below (which adds headlabel/xlabel)
                # cannot fire for this edge; otherwise it keeps head/tail labels.
                arg_labeler_may_fire = not child_is_collapsed_module and bool(
                    _should_mark_arguments_on_edge(self, metadata_base_for_pass, show_buffer_layers)
                )
                _label_rolled_call_indexs(
                    metadata_base_for_pass,
                    parent_base_for_pass,
                    edge_dict,
                    is_self_loop=edge_is_self_loop,
                    rankdir=rankdir,
                    allow_midpoint_merge="label" not in edge_dict and not arg_labeler_may_fire,
                )

        # Label the arguments to the next node if multiple inputs
        if (
            not edge_is_self_loop
            and not child_is_collapsed_module
            and metadata_child is not None
            and not edge_has_boundary
        ):
            _label_node_arguments_if_needed(
                self,
                _base_node_for_metadata(parent_node),
                _base_node_for_metadata(metadata_child),
                edge_dict,
                show_buffer_layers,
                render_edge.argument_label,
            )

        for arg_name, arg_val in overrides.edge.items():  # type: ignore[union-attr]
            if callable(arg_val):
                edge_dict[arg_name] = str(arg_val(self, parent_node, metadata_child or child_node))
            else:
                edge_dict[arg_name] = str(arg_val)

        visual_signature = _rendered_edge_signature(edge_dict) + (
            _intentional_parallel_edge_key(
                parent_node,
                child_node,
                render_edge,
                tail_name,
                head_name,
                vis_mode,
            ),
        )
        visual_dedupe_key = (tail_name, head_name, visual_signature)
        if visual_dedupe_key in edges_used:
            continue
        edges_used.add(visual_dedupe_key)

        if module != -1:
            module_key = cast(str, module)
            module_edge_dict[module_key]["edges"].append(edge_dict)
            if parent_node.has_input_ancestor or child_node.has_input_ancestor:
                module_edge_dict[module_key]["has_input_ancestor"] = True
                for module in parent_node.modules:
                    module_key = module.split(":")[0] if vis_mode == "rolled" else module
                    module_edge_dict[module_key]["has_input_ancestor"] = True
                    if module_key == module:
                        break
                for module in child_node.modules:
                    module_key = module.split(":")[0] if vis_mode == "rolled" else module
                    module_edge_dict[module_key]["has_input_ancestor"] = True
                    if module_key == module:
                        break
        else:
            graphviz_graph.edge(**edge_dict)

        if captured_forward_edges is not None:
            captured_forward_edges.append(
                CapturedForwardEdge(
                    source_label=parent_node.layer_label,
                    target_label=child_node.layer_label,
                    tail_name=tail_name,
                    head_name=head_name,
                    source_step=int(getattr(parent_node, "step_index", 0) or 0),
                    target_step=int(getattr(child_node, "step_index", 0) or 0),
                    source_node=parent_node,
                    target_node=child_node,
                    module_key=edge_module_key,
                    occurrence_key=render_edge.occurrence_key,
                )
            )

        # Finally, add a backwards edge if both tensors have stored grads.
        if not (isinstance(parent_node, BoundaryNode) or isinstance(child_node, BoundaryNode)):
            _add_grad_edge(
                self,
                parent_node,
                child_node,
                edge_style,
                edge_module_key,
                module_edge_dict,
                graphviz_graph,
                overrides,  # type: ignore[arg-type]
            )


def _projected_antiparallel_edge_attrs() -> dict[str, str]:
    """Return explicit DOT attrs for projected anti-parallel fold edges.

    Returns
    -------
    dict[str, str]
        Non-constraining edge style used to mark projection artifacts.
    """

    return {
        "color": "#B36B00",
        "fontcolor": "#8A5200",
        "style": "dashed",
        "constraint": "false",
        "tooltip": "Projected anti-parallel fold edge",
    }


def _is_rolled_loop_carried_self_edge(
    parent_node: GraphNode,
    child_node: GraphNode,
    vis_mode: str,
) -> bool:
    """Return whether a same-endpoint edge represents rolled loop-carried flow.

    Parameters
    ----------
    parent_node:
        Rendered source node before endpoint collapsing.
    child_node:
        Rendered target node before endpoint collapsing.
    vis_mode:
        Active visualization mode.

    Returns
    -------
    bool
        True when the rolled edge advances from an earlier pass to a later pass.
    """

    if vis_mode != "rolled":
        return False
    parent_base = _base_node_for_metadata(parent_node)
    child_base = _base_node_for_metadata(child_node)
    if not isinstance(parent_base, Layer) or not isinstance(child_base, Layer):
        return False
    parent_passes = parent_base.child_ops_per_layer.get(child_base.layer_label, [])
    child_passes = child_base.parent_ops_per_layer.get(parent_base.layer_label, [])
    return any(
        child_pass > parent_pass
        for parent_pass, child_pass in zip(parent_passes, child_passes, strict=False)
    )


def _self_loop_is_single_op_module(trace: "Trace", node: GraphNode) -> bool:
    """Return whether a self-loop belongs to an atomic one-op module.

    Parameters
    ----------
    trace:
        Owning trace.
    node:
        Rendered node carrying the self-loop.

    Returns
    -------
    bool
        True when the node's innermost module contains exactly one rendered op.
    """

    modules = list(getattr(node, "modules", []) or [])
    if not modules:
        return False
    address = str(modules[-1]).rsplit(":", 1)[0]
    if address not in trace.modules:
        return False
    return _module_has_single_rendered_op(cast(Any, trace.modules[address]))


def _label_node_arguments_if_needed(
    self: "Trace",
    parent_node: Union["Op", "Layer"],
    child_node: Union["Op", "Layer"],
    edge_dict: Dict[str, Any],
    show_buffer_layers: BufferVisibilityLiteral = "meaningful",
    occurrence_argument_label: str | None = None,
) -> None:
    """Add argument position labels to an edge when the child has multiple non-commutative parents.

    For nodes like ``sub(a, b)`` where argument order matters, labels like
    ``"arg 0"`` / ``"arg 1"`` are added to distinguish which parent feeds
    which argument.

    Note on substring false-positive risk: the lookup ``parent_node.layer_label == arg_label``
    uses exact equality, so substring matching is not an issue here.  However, the
    ``parent_arg_positions`` keys are positional and the check iterates all of them,
    so a parent appearing in multiple arg positions will get multiple labels joined
    with ``<br/>``.

    Args:
        parent_node: The parent node whose edge is being labeled.
        child_node: The child node receiving the edge.
        edge_dict: Mutable dict of edge attributes; ``"headlabel"`` or ``"xlabel"``
            may be added.
        show_buffer_layers: Buffer visibility mode (affects parent count).
        occurrence_argument_label: Optional single argument label for one repeated
            edge-use occurrence.
    """
    if not _should_mark_arguments_on_edge(self, child_node, show_buffer_layers):
        return

    if occurrence_argument_label is not None:
        arg_labels = [occurrence_argument_label]
    else:
        arg_labels = []
        for arg_type in ["args", "kwargs"]:
            for arg_loc, arg_label in child_node.parent_arg_positions[arg_type].items():
                if parent_node.layer_label == arg_label:
                    arg_labels.append(f"{arg_type[:-1]} {str(arg_loc)}")

    arg_labels = "<br/>".join(html_escape(label) for label in arg_labels)  # type: ignore[assignment]
    if not arg_labels:
        return
    arg_label = f"<<FONT POINT-SIZE='10'><b>{arg_labels}</b></FONT>>"
    _set_argument_edge_label(edge_dict, arg_label)


def _set_argument_edge_label(edge_dict: Dict[str, Any], arg_label: str) -> None:
    """Attach an argument-position label without overwriting semantic edge labels.

    Args:
        edge_dict:
            Mutable Graphviz edge attribute dict.
        arg_label:
            HTML label string describing edge argument positions.
    """
    if "headlabel" not in edge_dict:
        edge_dict["headlabel"] = arg_label
        return
    if "xlabel" not in edge_dict:
        edge_dict["xlabel"] = arg_label
        return
    if edge_dict["xlabel"] == arg_label:
        return
    edge_dict["xlabel"] = edge_dict["xlabel"][:-1] + "<br/>" + arg_label[1:]


def _should_mark_arguments_on_edge(
    self: "Trace",
    child_node: Union["Op", "Layer"],
    show_buffer_layers: BufferVisibilityLiteral = "meaningful",
) -> bool:
    """Returns True if argument position labels should be shown on the edge to child_node.

    Skips commutative functions (add, mul, cat, eq, ne) where arg order is
    interchangeable -- showing "arg 0" vs "arg 1" would be misleading.
    For non-commutative ops, labels are shown when the child has multiple
    visible parents.

    Args:
        child_node: The child node whose incoming edge is being considered.
        show_buffer_layers: Buffer visibility mode.
    """
    # Commutative ops: argument order doesn't matter, skip labels.
    if child_node.layer_type in COMMUTE_FUNCS:
        return False

    if isinstance(child_node, Op):
        return _should_mark_arguments_on_unrolled_edge(self, child_node, show_buffer_layers)
    elif isinstance(child_node, Layer):
        return _should_mark_arguments_on_rolled_edge(self, child_node, show_buffer_layers)


def _should_mark_arguments_on_unrolled_edge(
    self: "Trace",
    child_node: "Op",
    show_buffer_layers: BufferVisibilityLiteral = "meaningful",
) -> bool:
    """Returns True if argument labels should be shown on an unrolled graph edge.

    Args:
        child_node: The child Op node whose incoming edge is being considered.
        show_buffer_layers: Buffer visibility mode.
    """
    num_parents_shown = len(child_node.parents)

    if show_buffer_layers != "always":
        num_parents_shown -= sum(
            [
                int(
                    self[parent].is_buffer
                    and not _is_buffer_visible(self[parent], show_buffer_layers)
                )
                for parent in child_node.parents
            ]
        )

    if num_parents_shown > 1:
        return True
    else:
        return False


def _should_mark_arguments_on_rolled_edge(
    self: "Trace",
    child_node: "Layer",
    show_buffer_layers: BufferVisibilityLiteral = "meaningful",
) -> bool:
    """Returns True if argument labels should be shown on a rolled graph edge.

    Args:
        child_node: The child Layer node whose incoming edge is being considered.
        show_buffer_layers: Buffer visibility mode.
    """
    for call_index, pass_parents in child_node.parents_per_pass.items():
        num_parents_shown = len(pass_parents)
        if show_buffer_layers != "always":
            num_parents_shown -= sum(
                [
                    int(
                        self.layer_logs[parent].is_buffer
                        and not _is_buffer_visible(self.layer_logs[parent], show_buffer_layers)
                    )
                    for parent in pass_parents
                ]
            )
        if num_parents_shown > 1:
            return True

    return False


def _op_is_model_input_container_leaf(node: GraphNode) -> bool:
    """Return whether ``node`` is a model-input container leaf.

    Parameters
    ----------
    node:
        Candidate render node.

    Returns
    -------
    bool
        ``True`` when the node is an input leaf that already receives a
        container-node source edge in ``show_containers="nodes"`` mode.
    """

    if not isinstance(node, Op):
        return False
    trace = getattr(node, "source_trace", None)
    if trace is None:
        return False
    for record in getattr(trace, "_containers", {}).values():
        if not isinstance(record, ContainerRecord):
            continue
        for snapshot in record.snapshots:
            if snapshot.role != Role.MODEL_INPUT:
                continue
            for occurrence in snapshot.leaf_occurrences:
                if _occurrence_matches_op(trace, occurrence.producer_op_label, node):
                    return True
    return False


def _op_is_model_output_container_leaf(node: GraphNode) -> bool:
    """Return whether ``node`` is a final model-output container leaf.

    Parameters
    ----------
    node:
        Candidate render node.

    Returns
    -------
    bool
        ``True`` when a sink container-node edge should carry its key label.
    """

    if not isinstance(node, Op):
        return False
    trace = getattr(node, "source_trace", None)
    if trace is None:
        return False
    return getattr(node, "layer_label", None) in set(getattr(trace, "output_layers", ()) or ())


def _occurrence_matches_op(trace: "Trace", label: str | None, op: Op) -> bool:
    """Return whether a container occurrence producer label resolves to ``op``.

    Parameters
    ----------
    trace:
        Trace containing ``op``.
    label:
        Final or raw producer label from a container occurrence.
    op:
        Candidate operation.

    Returns
    -------
    bool
        ``True`` if the label identifies ``op``.
    """

    if label is None:
        return False
    if label in trace.ops and trace.ops[label] is op:
        return True
    return (
        getattr(op, "layer_label_raw", None) == label
        or getattr(op, "_layer_label_raw", None) == label
    )


def _html_container_edge_label(text: str) -> str:
    """Return the HTML midpoint label for a container leaf edge.

    Parameters
    ----------
    text:
        Plain key/index/field label.

    Returns
    -------
    str
        Graphviz HTML label string.
    """

    return _html_edge_label(text)


def _html_edge_label(text: str) -> str:
    """Return an HTML head/tail edge label with an even transparent margin.

    graphviz does not allocate layout space for head/tail labels, so plain text
    touches the node, arrowhead, or edge it sits beside (an arrowhead can even
    clip the first letter). Wrapping the text in a borderless one-cell table with
    ``CELLPADDING`` gives it a small, even margin on every side -- enough to read
    as belonging to that endpoint without crowding it, and far less than the full
    blank text line a ``\\n`` pad would add.

    ``text`` may originate from user-controlled data (e.g. a ``DictKey``/``HFKey``
    container label pulled from a model's output dict), so it is HTML-escaped
    before interpolation -- an unescaped ``<``, ``>``, or ``&`` breaks Graphviz's
    HTML-like label parser and raises ``GraphvizRenderError``.
    """

    text = html_escape(text)
    return (
        f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="{_EDGE_LABEL_PAD}">'
        f'<TR><TD><FONT POINT-SIZE="{_EDGE_LABEL_FONT_SIZE}">{text}</FONT></TD></TR></TABLE>>'
    )


def _html_combined_recurrence_label(top: str, bottom: Optional[str] = None) -> str:
    """Return an HTML combined recurrence label with left/right spacers.

    Used for a node's self-loop arc, a merged recurrence back-edge, and merged
    buffer-edge annotations; the combined ``In``/``Out`` label sits beside the
    edge, and wide side spacer cells keep it clear of the node border and the
    edge line.  When ``bottom`` is ``None`` the label is a single line (a
    buffer edge may carry only one of ``In``/``Out``).

    ``top``/``bottom`` are escaped individually (not after joining) so the
    literal ``<BR/>`` line-break tag inserted between them is preserved.
    """

    top = html_escape(top)
    text = top if bottom is None else f"{top}<BR/>{html_escape(bottom)}"
    return (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
        f'<TR><TD WIDTH="{_SELF_LOOP_LABEL_HGAP}"></TD>'
        f'<TD><FONT POINT-SIZE="{_EDGE_LABEL_FONT_SIZE}">{text}</FONT></TD>'
        f'<TD WIDTH="{_SELF_LOOP_LABEL_HGAP}"></TD></TR></TABLE>>'
    )


def _is_rolled_recurrence_back_edge(child_node: "Layer", parent_node: "Layer") -> bool:
    """Returns True if a rolled edge runs backwards in execution order.

    A recurrence back-edge feeds a layer whose first op executes EARLIER than
    the source layer's first op (e.g. ``tanh_1_2 -> linear_1_1`` in an RNN
    cell).  ``Layer.step_index`` records exactly that first-op position in the
    global execution order, so a structural comparison suffices -- no label
    parsing.  Input/buffer layers (``step_index`` 0) and output layers
    (``step_index`` is the final op count, never smaller than a real parent's)
    are excluded by requiring both indices to be positive, so input/output and
    buffer-write edges are never flagged.

    Args:
        child_node: The destination Layer of the edge.
        parent_node: The source Layer of the edge.
    """
    child_step = child_node.step_index
    parent_step = parent_node.step_index
    if not isinstance(child_step, int) or not isinstance(parent_step, int):
        return False
    return 0 < child_step < parent_step


def _layer_has_rolled_self_loop(node: "Layer") -> bool:
    """Returns True if a Layer feeds itself across passes (rolled self-loop edge).

    Args:
        node: The Layer to check.
    """
    return node.layer_label in node.child_ops_per_layer


def _is_rolled_congested_recurrence_forward_edge(child_node: "Layer", parent_node: "Layer") -> bool:
    """Returns True for the forward partner of a recurrence back-edge whose band
    is congested by a self-loop.

    The forward edge and its merged back-edge run anti-parallel; when either
    endpoint ALSO carries a recurrence self-loop, three-plus near-parallel
    curves crowd that band and the forward edge's head/tail labels collide with
    the back-edge's spline and arrowhead.  Merging the forward edge's
    annotations into a midpoint ``label`` (which Graphviz reserves dummy-node
    space for) pushes the curves apart, the same cure as the back-edge merge.
    Plain two-node recurrences (no self-loop) keep head/tail labels: their band
    holds only two curves and lays out cleanly.

    Args:
        child_node: The destination Layer of the edge.
        parent_node: The source Layer of the edge.
    """
    return (
        _is_rolled_recurrence_back_edge(parent_node, child_node)
        and parent_node.layer_label in child_node.child_ops_per_layer
        and (_layer_has_rolled_self_loop(parent_node) or _layer_has_rolled_self_loop(child_node))
    )


def _is_rolled_buffer_edge(child_node: "Layer", parent_node: "Layer") -> bool:
    """Returns True if either endpoint of a rolled edge is a buffer layer.

    Buffer layers carry the structural ``is_buffer`` flag (glossary:
    ``is_buffer_source``); their ``step_index`` is 0, so
    ``_is_rolled_recurrence_back_edge`` deliberately never matches them.  A
    buffer's read edge and write edge run anti-parallel in a narrow band (the
    same geometry the back-edge midpoint merge solves), so buffer-incident
    edges get the same midpoint treatment for their ``In``/``Out`` annotations.

    Args:
        child_node: The destination Layer of the edge.
        parent_node: The source Layer of the edge.
    """
    return bool(child_node.is_buffer) or bool(parent_node.is_buffer)


def _rolled_pass_label_placement(
    child_node: "Layer",
    parent_node: "Layer",
    out_label: Optional[str],
    in_label: Optional[str],
) -> Optional[Tuple[str, str]]:
    """Pick explicit (labeldistance, labelangle) for an at-risk head/tail label.

    Returns None (keep graphviz's default endpoint placement) for edges that
    lay out straight; returns tuned tangent-relative placement attrs for the
    structural classes whose splines bow near the labeled endpoint (see the
    ``_ROLLED_*_LABEL_PLACEMENT`` constants).  Only single-annotation edges are
    conditioned: the attrs are per-edge and move BOTH endpoint labels, and
    dual-annotation edges lay out straight (their recurrence partners merge to
    midpoint labels instead).

    Args:
        child_node: The destination Layer of the edge.
        parent_node: The source Layer of the edge.
        out_label: The ``Out ...`` annotation, if the edge carries one.
        in_label: The ``In ...`` annotation, if the edge carries one.
    """
    has_head = in_label is not None
    has_tail = out_label is not None
    if has_head == has_tail:
        return None
    p_step = parent_node.step_index
    c_step = child_node.step_index
    if not isinstance(p_step, int) or not isinstance(c_step, int):
        return None
    span = c_step - p_step
    forward = 0 < p_step < c_step
    # Body edge of a >=3-op cycle: both endpoints recurrent, no direct
    # reverse edge (a direct reverse edge means a two-op cycle, which lays
    # out straight and is handled by the back-edge midpoint merge).
    if (
        forward
        and parent_node.num_passes > 1
        and child_node.num_passes > 1
        and parent_node.layer_label not in child_node.child_ops_per_layer
    ):
        return _ROLLED_CYCLE_HEAD_LABEL_PLACEMENT if has_head else _ROLLED_OBLIQUE_LABEL_PLACEMENT
    # Multi-step skip edge attached to a self-loop layer (bowed long curve);
    # edges into output layers run straight to the top rank and stay default.
    if (
        0 <= p_step < c_step
        and 2 <= span <= 4
        and (_layer_has_rolled_self_loop(parent_node) or _layer_has_rolled_self_loop(child_node))
        and child_node.layer_type != "output"
    ):
        return _ROLLED_OBLIQUE_LABEL_PLACEMENT
    # Adjacent forward edge into a self-loop layer: the self-loop arc sits
    # where the default head label would go.
    if has_head and forward and span == 1 and _layer_has_rolled_self_loop(child_node):
        return _ROLLED_SELF_LOOP_HEAD_LABEL_PLACEMENT
    return None


def _label_rolled_call_indexs(
    child_node: "Layer",
    parent_node: "Layer",
    edge_dict: Dict[str, Any],
    *,
    is_self_loop: bool = False,
    rankdir: str = "BT",
    allow_midpoint_merge: bool = False,
) -> None:
    """Add pass-number annotations to edges in rolled mode.

    In rolled mode, a single edge may represent connections from different
    ops.  When edges vary across ops (``edges_vary_across_ops``),
    tail and head labels show which ops the edge applies to, e.g.,
    ``"Out 1,3"`` / ``"In 2,4"``.  Uses ``int_list_to_compact_str`` for
    concise range notation (e.g., ``"1-3"`` instead of ``"1,2,3"``).

    Labels are emitted as HTML-like tables with small spacer cells so the text
    keeps an even gap from the node and the arrowhead without crowding it (a plain
    head/tail label is not allocated layout space and would touch the node).

    Self-loops are special-cased: their head/tail labels crowd against the node
    and (unlike forward edges) a recurrence self-edge never carries argument or
    conditional midpoint labels.  So the ``In``/``Out`` annotations are merged
    into a single midpoint ``label``, which Graphviz reserves layout space for
    (it is modeled as a dummy node), eliminating the overlap.  The ``In`` line is
    placed above ``Out`` for bottom-up graphs (flow points up) and flipped for
    top-down ones.

    Recurrence back-edges between distinct nodes get the same merge when the
    caller marks it safe (``allow_midpoint_merge``): a back-edge runs
    anti-parallel to the forward edge between the same two nodes, so four
    head/tail labels would fight for the narrow gap between the two near-parallel
    edges and collide with the arrowheads.  Merging into one midpoint ``label``
    both clears that gap and -- because Graphviz reserves a dummy-node spot for
    midpoint labels -- pushes the anti-parallel edges apart.  Forward edges keep
    head/tail labels: they may carry argument or conditional midpoint labels
    that a combined label would collide with, which is also why the caller must
    vouch (no existing ``label``, argument labeler cannot fire) before a
    back-edge merges.

    Buffer-incident edges (either endpoint has ``is_buffer``) merge under the
    same caller gate, including when only ONE of ``In``/``Out`` is present
    (rendered as a one-line midpoint label).  Buffer layers have
    ``step_index`` 0, so the back-edge check never matches them, yet a
    buffer's read and write edges run anti-parallel in the same narrow band --
    and the read edge curves enough that even its own head label collides with
    its own spline.

    Head/tail labels that stay (no merge) may get explicit per-edge
    ``labeldistance``/``labelangle`` attrs when the edge belongs to a
    structural class whose spline bows near the labeled endpoint; see
    ``_rolled_pass_label_placement``.

    Args:
        child_node: The child Layer node.
        parent_node: The parent Layer node.
        edge_dict: Mutable dict of edge attributes; taillabel/headlabel/label may be added.
        is_self_loop: Whether this edge is a node's recurrence self-loop.
        rankdir: Graphviz rank direction, used to order a combined label's lines.
        allow_midpoint_merge: Whether a non-self-loop edge may safely take a
            midpoint ``label`` (no conditional label present, no argument label
            coming); only such recurrence back-edges merge their annotations.
    """
    parent_call_indexs = parent_node.child_ops_per_layer[child_node.layer_label]
    child_call_indexs = child_node.parent_ops_per_layer[parent_node.layer_label]
    out_label = (
        f"Out {int_list_to_compact_str(parent_call_indexs)}"
        if parent_node.edges_vary_across_ops
        else None
    )
    in_label = (
        f"In {int_list_to_compact_str(child_call_indexs)}"
        if child_node.edges_vary_across_ops
        else None
    )

    is_buffer_edge = not is_self_loop and _is_rolled_buffer_edge(child_node, parent_node)
    merge_into_midpoint = is_self_loop or (
        allow_midpoint_merge
        and (
            is_buffer_edge
            or _is_rolled_recurrence_back_edge(child_node, parent_node)
            or _is_rolled_congested_recurrence_forward_edge(child_node, parent_node)
        )
    )
    if merge_into_midpoint and out_label is not None and in_label is not None:
        top, bottom = (out_label, in_label) if rankdir == "TB" else (in_label, out_label)
        edge_dict["label"] = _html_combined_recurrence_label(top, bottom)
        return
    single_label = out_label if out_label is not None else in_label
    if is_buffer_edge and allow_midpoint_merge and single_label is not None:
        # A buffer edge with a single annotation still merges: its read and
        # write edges run anti-parallel in a narrow band, where a head/tail
        # label collides with the opposing edge's spline and arrowhead.
        edge_dict["label"] = _html_combined_recurrence_label(single_label)
        return

    if out_label is not None:
        edge_dict["taillabel"] = _html_edge_label(out_label)
    if in_label is not None:
        edge_dict["headlabel"] = _html_edge_label(in_label)
    if (out_label is not None or in_label is not None) and allow_midpoint_merge:
        # ``allow_midpoint_merge`` doubles as the caller's promise that the
        # argument labeler cannot add a headlabel later, so the per-edge
        # placement attrs below can only ever move OUR pass labels.
        placement = _rolled_pass_label_placement(child_node, parent_node, out_label, in_label)
        if placement is not None:
            edge_dict["labeldistance"], edge_dict["labelangle"] = placement


def _get_lowest_module_for_two_render_nodes(
    node1: GraphNode,
    node2: GraphNode,
    both_nodes_collapsed_modules: bool,
    vis_call_depth: int,
) -> Union[str, int]:
    """Find the deepest module subgraph for render nodes including boundaries."""

    return _get_lowest_module_for_two_nodes(
        cast(Union["Op", "Layer"], node1),
        cast(Union["Op", "Layer"], node2),
        both_nodes_collapsed_modules,
        vis_call_depth,
    )


def _get_lowest_module_for_two_nodes(
    node1: Union["Op", "Layer"],
    node2: Union["Op", "Layer"],
    both_nodes_collapsed_modules: bool,
    vis_call_depth: int,
) -> Union[str, int]:
    """Find the deepest module subgraph that contains both nodes.

    Used to place an edge into the correct Graphviz cluster (subgraph).
    Edges between nodes in the same module cluster are drawn inside that
    cluster; edges crossing module boundaries are drawn at the level of
    the lowest common ancestor module.

    Returns -1 when no module contains both nodes (the edge belongs to the
    top-level graph, not any subgraph).

    Special handling:
    - ``is_atomic_module`` nodes are adjusted to their parent
      scope (they represent the module's output, rendered one level up).
    - Rolled mode: pass suffixes are stripped from module names so that all
      ops share the same cluster.
    - Both-collapsed case: when both nodes are collapsed module boxes, the
      containing module must be at least one level above the collapse depth.

    Args:
        node1: The first node.
        node2: The second node.
        both_nodes_collapsed_modules: Whether both nodes are collapsed module boxes.
        vis_call_depth: How many levels deep to visualize.

    Returns:
        Module name (str) for the containing cluster, or -1 for top-level.
    """
    node1_modules = node1.modules[:]
    node2_modules = node2.modules[:]

    if isinstance(node1, Layer) or isinstance(node2, Layer):
        node1_modules = [module.split(":")[0] for module in node1_modules]
        node2_modules = [module.split(":")[0] for module in node2_modules]

    if node1.is_atomic_module:
        node1_nested_modules = node1_modules[:-1]
    else:
        node1_nested_modules = node1_modules[:]

    if (
        (len(node1_modules) == 0)
        or (len(node2_modules) == 0)
        or (node1_modules[0] != node2_modules[0])
    ):
        return -1  # no submodule contains them both.

    if node1 == node2:
        if node1.is_atomic_module and (len(node1_modules) == 1):
            return -1
        elif node1.is_atomic_module and (len(node1_modules) > 1):
            module = node1_modules[-2]
        else:
            module = node1_modules[-1]
        return cast(str, module)

    if both_nodes_collapsed_modules:
        if (vis_call_depth == 1) or (len(node1_nested_modules) == 1):
            return -1
        if len(node1_modules) < vis_call_depth or len(node2_modules) < vis_call_depth:
            return -1
        if node1_modules[vis_call_depth - 1] == node2_modules[vis_call_depth - 1]:
            module = node1_modules[vis_call_depth - 2]
            return cast(str, module)

    module = node1_modules[0]
    for m in range(min(len(node1_modules), len(node2_modules))):
        if node1_modules[m] != node2_modules[m]:
            break
        module = node1_modules[m]

    return cast(str, module)


__all__ = [
    "_add_edges_for_node",
    "_add_intervention_hook_nodes",
    "_add_legend_to_graphviz",
    "_buffer_name_segment",
    "_collapsed_module_owner_key",
    "_collapsed_module_remainder_stats",
    "_collapsed_module_should_show_remainder",
    "_edge_touches_run_fold",
    "_get_lowest_module_for_two_nodes",
    "_get_lowest_module_for_two_render_nodes",
    "_get_node_by_label",
    "_html_combined_recurrence_label",
    "_html_container_edge_label",
    "_html_edge_label",
    "_intentional_parallel_edge_key",
    "_intervention_hook_node_name",
    "_is_buffer_visible",
    "_is_noise_buffer",
    "_is_rolled_buffer_edge",
    "_is_rolled_congested_recurrence_forward_edge",
    "_is_rolled_loop_carried_self_edge",
    "_is_rolled_recurrence_back_edge",
    "_is_run_fold_representative",
    "_label_node_arguments_if_needed",
    "_label_rolled_call_indexs",
    "_layer_has_rolled_self_loop",
    "_occurrence_matches_op",
    "_op_is_model_input_container_leaf",
    "_op_is_model_output_container_leaf",
    "_projected_antiparallel_edge_attrs",
    "_queue_run_fold_ellipsis_node",
    "_raw_op_label",
    "_render_node_label",
    "_rendered_edge_signature",
    "_rolled_pass_label_placement",
    "_run_fold_ancestor_for_node",
    "_run_fold_ellipsis_label",
    "_run_fold_ellipsis_node_name",
    "_run_fold_ellipsis_owner_key",
    "_run_fold_hidden_endpoint",
    "_segment_for_node",
    "_self_loop_is_single_op_module",
    "_set_argument_edge_label",
    "_should_mark_arguments_on_edge",
    "_should_mark_arguments_on_rolled_edge",
    "_should_mark_arguments_on_unrolled_edge",
    "_surfaced_own_output_ops",
]
