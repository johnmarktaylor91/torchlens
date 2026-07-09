"""Focus, skip, container, and sibling setup helpers for Graphviz rendering."""

# ruff: noqa: F403, F405

from ._render_common import *
from ._render_leaf import *
from ._render_edges import *
from ._render_nodes import *


def _decode_graphviz_stderr(error: subprocess.CalledProcessError) -> str:
    """Return a readable stderr string from a Graphviz process failure.

    Parameters
    ----------
    error:
        Process error raised by ``subprocess.run(..., check=True)``.

    Returns
    -------
    str
        Decoded stderr text, or a fallback process summary when stderr is empty.
    """

    stderr = error.stderr
    if isinstance(stderr, bytes):
        decoded = stderr.decode(errors="replace")
    elif stderr is None:
        decoded = ""
    else:
        decoded = str(stderr)
    decoded = decoded.strip()
    if decoded:
        return decoded
    return f"Graphviz exited with status {error.returncode}."


if TYPE_CHECKING:
    from ..data_classes.module import Module
    from ..data_classes.trace import Trace
    from .auto_collapse import ModuleRepeatFold


def _code_panel_composition_available(file_format: str, engine: str) -> bool:
    """Return whether a code panel can be composed side by side for this output.

    Side-by-side composition renders the graph and the code panel as separate
    SVGs and joins them, which keeps the graph's proportions untouched (the code
    no longer participates in Graphviz layout). It needs a vector-capable target
    format and the ``cairosvg`` rasterizer for non-SVG outputs.
    """

    if engine == "rank" or file_format not in _CODE_PANEL_COMPOSED_FORMATS:
        return False
    if file_format == "svg":
        return True
    try:
        import cairosvg  # noqa: F401
    except Exception:
        return False
    return True


def _svg_attrs_to_dict(attrs_text: str) -> dict[str, str]:
    """Parse simple SVG tag attributes into a dictionary.

    Parameters
    ----------
    attrs_text:
        Raw attribute text from an SVG tag.

    Returns
    -------
    dict[str, str]
        Attribute values keyed by attribute name.
    """

    return {
        match.group("name"): html.unescape(match.group("value"))
        for match in _SVG_ATTR_RE.finditer(attrs_text)
    }


def _is_non_file_svg_href(href: str) -> bool:
    """Return whether an SVG href should not be treated as a local file.

    Parameters
    ----------
    href:
        SVG image href value.

    Returns
    -------
    bool
        True for data URIs, fragments, and network URLs.
    """

    lowered = href.lower()
    return (
        lowered.startswith("data:")
        or lowered.startswith("#")
        or lowered.startswith("http://")
        or lowered.startswith("https://")
        or lowered.startswith("file:")
    )


def _resolve_svg_image_path(href: str) -> Path:
    """Resolve an SVG href to a local path.

    Parameters
    ----------
    href:
        SVG image href value.

    Returns
    -------
    Path
        Local image path. The caller handles missing or unreadable files.
    """

    candidate = Path(href).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate


def _replace_svg_attr_value(tag: str, attr_name: str, value: str) -> str:
    """Replace an attribute value in an SVG tag.

    Parameters
    ----------
    tag:
        SVG tag text.
    attr_name:
        Attribute name to replace.
    value:
        Replacement attribute value.

    Returns
    -------
    str
        Updated SVG tag.
    """

    escaped_value = html.escape(value, quote=True)
    pattern = re.compile(
        rf"""(?P<prefix>\b{re.escape(attr_name)}\s*=\s*)(?P<quote>["']).*?(?P=quote)"""
    )
    return pattern.sub(rf"\g<prefix>\g<quote>{escaped_value}\g<quote>", tag, count=1)


def _svg_image_mime_type(path: Path) -> str:
    """Return the data-URI MIME type for an SVG image file.

    Parameters
    ----------
    path:
        Local image path.

    Returns
    -------
    str
        MIME type for the data URI.
    """

    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".gif":
        return "image/gif"
    if suffix == ".svg":
        return "image/svg+xml"
    return "image/png"


def _svg_image_placeholder(attrs: Mapping[str, str]) -> str:
    """Return SVG text for an unreadable image placeholder.

    Parameters
    ----------
    attrs:
        Parsed attributes from the original SVG image tag.

    Returns
    -------
    str
        Replacement SVG text element.
    """

    x = _svg_numeric_attr(attrs, "x")
    y = _svg_numeric_attr(attrs, "y")
    width = _svg_numeric_attr(attrs, "width")
    height = _svg_numeric_attr(attrs, "height")
    text_x = x + (width / 2.0)
    text_y = y + (height / 2.0)
    return (
        f'<text text-anchor="middle" x="{text_x:.2f}" y="{text_y:.2f}" '
        'font-family="Times,serif" font-size="10.00">preview unavailable</text>'
    )


def _svg_numeric_attr(attrs: Mapping[str, str], name: str) -> float:
    """Read a numeric SVG attribute, defaulting to zero.

    Parameters
    ----------
    attrs:
        Parsed SVG attributes.
    name:
        Attribute name to parse.

    Returns
    -------
    float
        Parsed number, or zero when absent/unparseable.
    """

    value = attrs.get(name, "0")
    match = re.match(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value)
    if match is None:
        return 0.0
    return float(match.group(0))


def _format_backward_filter_caption(pass_filter: BackwardPassFilter) -> str:
    """Return a compact caption suffix for a backward-pass filter.

    Parameters
    ----------
    pass_filter:
        Normalized backward-pass filter.

    Returns
    -------
    str
        Human-readable caption suffix.
    """

    if pass_filter is None:
        return ""
    return f" shown: {int_list_to_compact_str(sorted(pass_filter))}"


def _build_skip_filtered_edge_map(
    trace: "Trace",
    entries_to_plot: Mapping[str, GraphNode],
    *,
    vis_mode: str,
    show_buffer_layers: BufferVisibilityLiteral,
    skip_fn: SkipFn | None,
) -> tuple[dict[str, list[RenderEdge]], set[str]]:
    """Build skip-aware outgoing edges for each rendered node.

    Parameters
    ----------
    trace:
        Owning Trace.
    entries_to_plot:
        Candidate nodes for the current visualization mode.
    vis_mode:
        ``"unrolled"`` or ``"rolled"``.
        show_buffer_layers:
        Buffer visibility mode.
    skip_fn:
        Optional user predicate receiving aggregate Layer objects.

    Returns
    -------
    tuple[dict[str, list[RenderEdge]], set[str]]
        Outgoing edge map and skipped node labels.
    """

    visible_entries = {
        label: node
        for label, node in entries_to_plot.items()
        if not node.is_buffer or _is_buffer_visible(node, show_buffer_layers)
    }
    skipped_labels: set[str] = set()
    if skip_fn is not None:
        for node in visible_entries.values():
            if isinstance(node, BoundaryNode):
                continue
            layer_log = _layer_log_for_node(trace, node)
            if not skip_fn(layer_log):
                continue
            if layer_log.is_input or layer_log.is_output:
                raise ValueError(
                    f"skip_fn cannot skip input or output layer '{layer_log.layer_label}'."
                )
            skipped_labels.add(_render_node_label(node, vis_mode))

    edge_map: dict[str, list[RenderEdge]] = {}
    for node in visible_entries.values():
        if _render_node_label(node, vis_mode) in skipped_labels:
            continue
        edge_map[_render_node_label(node, vis_mode)] = _expand_edges_through_skipped(
            trace,
            node,
            visible_entries,
            skipped_labels,
            vis_mode,
        )
    return edge_map, skipped_labels


def _is_hidden_buffer_update_node(
    trace: "Trace",
    node: GraphNode,
    entries_to_plot: Mapping[str, GraphNode],
    show_buffer_layers: BufferVisibilityLiteral,
    vis_mode: str,
) -> bool:
    """Return whether ``node`` only updates buffers hidden by the visibility mode.

    Parameters
    ----------
    trace:
        Trace containing the rendered nodes.
    node:
        Candidate non-buffer update operation.
    entries_to_plot:
        Nodes visible before buffer filtering.
    show_buffer_layers:
        Active buffer visibility mode.
    vis_mode:
        ``"unrolled"`` or ``"rolled"``.

    Returns
    -------
    bool
        True when every parent and child endpoint is a hidden buffer.
    """

    if isinstance(node, BoundaryNode) or node.is_buffer or show_buffer_layers == "always":
        return False
    endpoint_labels = list(node.parents) + list(node.children)
    if not endpoint_labels:
        return False
    entries_by_layer_label = {entry.layer_label: entry for entry in entries_to_plot.values()}
    endpoints: list[GraphNode] = []
    for label in endpoint_labels:
        endpoint = entries_to_plot.get(label) or entries_by_layer_label.get(label)
        if endpoint is None and vis_mode == "unrolled":
            endpoint = trace.layer_dict_all_keys.get(label)
        if endpoint is None:
            return False
        endpoints.append(endpoint)
    return all(
        endpoint.is_buffer and not _is_buffer_visible(endpoint, show_buffer_layers)
        for endpoint in endpoints
    )


def _entries_to_plot_for_context(
    trace: "Trace",
    vis_mode: VisModeLiteral,
) -> dict[str, GraphNode]:
    """Return renderer entries for ``vis_mode`` without focus rewriting.

    Parameters
    ----------
    trace:
        Trace being rendered.
    vis_mode:
        ``"unrolled"`` or ``"rolled"``.

    Returns
    -------
    dict[str, GraphNode]
        Render entries matching the forward renderer's default path.
    """

    if vis_mode == "unrolled":
        return dict(trace.layer_dict_main_keys)
    if vis_mode == "rolled":
        return dict(trace.layer_logs)
    raise ValueError("vis_mode must be either 'rolled' or 'unrolled'")


def _enumerate_base_rendered_node_emissions(
    trace: "Trace",
    entries_to_plot: Mapping[str, GraphNode],
    *,
    skipped_labels: set[str],
    vis_mode: str,
    vis_call_depth: int,
    show_buffer_layers: BufferVisibilityLiteral,
    collapse_fn: CollapseFn | None,
    repeat_folds: Mapping[str, "ModuleRepeatFold"] | None,
    show_containers: ShowContainersLiteral,
    collapsed_container_nodes: Mapping[str, str],
) -> tuple[RenderedNodeEmission, ...]:
    """Enumerate non-ellipsis nodes emitted by the forward renderer.

    Parameters
    ----------
    trace:
        Trace being rendered.
    entries_to_plot:
        Candidate render entries.
    skipped_labels:
        Labels hidden by skip handling.
    vis_mode:
        ``"unrolled"`` or ``"rolled"``.
    vis_call_depth:
        Module depth threshold.
    show_buffer_layers:
        Normalized buffer visibility.
    collapse_fn:
        Active collapse predicate.
    repeat_folds:
        Active run folds.
    show_containers:
        Container overlay mode.
    collapsed_container_nodes:
        Container leaf to summary-node mapping.

    Returns
    -------
    tuple[RenderedNodeEmission, ...]
        Base rendered node emissions.
    """

    emitted_names: set[str] = set()
    emissions: list[RenderedNodeEmission] = []
    for node in entries_to_plot.values():
        if _render_node_label(node, vis_mode) in skipped_labels:
            continue
        if node.is_buffer and not _is_buffer_visible(node, show_buffer_layers):
            continue
        emission = _base_rendered_node_emission(
            trace,
            node,
            vis_mode=vis_mode,
            vis_call_depth=vis_call_depth,
            collapse_fn=collapse_fn,
            repeat_folds=repeat_folds,
            show_containers=show_containers,
            collapsed_container_nodes=collapsed_container_nodes,
        )
        if emission is None:
            continue
        if emission.kind != "hidden_run_member":
            if emission.name in emitted_names:
                continue
            emitted_names.add(emission.name)
        emissions.append(emission)
    return tuple(emissions)


def _base_rendered_node_emission(
    trace: "Trace",
    node: GraphNode,
    *,
    vis_mode: str,
    vis_call_depth: int,
    collapse_fn: CollapseFn | None,
    repeat_folds: Mapping[str, "ModuleRepeatFold"] | None,
    show_containers: ShowContainersLiteral,
    collapsed_container_nodes: Mapping[str, str],
) -> RenderedNodeEmission | None:
    """Return the base node emission for one render node.

    Parameters
    ----------
    trace:
        Trace being rendered.
    node:
        Candidate render node.
    vis_mode:
        ``"unrolled"`` or ``"rolled"``.
    vis_call_depth:
        Module depth threshold.
    collapse_fn:
        Active collapse predicate.
    repeat_folds:
        Active run folds.
    show_containers:
        Container overlay mode.
    collapsed_container_nodes:
        Container leaf to summary-node mapping.

    Returns
    -------
    RenderedNodeEmission | None
        Emitted node, or ``None`` when hidden by run/container folding.
    """

    collapse_address = _collapse_address_for_node(
        trace,
        node,
        vis_mode=vis_mode,
        collapse_fn=collapse_fn,
        max_module_depth=vis_call_depth,
    )
    fold_ancestor_address = _run_fold_ancestor_for_node(node, repeat_folds)
    if fold_ancestor_address is not None:
        collapse_address = fold_ancestor_address
    if collapse_address is not None:
        if repeat_folds is not None and not _is_run_fold_representative(
            collapse_address, repeat_folds
        ):
            return RenderedNodeEmission(
                name=_render_node_label(node, vis_mode).replace(":", "pass"),
                kind="hidden_run_member",
                node=node,
                module_address=collapse_address.rsplit(":", 1)[0],
                call=collapse_address,
                fold=_run_fold_for_address(collapse_address, repeat_folds),
            )
        name = _run_fold_graph_node_name(collapse_address, vis_mode, repeat_folds)
        address = collapse_address.rsplit(":", 1)[0]
        return RenderedNodeEmission(
            name=name,
            kind="module_box",
            node=node,
            module_address=address,
            call=collapse_address,
            fold=_run_fold_for_address(address, repeat_folds),
        )
    name = _render_node_label(node, vis_mode).replace(":", "pass")
    if show_containers in {"collapsed", "auto"} and name in collapsed_container_nodes:
        return None
    if isinstance(node, BoundaryNode):
        return RenderedNodeEmission(
            name=name,
            kind="boundary",
            node=node,
            op_label=node.layer_label,
            boundary_kind=node.boundary_kind,
        )
    return RenderedNodeEmission(name=name, kind="raw_op", node=node, op_label=node.layer_label)


def _enumerate_run_fold_ellipsis_emissions(
    trace: "Trace",
    entries_to_plot: Mapping[str, GraphNode],
    *,
    edge_map: Mapping[str, Sequence[RenderEdge]],
    skipped_labels: set[str],
    vis_mode: str,
    vis_call_depth: int,
    show_buffer_layers: BufferVisibilityLiteral,
    collapse_fn: CollapseFn | None,
    repeat_folds: Mapping[str, "ModuleRepeatFold"] | None,
    collapsed_container_nodes: Mapping[str, str],
) -> tuple[RenderedNodeEmission, ...]:
    """Enumerate repeat-fold ellipsis nodes triggered by rendered edges.

    Parameters
    ----------
    trace:
        Trace being rendered.
    entries_to_plot:
        Candidate render entries.
    edge_map:
        Skip-filtered edge map.
    skipped_labels:
        Labels hidden by skip handling.
    vis_mode:
        ``"unrolled"`` or ``"rolled"``.
    vis_call_depth:
        Module depth threshold.
    show_buffer_layers:
        Normalized buffer visibility.
    collapse_fn:
        Active collapse predicate.
    repeat_folds:
        Active run folds.
    collapsed_container_nodes:
        Container leaf to summary-node mapping.

    Returns
    -------
    tuple[RenderedNodeEmission, ...]
        Ellipsis nodes in first-trigger order.
    """

    if not repeat_folds:
        return ()
    emitted: set[str] = set()
    emissions: list[RenderedNodeEmission] = []
    for parent_node in entries_to_plot.values():
        if _render_node_label(parent_node, vis_mode) in skipped_labels:
            continue
        if parent_node.is_buffer and not _is_buffer_visible(parent_node, show_buffer_layers):
            continue
        parent_endpoint = _collapsed_endpoint_for_emission(
            trace,
            parent_node,
            vis_mode=vis_mode,
            vis_call_depth=vis_call_depth,
            collapse_fn=collapse_fn,
            repeat_folds=repeat_folds,
        )
        parent_name = parent_endpoint or _render_node_label(parent_node, vis_mode).replace(
            ":", "pass"
        )
        for render_edge in edge_map.get(_render_node_label(parent_node, vis_mode), ()):
            child_node = render_edge.target
            if child_node.is_buffer and not _is_buffer_visible(child_node, show_buffer_layers):
                continue
            child_render_name = _render_node_label(child_node, vis_mode).replace(":", "pass")
            child_endpoint = _collapsed_endpoint_for_emission(
                trace,
                child_node,
                vis_mode=vis_mode,
                vis_call_depth=vis_call_depth,
                collapse_fn=collapse_fn,
                repeat_folds=repeat_folds,
            )
            child_name = (
                collapsed_container_nodes.get(child_render_name)
                or child_endpoint
                or child_render_name
            )
            parent_fold = _run_fold_hidden_endpoint(parent_endpoint, repeat_folds)
            child_fold = _run_fold_hidden_endpoint(child_endpoint, repeat_folds)
            if parent_fold is not None and child_fold is parent_fold:
                continue
            fold = parent_fold or child_fold
            if fold is None:
                continue
            tail_name = (
                _run_fold_graph_node_name(parent_endpoint, vis_mode, repeat_folds)
                if parent_endpoint
                else parent_name
            )
            head_name = (
                _run_fold_graph_node_name(child_endpoint, vis_mode, repeat_folds)
                if child_endpoint
                else child_name
            )
            if tail_name == head_name:
                continue
            representative_name = _run_fold_graph_node_name(
                f"{fold.representative}:1",
                vis_mode,
                {fold.representative: fold},
            )
            ellipsis_name = _run_fold_ellipsis_node_name(representative_name)
            if ellipsis_name in emitted:
                continue
            emitted.add(ellipsis_name)
            emissions.append(
                RenderedNodeEmission(
                    name=ellipsis_name,
                    kind="run_fold_ellipsis",
                    fold=fold,
                )
            )
    return tuple(emissions)


def _collapsed_endpoint_for_emission(
    trace: "Trace",
    node: GraphNode,
    *,
    vis_mode: str,
    vis_call_depth: int,
    collapse_fn: CollapseFn | None,
    repeat_folds: Mapping[str, "ModuleRepeatFold"] | None,
) -> str | None:
    """Return pass-qualified collapsed endpoint for node-universe enumeration.

    Parameters
    ----------
    trace:
        Trace being rendered.
    node:
        Candidate render node.
    vis_mode:
        ``"unrolled"`` or ``"rolled"``.
    vis_call_depth:
        Module depth threshold.
    collapse_fn:
        Active collapse predicate.
    repeat_folds:
        Active run folds.

    Returns
    -------
    str | None
        Collapsed endpoint address, including hidden repeat-fold members.
    """

    endpoint = _collapse_address_for_node(
        trace,
        node,
        vis_mode=vis_mode,
        collapse_fn=collapse_fn,
        max_module_depth=vis_call_depth,
    )
    fold_ancestor = _run_fold_ancestor_for_node(node, repeat_folds)
    return fold_ancestor if fold_ancestor is not None else endpoint


def _container_role(node: BaseGraphNode) -> str | None:
    """Return the node's role within its container, if present."""

    path = tuple(getattr(node, "container_path", ()) or ())
    if not path:
        return None
    return _container_component_role(path[-1])


def _container_leaf_groups(
    entries_to_plot: Mapping[str, GraphNode],
    *,
    vis_mode: str,
) -> dict[str, list[GraphNode]]:
    """Group rendered container leaves by semantic container id."""

    groups: dict[str, list[GraphNode]] = defaultdict(list)
    for node in entries_to_plot.values():
        if vis_mode != "unrolled" or not isinstance(node, Op):
            continue
        group_id = _container_group_id(node)
        if group_id is not None:
            groups[group_id].append(node)
    return groups


def _collapsed_container_leaf_nodes(
    trace: "Trace",
    entries_to_plot: Mapping[str, GraphNode],
    *,
    vis_mode: str,
    show_containers: ShowContainersLiteral,
    container_max_inline: int,
    pending_nodes: list[dict[str, Any]],
) -> dict[str, str]:
    """Return leaf-to-summary node names hidden by homogeneous collapse."""

    if show_containers not in {"collapsed", "auto", "nodes"} or vis_mode != "unrolled":
        return {}
    hidden: dict[str, str] = {}
    for leaves in _container_leaf_groups(entries_to_plot, vis_mode=vis_mode).values():
        if len(leaves) <= container_max_inline or not _container_leaf_shapes_identical(leaves):
            continue
        group_id = cast(str, _container_group_id(cast(BaseGraphNode, leaves[0])))
        summary_node = _collapsed_container_node_name(group_id)
        for leaf in leaves:
            hidden[_render_node_label(leaf, vis_mode).replace(":", "pass")] = summary_node
        _add_collapsed_container_node(pending_nodes, leaves, vis_mode=vis_mode)
    return hidden


def _container_leaf_shapes_identical(leaves: Sequence[GraphNode]) -> bool:
    """Return whether all container leaves share one shape."""

    shapes = {tuple(getattr(leaf, "shape", ()) or ()) for leaf in leaves}
    return len(shapes) == 1


def _container_nodes_and_overlay_edges(
    trace: "Trace",
    collapsed_container_nodes: Mapping[str, str],
    *,
    vis_call_depth: int,
    collapse_fn: CollapseFn | None,
) -> tuple[list[ContainerOverlayNode], list[ContainerOverlayEdge]]:
    """Return render-only container nodes and non-dataflow overlay edges.

    Parameters
    ----------
    trace:
        Trace whose registry-backed containers should be rendered.
    collapsed_container_nodes:
        Mapping from rendered leaf node names to collapsed summary node names.
    vis_call_depth:
        Maximum module nesting levels to show before collapsed-module summaries.
    collapse_fn:
        Optional custom module-collapse predicate.

    Returns
    -------
    tuple[list[ContainerOverlayNode], list[ContainerOverlayEdge]]
        Node argument dictionaries and overlay edge specs. The edge specs are
        intentionally separate from real dataflow edge queues.
    """

    nodes_by_name: dict[str, ContainerOverlayNode] = {}
    edges: list[ContainerOverlayEdge] = []
    edge_keys: set[tuple[str, str, str, str]] = set()
    records = getattr(trace, "_containers", {})
    for record in records.values():
        if not isinstance(record, ContainerRecord):
            continue
        node_name = _container_record_node_name(record)
        selected_snapshot: ContainerSnapshot | None = None
        for snapshot in record.snapshots:
            if snapshot.role not in {Role.MODEL_INPUT, Role.MODEL_OUTPUT, Role.CALL_OUTPUT}:
                continue
            if selected_snapshot is None:
                selected_snapshot = snapshot
                owner_key = _container_record_owner_key(
                    trace,
                    record,
                    vis_call_depth=vis_call_depth,
                    collapse_fn=collapse_fn,
                )
                nodes_by_name.setdefault(
                    node_name,
                    ContainerOverlayNode(
                        args=_container_record_node_args(record, snapshot, node_name),
                        owner_key=owner_key,
                    ),
                )
            if snapshot.role in {Role.MODEL_INPUT, Role.MODEL_OUTPUT}:
                _append_boundary_container_edges(
                    trace,
                    snapshot,
                    node_name,
                    collapsed_container_nodes,
                    edge_keys,
                    edges,
                )
            elif snapshot.role == Role.CALL_OUTPUT:
                _append_member_of_container_edges(
                    trace,
                    snapshot,
                    node_name,
                    collapsed_container_nodes,
                    edge_keys,
                    edges,
                )
    return list(nodes_by_name.values()), edges


def _container_record_owner_key(
    trace: "Trace",
    record: ContainerRecord,
    *,
    vis_call_depth: int,
    collapse_fn: CollapseFn | None,
) -> str | None:
    """Return a single module owner for a call-output container record.

    Parameters
    ----------
    trace:
        Trace used for resolving occurrence producer labels.
    record:
        Container registry record.
    vis_call_depth:
        Maximum module nesting levels to show before collapsed-module summaries.
    collapse_fn:
        Optional custom module-collapse predicate.

    Returns
    -------
    str | None
        Owner module-cluster key, or ``None`` when the record is a boundary
        container or its leaves do not share one module owner.
    """

    if any(snapshot.role in {Role.MODEL_INPUT, Role.MODEL_OUTPUT} for snapshot in record.snapshots):
        return None
    owners: set[str | int] = set()
    for snapshot in record.snapshots:
        if snapshot.role != Role.CALL_OUTPUT:
            continue
        for occurrence in snapshot.leaf_occurrences:
            op = _op_for_occurrence_label(trace, occurrence.producer_op_label)
            if op is None:
                continue
            owners.add(
                _owner_module_key_for_node(
                    trace,
                    op,
                    vis_mode="unrolled",
                    vis_call_depth=vis_call_depth,
                    collapse_fn=collapse_fn,
                )
            )
    if len(owners) != 1:
        return None
    owner = next(iter(owners))
    return cast(str, owner) if owner != -1 else None


def _append_boundary_container_edges(
    trace: "Trace",
    snapshot: ContainerSnapshot,
    node_name: str,
    collapsed_container_nodes: Mapping[str, str],
    edge_keys: set[tuple[str, str, str, str]],
    edges: list[ContainerOverlayEdge],
) -> None:
    """Append source or sink container-node overlay edges.

    Parameters
    ----------
    trace:
        Trace used for resolving occurrence producer labels.
    snapshot:
        Boundary snapshot whose leaves should connect to the container node.
    node_name:
        Rendered container node name.
    collapsed_container_nodes:
        Mapping from rendered leaf node names to collapsed summary node names.
    edge_keys:
        Overlay-only dedupe keys.
    edges:
        Mutable overlay edge sink.
    """

    for occurrence in snapshot.leaf_occurrences:
        leaf_node = _render_node_name_for_occurrence(trace, occurrence.producer_op_label)
        if leaf_node is None and snapshot.role == Role.MODEL_OUTPUT:
            leaf_node = _render_output_node_name_for_path(trace, occurrence.path)
        if leaf_node is None:
            continue
        leaf_node = collapsed_container_nodes.get(leaf_node, leaf_node)
        label = _container_path_leaf_label(occurrence.path)
        attrs = _container_boundary_edge_attrs(label)
        if snapshot.role == Role.MODEL_INPUT:
            _append_container_overlay_edge(
                edge_keys,
                edges,
                namespace="container-source",
                tail_name=node_name,
                head_name=leaf_node,
                attrs=attrs,
            )
        elif snapshot.role == Role.MODEL_OUTPUT:
            _append_container_overlay_edge(
                edge_keys,
                edges,
                namespace="container-sink",
                tail_name=leaf_node,
                head_name=node_name,
                attrs=attrs,
            )


def _append_member_of_container_edges(
    trace: "Trace",
    snapshot: ContainerSnapshot,
    node_name: str,
    collapsed_container_nodes: Mapping[str, str],
    edge_keys: set[tuple[str, str, str, str]],
    edges: list[ContainerOverlayEdge],
) -> None:
    """Append dashed producer-to-container grouping ties for mid-graph outputs.

    Parameters
    ----------
    trace:
        Trace used for resolving occurrence producer labels.
    snapshot:
        Call-output snapshot whose producers should be associated to a
        render-only container node.
    node_name:
        Rendered container node name.
    collapsed_container_nodes:
        Mapping from rendered leaf node names to collapsed summary node names.
    edge_keys:
        Overlay-only dedupe keys.
    edges:
        Mutable overlay edge sink.
    """

    for occurrence in snapshot.leaf_occurrences:
        producer_node = _render_node_name_for_occurrence(trace, occurrence.producer_op_label)
        if producer_node is None:
            continue
        producer_node = collapsed_container_nodes.get(producer_node, producer_node)
        _append_container_overlay_edge(
            edge_keys,
            edges,
            namespace="container-member",
            tail_name=producer_node,
            head_name=node_name,
            attrs=_container_member_edge_attrs(),
        )


def _container_record_node_name(record: ContainerRecord) -> str:
    """Return a stable Graphviz node name for one rendered container record.

    Parameters
    ----------
    record:
        Container registry record.

    Returns
    -------
    str
        Graphviz node identifier.
    """

    return f"container_node_{record.ordinal}"


def _container_record_node_args(
    record: ContainerRecord,
    snapshot: ContainerSnapshot,
    node_name: str,
) -> dict[str, str]:
    """Return Graphviz node args for a collapsed labeled container node.

    Parameters
    ----------
    record:
        Container registry record.
    snapshot:
        Snapshot being rendered.
    node_name:
        Graphviz node identifier.

    Returns
    -------
    dict[str, str]
        Node attributes.
    """

    return {
        "name": node_name,
        "label": render_lines_to_html([_container_record_label(record, snapshot)]),
        "shape": "box",
        "style": "filled,dashed",
        "fillcolor": "white",
        "color": "#777777",
        "fontcolor": "black",
        "ordering": "out",
        "group": f"container_record_{record.ordinal}",
        "fixedsize": "false",
        "margin": "0.22,0.11",
    }


def _container_record_label(record: ContainerRecord, snapshot: ContainerSnapshot) -> str:
    """Return the visible type-and-role label for a container node.

    Parameters
    ----------
    record:
        Container registry record.
    snapshot:
        Snapshot being rendered.

    Returns
    -------
    str
        Label such as ``"dict[3] (model input)"``.
    """

    role = snapshot.role.value.replace("_", " ")
    return f"{_container_spec_label(snapshot.spec, fallback=record.object_kind)} ({role})"


def _container_spec_label(spec: ContainerSpec, *, fallback: str) -> str:
    """Return a concise container type label.

    Parameters
    ----------
    spec:
        Captured container spec.
    fallback:
        Portable object kind used if the spec lacks type details.

    Returns
    -------
    str
        Human-readable container type and immediate arity.
    """

    if spec.type_qualname:
        return spec.type_qualname
    if spec.kind in {"dict", "tuple", "list"} and spec.length is not None:
        return f"{spec.kind}[{spec.length}]"
    if spec.length is not None:
        return f"{spec.kind}[{spec.length}]"
    return fallback.rsplit(".", 1)[-1]


def _container_member_edge_attrs() -> dict[str, str]:
    """Return Graphviz attrs for a mid-graph member-of tie.

    Returns
    -------
    dict[str, str]
        Arrowless, non-constraining association edge attributes.
    """

    return {
        "color": "#999999",
        "fontcolor": "#777777",
        "style": "dashed",
        "arrowhead": "none",
        "arrowsize": ".6",
        "constraint": "false",
    }


def _container_clusters_for_graphviz(
    trace: "Trace",
    entries_to_plot: Mapping[str, GraphNode],
    *,
    vis_mode: str,
    vis_call_depth: int,
    collapse_fn: CollapseFn | None,
    collapsed_container_nodes: Mapping[str, str],
) -> list[ContainerClusterSpec]:
    """Return single-owner container clusters for the active draw mode."""

    clusters: list[ContainerClusterSpec] = []
    for group_id, leaves in _container_leaf_groups(entries_to_plot, vis_mode=vis_mode).items():
        node_names = tuple(
            _render_node_label(leaf, vis_mode).replace(":", "pass") for leaf in leaves
        )
        if any(node_name in collapsed_container_nodes for node_name in node_names):
            continue
        owner_key = _single_container_owner(
            trace,
            leaves,
            vis_mode=vis_mode,
            vis_call_depth=vis_call_depth,
            collapse_fn=collapse_fn,
        )
        if owner_key is None:
            continue
        kind = _container_kind(cast(BaseGraphNode, leaves[0])) or "container"
        clusters.append(
            ContainerClusterSpec(
                cluster_id=_collapsed_container_node_name(group_id),
                owner_key=owner_key,
                node_names=node_names,
                title=f"{kind} {_container_path_label(())}",
                kind=kind,
            )
        )
    return clusters


def _single_container_owner(
    trace: "Trace",
    leaves: Sequence[GraphNode],
    *,
    vis_mode: str,
    vis_call_depth: int,
    collapse_fn: CollapseFn | None,
) -> str | int | None:
    """Return the sole active module-cluster owner for ``leaves``."""

    owners: set[str | int] = set()
    for leaf in leaves:
        owner = _owner_module_key_for_node(
            trace,
            leaf,
            vis_mode=vis_mode,
            vis_call_depth=vis_call_depth,
            collapse_fn=collapse_fn,
        )
        owners.add(owner)
    if len(owners) != 1:
        return None
    owner = next(iter(owners))
    return owner if owner != -1 else None


def _owner_module_key_for_node(
    trace: "Trace",
    node: GraphNode,
    *,
    vis_mode: str,
    vis_call_depth: int,
    collapse_fn: CollapseFn | None,
) -> str | int:
    """Return the module cluster key that owns a rendered node."""

    collapse_address = _collapse_address_for_node(
        trace,
        node,
        vis_mode=vis_mode,
        collapse_fn=collapse_fn,
        max_module_depth=vis_call_depth,
    )
    if collapse_address is not None:
        return collapse_address if vis_mode == "unrolled" else collapse_address.split(":", 1)[0]
    modules = list(getattr(node, "modules", ()) or ())
    if getattr(node, "is_atomic_module", False) and modules:
        modules = modules[:-1]
    if not modules:
        return -1
    owner = modules[-1]
    return owner if vis_mode == "unrolled" else owner.split(":", 1)[0]


def _build_module_focus_entries(
    trace: "Trace",
    entries_to_plot: Mapping[str, GraphNode],
    target_module: "Module",
    *,
    vis_mode: str,
) -> dict[str, GraphNode]:
    """Return render entries focused on one module plus synthetic boundaries.

    Parameters
    ----------
    trace:
        Trace being rendered.
    entries_to_plot:
        Original entries for the current render mode.
    target_module:
        Module whose internal forward operations should be shown.
    vis_mode:
        ``"unrolled"`` or ``"rolled"``.

    Returns
    -------
    dict[str, GraphNode]
        Focused entries with boundary nodes inserted.

    Raises
    ------
    ValueError
        If the module contains no rendered layers.
    """

    focus_labels = {
        node.layer_label
        for node in entries_to_plot.values()
        if _node_is_inside_module(node, target_module.address)
    }
    if not focus_labels:
        raise ValueError(
            f"Module '{target_module.address}' has no layers to render. "
            "Empty modules cannot be focused."
        )

    focused_entries: dict[str, GraphNode] = {
        label: _copy_focus_node(node)
        for label, node in entries_to_plot.items()
        if node.layer_label in focus_labels
    }
    entries_by_layer_label = {node.layer_label: node for node in entries_to_plot.values()}
    input_boundaries: dict[str, BoundaryNode] = {}
    output_boundaries: dict[str, BoundaryNode] = {}

    for render_node in list(focused_entries.values()):
        node = cast(FocusNode, render_node)
        new_parents: list[str] = []
        for parent_label in node.parents:
            if parent_label in focus_labels:
                new_parents.append(parent_label)
                continue
            parent_node = entries_to_plot.get(parent_label) or entries_by_layer_label.get(
                parent_label
            )
            if parent_node is None:
                continue
            boundary = _get_or_create_boundary_node(
                input_boundaries,
                parent_node,
                target_module,
                vis_mode=vis_mode,
                boundary_kind="input",
                child_label=node.layer_label,
            )
            if node.layer_label not in boundary.children:
                boundary.children.append(node.layer_label)
            new_parents.append(boundary.layer_label)
        node.parents = new_parents

        new_children: list[str] = []
        for child_label in node.children:
            if child_label in focus_labels:
                new_children.append(child_label)
                continue
            child_node = entries_to_plot.get(child_label) or entries_by_layer_label.get(child_label)
            if child_node is None:
                continue
            boundary = _get_or_create_boundary_node(
                output_boundaries,
                child_node,
                target_module,
                vis_mode=vis_mode,
                boundary_kind="output",
                parent_label=node.layer_label,
            )
            if node.layer_label not in boundary.parents:
                boundary.parents.append(node.layer_label)
            new_children.append(boundary.layer_label)
        node.children = new_children

    _simplify_boundary_labels(input_boundaries, "input")
    _simplify_boundary_labels(output_boundaries, "output")
    for boundary_dict in (input_boundaries, output_boundaries):
        for label, boundary in boundary_dict.items():
            focused_entries[label] = boundary

    return focused_entries


def _node_is_inside_module(node: GraphNode, address: str) -> bool:
    """Return whether ``node`` ran inside ``address``."""

    return any(module.split(":", 1)[0] == address for module in node.modules)


def _copy_focus_node(node: GraphNode) -> GraphNode:
    """Return a shallow render copy whose edge lists can be rewritten."""

    if isinstance(node, BoundaryNode):
        return copy.copy(node)
    if isinstance(node, FocusNode):
        original = node.original
    else:
        original = node
    return FocusNode(
        original=original,
        parents=list(node.parents),
        children=list(node.children),
        modules=list(node.modules),
    )


def _get_or_create_boundary_node(
    boundary_nodes: dict[str, BoundaryNode],
    external_node: GraphNode,
    target_module: "Module",
    *,
    vis_mode: str,
    boundary_kind: str,
    child_label: str | None = None,
    parent_label: str | None = None,
) -> BoundaryNode:
    """Create or return a boundary node for one external layer."""

    external_label = external_node.layer_label.replace(":", "pass")
    boundary_label = f"__module_focus_{boundary_kind}_{external_label}"
    boundary = boundary_nodes.get(boundary_label)
    if boundary is not None:
        return boundary

    module_path = _boundary_module_path(target_module, vis_mode)
    boundary = BoundaryNode(
        layer_label=boundary_label,
        display_label=f"ext: {external_node.layer_label}",
        boundary_kind=boundary_kind,
        children=[] if child_label is None else [child_label],
        parents=[] if parent_label is None else [parent_label],
        modules=module_path,
    )
    boundary_nodes[boundary_label] = boundary
    return boundary


def _boundary_module_path(target_module: "Module", vis_mode: str) -> list[str]:
    """Return a module path for focus boundary node cluster placement."""

    module_path = []
    parts = target_module.address.split(".") if target_module.address != "self" else ["self"]
    for idx in range(len(parts)):
        address = ".".join(parts[: idx + 1])
        module_path.append(address if vis_mode == "rolled" else f"{address}:1")
    return module_path


def _simplify_boundary_labels(
    boundary_nodes: dict[str, BoundaryNode],
    fallback_label: str,
) -> None:
    """Use simple labels when a focus side has exactly one boundary."""

    if len(boundary_nodes) != 1:
        return
    only_boundary = next(iter(boundary_nodes.values()))
    only_boundary.display_label = fallback_label


def _arg_path_key(arg_path: Iterable[Any]) -> tuple[Any, ...]:
    """Return a hashable primitive key for an edge-use argument path.

    Parameters
    ----------
    arg_path:
        Edge-use path components.

    Returns
    -------
    tuple[Any, ...]
        Primitive key preserving the recorded path order.
    """

    key: list[Any] = []
    for component in arg_path:
        if isinstance(component, TupleIndex):
            key.append(("tuple", component.index))
        elif isinstance(component, DictKey):
            key.append(("dict", component.key))
        elif isinstance(component, NamedField):
            key.append(("field", component.name))
        elif isinstance(component, HFKey):
            key.append(("hf", component.key))
        elif isinstance(component, DataclassField):
            key.append(("dataclass", component.name))
        elif isinstance(component, OutputPathComponent):
            key.append((component.__class__.__name__, repr(component)))
        else:
            key.append(component)
    return tuple(key)


def _arg_path_label_value(arg_path: Iterable[Any]) -> str:
    """Return the display value for an edge-use argument path.

    Parameters
    ----------
    arg_path:
        Edge-use path components.

    Returns
    -------
    str
        Compact label suffix such as ``"0"`` or ``"(0, 1)"``.
    """

    values: list[Any] = []
    for component in arg_path:
        if isinstance(component, TupleIndex):
            values.append(component.index)
        elif isinstance(component, DictKey):
            values.append(component.key)
        elif isinstance(component, NamedField):
            values.append(component.name)
        elif isinstance(component, HFKey):
            values.append(component.key)
        elif isinstance(component, DataclassField):
            values.append(component.name)
        else:
            values.append(component)
    if len(values) == 1:
        return str(values[0])
    return str(tuple(values))


def _edge_use_argument_label(edge_use: Any) -> str:
    """Return the argument label for one recorded edge use.

    Parameters
    ----------
    edge_use:
        Edge-use record carrying ``arg_kind`` and ``arg_path`` attributes.

    Returns
    -------
    str
        Label text matching existing argument-edge wording.
    """

    prefix = "arg" if edge_use.arg_kind == "positional" else "kwarg"
    return f"{prefix} {_arg_path_label_value(edge_use.arg_path)}"


def _render_edge_occurrences(
    parent_node: GraphNode,
    child_node: GraphNode,
    vis_mode: str,
) -> tuple[tuple[tuple[Any, ...], str | None], ...]:
    """Return rendered occurrences for a parent-child edge.

    Parameters
    ----------
    parent_node:
        Rendered source node.
    child_node:
        Rendered target node.
    vis_mode:
        ``"unrolled"`` emits one edge per repeated argument occurrence.
        ``"rolled"`` keeps a single aggregate edge between rolled endpoints.

    Returns
    -------
    tuple[tuple[tuple[Any, ...], str | None], ...]
        ``(occurrence_key, argument_label)`` pairs. Repeated same-parent argument
        uses yield multiple pairs; ordinary edges yield one pair.
    """

    parent_layer_label = parent_node.layer_label
    child_layer_label = child_node.layer_label
    parent_render_label = _render_node_label(parent_node, vis_mode)
    child_render_label = _render_node_label(child_node, vis_mode)
    if vis_mode == "rolled":
        return ((("edge", parent_render_label, child_render_label), None),)

    parent_render_label = _render_node_label(parent_node, "unrolled")
    child_render_label = _render_node_label(child_node, "unrolled")
    if isinstance(child_node, Op):
        matches = [
            edge_use
            for edge_use in child_node.edge_uses
            if (
                edge_use.parent_label == parent_layer_label
                and edge_use.child_label == child_layer_label
            )
        ]
        if matches:
            return tuple(
                (
                    (
                        "edge_use",
                        edge_use.parent_label,
                        edge_use.child_label,
                        edge_use.arg_kind,
                        _arg_path_key(edge_use.arg_path),
                        edge_use.child_func_call_id,
                        index,
                    ),
                    _edge_use_argument_label(edge_use),
                )
                for index, edge_use in enumerate(matches)
            )

    occurrences: list[tuple[tuple[Any, ...], str | None]] = []
    positions = getattr(child_node, "parent_arg_positions", None)
    if isinstance(positions, Mapping):
        for arg_type in ("args", "kwargs"):
            arg_positions = positions.get(arg_type, {})
            if not isinstance(arg_positions, Mapping):
                continue
            for arg_loc, arg_label in arg_positions.items():
                if arg_label != parent_layer_label:
                    continue
                prefix = "arg" if arg_type == "args" else "kwarg"
                occurrence_key = (
                    "parent_arg_position",
                    child_layer_label,
                    arg_type,
                    repr(arg_loc),
                )
                occurrences.append((occurrence_key, f"{prefix} {arg_loc}"))
    if occurrences:
        return tuple(occurrences)
    return ((("edge", parent_render_label, child_render_label), None),)


def _expand_edges_through_skipped(
    trace: "Trace",
    parent_node: GraphNode,
    visible_entries: dict[str, GraphNode],
    skipped_labels: set[str],
    vis_mode: str,
) -> list[RenderEdge]:
    """Expand one node's outgoing edges through skipped successor chains.

    Parameters
    ----------
    trace:
        Owning Trace.
    parent_node:
        Source node whose outgoing edges should be expanded.
    visible_entries:
        Visible nodes before applying ``skip_fn``.
    skipped_labels:
        Labels elided by ``skip_fn``.
    vis_mode:
        ``"unrolled"`` or ``"rolled"``.

    Returns
    -------
    list[RenderEdge]
        Deduplicated non-skipped targets.
    """

    visible_entries_by_layer = {node.layer_label: node for node in visible_entries.values()}
    by_target: dict[tuple[str, tuple[Any, ...]], RenderEdge] = {}
    for child_label in parent_node.children:
        child_node = visible_entries.get(child_label) or visible_entries_by_layer.get(child_label)
        if child_node is None and vis_mode == "unrolled":
            child_node = trace.layer_dict_all_keys.get(child_label)
        if child_node is None:
            continue
        parent_label = _render_node_label(parent_node, vis_mode)
        child_render_label = _render_node_label(child_node, vis_mode)
        if child_render_label == parent_label and child_render_label not in skipped_labels:
            reached = [child_node]
        else:
            reached = _walk_skipped_successors(
                trace,
                child_node,
                visible_entries,
                skipped_labels,
                vis_mode,
                seen={parent_label},
            )
        for target_node in reached:
            first_child = child_node
            target_label = _render_node_label(target_node, vis_mode)
            if target_node is first_child:
                occurrences = _render_edge_occurrences(parent_node, first_child, vis_mode)
            else:
                occurrences = ((("skipped", parent_label, target_label), None),)
            for occurrence_key, argument_label in occurrences:
                map_key = (target_label, occurrence_key)
                existing = by_target.get(map_key)
                if existing is None:
                    by_target[map_key] = RenderEdge(
                        target_node,
                        first_child,
                        occurrence_key,
                        argument_label,
                    )
                elif existing.metadata_child is not first_child:
                    by_target[map_key] = RenderEdge(target_node, None, occurrence_key, None)
    return list(by_target.values())


def _walk_skipped_successors(
    trace: "Trace",
    node: GraphNode,
    visible_entries: dict[str, GraphNode],
    skipped_labels: set[str],
    vis_mode: str,
    seen: set[str],
) -> list[GraphNode]:
    """Return non-skipped descendants reached through skipped chains.

    Parameters
    ----------
    trace:
        Owning Trace.
    node:
        Current node in the traversal.
    visible_entries:
        Visible nodes before applying ``skip_fn``.
    skipped_labels:
        Labels elided by ``skip_fn``.
    vis_mode:
        ``"unrolled"`` or ``"rolled"``.
    seen:
        Labels already visited on this traversal branch.

    Returns
    -------
    list[GraphNode]
        Non-skipped reachable nodes.
    """

    node_label = _render_node_label(node, vis_mode)
    if node_label in seen:
        return []
    seen.add(node_label)
    if node_label not in skipped_labels:
        return [node]
    reached: list[GraphNode] = []
    visible_entries_by_layer = {entry.layer_label: entry for entry in visible_entries.values()}
    for child_label in node.children:
        child_node = visible_entries.get(child_label) or visible_entries_by_layer.get(child_label)
        if child_node is None and vis_mode == "unrolled":
            child_node = trace.layer_dict_all_keys.get(child_label)
        if child_node is None:
            continue
        reached.extend(
            _walk_skipped_successors(
                trace,
                child_node,
                visible_entries,
                skipped_labels,
                vis_mode,
                seen=set(seen),
            )
        )
    return reached


def _build_sibling_order_chains(
    captured_edges: list[CapturedForwardEdge],
) -> tuple[SiblingOrderChain, ...]:
    """Build candidate sibling chains from captured rendered edges."""

    if not _has_rendered_fanout(captured_edges):
        return ()

    rendered_parents: dict[str, set[str]] = defaultdict(set)
    by_source: dict[tuple[str, str], list[CapturedForwardEdge]] = defaultdict(list)
    for edge in captured_edges:
        rendered_parents[edge.head_name].add(edge.tail_name)
        by_source[(edge.source_label, edge.tail_name)].append(edge)

    chains: list[SiblingOrderChain] = []
    for (source_label, source_name), source_edges in sorted(by_source.items()):
        distinct_targets: dict[str, CapturedForwardEdge] = {}
        # Conditional fanouts are ordered too: the by-execution-step ordering below places
        # the (earlier) branch test on the left and the (later) taken arm on the right, i.e.
        # the "if left / then right" reading. This used to be skipped, leaving conditional
        # branch layout arbitrary.
        for edge in source_edges:
            distinct_targets.setdefault(edge.head_name, edge)
        if len(distinct_targets) < 2:
            continue
        kept_edges = [
            edge
            for target_name, edge in distinct_targets.items()
            if rendered_parents[target_name] == {source_name}
        ]
        if len(kept_edges) < 2:
            continue
        kept_edges.sort(key=lambda edge: (edge.target_step, edge.head_name))
        chains.append(
            SiblingOrderChain(
                source_label=source_label,
                source_name=source_name,
                targets=tuple(edge.head_name for edge in kept_edges),
                target_labels=tuple(edge.target_label for edge in kept_edges),
                lca_key=_sibling_chain_lca_key(kept_edges),
            )
        )
    return tuple(chains)


def _has_rendered_fanout(captured_edges: list[CapturedForwardEdge]) -> bool:
    """Return whether any rendered source has at least two distinct children."""

    children_by_source: dict[tuple[str, str], set[str]] = defaultdict(set)
    for edge in captured_edges:
        key = (edge.source_label, edge.tail_name)
        children_by_source[key].add(edge.head_name)
        if len(children_by_source[key]) >= 2:
            return True
    return False


def _sibling_chain_lca_key(edges: list[CapturedForwardEdge]) -> str | int:
    """Return the rendered module key shared by all sibling edges."""

    if not edges:
        return -1
    first_key = edges[0].module_key
    if all(edge.module_key == first_key for edge in edges):
        return first_key
    return -1


def _filter_sibling_chains_to_rendered_nodes(
    chains: tuple[SiblingOrderChain, ...],
    rendered_nodes: Mapping[str, tuple[float, float]],
) -> tuple[SiblingOrderChain, ...]:
    """Keep sibling chains whose source and targets survived DOT rendering.

    Predicate-based module collapse can deduplicate or absorb candidate
    endpoints after edge capture. Sibling ordering is a layout refinement, so
    chains whose rendered nodes are gone are ignored rather than failing the
    render.
    """

    return tuple(
        chain
        for chain in chains
        if chain.source_name in rendered_nodes
        and all(target in rendered_nodes for target in chain.targets)
    )


def _flow_span(tail_xy: tuple[float, float], head_xy: tuple[float, float], rankdir: str) -> float:
    """Return flow-axis span between two node coordinates."""

    if rankdir in {"LR", "RL"}:
        return abs(tail_xy[0] - head_xy[0])
    return abs(tail_xy[1] - head_xy[1])


def _assert_sibling_backstops(
    baseline: PlainLayout,
    injected: PlainLayout,
    chains: tuple[SiblingOrderChain, ...],
    captured_edges: list[CapturedForwardEdge],
) -> None:
    """Assert sibling-ordering structural backstops."""

    real_edges = {(edge.tail_name, edge.head_name) for edge in captured_edges}
    assert len(baseline.nodes) == len(injected.nodes)
    for chain in chains:
        for target in chain.targets:
            assert target in baseline.nodes
        for left, right in zip(chain.targets, chain.targets[1:]):
            assert (left, right) not in real_edges
            assert (right, left) not in real_edges


def _inject_sibling_rank_groups(source: str, chains: tuple[SiblingOrderChain, ...]) -> str:
    """Inject surviving sibling rank groups into baseline DOT source."""

    result = source
    top_level_groups = [chain for chain in chains if chain.lca_key == -1]
    if top_level_groups:
        result = _insert_before_final_brace(result, _rank_group_lines(top_level_groups, ""))

    by_cluster: dict[str, list[SiblingOrderChain]] = defaultdict(list)
    for chain in chains:
        if chain.lca_key != -1:
            by_cluster[cast(str, chain.lca_key)].append(chain)
    fallback_chains: list[SiblingOrderChain] = []
    for cluster_key, cluster_chains in by_cluster.items():
        cluster_name = f"cluster_{cluster_key.replace(':', '_pass')}"
        result, did_emit = _insert_into_cluster(
            result,
            cluster_name,
            _rank_group_lines(cluster_chains, "    "),
        )
        if not did_emit:
            # Cluster-name string surgery missed (e.g. an unexpected rendered key).
            # Fall back to top-level emission, which is verify-safe (a distorting
            # chain is dropped by the per-chain stretch check), rather than crash
            # the render with a bare assert.
            fallback_chains.extend(cluster_chains)
    if fallback_chains:
        result = _insert_before_final_brace(result, _rank_group_lines(fallback_chains, ""))
    return result


def _rank_group_lines(chains: Sequence[SiblingOrderChain], indent: str) -> str:
    """Return DOT lines for sibling rank groups."""

    lines: list[str] = []
    for chain in chains:
        lines.append(f"{indent}// tl:sibling-order:start")
        lines.append(f"{indent}{{")
        lines.append(f"{indent}    rank=same")
        for target in chain.targets:
            lines.append(f"{indent}    {quote_dot_id(target)}")
        for left, right in zip(chain.targets, chain.targets[1:]):
            lines.append(
                f"{indent}    {quote_dot_id(left)} -> {quote_dot_id(right)} "
                '[style=invis weight=100 comment="tl:sibling-order"]'
            )
        lines.append(f"{indent}}}")
        lines.append(f"{indent}// tl:sibling-order:end")
    return "\n".join(lines) + "\n"


def _emit_sibling_rank_group(graph: graphviz.Digraph, chain: SiblingOrderChain) -> None:
    """Emit one sibling rank group into ``graph``."""

    # Bracket the markers AROUND the whole subgraph (in the parent body) so that
    # ``_strip_sibling_rank_groups`` removes the entire ``{ rank=same ... }`` block,
    # not just its interior (which would leave an orphan empty wrapper).
    graph.body.append("\t// tl:sibling-order:start\n")
    with graph.subgraph() as rank_group:
        rank_group.attr(rank="same")
        for target in chain.targets:
            rank_group.node(target)
        for left, right in zip(chain.targets, chain.targets[1:]):
            rank_group.edge(
                left,
                right,
                style="invis",
                weight="100",
                comment="tl:sibling-order",
            )
    graph.body.append("\t// tl:sibling-order:end\n")


def _insert_before_final_brace(source: str, insertion: str) -> str:
    """Insert text before the final top-level DOT brace."""

    index = source.rfind("}")
    if index == -1:
        return source + insertion
    return source[:index] + insertion + source[index:]


def _insert_into_cluster(source: str, cluster_name: str, insertion: str) -> tuple[str, bool]:
    """Insert text immediately after a cluster's opening brace."""

    markers = (f"subgraph {cluster_name} {{", f"subgraph {quote_dot_id(cluster_name)} {{")
    index = -1
    for marker in markers:
        index = source.find(marker)
        if index != -1:
            break
    if index == -1:
        return source, False
    insert_at = source.find("\n", index)
    if insert_at == -1:
        return source, False
    return source[: insert_at + 1] + insertion + source[insert_at + 1 :], True


def _emit_container_cluster(
    parent_graph: graphviz.Digraph,
    container_cluster: ContainerClusterSpec,
) -> None:
    """Emit one opt-in container cluster inside an existing module cluster."""

    with parent_graph.subgraph(name=f"cluster_{container_cluster.cluster_id}") as cluster:
        cluster.attr(
            **make_module_cluster_attrs(
                title=container_cluster.title,
                module_type=container_cluster.kind,
                line_style="dotted",
                penwidth=1.0,
                fillcolor="white",
            )
        )
        for node_name in container_cluster.node_names:
            cluster.node(node_name)


def _get_max_call_depth(
    top_modules: list[str],
    module_edge_dict: Dict[str, Any],
    module_submodule_dict: Dict[str, list[str]],
) -> int:
    """Recursively computes the maximum module nesting depth in the model hierarchy.

    Used to determine subgraph layout depth for graphviz rendering. Works by
    crawling down the stack of modules till it hits one with no children and at least one edge.

    Args:
        top_modules: modules at highest level of nesting
        module_edge_dict: Edges in each module.
        module_submodule_dict: Mapping from each module to any children.

    Returns:
        Max nesting depth.
    """
    max_call_depth = 1
    module_depth_stack = [(graph, 1) for graph in top_modules]

    while len(module_depth_stack) > 0:
        module, module_depth = module_depth_stack.pop()
        module_edges = module_edge_dict[module]["edges"]
        module_submodules = module_submodule_dict[module]

        if (len(module_edges) == 0) and (
            len(module_submodules) == 0
        ):  # can ignore if no edges and no children.
            continue
        elif (len(module_edges) > 0) and (len(module_submodules) == 0):
            max_call_depth = max([module_depth, max_call_depth])
        elif (len(module_edges) == 0) and (len(module_submodules) > 0):
            module_depth_stack.extend(
                [(module_child, module_depth + 1) for module_child in module_submodules]
            )
        else:
            max_call_depth = max([module_depth, max_call_depth])
            module_depth_stack.extend(
                [(module_child, module_depth + 1) for module_child in module_submodules]
            )
    return max_call_depth


__all__ = [
    "_append_boundary_container_edges",
    "_append_member_of_container_edges",
    "_arg_path_key",
    "_arg_path_label_value",
    "_assert_sibling_backstops",
    "_base_rendered_node_emission",
    "_boundary_module_path",
    "_build_module_focus_entries",
    "_build_sibling_order_chains",
    "_build_skip_filtered_edge_map",
    "_code_panel_composition_available",
    "_collapsed_container_leaf_nodes",
    "_collapsed_endpoint_for_emission",
    "_container_clusters_for_graphviz",
    "_container_leaf_groups",
    "_container_leaf_shapes_identical",
    "_container_member_edge_attrs",
    "_container_nodes_and_overlay_edges",
    "_container_record_label",
    "_container_record_node_args",
    "_container_record_node_name",
    "_container_record_owner_key",
    "_container_role",
    "_container_spec_label",
    "_copy_focus_node",
    "_decode_graphviz_stderr",
    "_edge_use_argument_label",
    "_emit_container_cluster",
    "_emit_sibling_rank_group",
    "_entries_to_plot_for_context",
    "_enumerate_base_rendered_node_emissions",
    "_enumerate_run_fold_ellipsis_emissions",
    "_expand_edges_through_skipped",
    "_filter_sibling_chains_to_rendered_nodes",
    "_flow_span",
    "_format_backward_filter_caption",
    "_get_max_call_depth",
    "_get_or_create_boundary_node",
    "_has_rendered_fanout",
    "_inject_sibling_rank_groups",
    "_insert_before_final_brace",
    "_insert_into_cluster",
    "_is_non_file_svg_href",
    "_is_hidden_buffer_update_node",
    "_node_is_inside_module",
    "_owner_module_key_for_node",
    "_rank_group_lines",
    "_render_edge_occurrences",
    "_replace_svg_attr_value",
    "_resolve_svg_image_path",
    "_sibling_chain_lca_key",
    "_simplify_boundary_labels",
    "_single_container_owner",
    "_svg_attrs_to_dict",
    "_svg_image_mime_type",
    "_svg_image_placeholder",
    "_svg_numeric_attr",
    "_walk_skipped_successors",
]
