"""Pure-Python rank layout backend for large computational graphs.

Graphviz ``dot`` remains the default renderer for local-topology graphs.  For
graphs with long-range edges, this module provides a cheap cost estimator and a
direct DOT/SVG writer backed by Kahn topological ranks.
"""

from __future__ import annotations

import os
import re
import subprocess
import warnings
from collections import defaultdict, deque
from typing import Any

from .._label_format import format_memory, format_shape
from .._render_utils import _open_file_quietly

SPAN_LOCAL = 12
# Calibrated 2026-06-11: local 5k-node chains cost about 5k and dot rendered
# in ~14s; 3.5k-node hub graphs with 24 long edges cost about 42k and dot
# exceeded 30s. 20k keeps local topology on dot and sends hub topology to rank.
RANK_LAYOUT_COST_THRESHOLD = 20_000
RANK_LAYOUT_NOTICE = (
    "TorchLens auto-selected rank layout (estimated layout cost={cost}, "
    "threshold={threshold}). Reduce graph complexity with vis_call_depth, "
    "rolled mode, or module= focus; force Graphviz dot with "
    "vis_node_placement='dot' if you are willing to wait minutes."
)
_NEATO_TIMEOUT = 120
_DEFAULT_NODE_WIDTH = 200  # points — fallback when label isn't available
_DEFAULT_NODE_HEIGHT = 60  # points — fallback when label isn't available

# Spline edge-routing thresholds for the ``neato -n`` render path.
#
# neato spline routing (``-Gsplines=true``) is super-linear in the number of
# *edge crossings* the routed layout contains.  Crossings grow with BOTH the raw
# node count (more nodes packed in 2D -> more geometric crossings even at a low
# logical edge/node ratio) AND with fan-out density (DenseNet concat fan-out,
# attention-heavy dense conv blocks).  A graph only reaches this
# neato path at all once its estimated layout cost already exceeds
# ``RANK_LAYOUT_COST_THRESHOLD`` (20k), i.e. it is large or hub-heavy by
# construction -- precisely the regime where splines choke.
#
# Empirically calibrated 2026-06-20 against three real failures on this path:
#   convnextv2_huge       785 nodes / 928 edges  (ratio 1.18) -> spline TIMEOUT
#   smp_Unet_densenet201  772 nodes / 2578 edges (ratio 3.34) -> spline TIMEOUT
#   high-res attention model 2359 nodes / 2598 edges(ratio 1.10) -> rtree overflow
# All three MUST degrade to straight "line" edges.  convnext shows the node
# count alone is the dominant signal: at ~1.18 edges/node it still timed out, so
# the node ceiling must sit below 785.
#
# We therefore degrade to straight ``line`` edges when EITHER the graph has many
# nodes OR it is edge-dense.  Sparse small graphs never even reach this path
# (they render through Graphviz ``dot``), so this only affects already-large
# rank-layout graphs and never the common small-model case.
_SPLINE_NODE_LIMIT = 700
# Density gate: catch genuinely dense mid-size graphs (300-700 nodes) whose
# fan-out blows up crossings before they hit the node ceiling.
_SPLINE_DENSITY_RATIO = 1.5
# Below this node count even a dense layout routes cheaply, so keep splines.
_SPLINE_DENSITY_MIN_NODES = 300


def _choose_spline_mode(num_nodes: int, num_edges: int) -> str:
    """Pick the neato spline mode for a graph of the given size/density.

    Returns ``"true"`` (curved spline routing — pretty, but super-linear in
    edge crossings) for small/sparse graphs, and ``"line"`` (straight segments —
    O(edges), robust) for large or edge-dense graphs.

    A graph degrades to ``"line"`` when it has at least
    :data:`_SPLINE_NODE_LIMIT` nodes (node count dominates crossing count), OR
    when it is "edge-dense" — its edges-per-node ratio exceeds
    :data:`_SPLINE_DENSITY_RATIO` and it has at least
    :data:`_SPLINE_DENSITY_MIN_NODES` nodes.  Tiny graphs always keep splines.
    """
    if num_nodes >= _SPLINE_NODE_LIMIT:
        return "line"
    if num_nodes >= _SPLINE_DENSITY_MIN_NODES and num_edges > num_nodes * _SPLINE_DENSITY_RATIO:
        return "line"
    return "true"


def _compute_topological_layout(
    node_data: dict[str, dict[str, Any]],
    all_edges: list[dict[str, Any]],
    node_label_sizes: dict[str, tuple[float, float]],
    module_direct_nodes: dict[str, list[str]],
    module_child_map: dict[str, set[str]],
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float, float, float]], float]:
    """Compute node positions via topological rank layout.

    Parameters
    ----------
    node_data:
        Mapping from DOT node name to graph attributes and source layer label.
    all_edges:
        List of edge dictionaries with ``tail_name`` and ``head_name``.
    node_label_sizes:
        Mapping from source layer label to estimated node size in points.
    module_direct_nodes:
        Mapping from module key to directly contained DOT node names.
    module_child_map:
        Mapping from module key to child module keys.

    Returns
    -------
    tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float, float, float]], float]
        Positions keyed by source layer label, compound module boxes, and maximum y coordinate.
    """
    all_node_labels = set(nd["node_label"] for nd in node_data.values())

    # Build adjacency from DOT-level edges.
    children_of: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = defaultdict(int)
    for e in all_edges:
        src = e.get("tail_name") or e["tail_name"]
        tgt = e.get("head_name") or e["head_name"]
        if not isinstance(src, str) or not isinstance(tgt, str):
            continue
        src_eid = node_data.get(src, {}).get("node_label")
        tgt_eid = node_data.get(tgt, {}).get("node_label")
        if (
            isinstance(src_eid, str)
            and isinstance(tgt_eid, str)
            and src_eid in all_node_labels
            and tgt_eid in all_node_labels
        ):
            children_of[src_eid].append(tgt_eid)
            in_degree[tgt_eid] += 1

    # Kahn's algorithm for topological depth assignment.
    depth: dict[str, int] = {}
    queue: deque[str] = deque()
    for nid in all_node_labels:
        if in_degree[nid] == 0:
            depth[nid] = 0
            queue.append(nid)

    while queue:
        nid = queue.popleft()
        for child in children_of[nid]:
            new_depth = depth[nid] + 1
            if child not in depth or new_depth > depth[child]:
                depth[child] = new_depth
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    # Unreached nodes (cycles or disconnected) get depth 0.
    for nid in all_node_labels:
        if nid not in depth:
            depth[nid] = 0

    # Group by depth rank.
    ranks: dict[int, list[str]] = defaultdict(list)
    for nid, d in depth.items():
        ranks[d].append(nid)

    # Sort nodes within each rank by module membership for visual grouping.
    node_label_module: dict[str, str] = {}
    for mod_key, dot_names in module_direct_nodes.items():
        for dn in dot_names:
            node_info = node_data.get(dn)
            if node_info:
                node_label_module[node_info["node_label"]] = mod_key

    for d in ranks:
        ranks[d].sort(key=lambda nid: node_label_module.get(nid, ""))

    # Compute positions.  Y = depth rank, X = position within rank.
    spacing_y = 120  # points between ranks
    spacing_x = 30  # points between node edges within a rank
    positions: dict[str, tuple[float, float]] = {}

    for d, nodes in sorted(ranks.items()):
        x_cursor = 0.0
        for nid in nodes:
            w, h = node_label_sizes.get(nid, (_DEFAULT_NODE_WIDTH, _DEFAULT_NODE_HEIGHT))
            cx = x_cursor + w / 2
            cy = d * spacing_y + h / 2
            positions[nid] = (cx, cy)
            x_cursor += w + spacing_x

    max_y = max((y for _, y in positions.values()), default=0) + _DEFAULT_NODE_HEIGHT

    # Compute module bounding boxes from node positions.
    # Collect all source labels in each module, including nested children.
    def _collect_module_node_labels(mod_key: str) -> set[str]:
        ids: set[str] = set()
        for dn in module_direct_nodes.get(mod_key, []):
            nd = node_data.get(dn)
            if nd and nd["node_label"] in positions:
                ids.add(nd["node_label"])
        for child_mod in module_child_map.get(mod_key, set()):
            ids.update(_collect_module_node_labels(child_mod))
        return ids

    compound_bboxes = {}
    padding = 60  # points around contained nodes

    all_mod_keys = set(module_direct_nodes.keys()) | set(module_child_map.keys())
    for mod_key in all_mod_keys:
        module_node_labels = _collect_module_node_labels(mod_key)
        if not module_node_labels:
            continue
        xs = []
        ys = []
        for eid in module_node_labels:
            cx, cy = positions[eid]
            w, h = node_label_sizes.get(eid, (_DEFAULT_NODE_WIDTH, _DEFAULT_NODE_HEIGHT))
            xs.extend([cx - w / 2, cx + w / 2])
            ys.extend([cy - h / 2, cy + h / 2])
        min_x, max_x_val = min(xs) - padding, max(xs) + padding
        min_y, max_y_val = min(ys) - padding, max(ys) + padding
        mod_addr = mod_key.split(":")[0] if ":" in mod_key else mod_key
        group_id = f"group_{mod_addr}"
        compound_bboxes[group_id] = (
            min_x,
            min_y,
            max_x_val - min_x,
            max_y_val - min_y,
        )

    return positions, compound_bboxes, max_y


def compute_rank_depths(node_labels: set[str], edges: list[tuple[str, str]]) -> dict[str, int]:
    """Compute topological depths for a render graph.

    Parameters
    ----------
    node_labels:
        Labels present in the render graph.
    edges:
        Directed edges between render labels.

    Returns
    -------
    dict[str, int]
        Maximum upstream depth for each render label; cyclic or disconnected
        leftovers receive depth 0.
    """
    children_of: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = defaultdict(int)
    for source, target in edges:
        if source in node_labels and target in node_labels:
            children_of[source].append(target)
            in_degree[target] += 1

    depth: dict[str, int] = {}
    queue: deque[str] = deque()
    for node_label in node_labels:
        if in_degree[node_label] == 0:
            depth[node_label] = 0
            queue.append(node_label)

    while queue:
        node_label = queue.popleft()
        for child in children_of[node_label]:
            depth[child] = max(depth.get(child, 0), depth[node_label] + 1)
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    for node_label in node_labels:
        depth.setdefault(node_label, 0)
    return depth


def estimate_rank_layout_cost(
    node_labels: set[str],
    edges: list[tuple[str, str]],
    span_local: int = SPAN_LOCAL,
) -> int:
    """Estimate Graphviz dot layout cost from rank-spanning edges.

    Parameters
    ----------
    node_labels:
        Labels present in the render graph.
    edges:
        Directed edges between render labels.
    span_local:
        Maximum rank span considered local and inexpensive.

    Returns
    -------
    int
        ``num_nodes + sum(rank_span for rank_span > span_local)``.
    """
    depths = compute_rank_depths(node_labels, edges)
    cost = len(node_labels)
    for source, target in edges:
        if source not in depths or target not in depths:
            continue
        rank_span = abs(depths[target] - depths[source])
        if rank_span > span_local:
            cost += rank_span
    return cost


def get_node_placement_engine(vis_node_placement: str, layout_cost: int) -> str:
    """Resolve the requested node placement engine.

    Parameters
    ----------
    vis_node_placement:
        User preference: ``"auto"``, ``"dot"``, or ``"rank"``.
    layout_cost:
        Precomputed render-graph cost estimate.

    Returns
    -------
    str
        ``"dot"`` or ``"rank"``.
    """
    if vis_node_placement in {"dot", "rank"}:
        return vis_node_placement
    if vis_node_placement != "auto":
        raise ValueError("vis_node_placement must be one of 'auto', 'dot', or 'rank'.")
    if layout_cost > RANK_LAYOUT_COST_THRESHOLD:
        return "rank"
    return "dot"


def _estimate_node_size(label: str) -> tuple[float, float]:
    """Estimate graphviz node dimensions in points from an HTML label.

    Splits on ``<br/>`` to count lines, strips HTML tags to measure character
    width.  Returns (width, height) in points.
    """
    text = label.strip()
    if text.startswith("<") and text.endswith(">"):
        text = text[1:-1]

    # Split on <br/>, <br>, <BR/> etc.
    lines = re.split(r"<br\s*/?>", text, flags=re.IGNORECASE)
    lines = [re.sub(r"<[^>]+>", "", line).strip() for line in lines]
    lines = [ln for ln in lines if ln]

    if not lines:
        return _DEFAULT_NODE_WIDTH, _DEFAULT_NODE_HEIGHT

    max_chars = max(len(ln) for ln in lines)
    n_lines = len(lines)

    # Generous estimate — neato renders text wider than raw char count suggests
    # due to font metrics, bold text, and internal node padding.
    # ~8.5 points per char + 60pt padding, ~22pt per line + 30pt padding
    width = max(max_chars * 8.5 + 60, 150)
    height = max(n_lines * 22 + 30, 60)

    return width, height


def _dot_quote(value: str) -> str:
    """Quote a DOT attribute value, preserving HTML labels."""
    if value.startswith("<") and value.endswith(">"):
        return value
    return f'"{value}"'


def _dot_id(name: str) -> str:
    """Format a node name for DOT, quoting if needed."""
    _KW = {"graph", "digraph", "subgraph", "node", "edge", "strict"}
    if re.match(r"^[a-zA-Z_]\w*$", name) and name not in _KW:
        return name
    return f'"{name}"'


def render_rank_layout(
    trace: Any,
    entries_to_plot: dict[str, Any],
    vis_mode: str,
    vis_call_depth: int,
    show_buffer_layers: bool,
    overrides: Any,
    node_mode: Any,
    node_spec_fn: Any,
    collapsed_node_spec_fn: Any,
    collapse_fn: Any,
    skip_fn: Any,
    edge_map: dict[str, Any] | None,
    skipped_labels: set[str],
    vis_outpath: str,
    vis_fileformat: str,
    vis_save_only: bool,
    graph_caption: str,
    rankdir: str,
) -> str:
    """Render a graph with the pure-Python rank layout.

    Byops ``graphviz.Digraph`` construction entirely.  Generates DOT text
    directly with:

    - Node styling matching the dot path (same labels, colors, shapes)
    - ``subgraph cluster_*`` blocks for module hierarchy (the boxes)
    - Rank-layout node positions (``pos="x,y!"``)
    - Edge styling (color, solid/dashed, arg labels)

    Renders with ``neato -n`` (pre-positioned layout that respects clusters).

    Args:
        trace: The Trace instance.
        entries_to_plot: Dict of node_barcode -> Op/Layer.
        vis_mode: ``'unrolled'`` or ``'rolled'``.
        vis_call_depth: Module nesting depth for collapsed modules.
        show_buffer_layers: Whether to include buffer layers.
        overrides: VisualizationOverrides instance.
        node_mode: Node-mode preset name.
        vis_outpath: Output file path (without extension).
        vis_fileformat: Output format (pdf, png, svg, etc.).
        vis_save_only: If True, don't open viewer.
        graph_caption: HTML label for the graph title.
        rankdir: Graphviz rank direction (BT, TB, LR).

    Returns:
        The generated DOT source string.

    Raises:
        RuntimeError: If neato rendering fails.
    """
    from collections import defaultdict

    # Late imports to avoid circular dependency
    from ..rendering import (
        _collapse_address_for_node,
        _get_node_address_shape_color,
        _get_node_bg_color,
        _apply_node_spec_fn,
        _node_spec_to_graphviz_args,
        compute_default_node_lines,
        TRAINABLE_PARAMS_BG_COLOR,
        FROZEN_PARAMS_BG_COLOR,
        DEFAULT_BG_COLOR,
        COMMUTE_FUNCS,
        _render_node_label,
    )
    from .._render_utils import compute_module_penwidth
    from ..modes import COLLAPSED_MODE_REGISTRY
    from ..node_spec import NodeSpec

    # ── Phase 1: Collect node styling, module assignments, and edges ──

    # graph_node_label -> {"attrs": {dot attrs}, "node_label": layer_label}
    node_data: dict[str, dict[str, Any]] = {}
    # module_key -> [node_names directly in this module (not nested deeper)]
    module_direct_nodes: dict[str, list[str]] = defaultdict(list)
    # module_key -> set of child module_keys
    module_child_map: dict[str, set[str]] = defaultdict(set)
    # module_key -> True if any contained node has input_ancestor
    module_has_ancestor: dict[str, bool] = defaultdict(bool)
    # Nodes not inside any module
    root_node_names: list[str] = []
    # Edge data: list of dicts with tail_name, head_name, color, style, ...
    all_edges: list[dict[str, Any]] = []
    collapsed_set: set[str] = set()
    edges_used: set[tuple[str, str, tuple[Any, ...]]] = set()

    def _module_keys_for_node(node: Any, is_collapsed_mod: bool) -> list[str]:
        """Get module hierarchy keys for a node."""
        if is_collapsed_mod:
            mods = list(node.modules[: vis_call_depth - 1])
        else:
            mods = list(node.modules)
        if vis_mode == "rolled":
            return list(dict.fromkeys(m.split(":")[0] for m in mods))
        return mods

    def _assign_to_hierarchy(
        graph_node_label: str, mod_keys: list[str], has_ancestor: bool
    ) -> None:
        """Place a node into the module tree."""
        if mod_keys:
            module_direct_nodes[mod_keys[-1]].append(graph_node_label)
            for i in range(len(mod_keys) - 1):
                module_child_map[mod_keys[i]].add(mod_keys[i + 1])
            # Propagate has_input_ancestor up
            if has_ancestor:
                for mk in mod_keys:
                    module_has_ancestor[mk] = True
        else:
            root_node_names.append(graph_node_label)

    for _barcode, node in entries_to_plot.items():
        if node.layer_label in skipped_labels:
            continue
        if node.is_buffer and not show_buffer_layers:
            continue

        collapse_address = _collapse_address_for_node(
            trace,
            node,
            collapse_fn=collapse_fn,
            max_module_depth=vis_call_depth,
        )
        is_collapsed = collapse_address is not None

        if is_collapsed:
            mod_w_pass = collapse_address
            if mod_w_pass is None:
                continue
            mod_parts = mod_w_pass.rsplit(":", 1)
            mod_addr, call_index = mod_parts
            graph_node_label = "pass".join(mod_parts) if vis_mode == "unrolled" else mod_addr
            node_label = node.layer_label

            if graph_node_label not in collapsed_set:
                collapsed_set.add(graph_node_label)
                ml = trace.modules[mod_addr]
                mod_out = trace[mod_w_pass]

                if vis_mode == "unrolled":
                    mpl = trace.modules[mod_w_pass]
                    n_tensors = mpl.num_layers
                    has_anc = any(trace[la].has_input_ancestor for la in mpl.ops)
                else:
                    n_tensors = ml.num_layers
                    has_anc = any(trace[la].has_input_ancestor for la in ml.layer_labels)

                np_ = ml.num_calls
                if np_ == 1:
                    title = f"<b>@{mod_addr}</b>"
                elif vis_mode == "unrolled":
                    title = f"<b>@{mod_addr}:{call_index}</b>"
                else:
                    title = f"<b>@{mod_addr} (x{np_})</b>"

                out_shape: tuple[Any, ...] = mod_out.shape or ()
                ss = format_shape(out_shape)

                npar = ml.num_params
                npt = ml.num_params_trainable
                npf = ml.num_params_frozen
                if npar > 0:
                    bg = (
                        TRAINABLE_PARAMS_BG_COLOR
                        if npf == 0
                        else FROZEN_PARAMS_BG_COLOR
                        if npt == 0
                        else f"{TRAINABLE_PARAMS_BG_COLOR}:{FROZEN_PARAMS_BG_COLOR}"
                    )
                else:
                    bg = DEFAULT_BG_COLOR

                if npar == 0:
                    pd = "0 parameters"
                elif npf == 0:
                    pd = f"{npar} params (all trainable)"
                elif npt == 0:
                    pd = f"{npar} params (all frozen)"
                else:
                    pd = f"{npar} params ({npt} trainable, {npf} frozen)"

                ls = "solid" if has_anc else "dashed"
                default_spec = NodeSpec(
                    lines=[
                        title.replace("<b>", "").replace("</b>", ""),
                        ml.class_name,
                        f"{ss}, {format_memory(mod_out.activation_memory)}",
                        f"{n_tensors} layers total",
                        pd,
                    ],
                    shape="box3d",
                    fillcolor=bg,
                    fontcolor="black",
                    color="black",
                    style=f"filled,{ls}",
                    extra_attrs={"ordering": "out"},
                )
                mode_fn = COLLAPSED_MODE_REGISTRY[node_mode]
                mode_result = mode_fn(ml, default_spec)
                mode_spec = default_spec if mode_result is None else mode_result
                if collapsed_node_spec_fn is not None:
                    result = collapsed_node_spec_fn(ml, mode_spec)
                    spec = mode_spec if result is None else result
                else:
                    spec = mode_spec
                attrs = _node_spec_to_graphviz_args(spec)
                if spec.fillcolor is not None and ":" in spec.fillcolor:
                    attrs["gradangle"] = "0"

                node_data[graph_node_label] = {"attrs": attrs, "node_label": node_label}
                mod_keys = _module_keys_for_node(node, True)
                _assign_to_hierarchy(graph_node_label, mod_keys, has_anc)

            node_color = "black"
        else:
            # Regular layer node
            graph_node_label = node.layer_label.replace(":", "pass")
            node_label = node.layer_label

            addr, shape, node_color = _get_node_address_shape_color(trace, node, show_buffer_layers)
            bg = _get_node_bg_color(trace, node)
            ls = "solid" if node.has_input_ancestor else "dashed"
            default_spec = NodeSpec(
                lines=compute_default_node_lines(node, addr, vis_mode),
                shape=shape,
                fillcolor=bg,
                fontcolor=node_color,
                color=node_color,
                style=f"filled,{ls}",
                extra_attrs={"ordering": "out"},
            )
            spec = _apply_node_spec_fn(trace, node, default_spec, node_mode, node_spec_fn)
            attrs = _node_spec_to_graphviz_args(spec)
            if spec.fillcolor is not None and ":" in spec.fillcolor:
                attrs["gradangle"] = "0"

            node_data[graph_node_label] = {"attrs": attrs, "node_label": node_label}
            mod_keys = _module_keys_for_node(node, False)
            _assign_to_hierarchy(graph_node_label, mod_keys, node.has_input_ancestor)

        # ── Collect edges (this node -> skip-filtered children) ──
        for render_edge in (edge_map or {}).get(_render_node_label(node, vis_mode), []):
            child_node = render_edge.target
            metadata_child = render_edge.metadata_child
            if child_node.is_buffer and not show_buffer_layers:
                continue

            # Resolve tail name
            if is_collapsed:
                tail_name = graph_node_label
            else:
                tail_name = node.layer_label.replace(":", "pass")

            # Resolve head name
            child_collapse_address = _collapse_address_for_node(
                trace,
                child_node,
                collapse_fn=collapse_fn,
                max_module_depth=vis_call_depth,
            )
            child_is_collapsed = child_collapse_address is not None
            if child_is_collapsed:
                c_mod_w_pass = child_collapse_address
                if c_mod_w_pass is None:
                    continue
                c_parts = c_mod_w_pass.rsplit(":", 1)
                head_name = "pass".join(c_parts) if vis_mode == "unrolled" else c_parts[0]
            else:
                head_name = child_node.layer_label.replace(":", "pass")

            # Intra-module skip for two collapsed nodes in the same module
            if is_collapsed and child_is_collapsed and tail_name != head_name:
                p_mods = node.modules[:]
                c_mods = child_node.modules[:]
                if node.is_atomic_module:
                    p_mods = p_mods[:-1]
                if child_node.is_atomic_module:
                    c_mods = c_mods[:-1]
                if p_mods[:vis_call_depth] == c_mods[:vis_call_depth]:
                    continue

            dedupe_key = (tail_name, head_name, render_edge.occurrence_key)
            if dedupe_key in edges_used:
                continue
            if tail_name == head_name and metadata_child is not child_node:
                continue
            edges_used.add(dedupe_key)

            edge_style = "solid" if node.has_input_ancestor else "dashed"
            edge = {
                "tail_name": tail_name,
                "head_name": head_name,
                "color": node_color,
                "style": edge_style,
                "arrowsize": ".7",
            }

            # Arg labels for non-commutative ops with multiple parents
            if (
                not child_is_collapsed
                and metadata_child is not None
                and metadata_child.layer_type not in COMMUTE_FUNCS
            ):
                _add_arg_label(
                    node,
                    metadata_child,
                    edge,
                    trace,
                    show_buffer_layers,
                    render_edge.argument_label,
                )

            for k, v in (overrides.edge or {}).items():
                if callable(v):
                    edge[k] = str(v(trace, node, metadata_child or child_node))
                else:
                    edge[k] = str(v)

            all_edges.append(edge)

    # ── Phase 2: Rank layout ──
    node_label_sizes: dict[str, tuple[float, float]] = {}
    for dot_name, nd in node_data.items():
        node_label = nd["node_label"]
        label = nd["attrs"].get("label", "")
        node_label_sizes[node_label] = _estimate_node_size(str(label))

    num_rank_nodes = len(node_data)
    positions, compound_bboxes, max_y = _compute_topological_layout(
        node_data, all_edges, node_label_sizes, module_direct_nodes, module_child_map
    )

    # ── Phase 3: Generate DOT with clusters and positions ──

    lines = []
    lines.append("digraph {")
    lines.append(
        f"  graph [rankdir={rankdir} label={graph_caption} labelloc=t labeljust=left ordering=out]"
    )
    lines.append("  node [ordering=out]")

    def _node_line(name: str, indent: int = 1) -> str:
        """Generate a DOT node declaration with position and size."""
        nd = node_data[name]
        parts = []
        for k, v in nd["attrs"].items():
            parts.append(f"{k}={_dot_quote(str(v))}")
        node_label = nd["node_label"]
        if node_label in positions:
            x, y = positions[node_label]
            # neato -n expects pos in points (not inches).
            parts.append(f'pos="{x:.1f},{(max_y - y):.1f}!"')
        prefix = "  " * indent
        return f"{prefix}{_dot_id(name)} [{' '.join(parts)}]"

    # Compute max module depth for penwidth scaling
    all_mod_keys = set(module_direct_nodes.keys()) | set(module_child_map.keys())

    def _max_depth(mod_key: str, depth: int = 0, visited: set[str] | None = None) -> int:
        if visited is None:
            visited = set()
        if mod_key in visited:
            return depth
        visited.add(mod_key)
        children = module_child_map.get(mod_key, set())
        if not children:
            return depth
        return max(_max_depth(c, depth + 1, visited) for c in children)

    # Find top-level modules (not children of any other)
    all_children = set()
    for children in module_child_map.values():
        all_children.update(children)
    top_modules = sorted(all_mod_keys - all_children)

    max_nest = max((_max_depth(m) for m in top_modules), default=0) + 1

    def _write_cluster(mod_key: str, depth: int, indent: int) -> None:
        """Recursively write a cluster subgraph with its nodes and children."""
        prefix = "  " * indent
        safe = mod_key.replace(":", "_pass").replace(".", "_")
        lines.append(f"{prefix}subgraph cluster_{safe} {{")

        mod_addr = mod_key.split(":")[0] if ":" in mod_key else mod_key
        try:
            ml = trace.modules[mod_addr]
        except (KeyError, IndexError):
            ml = None
        mod_type = ml.class_name if ml else "Module"
        np_ = ml.num_calls if ml else 1

        if vis_mode == "unrolled" and np_ > 1 and ":" in mod_key:
            title = mod_key
        elif vis_mode == "rolled" and np_ > 1:
            title = f"{mod_addr} (x{np_})"
        else:
            title = mod_addr

        pw = compute_module_penwidth(depth, max_nest)
        ls = "solid" if module_has_ancestor.get(mod_key) else "dashed"

        cluster_label = f'<<B>@{title}</B><br align="left"/>({mod_type})<br align="left"/>>'

        # Apply module overrides
        mod_attrs = {
            "label": cluster_label,
            "labelloc": "b",
            "style": f"filled,{ls}",
            "fillcolor": "white",
            "penwidth": f"{pw:.1f}",
        }
        for k, v in (overrides.module or {}).items():
            mod_attrs[k] = str(v(trace, mod_key)) if callable(v) else str(v)

        group_id = f"group_{mod_addr}"
        if group_id in compound_bboxes:
            ex, ey, ew, eh = compound_bboxes[group_id]
            # Convert rank-layout coords (y-down) to graphviz bb (y-up).
            bb_llx = ex
            bb_lly = max_y - ey - eh
            bb_urx = ex + ew
            bb_ury = max_y - ey
            mod_attrs["bb"] = f"{bb_llx:.1f},{bb_lly:.1f},{bb_urx:.1f},{bb_ury:.1f}"

        for k, v in mod_attrs.items():
            lines.append(f"{prefix}  {k}={_dot_quote(str(v))}")

        # Nodes directly in this module
        for nn in module_direct_nodes.get(mod_key, []):
            if nn in node_data:
                lines.append(_node_line(nn, indent + 1))

        # Child module clusters
        for child in sorted(module_child_map.get(mod_key, [])):
            _write_cluster(child, depth + 1, indent + 1)

        lines.append(f"{prefix}}}")

    # Root-level nodes (not in any module)
    for nn in root_node_names:
        if nn in node_data:
            lines.append(_node_line(nn))

    # Module cluster hierarchy
    for mod in top_modules:
        _write_cluster(mod, 0, 1)

    # Edges (at top level — neato -n routes them fine).
    # Capture the count BEFORE the loop mutates each edge dict (it pops keys).
    num_edges = len(all_edges)
    for edge in all_edges:
        tail = _dot_id(edge.pop("tail_name"))
        head = _dot_id(edge.pop("head_name"))
        parts = [f"{k}={_dot_quote(str(v))}" for k, v in edge.items()]
        lines.append(f"  {tail} -> {head} [{' '.join(parts)}]")

    lines.append("}")
    dot_source = "\n".join(lines)

    # ── Phase 4: Render with neato -n ──
    if num_rank_nodes > 25000 and vis_fileformat != "svg":
        warnings.warn(
            f"Graph has {num_rank_nodes} nodes. PDF/PNG rendering may produce "
            f"empty output at this scale. Consider using vis_fileformat='svg' "
            f"for large graphs; SVG files are zoomable in any browser."
        )

    source_path = f"{vis_outpath}.dot"
    with open(source_path, "w") as f:
        f.write(dot_source)

    rendered_path = f"{vis_outpath}.{vis_fileformat}"
    num_nodes = len(node_data)
    # Spline routing cost is driven by edge crossings, not node count, so the
    # heuristic keys off BOTH size and edge density (see _choose_spline_mode).
    spline_mode = _choose_spline_mode(num_nodes, num_edges)
    render_timeout = max(_NEATO_TIMEOUT, int(num_nodes * 0.01))
    try:
        _run_neato_with_fallbacks(
            rendered_path=rendered_path,
            source_path=source_path,
            vis_fileformat=vis_fileformat,
            spline_mode=spline_mode,
            render_timeout=render_timeout,
        )
        if not vis_save_only:
            _open_file_quietly(rendered_path)
    finally:
        if os.path.exists(source_path):
            os.remove(source_path)

    return dot_source


# neato builds an rtree spatial index over node/label boxes during edge
# routing; its box coordinates are 16-bit-ish and it aborts with "area too
# large for rtree" when a box (or the whole canvas) overflows that range.
# Very-high-resolution models (512px input -> 2747 nodes spread across a huge
# pinned canvas) trip this.  ``-Gsize``/``-Gratio`` only
# rescale the OUTPUT viewport, not the coordinates fed to the rtree, so they do
# NOT fix it.  We instead shrink the pinned coordinates themselves so the whole
# drawing fits inside this ceiling before re-running neato.
_RTREE_COORD_CEILING = 28000.0
_RTREE_POS_RE = re.compile(r'pos="(-?[\d.]+),(-?[\d.]+)!"')
_RTREE_BB_RE = re.compile(r'bb="(-?[\d.]+),(-?[\d.]+),(-?[\d.]+),(-?[\d.]+)"')
_RTREE_DIM_RE = re.compile(r"(width|height)=([\d.]+)")


def _rescale_dot_for_rtree(dot_source: str, ceiling: float = _RTREE_COORD_CEILING) -> str | None:
    """Shrink pinned coordinates so the layout fits in neato's rtree range.

    Scans ``pos="x,y!"`` pins for the maximum coordinate magnitude.  If it
    already fits under ``ceiling`` returns ``None`` (no rescale needed).
    Otherwise scales every pinned ``pos``, cluster ``bb`` box, and node
    ``width``/``height`` (inches) by ``ceiling / max_coord`` so the geometry is
    preserved (uniform scale) but the canvas fits.  Returns the rewritten DOT.
    """
    max_coord = 0.0
    for m in _RTREE_POS_RE.finditer(dot_source):
        max_coord = max(max_coord, abs(float(m.group(1))), abs(float(m.group(2))))
    if max_coord <= ceiling or max_coord == 0.0:
        return None
    scale = ceiling / max_coord

    def _scale_pos(m: re.Match) -> str:
        return f'pos="{float(m.group(1)) * scale:.1f},{float(m.group(2)) * scale:.1f}!"'

    def _scale_bb(m: re.Match) -> str:
        vals = [float(m.group(i)) * scale for i in range(1, 5)]
        return 'bb="' + ",".join(f"{v:.1f}" for v in vals) + '"'

    def _scale_dim(m: re.Match) -> str:
        return f"{m.group(1)}={float(m.group(2)) * scale:.4f}"

    out = _RTREE_POS_RE.sub(_scale_pos, dot_source)
    out = _RTREE_BB_RE.sub(_scale_bb, out)
    out = _RTREE_DIM_RE.sub(_scale_dim, out)
    return out


def _run_neato(
    *,
    rendered_path: str,
    source_path: str,
    vis_fileformat: str,
    spline_mode: str,
    render_timeout: int,
    extra_gattrs: tuple[str, ...] = (),
) -> subprocess.CompletedProcess:
    """Invoke ``neato -n`` once with the given spline mode and graph attrs."""
    cmd = [
        "neato",
        "-n",
        f"-Gsplines={spline_mode}",
        *extra_gattrs,
        f"-T{vis_fileformat}",
        "-o",
        rendered_path,
        source_path,
    ]
    return subprocess.run(cmd, timeout=render_timeout, capture_output=True, text=True)


def _run_neato_with_fallbacks(
    *,
    rendered_path: str,
    source_path: str,
    vis_fileformat: str,
    spline_mode: str,
    render_timeout: int,
) -> None:
    """Render with ``neato -n``, degrading gracefully on two known failures.

    1. **rtree overflow** — very high-resolution models produce a layout whose
       canvas overflows neato's spatial-index coordinate limit and it exits
       non-zero with ``"area too large for rtree"``.  We retry once with
       straight-line edges and the pinned coordinates uniformly down-scaled
       (see ``_rescale_dot_for_rtree``) so the canvas fits in range.
    2. **spline timeout** — dense graphs that slipped past the density gate can
       still blow the timeout while routing splines.  We retry once with
       straight-line edges (O(edges), no crossing search) before giving up.

    Raises ``RuntimeError`` if the (post-fallback) render still fails, and
    re-raises ``subprocess.TimeoutExpired`` if the straight-line retry also
    times out.
    """
    try:
        result = _run_neato(
            rendered_path=rendered_path,
            source_path=source_path,
            vis_fileformat=vis_fileformat,
            spline_mode=spline_mode,
            render_timeout=render_timeout,
        )
    except subprocess.TimeoutExpired:
        # Spline routing blew the budget — straight lines are crossing-free and
        # far cheaper. Retry once with "line"; let a second timeout propagate.
        if spline_mode == "line":
            raise
        warnings.warn(
            "neato spline routing timed out; retrying with straight-line edges "
            "(-Gsplines=line). The graph is rendered with straight edges."
        )
        result = _run_neato(
            rendered_path=rendered_path,
            source_path=source_path,
            vis_fileformat=vis_fileformat,
            spline_mode="line",
            render_timeout=render_timeout,
        )

    if result.returncode != 0 and "rtree" in (result.stderr or "").lower():
        # The pinned canvas overflowed neato's rtree coordinate range. Shrink the
        # coordinates themselves (uniform scale preserves the layout) so they fit,
        # and retry once with straight-line edges. -Gsize/-Gratio do NOT help here
        # because they only rescale the output viewport, not the rtree input.
        rescaled = _rescale_dot_for_rtree(open(source_path, encoding="utf-8").read())
        if rescaled is not None:
            warnings.warn(
                "neato layout exceeded the rtree coordinate limit; retrying with "
                "straight-line edges and down-scaled pinned coordinates so the "
                "canvas fits. Geometry is preserved (uniform scale)."
            )
            with open(source_path, "w", encoding="utf-8") as f:
                f.write(rescaled)
            result = _run_neato(
                rendered_path=rendered_path,
                source_path=source_path,
                vis_fileformat=vis_fileformat,
                spline_mode="line",
                render_timeout=render_timeout,
            )

    if result.returncode != 0:
        raise RuntimeError(f"neato rendering failed (exit {result.returncode}):\n{result.stderr}")


def _add_arg_label(
    parent_node: Any,
    child_node: Any,
    edge_dict: dict[str, Any],
    trace: Any,
    show_buffer_layers: bool,
    occurrence_argument_label: str | None = None,
) -> None:
    """Add argument position labels to an edge when the child has multiple parents.

    Simplified version of ``rendering._label_node_arguments_if_needed`` for the
    direct rank-layout path.
    """
    from ...data_classes.layer import Layer
    from ...data_classes.op import Op

    # Count visible parents
    num_parents = len(child_node.parents)
    if not show_buffer_layers:
        for pl in child_node.parents:
            if isinstance(child_node, Op):
                if trace[pl].is_buffer:
                    num_parents -= 1
            elif isinstance(child_node, Layer):
                if trace.layer_logs[pl].is_buffer:
                    num_parents -= 1
    if num_parents <= 1:
        return

    if occurrence_argument_label is not None:
        arg_labels = [occurrence_argument_label]
    else:
        arg_labels = []
        for arg_type in ["args", "kwargs"]:
            for arg_loc, arg_label in child_node.parent_arg_positions[arg_type].items():
                if parent_node.layer_label == arg_label:
                    arg_labels.append(f"{arg_type[:-1]} {arg_loc}")

    if arg_labels:
        label_str = "<br/>".join(arg_labels)
        edge_dict["label"] = f"<<FONT POINT-SIZE='10'><b>{label_str}</b></FONT>>"
