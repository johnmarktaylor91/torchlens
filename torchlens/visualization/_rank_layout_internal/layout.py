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

from .._render_utils import _open_file_quietly, html_escape
from .._render_utils import compute_module_penwidth
from ..code_panel import _code_panel_label
from ..render_ir import RenderIR, RenderIRDotStatement

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


def _dot_escape(value: str) -> str:
    """Backslash-escape a raw string for embedding inside a quoted DOT string.

    Mirrors ``graphviz.quoting.quote()``'s escaping semantics: escape a
    literal backslash ``\\`` FIRST, then a literal double-quote ``"``.  Order
    matters -- escaping the backslash first means the backslash newly
    introduced to escape a ``"`` is never itself re-escaped by a later pass.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _dot_quote(value: str) -> str:
    """Quote a DOT attribute value, preserving HTML labels."""
    if value.startswith("<") and value.endswith(">"):
        return value
    return f'"{_dot_escape(value)}"'


def _dot_id(name: str) -> str:
    """Format a node name for DOT, quoting if needed."""
    _KW = {"graph", "digraph", "subgraph", "node", "edge", "strict"}
    if re.match(r"^[a-zA-Z_]\w*$", name) and name.lower() not in _KW:
        return name
    return f'"{_dot_escape(name)}"'


def _rank_node_statement(
    statements: tuple[RenderIRDotStatement, ...],
) -> RenderIRDotStatement | None:
    """Return the resolved node statement used by the rank renderer."""

    return next((statement for statement in statements if statement.kind == "node"), None)


def render_rank_layout(
    ir: RenderIR,
    vis_mode: str,
    vis_outpath: str,
    vis_fileformat: str,
    vis_save_only: bool,
    graph_caption: str,
    rankdir: str,
    code_panel_source: str | None = None,
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
        ir: Decision-complete render IR.
        vis_mode: ``'unrolled'`` or ``'rolled'``.
        vis_outpath: Output file path (without extension).
        vis_fileformat: Output format (pdf, png, svg, etc.).
        vis_save_only: If True, don't open viewer.
        graph_caption: HTML label for the graph title.
        rankdir: Graphviz rank direction (BT, TB, LR).
        code_panel_source: Optional source code to embed as a graph cluster.

    Returns:
        The generated DOT source string.

    Raises:
        RuntimeError: If neato rendering fails.
    """
    # ── Phase 1: Collect node styling, module assignments, and edges ──
    node_data: dict[str, dict[str, Any]] = {}
    rank_names: dict[str, str] = {}
    module_direct_nodes: dict[str, list[str]] = defaultdict(list)
    module_child_map: dict[str, set[str]] = defaultdict(set)
    module_has_ancestor: dict[str, bool] = defaultdict(bool)
    root_node_names: list[str] = []
    all_edges: list[dict[str, Any]] = []
    for node in ir.nodes:
        statement = _rank_node_statement(node.node_calls)
        if statement is None and node.owned_node_args:
            _, raw_attrs = node.owned_node_args[0]
            attrs = dict(raw_attrs)
            name = str(attrs.pop("name", node.name))
        elif statement is not None:
            attrs = dict(statement.attrs)
            name = str(attrs.pop("name", statement.args[0] if statement.args else node.name))
        else:
            continue
        rank_name = (
            (node.source_label or name).replace(":", "pass") if node.kind != "module_box" else name
        )
        rank_names[node.name] = rank_name
        node_data[rank_name] = {"attrs": attrs, "node_label": node.source_label or rank_name}
        if "solid" in str(attrs.get("style", "")):
            for region_key in node.region_path:
                module_has_ancestor[region_key] = True

    region_keys = {region.key for region in ir.regions if region.kind == "module"}
    region_by_key = {region.key: region for region in ir.regions if region.kind == "module"}
    owned_names: set[str] = set()
    for region in ir.regions:
        if region.kind != "module":
            continue
        if region.parent_key in region_keys:
            module_child_map[region.parent_key].add(region.key)
        for name in region.node_names:
            rank_name = rank_names.get(name, name)
            if rank_name in node_data:
                module_direct_nodes[region.key].append(rank_name)
                owned_names.add(rank_name)
    root_node_names.extend(name for name in node_data if name not in owned_names)

    for edge in ir.edges:
        attrs = dict(edge.attrs)
        attrs.pop("tail_name", None)
        attrs.pop("head_name", None)
        attrs.pop("fontcolor", None)
        attrs.pop("labelfontsize", None)
        all_edges.append(
            {
                "tail_name": rank_names.get(edge.source_unit, edge.tail_name or edge.source_unit),
                "head_name": rank_names.get(edge.target_unit, edge.head_name or edge.target_unit),
                **attrs,
            }
        )

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

    # Find top-level modules (not children of any other)
    all_children = set()
    for children in module_child_map.values():
        all_children.update(children)
    top_modules = sorted(all_mod_keys - all_children)

    def _max_depth(mod_key: str, depth: int = 0) -> int:
        """Return maximum descendant depth for rank cluster pen widths."""

        children = module_child_map.get(mod_key, set())
        return max((_max_depth(child, depth + 1) for child in children), default=depth)

    max_nest = max((_max_depth(module) for module in top_modules), default=0) + 1

    def _write_cluster(mod_key: str, depth: int, indent: int) -> None:
        """Recursively write a cluster subgraph with its nodes and children."""
        prefix = "  " * indent
        safe = mod_key.replace(":", "_pass").replace(".", "_")
        # ``safe`` only substitutes ``:``/``.`` -- it still carries through
        # arbitrary module-address text (e.g. an ``nn.ModuleDict`` key like
        # ``"a<b>&c"``). Route the subgraph identifier through ``_dot_id()``,
        # the same quoting helper every other raw-DOT identifier in this file
        # uses (node names at ``_node_line``, edge tail/head names below), so
        # characters illegal in an unquoted Graphviz ID don't get spliced
        # into raw DOT text. Quoting is safe for subgraph names too: neato
        # still recognizes the "cluster" prefix and applies cluster styling
        # whether or not the name is quoted.
        lines.append(f"{prefix}subgraph {_dot_id(f'cluster_{safe}')} {{")

        mod_addr = mod_key.split(":")[0] if ":" in mod_key else mod_key
        mod_attrs = dict(region_by_key[mod_key].style)
        mod_attrs["label"] = str(mod_attrs["label"]).replace("align='left'", 'align="left"')
        mod_attrs["style"] = "filled,solid" if module_has_ancestor.get(mod_key) else "filled,dashed"
        mod_attrs["penwidth"] = f"{compute_module_penwidth(depth, max_nest):.1f}"
        mod_attrs.pop("margin", None)

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

    if code_panel_source is not None:
        panel_x = 0.0
        panel_y = max_y + 180.0
        lines.append("  subgraph cluster_torchlens_code_panel {")
        lines.append('    label=""')
        lines.append('    style="filled,rounded"')
        lines.append('    fillcolor="#FAFAFA"')
        lines.append('    color="#A8A8A8"')
        lines.append('    margin="12"')
        lines.append(
            "    __tl_code_panel_node "
            f"[label={_code_panel_label(code_panel_source)} shape=plaintext "
            f'fontname="Courier" margin="0" pos="{panel_x:.1f},{panel_y:.1f}!"]'
        )
        lines.append("  }")

    # Module cluster hierarchy
    for mod in top_modules:
        _write_cluster(mod, 0, 1)

    # Edges (at top level — neato -n routes them fine).
    # Capture the count BEFORE the loop mutates each edge dict (it pops keys).
    num_edges = len(all_edges)
    for edge_data in all_edges:
        tail = _dot_id(edge_data.pop("tail_name"))
        head = _dot_id(edge_data.pop("head_name"))
        parts = [f"{k}={_dot_quote(str(v))}" for k, v in edge_data.items()]
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
    render_succeeded = False
    try:
        _run_neato_with_fallbacks(
            rendered_path=rendered_path,
            source_path=source_path,
            vis_fileformat=vis_fileformat,
            spline_mode=spline_mode,
            render_timeout=render_timeout,
        )
        render_succeeded = True
        if not vis_save_only:
            _open_file_quietly(rendered_path)
    finally:
        if render_succeeded and os.path.exists(source_path):
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
        label_str = "<br/>".join(html_escape(str(label)) for label in arg_labels)
        edge_dict["label"] = f"<<FONT POINT-SIZE='10'><b>{label_str}</b></FONT>>"
