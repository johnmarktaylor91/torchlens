"""ELK-based layout engine for large computational graphs.

When the graph has more nodes than ``_ELK_NODE_THRESHOLD``, the standard
Graphviz ``dot`` engine can hang or crash.  This module provides:

1. A resolution function (``get_node_placement_engine``) that picks the best
   available layout engine based on graph size and user preference.
2. An ELK layout pipeline: build an ELK JSON graph, run ``elkjs`` via Node.js,
   and inject the resulting positions back into a DOT source string so that
   ``neato -n`` can render it without re-computing layout.
3. Graceful fallback to Graphviz ``sfdp`` when Node.js/elkjs is unavailable.
"""

import functools
import gc
import json
import os
import re
import resource
import subprocess
import tempfile
import warnings
from typing import Optional

# Set the soft stack limit to match the hard limit once at import time.
# Child processes (Node.js) inherit this, removing the need for preexec_fn
# which forces fork+exec and COW-doubles virtual memory of large parent processes.
try:
    _soft, _hard = resource.getrlimit(resource.RLIMIT_STACK)
    if _hard == resource.RLIM_INFINITY:
        resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, _hard))
    elif _soft < _hard:
        resource.setrlimit(resource.RLIMIT_STACK, (_hard, _hard))
except (ValueError, resource.error):
    pass


def _available_memory_mb() -> int:
    """Return available system memory in MB, or 0 if unknown."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024  # KB -> MB
    except (OSError, ValueError, IndexError):
        pass
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return (pages * page_size) // (1024 * 1024)
    except (ValueError, OSError):
        pass
    return 0


_ELK_NODE_THRESHOLD = 3500
_ELK_TIMEOUT = 120  # seconds for Node.js subprocess
_SFDP_TIMEOUT = 120  # seconds for sfdp/neato subprocess
_DEFAULT_NODE_WIDTH = 200  # points — fallback when label isn't available
_DEFAULT_NODE_HEIGHT = 60  # points — fallback when label isn't available

# Inline Node.js script that reads ELK JSON from a temp file (path in
# _TL_JSON_PATH env var) or stdin, runs layout, writes to stdout.
_ELK_LAYOUT_SCRIPT = r"""
const { Worker } = require('worker_threads');
const fs = require('fs');

// Run ELK layout in a worker thread with a large stack via resourceLimits.
// resourceLimits.stackSizeMb is far more reliable than the --stack-size V8
// flag for preventing "Maximum call stack size exceeded" in deeply recursive
// ELK layout on large graphs (100k+ nodes).
const stackMb = parseInt(process.env._TL_STACK_MB || '64', 10);
const heapMb = parseInt(process.env._TL_HEAP_MB || '16384', 10);

const workerCode = `
const { parentPort, workerData } = require('worker_threads');
const ELK = require('elkjs');
const elk = new ELK();
const graph = JSON.parse(workerData);
elk.layout(graph).then((result) => {
    parentPort.postMessage(JSON.stringify(result));
}).catch((err) => { throw err; });
`;

function runLayout(input) {
    const worker = new Worker(workerCode, {
        eval: true,
        workerData: input,
        resourceLimits: {
            stackSizeMb: stackMb,
            maxOldGenerationSizeMb: heapMb,
            maxYoungGenerationSizeMb: Math.min(2048, Math.floor(heapMb / 8)),
        },
    });
    worker.on('message', (result) => {
        process.stdout.write(result);
    });
    worker.on('error', (err) => {
        process.stderr.write(err.toString());
        process.exit(1);
    });
}

const jsonPath = process.env._TL_JSON_PATH;
if (jsonPath) {
    runLayout(fs.readFileSync(jsonPath, 'utf8'));
} else {
    let input = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', (chunk) => { input += chunk; });
    process.stdin.on('end', () => { runLayout(input); });
}
"""


def _find_node_binary() -> "Optional[str]":
    """Locate the ``node`` binary, probing nvm paths if not on PATH."""
    import os
    import shutil

    node_bin = shutil.which("node")
    if node_bin is not None:
        return node_bin

    # nvm installs node under ~/.nvm/versions/node/<version>/bin/node but
    # non-interactive shells often don't have nvm's PATH additions.
    nvm_dir = os.environ.get("NVM_DIR", os.path.expanduser("~/.nvm"))
    versions_dir = os.path.join(nvm_dir, "versions", "node")
    if os.path.isdir(versions_dir):
        # Pick the newest installed version.
        versions = sorted(os.listdir(versions_dir), reverse=True)
        for v in versions:
            candidate = os.path.join(versions_dir, v, "bin", "node")
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
    return None


def _node_env() -> dict:
    """Build environment dict for Node.js subprocesses.

    Sets NODE_PATH to include the global node_modules directory so that
    globally-installed packages (like elkjs via ``npm install -g``) are
    found even when using nvm or non-standard Node.js installs.
    """
    import os

    env = os.environ.copy()
    node_bin = _find_node_binary()
    if node_bin is None:
        return env

    # Ensure the node binary's directory is on PATH for subprocess calls.
    node_dir = os.path.dirname(node_bin)
    path = env.get("PATH", "")
    if node_dir not in path.split(os.pathsep):
        env["PATH"] = f"{node_dir}{os.pathsep}{path}" if path else node_dir

    # Derive global node_modules: <prefix>/lib/node_modules
    prefix_dir = os.path.dirname(os.path.dirname(os.path.realpath(node_bin)))
    global_modules = os.path.join(prefix_dir, "lib", "node_modules")
    existing = env.get("NODE_PATH", "")
    if global_modules not in existing:
        env["NODE_PATH"] = f"{global_modules}:{existing}" if existing else global_modules
    return env


@functools.lru_cache(maxsize=1)
def elk_available() -> bool:
    """Check whether Node.js and elkjs are installed."""
    try:
        result = subprocess.run(
            ["node", "-e", "require('elkjs')"],
            capture_output=True,
            timeout=10,
            env=_node_env(),
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_node_placement_engine(vis_node_placement: str, num_nodes: int) -> str:
    """Resolve ``vis_node_placement`` to a concrete engine name.

    Args:
        vis_node_placement: User preference: ``"auto"``, ``"dot"``, ``"elk"``, or ``"sfdp"``.
        num_nodes: Number of nodes in the graph.

    Returns:
        One of ``"dot"``, ``"elk"``, ``"sfdp"``.
    """
    if vis_node_placement == "dot":
        return "dot"
    if vis_node_placement == "sfdp":
        return "sfdp"
    if vis_node_placement == "elk":
        if elk_available():
            return "elk"
        warnings.warn(
            "elkjs not available — falling back to sfdp. "
            "Install with: npm install -g elkjs\n"
            "Requires Node.js (https://nodejs.org/)."
        )
        return "sfdp"
    # auto mode
    if num_nodes < _ELK_NODE_THRESHOLD:
        return "dot"
    if elk_available():
        return "elk"
    warnings.warn(
        f"Graph has {num_nodes} nodes (threshold: {_ELK_NODE_THRESHOLD}). "
        f"Graphviz dot may hang. Using sfdp instead.\n"
        f"For better large-graph layout, install elkjs: npm install -g elkjs\n"
        f"Requires Node.js (https://nodejs.org/)."
    )
    return "sfdp"


def build_elk_graph_hierarchical(entries_to_plot, show_buffer_layers: bool = False) -> dict:
    """Build an ELK JSON graph with module hierarchy from model entries.

    Creates nested ELK compound nodes that mirror the module containment
    structure, so ELK's layered algorithm can produce a layout with proper
    module grouping (analogous to Graphviz subgraph clusters).

    Args:
        entries_to_plot: Dict of node_barcode -> LayerPassLog/LayerLog
            (same as used by render_graph).
        show_buffer_layers: Whether to include buffer layers.

    Returns:
        ELK graph dict with hierarchical children, ready for ``run_elk_layout``.
    """
    from collections import defaultdict

    # Step 1: Collect all nodes and their module paths.
    # module_path is containing_modules with pass info stripped.
    node_module_map = {}  # node_label -> [module_addr, ...]
    node_labels = []
    edges = []
    edge_id = 0

    for node_barcode, node in entries_to_plot.items():
        if node.is_buffer_layer and not show_buffer_layers:
            continue
        label = node.layer_label
        node_labels.append(label)

        # Strip pass numbers from module addresses for grouping.
        modules = []
        for mod in node.containing_modules:
            addr = mod.split(":")[0]
            if addr not in modules:
                modules.append(addr)
        node_module_map[label] = modules

        # Collect edges from parent layers.
        for parent_label in node.parent_layers:
            edges.append((parent_label, label, f"e{edge_id}"))
            edge_id += 1

    # Step 2: Build module tree structure.
    # Each module becomes an ELK compound node containing its direct children.
    all_modules = set()
    module_children = defaultdict(set)  # module_addr -> set of child module addrs
    for modules in node_module_map.values():
        for i, mod in enumerate(modules):
            all_modules.add(mod)
            if i > 0:
                module_children[modules[i - 1]].add(mod)

    # Step 3: Assign each node to its deepest module (or root).
    module_nodes = defaultdict(list)  # module_addr -> [node_labels]
    root_nodes = []
    for label in node_labels:
        modules = node_module_map[label]
        if modules:
            module_nodes[modules[-1]].append(label)
        else:
            root_nodes.append(label)

    # Step 4: Build ELK graph recursively.
    def _make_elk_node(label):
        return {"id": label, "width": _DEFAULT_NODE_WIDTH, "height": _DEFAULT_NODE_HEIGHT}

    def _make_elk_group(module_addr, visited=None):
        if visited is None:
            visited = set()
        if module_addr in visited:
            return None
        visited.add(module_addr)

        children = []
        # Add direct leaf nodes in this module.
        for label in module_nodes.get(module_addr, []):
            children.append(_make_elk_node(label))

        # Add child modules as compound nodes.
        for child_mod in sorted(module_children.get(module_addr, [])):
            child_group = _make_elk_group(child_mod, visited)
            if child_group and child_group.get("children"):
                children.append(child_group)

        if not children:
            return None

        return {
            "id": f"group_{module_addr}",
            "layoutOptions": {
                "elk.padding": "[top=60,left=40,bottom=70,right=40]",
            },
            "children": children,
        }

    # Step 5: Assemble root-level ELK graph.
    top_children = []
    # Add root-level nodes (no module containment).
    for label in root_nodes:
        top_children.append(_make_elk_node(label))

    # Find top-level modules (not children of any other module).
    child_modules = set()
    for children in module_children.values():
        child_modules.update(children)
    top_modules = all_modules - child_modules

    for mod in sorted(top_modules):
        group = _make_elk_group(mod)
        if group and group.get("children"):
            top_children.append(group)

    # Any orphan modules (modules that are top-level but have no direct children
    # because all their nodes are in sub-modules) — their nodes are already
    # included via the sub-module groups.

    # Step 6: Build edge list. ELK edges at root level reference node IDs
    # anywhere in the hierarchy.
    elk_edges = []
    node_set = set(node_labels)
    for source, target, eid in edges:
        if source in node_set and target in node_set:
            elk_edges.append({"id": eid, "sources": [source], "targets": [target]})

    return {
        "id": "root",
        "layoutOptions": {
            "elk.algorithm": "layered",
            "elk.direction": "UP",
            "elk.spacing.nodeNode": "50",
            "elk.layered.spacing.nodeNodeBetweenLayers": "80",
            "elk.spacing.edgeNode": "30",
            "elk.edgeRouting": "ORTHOGONAL",
            "elk.hierarchyHandling": "INCLUDE_CHILDREN",
        },
        "children": top_children,
        "edges": elk_edges,
    }


def build_elk_graph(dot_source: str) -> dict:
    """Parse a Graphviz DOT source string and build an ELK JSON graph.

    Extracts node IDs and edges from the DOT source to build a flat ELK graph
    (no hierarchy — module clusters are skipped for large graphs).

    Args:
        dot_source: The Graphviz DOT source string.

    Returns:
        ELK graph dict ready for ``run_elk_layout``.
    """
    elk_nodes = []
    elk_edges = []
    node_ids = set()

    # Skip Graphviz keywords that look like node declarations.
    _DOT_KEYWORDS = {"graph", "digraph", "subgraph", "node", "edge", "strict"}

    # Match node declarations — both quoted ("node_id" [...]) and unquoted (node_id [...]).
    node_pattern = re.compile(r'^\s*(?:"([^"]+)"|(\w+))\s*\[', re.MULTILINE)
    for m in node_pattern.finditer(dot_source):
        node_id = m.group(1) or m.group(2)
        if node_id in _DOT_KEYWORDS:
            continue
        if node_id not in node_ids:
            node_ids.add(node_id)
            elk_nodes.append(
                {
                    "id": node_id,
                    "width": _DEFAULT_NODE_WIDTH,
                    "height": _DEFAULT_NODE_HEIGHT,
                }
            )

    # Match edges — both quoted and unquoted node IDs.
    edge_pattern = re.compile(r'(?:"([^"]+)"|(\w+))\s*->\s*(?:"([^"]+)"|(\w+))')
    edge_id = 0
    for m in edge_pattern.finditer(dot_source):
        source = m.group(1) or m.group(2)
        target = m.group(3) or m.group(4)
        elk_edges.append(
            {
                "id": f"e{edge_id}",
                "sources": [source],
                "targets": [target],
            }
        )
        edge_id += 1

    return {
        "id": "root",
        "layoutOptions": {
            "elk.algorithm": "layered",
            "elk.direction": "UP",
            "elk.spacing.nodeNode": "40",
            "elk.layered.spacing.nodeNodeBetweenLayers": "60",
            "elk.spacing.edgeNode": "30",
            "elk.edgeRouting": "ORTHOGONAL",
        },
        "children": elk_nodes,
        "edges": elk_edges,
    }


def run_elk_layout(elk_graph: dict, timeout: Optional[int] = None) -> dict:
    """Run ELK layout via Node.js subprocess.

    Args:
        elk_graph: ELK JSON graph dict.
        timeout: Subprocess timeout in seconds (default: ``_ELK_TIMEOUT``).

    Returns:
        ELK graph dict with ``x``, ``y`` positions on each node.

    Raises:
        RuntimeError: If the Node.js subprocess fails.
    """
    if timeout is None:
        timeout = _ELK_TIMEOUT

    graph_json = json.dumps(elk_graph)
    # Free the Python dict — we only need the JSON string from here.
    elk_graph.clear()

    graph_kb = len(graph_json) // 1024
    # Cap heap at 64GB — V8 only allocates what it actually needs, and
    # unbounded values (e.g. 5.6TB for 1M nodes) are nonsensical.
    heap_mb = min(65536, max(16384, graph_kb * 48))

    # Further cap to available system memory (leave 4GB for Python + OS).
    avail_mb = _available_memory_mb()
    if avail_mb > 0:
        heap_mb = min(heap_mb, max(4096, avail_mb - 4096))

    # Worker thread stack via resourceLimits.stackSizeMb (MB).
    # Floor of 4096 MB (matches CHANGELOG), cap at 8192 MB.
    stack_mb = min(8192, max(4096, graph_kb // 8))

    env = _node_env()
    env["_TL_STACK_MB"] = str(stack_mb)
    env["_TL_HEAP_MB"] = str(heap_mb)

    # Write JSON to a temp file so Node.js reads from disk instead of stdin.
    # This lets us free the graph_json string before the subprocess runs,
    # avoiding holding ~120MB+ in Python memory during ELK layout.
    json_fd, json_path = tempfile.mkstemp(suffix=".json", prefix="tl_elk_")
    try:
        with os.fdopen(json_fd, "w") as f:
            f.write(graph_json)
        del graph_json
        env["_TL_JSON_PATH"] = json_path

        # Reclaim garbage before the memory-heavy subprocess.
        gc.collect()

        try:
            result = subprocess.run(
                [
                    "node",
                    f"--max-old-space-size={heap_mb}",
                    "-e",
                    _ELK_LAYOUT_SCRIPT,
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        except FileNotFoundError:
            raise RuntimeError("Node.js not found. Install from https://nodejs.org/")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"ELK layout timed out after {timeout}s")
    finally:
        try:
            os.unlink(json_path)
        except OSError:
            pass

    if result.returncode != 0:
        detail = (
            result.stderr.strip()
            if result.stderr
            else (
                f"Node.js exited with code {result.returncode} (no stderr). "
                f"Likely OOM — JSON was {graph_kb} KB, heap was {heap_mb} MB"
                f"{f', system had {avail_mb} MB available' if avail_mb > 0 else ''}. "
                f"Try reducing vis_nesting_depth to collapse modules."
            )
        )
        raise RuntimeError(f"ELK layout failed: {detail}")

    return json.loads(result.stdout)


def inject_elk_positions(dot_source: str, positioned_graph: dict) -> str:
    """Inject ELK-computed positions into a DOT source string.

    Adds ``pos="x,y!"`` attributes to each node so that ``neato -n`` will
    use the pre-computed positions without re-running layout.

    Args:
        dot_source: Original DOT source string.
        positioned_graph: ELK output with ``x``, ``y`` on each child node.

    Returns:
        Modified DOT source with position attributes injected.
    """
    # Build position lookup from ELK output, recursing into compound nodes.
    positions = {}

    def _collect_positions(node, offset_x=0, offset_y=0):
        """Recurse into ELK compound nodes, accumulating absolute positions."""
        for child in node.get("children", []):
            abs_x = offset_x + child.get("x", 0)
            abs_y = offset_y + child.get("y", 0)
            if child["id"].startswith("group_"):
                # Compound node — recurse into its children.
                _collect_positions(child, abs_x, abs_y)
            else:
                # Leaf node — record center position.
                cx = abs_x + child.get("width", _DEFAULT_NODE_WIDTH) / 2
                cy = abs_y + child.get("height", _DEFAULT_NODE_HEIGHT) / 2
                positions[child["id"]] = (cx, cy)

    _collect_positions(positioned_graph)

    if not positions:
        return dot_source

    # Find the maximum y to flip coordinates (ELK y grows down, Graphviz up).
    max_y = max(y for _, y in positions.values())

    _DOT_KEYWORDS = {"graph", "digraph", "subgraph", "node", "edge", "strict"}

    def _inject_pos(match):
        # match has groups: (quoted_id, unquoted_id, attrs)
        node_id = match.group(1) or match.group(2)
        attrs = match.group(3)
        original_id_str = match.group(0).split("[")[0]  # preserve original quoting
        if node_id in _DOT_KEYWORDS:
            return match.group(0)
        if node_id in positions:
            x, y = positions[node_id]
            # Flip y-axis; neato -n expects pos in points.
            pos_str = f"{x:.1f},{(max_y - y):.1f}!"
            attrs = attrs.rstrip("]") + f' pos="{pos_str}"]'
        return f"{original_id_str}[{attrs}"

    # Replace node declarations — both quoted and unquoted IDs.
    result = re.sub(
        r'(?:"([^"]+)"|(\w+))\s*\[([^\]]*\])',
        _inject_pos,
        dot_source,
    )
    return result


def render_with_sfdp(
    source_path: str,
    vis_outpath: str,
    vis_fileformat: str,
    vis_save_only: bool = False,
    timeout: Optional[int] = None,
) -> None:
    """Render a DOT source file using Graphviz sfdp engine.

    Args:
        source_path: Path to the DOT source file.
        vis_outpath: Output path (without extension).
        vis_fileformat: Output format (pdf, png, etc.).
        vis_save_only: If True, don't open viewer.
        timeout: Subprocess timeout in seconds.
    """
    import graphviz

    if timeout is None:
        timeout = _SFDP_TIMEOUT

    rendered_path = f"{vis_outpath}.{vis_fileformat}"
    cmd = ["sfdp", f"-T{vis_fileformat}", "-Goverlap=prism", "-o", rendered_path, source_path]
    subprocess.run(cmd, timeout=timeout, check=True, capture_output=True)
    if not vis_save_only:
        graphviz.backend.viewing.view(rendered_path)


def render_with_elk(
    dot_source: str,
    source_path: str,
    vis_outpath: str,
    vis_fileformat: str,
    vis_save_only: bool = False,
    entries_to_plot=None,
    show_buffer_layers: bool = False,
) -> None:
    """Render using ELK for layout and neato -n for drawing.

    Args:
        dot_source: The DOT source string.
        source_path: Path where DOT source was saved.
        vis_outpath: Output path (without extension).
        vis_fileformat: Output format (pdf, png, etc.).
        vis_save_only: If True, don't open viewer.
        entries_to_plot: If provided, build hierarchical ELK graph with
            module grouping. Otherwise falls back to flat DOT parsing.
        show_buffer_layers: Whether to include buffer layers.
    """
    import graphviz

    if entries_to_plot is not None:
        elk_graph = build_elk_graph_hierarchical(entries_to_plot, show_buffer_layers)
    else:
        elk_graph = build_elk_graph(dot_source)
    positioned = run_elk_layout(elk_graph)
    positioned_source = inject_elk_positions(dot_source, positioned)

    # Write positioned source and render with neato -n (use pre-computed positions).
    positioned_path = source_path + ".positioned"
    with open(positioned_path, "w") as f:
        f.write(positioned_source)

    rendered_path = f"{vis_outpath}.{vis_fileformat}"
    cmd = [
        "neato",
        "-n",
        "-Gsplines=true",
        f"-T{vis_fileformat}",
        "-o",
        rendered_path,
        positioned_path,
    ]
    try:
        subprocess.run(cmd, timeout=_SFDP_TIMEOUT, check=True, capture_output=True)
        if not vis_save_only:
            graphviz.backend.viewing.view(rendered_path)
    finally:
        import os

        if os.path.exists(positioned_path):
            os.remove(positioned_path)


def _seed_stress_positions(elk_graph: dict, edges: list) -> None:
    """Assign initial positions to ELK nodes via topological sort.

    For the stress algorithm, initial positions bias the final layout.
    By assigning y-coordinates from topological depth and spreading
    x-coordinates within each rank, we get directional flow (inputs
    at bottom, outputs at top) even though stress doesn't natively
    support ``elk.direction``.

    Mutates ``elk_graph`` in place, setting ``x`` and ``y`` on leaf nodes.
    """
    from collections import defaultdict, deque

    # Collect all leaf node IDs from the ELK graph.
    leaf_ids = set()

    def _collect_leaves(node):
        for ch in node.get("children", []):
            if ch["id"].startswith("group_"):
                _collect_leaves(ch)
            else:
                leaf_ids.add(ch["id"])

    _collect_leaves(elk_graph)

    # Build adjacency from edges. Edge dicts have "sources" and "targets"
    # from render_elk_direct's all_edges, which are DOT-level dicts with
    # string "source"/"target" keys (not ELK format).
    children_of = defaultdict(list)
    in_degree: dict[str, int] = defaultdict(int)
    for e in edges:
        src = e.get("tail_name")
        tgt = e.get("head_name")
        if src in leaf_ids and tgt in leaf_ids:
            children_of[src].append(tgt)
            in_degree[tgt] += 1

    # Kahn's algorithm for topological depth assignment.
    depth = {}
    queue: deque[str] = deque()
    for nid in leaf_ids:
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

    # Nodes not reached (cycles or disconnected) get depth 0.
    for nid in leaf_ids:
        if nid not in depth:
            depth[nid] = 0

    # Group by depth, spread x within each rank.
    ranks = defaultdict(list)
    for nid, d in depth.items():
        ranks[d].append(nid)

    spacing_y = 100  # points between ranks
    spacing_x = 250  # points between nodes in same rank

    positions = {}
    for d, nodes in ranks.items():
        for i, nid in enumerate(nodes):
            x = i * spacing_x
            y = d * spacing_y  # deeper = higher y (ELK y-down = bottom of graph)
            positions[nid] = (x, y)

    # Inject positions into ELK leaf nodes.
    def _inject(node):
        for ch in node.get("children", []):
            if ch["id"].startswith("group_"):
                _inject(ch)
            elif ch["id"] in positions:
                ch["x"], ch["y"] = positions[ch["id"]]

    _inject(elk_graph)


def _estimate_node_size(label: str) -> tuple:
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


def render_elk_direct(
    model_log,
    entries_to_plot: dict,
    vis_mode: str,
    vis_nesting_depth: int,
    show_buffer_layers: bool,
    overrides,
    vis_outpath: str,
    vis_fileformat: str,
    vis_save_only: bool,
    graph_caption: str,
    rankdir: str,
) -> str:
    """Fast ELK rendering with module cluster boxes.

    Bypasses ``graphviz.Digraph`` construction entirely.  Generates DOT text
    directly with:

    - Node styling matching the dot path (same labels, colors, shapes)
    - ``subgraph cluster_*`` blocks for module hierarchy (the boxes)
    - ELK-computed node positions (``pos="x,y!"``)
    - Edge styling (color, solid/dashed, arg labels)

    Renders with ``neato -n`` (pre-positioned layout that respects clusters).

    Args:
        model_log: The ModelLog instance.
        entries_to_plot: Dict of node_barcode -> LayerPassLog/LayerLog.
        vis_mode: ``'unrolled'`` or ``'rolled'``.
        vis_nesting_depth: Module nesting depth for collapsed modules.
        show_buffer_layers: Whether to include buffer layers.
        overrides: VisualizationOverrides instance.
        vis_outpath: Output file path (without extension).
        vis_fileformat: Output format (pdf, png, svg, etc.).
        vis_save_only: If True, don't open viewer.
        graph_caption: HTML label for the graph title.
        rankdir: Graphviz rank direction (BT, TB, LR).

    Returns:
        The generated DOT source string.

    Raises:
        RuntimeError: If ELK layout or neato rendering fails.
    """
    import os
    from collections import defaultdict

    # Late imports to avoid circular dependency
    from .rendering import (
        _is_collapsed_module,
        _get_node_address_shape_color,
        _get_node_bg_color,
        _make_node_label,
        TRAINABLE_PARAMS_BG_COLOR,
        FROZEN_PARAMS_BG_COLOR,
        DEFAULT_BG_COLOR,
        COMMUTE_FUNCS,
        MIN_MODULE_PENWIDTH,
        PENWIDTH_RANGE,
    )

    # ── Phase 1: Collect node styling, module assignments, and edges ──

    # node_name -> {"attrs": {dot attrs}, "elk_id": layer_label}
    node_data = {}
    # module_key -> [node_names directly in this module (not nested deeper)]
    module_direct_nodes = defaultdict(list)
    # module_key -> set of child module_keys
    module_child_map = defaultdict(set)
    # module_key -> True if any contained node has input_ancestor
    module_has_ancestor = defaultdict(bool)
    # Nodes not inside any module
    root_node_names = []
    # Edge data: list of dicts with tail_name, head_name, color, style, ...
    all_edges = []
    collapsed_set = set()
    edges_used = set()

    def _module_keys_for_node(node, is_collapsed_mod):
        """Get module hierarchy keys for a node."""
        if is_collapsed_mod:
            mods = list(node.containing_modules[: vis_nesting_depth - 1])
        else:
            mods = list(node.containing_modules)
        if vis_mode == "rolled":
            return list(dict.fromkeys(m.split(":")[0] for m in mods))
        return mods

    def _assign_to_hierarchy(node_name, mod_keys, has_ancestor):
        """Place a node into the module tree."""
        if mod_keys:
            module_direct_nodes[mod_keys[-1]].append(node_name)
            for i in range(len(mod_keys) - 1):
                module_child_map[mod_keys[i]].add(mod_keys[i + 1])
            # Propagate has_input_ancestor up
            if has_ancestor:
                for mk in mod_keys:
                    module_has_ancestor[mk] = True
        else:
            root_node_names.append(node_name)

    for _barcode, node in entries_to_plot.items():
        if node.is_buffer_layer and not show_buffer_layers:
            continue

        is_collapsed = _is_collapsed_module(node, vis_nesting_depth)

        if is_collapsed:
            mod_w_pass = node.containing_modules[vis_nesting_depth - 1]
            mod_parts = mod_w_pass.rsplit(":", 1)
            mod_addr, pass_num = mod_parts
            node_name = "pass".join(mod_parts) if vis_mode == "unrolled" else mod_addr
            elk_id = node.layer_label

            if node_name not in collapsed_set:
                collapsed_set.add(node_name)
                ml = model_log.modules[mod_addr]
                mod_out = model_log[mod_w_pass]

                if vis_mode == "unrolled":
                    mpl = model_log.modules[mod_w_pass]
                    n_tensors = mpl.num_layers
                    has_anc = any(model_log[la].has_input_ancestor for la in mpl.layers)
                else:
                    n_tensors = ml.num_layers
                    has_anc = any(model_log[la].has_input_ancestor for la in ml.all_layers)

                np_ = ml.num_passes
                if np_ == 1:
                    title = f"<b>@{mod_addr}</b>"
                elif vis_mode == "unrolled":
                    title = f"<b>@{mod_addr}:{pass_num}</b>"
                else:
                    title = f"<b>@{mod_addr} (x{np_})</b>"

                out_shape: tuple = mod_out.tensor_shape or ()
                if len(out_shape) > 1:
                    ss = "x".join(str(s) for s in out_shape)
                elif len(out_shape) == 1:
                    ss = f"x{out_shape[0]}"
                else:
                    ss = "x1"

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
                lbl = (
                    f"<{title}<br/>{ml.module_class_name}<br/>"
                    f"{ss} ({mod_out.tensor_memory_str})<br/>"
                    f"{n_tensors} layers total<br/>{pd}>"
                )
                attrs = {
                    "label": lbl,
                    "fontcolor": "black",
                    "color": "black",
                    "style": f"filled,{ls}",
                    "fillcolor": bg,
                    "shape": "box3d",
                    "ordering": "out",
                }
                if ":" in bg:
                    attrs["gradientangle"] = "0"

                for k, v in (overrides.nested_node or {}).items():
                    attrs[k] = str(v(model_log, node)) if callable(v) else str(v)

                node_data[node_name] = {"attrs": attrs, "elk_id": elk_id}
                mod_keys = _module_keys_for_node(node, True)
                _assign_to_hierarchy(node_name, mod_keys, has_anc)

            node_color = "black"
        else:
            # Regular layer node
            node_name = node.layer_label.replace(":", "pass")
            elk_id = node.layer_label

            addr, shape, node_color = _get_node_address_shape_color(
                model_log, node, show_buffer_layers
            )
            bg = _get_node_bg_color(model_log, node)
            ls = "solid" if node.has_input_ancestor else "dashed"
            lbl = _make_node_label(node, addr, vis_mode)

            attrs = {
                "label": lbl,
                "fontcolor": node_color,
                "color": node_color,
                "style": f"filled,{ls}",
                "fillcolor": bg,
                "shape": shape,
                "ordering": "out",
            }
            if ":" in bg:
                attrs["gradientangle"] = "0"

            for k, v in (overrides.node or {}).items():
                attrs[k] = str(v(model_log, node)) if callable(v) else str(v)

            node_data[node_name] = {"attrs": attrs, "elk_id": elk_id}
            mod_keys = _module_keys_for_node(node, False)
            _assign_to_hierarchy(node_name, mod_keys, node.has_input_ancestor)

        # ── Collect edges (this node → its children) ──
        for child_label in node.child_layers:
            if vis_mode == "unrolled":
                child_node = model_log.layer_dict_main_keys.get(child_label)
            else:
                child_node = model_log.layer_logs.get(child_label)
            if child_node is None:
                continue
            if child_node.is_buffer_layer and not show_buffer_layers:
                continue

            # Resolve tail name
            if is_collapsed:
                tail_name = node_name
            else:
                tail_name = node.layer_label.replace(":", "pass")

            # Resolve head name
            child_is_collapsed = _is_collapsed_module(child_node, vis_nesting_depth)
            if child_is_collapsed:
                c_mod_w_pass = child_node.containing_modules[vis_nesting_depth - 1]
                c_parts = c_mod_w_pass.rsplit(":", 1)
                head_name = "pass".join(c_parts) if vis_mode == "unrolled" else c_parts[0]
            else:
                head_name = child_node.layer_label.replace(":", "pass")

            # Intra-module skip for two collapsed nodes in the same module
            if is_collapsed and child_is_collapsed and tail_name != head_name:
                p_mods = node.containing_modules[:]
                c_mods = child_node.containing_modules[:]
                if node.is_leaf_module_output:
                    p_mods = p_mods[:-1]
                if child_node.is_leaf_module_output:
                    c_mods = c_mods[:-1]
                if p_mods[:vis_nesting_depth] == c_mods[:vis_nesting_depth]:
                    continue

            if (tail_name, head_name) in edges_used:
                continue
            edges_used.add((tail_name, head_name))

            edge_style = "solid" if node.has_input_ancestor else "dashed"
            edge = {
                "tail_name": tail_name,
                "head_name": head_name,
                "color": node_color,
                "style": edge_style,
                "arrowsize": ".7",
            }

            # Arg labels for non-commutative ops with multiple parents
            if not child_is_collapsed and child_node.layer_type not in COMMUTE_FUNCS:
                _add_arg_label(node, child_node, edge, model_log, show_buffer_layers)

            for k, v in (overrides.edge or {}).items():
                if callable(v):
                    edge[k] = str(v(model_log, node, child_node))
                else:
                    edge[k] = str(v)

            all_edges.append(edge)

    # ── Phase 2: ELK layout ──

    # Build per-node size estimates from labels, so ELK spaces correctly.
    elk_id_sizes = {}
    for dot_name, nd in node_data.items():
        elk_id = nd["elk_id"]
        label = nd["attrs"].get("label", "")
        elk_id_sizes[elk_id] = _estimate_node_size(label)

    elk_graph = build_elk_graph_hierarchical(entries_to_plot, show_buffer_layers)

    # Override ELK node sizes with label-based estimates.
    def _patch_sizes(elk_node):
        for ch in elk_node.get("children", []):
            if ch["id"].startswith("group_"):
                _patch_sizes(ch)
            elif ch["id"] in elk_id_sizes:
                w, h = elk_id_sizes[ch["id"]]
                ch["width"] = w
                ch["height"] = h

    _patch_sizes(elk_graph)

    # Scale timeout with graph size: ~15ms per node, minimum 120s.
    # Empirical: 5k→10s, 25k→114s, scaling ~O(n^1.4).
    num_elk_nodes = len(node_data)

    # The layered algorithm (Sugiyama) uses O(n^2) memory for crossing
    # minimization — at ~100k+ nodes it triggers std::bad_alloc in elkjs.
    # Switch to stress-majorization which is O(n) memory, seeded with
    # topological positions so the layout preserves directional flow.
    if num_elk_nodes > 150000:
        elk_graph["layoutOptions"]["elk.algorithm"] = "stress"
        _seed_stress_positions(elk_graph, all_edges)

    elk_timeout = max(_ELK_TIMEOUT, int(num_elk_nodes * 0.015))
    positioned = run_elk_layout(elk_graph, timeout=elk_timeout)

    # Collect leaf node centers and compound node bounding boxes from ELK output.
    positions = {}  # leaf_id -> (center_x, center_y) in ELK coords
    compound_bboxes = {}  # "group_<mod>" -> (x, y, w, h) in ELK coords (absolute)

    def _collect_pos(elk_node, ox=0, oy=0):
        for ch in elk_node.get("children", []):
            ax = ox + ch.get("x", 0)
            ay = oy + ch.get("y", 0)
            if ch["id"].startswith("group_"):
                w = ch.get("width", 0)
                h = ch.get("height", 0)
                compound_bboxes[ch["id"]] = (ax, ay, w, h)
                _collect_pos(ch, ax, ay)
            else:
                w = ch.get("width", _DEFAULT_NODE_WIDTH)
                h = ch.get("height", _DEFAULT_NODE_HEIGHT)
                positions[ch["id"]] = (ax + w / 2, ay + h / 2)

    _collect_pos(positioned)
    # Use the root node's full height as the y-flip reference.
    root_h = positioned.get("height", 0)
    max_y = max(root_h, max((y for _, y in positions.values()), default=0))

    # ── Phase 3: Generate DOT with clusters and positions ──

    lines = []
    lines.append("digraph {")
    lines.append(
        f"  graph [rankdir={rankdir} label={graph_caption} labelloc=t labeljust=left ordering=out]"
    )
    lines.append("  node [ordering=out]")

    def _node_line(name, indent=1):
        """Generate a DOT node declaration with position and size."""
        nd = node_data[name]
        parts = []
        for k, v in nd["attrs"].items():
            parts.append(f"{k}={_dot_quote(str(v))}")
        elk_id = nd["elk_id"]
        if elk_id in positions:
            x, y = positions[elk_id]
            # neato -n expects pos in points (not inches).
            parts.append(f'pos="{x:.1f},{(max_y - y):.1f}!"')
        prefix = "  " * indent
        return f"{prefix}{_dot_id(name)} [{' '.join(parts)}]"

    # Compute max module depth for penwidth scaling
    all_mod_keys = set(module_direct_nodes.keys()) | set(module_child_map.keys())

    def _max_depth(mod_key, depth=0, visited=None):
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

    def _write_cluster(mod_key, depth, indent):
        """Recursively write a cluster subgraph with its nodes and children."""
        prefix = "  " * indent
        safe = mod_key.replace(":", "_pass").replace(".", "_")
        lines.append(f"{prefix}subgraph cluster_{safe} {{")

        mod_addr = mod_key.split(":")[0] if ":" in mod_key else mod_key
        try:
            ml = model_log.modules[mod_addr]
        except (KeyError, IndexError):
            ml = None
        mod_type = ml.module_class_name if ml else "Module"
        np_ = ml.num_passes if ml else 1

        if vis_mode == "unrolled" and np_ > 1 and ":" in mod_key:
            title = mod_key
        elif vis_mode == "rolled" and np_ > 1:
            title = f"{mod_addr} (x{np_})"
        else:
            title = mod_addr

        nf = (max_nest - depth) / max_nest if max_nest > 0 else 1
        pw = MIN_MODULE_PENWIDTH + nf * PENWIDTH_RANGE
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
            mod_attrs[k] = str(v(model_log, mod_key)) if callable(v) else str(v)

        # Inject bounding box from ELK compound node so neato uses exact size.
        elk_group_id = f"group_{mod_addr}"
        if elk_group_id in compound_bboxes:
            ex, ey, ew, eh = compound_bboxes[elk_group_id]
            # Convert ELK coords (y-down) to graphviz bb (y-up, in points).
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

    # Edges (at top level — neato -n routes them fine)
    for edge in all_edges:
        tail = _dot_id(edge.pop("tail_name"))
        head = _dot_id(edge.pop("head_name"))
        parts = [f"{k}={_dot_quote(str(v))}" for k, v in edge.items()]
        lines.append(f"  {tail} -> {head} [{' '.join(parts)}]")

    lines.append("}")
    dot_source = "\n".join(lines)

    # ── Phase 4: Render with neato -n ──

    if num_elk_nodes > 25000 and vis_fileformat != "svg":
        warnings.warn(
            f"Graph has {num_elk_nodes} nodes. PDF/PNG rendering may produce "
            f"empty output at this scale. Consider using vis_fileformat='svg' "
            f"for large graphs — SVG files are zoomable in any browser."
        )

    source_path = f"{vis_outpath}.dot"
    with open(source_path, "w") as f:
        f.write(dot_source)

    rendered_path = f"{vis_outpath}.{vis_fileformat}"
    # Spline routing is O(n^2) — use straight lines for large graphs.
    num_nodes = len(node_data)
    spline_mode = "true" if num_nodes < 1000 else "line"
    cmd = [
        "neato",
        "-n",
        f"-Gsplines={spline_mode}",
        f"-T{vis_fileformat}",
        "-o",
        rendered_path,
        source_path,
    ]
    render_timeout = max(_SFDP_TIMEOUT, int(num_nodes * 0.01))
    try:
        result = subprocess.run(cmd, timeout=render_timeout, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"neato rendering failed (exit {result.returncode}):\n{result.stderr}"
            )
        if not vis_save_only:
            import graphviz

            graphviz.backend.viewing.view(rendered_path)
    finally:
        if os.path.exists(source_path):
            os.remove(source_path)

    return dot_source


def _add_arg_label(parent_node, child_node, edge_dict, model_log, show_buffer_layers):
    """Add argument position labels to an edge when the child has multiple parents.

    Simplified version of ``rendering._label_node_arguments_if_needed`` for the
    fast ELK path.
    """
    from .rendering import LayerPassLog, LayerLog

    # Count visible parents
    num_parents = len(child_node.parent_layers)
    if not show_buffer_layers:
        for pl in child_node.parent_layers:
            if isinstance(child_node, LayerPassLog):
                if model_log[pl].is_buffer_layer:
                    num_parents -= 1
            elif isinstance(child_node, LayerLog):
                if model_log.layer_logs[pl].is_buffer_layer:
                    num_parents -= 1
    if num_parents <= 1:
        return

    arg_labels = []
    for arg_type in ["args", "kwargs"]:
        for arg_loc, arg_label in child_node.parent_layer_arg_locs[arg_type].items():
            if parent_node.layer_label == arg_label:
                arg_labels.append(f"{arg_type[:-1]} {arg_loc}")

    if arg_labels:
        label_str = "<br/>".join(arg_labels)
        edge_dict["label"] = f"<<FONT POINT-SIZE='10'><b>{label_str}</b></FONT>>"
