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
import json
import re
import subprocess
import warnings
from typing import Optional

_ELK_NODE_THRESHOLD = 3500
_ELK_TIMEOUT = 60  # seconds for Node.js subprocess
_SFDP_TIMEOUT = 120  # seconds for sfdp subprocess
_DEFAULT_NODE_WIDTH = 150  # points
_DEFAULT_NODE_HEIGHT = 40  # points

# Inline Node.js script that reads ELK JSON from stdin, runs layout, writes to stdout.
_ELK_LAYOUT_SCRIPT = r"""
const ELK = require('elkjs');
const elk = new ELK();

let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => { input += chunk; });
process.stdin.on('end', () => {
    const graph = JSON.parse(input);
    elk.layout(graph).then((result) => {
        process.stdout.write(JSON.stringify(result));
    }).catch((err) => {
        process.stderr.write(err.toString());
        process.exit(1);
    });
});
"""


def _node_env() -> dict:
    """Build environment dict for Node.js subprocesses.

    Sets NODE_PATH to include the global node_modules directory so that
    globally-installed packages (like elkjs via ``npm install -g``) are
    found even when using nvm or non-standard Node.js installs.
    """
    import os
    import shutil

    env = os.environ.copy()
    node_bin = shutil.which("node")
    if node_bin is None:
        return env

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
            "elk.spacing.nodeNode": "20",
            "elk.layered.spacing.nodeNodeBetweenLayers": "40",
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

    try:
        result = subprocess.run(
            ["node", "--stack-size=65536", "-e", _ELK_LAYOUT_SCRIPT],
            input=json.dumps(elk_graph),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_node_env(),
        )
    except FileNotFoundError:
        raise RuntimeError("Node.js not found. Install from https://nodejs.org/")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ELK layout timed out after {timeout}s")

    if result.returncode != 0:
        raise RuntimeError(f"ELK layout failed: {result.stderr}")

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
    # Build position lookup from ELK output.
    positions = {}
    for child in positioned_graph.get("children", []):
        node_id = child["id"]
        # ELK coordinates: top-left corner. Convert to center for Graphviz.
        x = child["x"] + child.get("width", _DEFAULT_NODE_WIDTH) / 2
        y = child["y"] + child.get("height", _DEFAULT_NODE_HEIGHT) / 2
        positions[node_id] = (x, y)

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
            # Flip y-axis and convert to inches (72 points per inch).
            pos_str = f"{x / 72:.4f},{(max_y - y) / 72:.4f}!"
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
    save_only: bool = False,
    timeout: Optional[int] = None,
) -> None:
    """Render a DOT source file using Graphviz sfdp engine.

    Args:
        source_path: Path to the DOT source file.
        vis_outpath: Output path (without extension).
        vis_fileformat: Output format (pdf, png, etc.).
        save_only: If True, don't open viewer.
        timeout: Subprocess timeout in seconds.
    """
    import graphviz

    if timeout is None:
        timeout = _SFDP_TIMEOUT

    rendered_path = f"{vis_outpath}.{vis_fileformat}"
    cmd = ["sfdp", f"-T{vis_fileformat}", "-Goverlap=prism", "-o", rendered_path, source_path]
    subprocess.run(cmd, timeout=timeout, check=True, capture_output=True)
    if not save_only:
        graphviz.backend.viewing.view(rendered_path)


def render_with_elk(
    dot_source: str,
    source_path: str,
    vis_outpath: str,
    vis_fileformat: str,
    save_only: bool = False,
) -> None:
    """Render using ELK for layout and neato -n for drawing.

    Args:
        dot_source: The DOT source string.
        source_path: Path where DOT source was saved.
        vis_outpath: Output path (without extension).
        vis_fileformat: Output format (pdf, png, etc.).
        save_only: If True, don't open viewer.
    """
    import graphviz

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
        if not save_only:
            graphviz.backend.viewing.view(rendered_path)
    finally:
        import os

        if os.path.exists(positioned_path):
            os.remove(positioned_path)
