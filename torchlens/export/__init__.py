"""Static export helpers for TorchLens logs."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any


def svg(log: Any, path: str | Path, *, editable: bool = True) -> Path:
    """Export a ModelLog graph as a lightweight SVG file.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog`` to export.
    path:
        Destination SVG path.
    editable:
        Whether to include stable IDs and semantic CSS classes.

    Returns
    -------
    Path
        Written SVG path.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = _static_graph_data(log)
    destination.write_text(_render_svg(data, editable=editable), encoding="utf-8")
    return destination


def html(log: Any, path: str | Path) -> Path:
    """Export a minimal self-contained HTML graph viewer.

    The output supports pan, zoom, and node hover without importing TorchLens'
    viewer or notebook extras and without loading network resources.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog`` to export.
    path:
        Destination HTML path.

    Returns
    -------
    Path
        Written HTML path.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = _static_graph_data(log)
    payload = json.dumps(data, separators=(",", ":"))
    destination.write_text(_render_html(payload), encoding="utf-8")
    return destination


def chrome_trace_diff(bundle: Any, path: str | Path) -> Path:
    """Export a Chrome trace timeline comparing bundle members.

    Parameters
    ----------
    bundle:
        TorchLens ``Bundle`` with a ``supergraph`` accessor.
    path:
        Destination JSON path.

    Returns
    -------
    Path
        Written JSON path.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "traceEvents": _chrome_trace_diff_events(bundle),
        "displayTimeUnit": "ms",
        "metadata": {
            "schema": "torchlens.chrome_trace_diff.v1",
            "members": list(bundle.names),
        },
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def _chrome_trace_diff_events(bundle: Any) -> list[dict[str, Any]]:
    """Return Chrome trace events for a bundle comparison.

    Parameters
    ----------
    bundle:
        Bundle to serialize.

    Returns
    -------
    list[dict[str, Any]]
        Chrome trace event records.
    """

    events: list[dict[str, Any]] = []
    deltas = bundle.norm_delta()
    pid_by_member = {name: index + 1 for index, name in enumerate(bundle.names)}
    for member_name in bundle.names:
        events.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": pid_by_member[member_name],
                "tid": 0,
                "args": {"name": member_name},
            }
        )
    for node_index, node_name in enumerate(bundle.supergraph.topological_order):
        node = bundle.supergraph.nodes[node_name]
        for member_name in getattr(node, "traces", []):
            layer = node.layer_refs.get(member_name)
            events.append(
                {
                    "name": node_name,
                    "cat": "torchlens.forward",
                    "ph": "X",
                    "pid": pid_by_member[member_name],
                    "tid": 0,
                    "ts": node_index * 1000,
                    "dur": 1000,
                    "args": {
                        "op_type": getattr(node, "op_type", ""),
                        "module_path": getattr(node, "module_path", None),
                        "module_type": getattr(node, "module_type", None),
                        "delta": deltas.get(node_name, {}).get(member_name),
                        "tensor_memory": getattr(layer, "tensor_memory", None),
                    },
                }
            )
    return events


def _static_graph_data(log: Any) -> dict[str, Any]:
    """Serialize a ModelLog into static graph data.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog`` to serialize.

    Returns
    -------
    dict[str, Any]
        Node and edge metadata for SVG/HTML exporters.
    """

    entries = list(
        getattr(log, "layer_list", None) or getattr(log, "layer_dict_main_keys", {}).values()
    )
    node_ids = {getattr(entry, "layer_label", "") for entry in entries}
    nodes: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        node_id = str(getattr(entry, "layer_label", f"node_{index}"))
        nodes.append(
            {
                "id": node_id,
                "label": node_id,
                "type": _node_type(entry),
                "shape": "x".join(str(dim) for dim in getattr(entry, "tensor_shape", ()) or ()),
                "memory": str(getattr(entry, "tensor_memory_str", "")),
                "x": 80 + (index % 8) * 180,
                "y": 80 + (index // 8) * 110,
            }
        )
    edges: list[dict[str, str]] = []
    for entry in entries:
        target = str(getattr(entry, "layer_label", ""))
        for parent in getattr(entry, "parent_layers", None) or []:
            if parent in node_ids:
                edges.append({"source": str(parent), "target": target})
    width = max((int(node["x"]) for node in nodes), default=0) + 160
    height = max((int(node["y"]) for node in nodes), default=0) + 100
    return {
        "title": getattr(log, "model_name", "TorchLens graph"),
        "nodes": nodes,
        "edges": edges,
        "width": width,
        "height": height,
    }


def _node_type(entry: Any) -> str:
    """Return the semantic node type for an exported entry.

    Parameters
    ----------
    entry:
        Layer-pass log entry.

    Returns
    -------
    str
        Semantic node type.
    """

    if getattr(entry, "is_input_layer", False):
        return "input"
    if getattr(entry, "is_output_layer", False):
        return "output"
    if getattr(entry, "is_buffer_layer", False):
        return "buffer"
    if getattr(entry, "is_terminal_bool_layer", False):
        return "bool"
    if int(getattr(entry, "num_params_total", 0) or 0) > 0:
        return "parameterized"
    return "operation"


def _render_svg(data: dict[str, Any], *, editable: bool) -> str:
    """Render serialized graph data as SVG.

    Parameters
    ----------
    data:
        Static graph data.
    editable:
        Whether to include stable IDs and semantic classes.

    Returns
    -------
    str
        SVG document.
    """

    node_by_id = {node["id"]: node for node in data["nodes"]}
    edge_markup = []
    for edge in data["edges"]:
        source = node_by_id[edge["source"]]
        target = node_by_id[edge["target"]]
        edge_id = f"tl-edge-{_safe_id(edge['source'])}-{_safe_id(edge['target'])}"
        attrs = f' id="{edge_id}" class="tl-edge"' if editable else ""
        edge_markup.append(
            f'<line{attrs} x1="{source["x"] + 60}" y1="{source["y"]}" '
            f'x2="{target["x"] - 60}" y2="{target["y"]}" />'
        )
    node_markup = []
    for node in data["nodes"]:
        node_id = f"tl-node-{_safe_id(node['id'])}"
        attrs = f' id="{node_id}" class="tl-node tl-node-{node["type"]}"' if editable else ""
        title = escape(f"{node['label']} {node['shape']} {node['memory']}".strip())
        node_markup.append(
            f'<g{attrs} transform="translate({node["x"]},{node["y"]})">'
            f"<title>{title}</title><rect x='-65' y='-28' width='130' height='56' rx='6' />"
            f"<text text-anchor='middle' y='-4'>{escape(node['label'])}</text>"
            f"<text text-anchor='middle' y='16'>{escape(node['shape'] or node['memory'])}</text></g>"
        )
    return (
        "<?xml version='1.0' encoding='utf-8'?>\n"
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{data['width']}' height='{data['height']}' "
        f"viewBox='0 0 {data['width']} {data['height']}'>"
        "<style>.tl-edge{stroke:#555;stroke-width:1.4}.tl-node rect{fill:#fff;stroke:#222;stroke-width:1.2}"
        ".tl-node-input rect{fill:#D9F0D3}.tl-node-output rect{fill:#F6D7C3}"
        ".tl-node-parameterized rect{fill:#DDEAF7}.tl-node-buffer rect{fill:#F7E7BA}"
        ".tl-node text{font:12px sans-serif;fill:#111;pointer-events:none}</style>"
        + "".join(edge_markup)
        + "".join(node_markup)
        + "</svg>"
    )


def _render_html(payload: str) -> str:
    """Render a self-contained HTML graph viewer.

    Parameters
    ----------
    payload:
        JSON graph payload.

    Returns
    -------
    str
        HTML document.
    """

    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>TorchLens graph</title>"
        "<style>html,body{margin:0;height:100%;overflow:hidden;font-family:system-ui,sans-serif}"
        "#tip{position:fixed;display:none;background:#111;color:white;padding:6px 8px;border-radius:4px;"
        "font-size:12px;pointer-events:none}.tl-edge{stroke:#666;stroke-width:1.4}.tl-node rect{fill:#fff;"
        "stroke:#222;stroke-width:1.2}.tl-node:hover rect{stroke:#0072B2;stroke-width:3}"
        ".tl-node-input rect{fill:#D9F0D3}.tl-node-output rect{fill:#F6D7C3}"
        ".tl-node-parameterized rect{fill:#DDEAF7}.tl-node-buffer rect{fill:#F7E7BA}"
        ".tl-node text{font:12px sans-serif;fill:#111;pointer-events:none}</style></head>"
        "<body><svg id='graph' width='100%' height='100%'><g id='viewport'></g></svg><div id='tip'></div>"
        f"<script>const graph={payload};"
        "const svg=document.getElementById('graph'),vp=document.getElementById('viewport'),tip=document.getElementById('tip');"
        "let scale=1,tx=20,ty=20,drag=false,last=[0,0];const byId=new Map(graph.nodes.map(n=>[n.id,n]));"
        "function el(n,a){const e=document.createElementNS('http://www.w3.org/2000/svg',n);for(const k in a)e.setAttribute(k,a[k]);return e}"
        "function draw(){graph.edges.forEach(ed=>{const s=byId.get(ed.source),t=byId.get(ed.target);"
        "vp.appendChild(el('line',{class:'tl-edge',x1:s.x+60,y1:s.y,x2:t.x-60,y2:t.y}));});"
        "graph.nodes.forEach(n=>{const g=el('g',{class:'tl-node tl-node-'+n.type,transform:`translate(${n.x},${n.y})`});"
        "g.appendChild(el('rect',{x:-65,y:-28,width:130,height:56,rx:6}));"
        "let a=el('text',{'text-anchor':'middle',y:-4});a.textContent=n.label;g.appendChild(a);"
        "let b=el('text',{'text-anchor':'middle',y:16});b.textContent=n.shape||n.memory;g.appendChild(b);"
        "g.onmousemove=e=>{tip.style.display='block';tip.style.left=e.clientX+12+'px';tip.style.top=e.clientY+12+'px';"
        "tip.textContent=[n.label,n.shape,n.memory].filter(Boolean).join('  ')};g.onmouseleave=()=>tip.style.display='none';vp.appendChild(g);});}"
        "function apply(){vp.setAttribute('transform',`translate(${tx},${ty}) scale(${scale})`)}"
        "svg.addEventListener('wheel',e=>{e.preventDefault();scale*=e.deltaY<0?1.1:.9;apply()},{passive:false});"
        "svg.addEventListener('mousedown',e=>{drag=true;last=[e.clientX,e.clientY]});"
        "window.addEventListener('mouseup',()=>drag=false);window.addEventListener('mousemove',e=>{if(!drag)return;"
        "tx+=e.clientX-last[0];ty+=e.clientY-last[1];last=[e.clientX,e.clientY];apply()});draw();apply();</script></body></html>"
    )


def _safe_id(value: str) -> str:
    """Return a CSS/SVG-safe identifier fragment.

    Parameters
    ----------
    value:
        Raw identifier.

    Returns
    -------
    str
        Sanitized identifier.
    """

    return "".join(char if char.isalnum() else "-" for char in value).strip("-")


__all__ = ["chrome_trace_diff", "html", "svg"]
