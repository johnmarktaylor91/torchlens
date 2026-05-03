"""Experimental Dagua-backed visualization bridge for TorchLens Trace objects.

This module translates TorchLens semantics into Dagua graph elements and
documents exactly which Trace fields are consumed by the visualization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from ...utils.display import human_readable_size
from ...visualization.node_spec import (
    INTERVENTION_CONE_COLOR,
    INTERVENTION_SITE_COLOR,
    intervention_site_and_cone_labels,
)

COMMUTATIVE_FUNCS = {"add", "mul", "cat", "eq", "ne"}

MODELLOG_FIELD_USAGE: dict[str, str] = {
    "model_name": "Graph caption title.",
    "num_tensors": "Graph caption tensor count.",
    "total_out_memory": "Graph caption out-memory summary.",
    "num_param_tensors": "Caption metadata for model scale.",
    "num_params": "Graph caption parameter summary.",
    "num_params_trainable": "Graph caption trainable/frozen breakdown.",
    "num_params_frozen": "Graph caption trainable/frozen breakdown.",
    "param_memory": "Graph caption parameter-memory summary.",
    "is_recurrent": "Theme recommendations and recurrent-edge styling.",
    "max_recurrent_loops": "Reference-gallery and recurrent summaries.",
    "modules": "Module hierarchy, cluster labels, collapsed summaries.",
    "has_grads": "Whether to add backward edges in unrolled mode.",
    "layer_dict_main_keys": "Unrolled node inventory.",
    "layer_logs": "Rolled node inventory.",
    "layer_list": "Fallback node inventory and audit coverage.",
    "_intervention_spec": "Intervention site and fire-record metadata overlay.",
    "_has_direct_writes": "Direct-write dirty flag surfaced in graph metadata when present.",
}

ENTRY_FIELD_USAGE: dict[str, str] = {
    "layer_label": "Stable node ID and default primary label.",
    "layer_label_no_pass": "Rolled-mode primary label.",
    "layer_type": "Operation label and type differentiation.",
    "func_name": "Operation label and argument-label suppression for commutative ops.",
    "call_index": "Unrolled pass suffix.",
    "num_calls": "Rolled repetition count and recurrent emphasis.",
    "shape": "Node secondary text.",
    "memory": "Node secondary text.",
    "is_input": "Input node styling.",
    "is_output": "Output node styling.",
    "is_buffer": "Buffer node styling and optional visibility filtering.",
    "buffer_address": "Buffer node secondary text.",
    "is_terminal_bool": "Boolean node styling and TRUE/FALSE header.",
    "bool_value": "Boolean node header.",
    "num_params": "Parameterized-op styling and summaries.",
    "num_params_trainable": "Trainable/frozen/mixed node styling.",
    "num_params_frozen": "Trainable/frozen/mixed node styling.",
    "param_shapes": "Compact parameter-shape text.",
    "has_input_ancestor": "Dashed style for data-independent nodes/edges.",
    "parents": "Forward-edge construction.",
    "children": "Fallback edge construction and skip detection.",
    "parent_arg_positions": "Argument-position edge labels.",
    "cond_branch_start_children": "IF-edge styling.",
    "cond_branch_then_children": "THEN-edge styling.",
    "recurrent_ops": "Recurrent-edge styling.",
    "min_distance_from_input": "Skip-edge heuristics.",
    "module": "Leaf-module line in labels.",
    "modules": "Cluster nesting.",
    "is_submodule_output": "Rectangle/module-output shape override.",
    "is_atomic_module_op": "Rectangle/module-output shape override.",
    "has_grad": "Backward-edge inclusion in unrolled mode.",
    "interventions": "Per-node intervention fire-record summary.",
}

MODULE_FIELD_USAGE: dict[str, str] = {
    "address": "Cluster identity.",
    "class_name": "Cluster label and collapsed summary text.",
    "address_parent": "Cluster hierarchy.",
    "address_depth": "Cluster depth-sensitive styling.",
    "num_calls": "Rolled/unrolled cluster summaries.",
    "num_layers": "Collapsed summary text.",
    "num_params": "Collapsed summary text.",
    "num_params_trainable": "Collapsed summary text.",
    "num_params_frozen": "Collapsed summary text.",
    "param_memory": "Collapsed summary text.",
    "layers": "Collapsed summary node membership.",
}


@dataclass
class TorchLensRenderAudit:
    """Explicit field-consumption report for TorchLens → Dagua rendering."""

    trace_used: dict[str, str]
    trace_unused: dict[str, str]
    entry_used: dict[str, str]
    entry_unused: dict[str, str]
    module_used: dict[str, str]
    module_unused: dict[str, str]

    def to_dict(self) -> dict[str, dict[str, str]]:
        """Return the audit as plain dictionaries.

        Returns
        -------
        dict[str, dict[str, str]]
            Used and unused field descriptions grouped by source object.
        """
        return {
            "trace_used": self.trace_used,
            "trace_unused": self.trace_unused,
            "entry_used": self.entry_used,
            "entry_unused": self.entry_unused,
            "module_used": self.module_used,
            "module_unused": self.module_unused,
        }


def _list_public_fields(obj: Any) -> list[str]:
    names: list[str] = []
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if callable(value):
            continue
        names.append(name)
    return sorted(set(names))


def _unused_justification(name: str) -> str:
    if "out" in name or "grad" in name:
        return "Raw tensor contents are not rendered directly; visualization uses summaries, not full tensor values."
    if "time" in name or "flops" in name or "macs" in name:
        return "Performance metadata is useful for profiling, but not shown in the default structural visualization."
    if "rng" in name or "source" in name or "docstring" in name or "signature" in name:
        return "Debugging/source-inspection metadata is out of the default visual language and remains available for future overlays."
    if "lookup" in name or "labels" in name or name.endswith("_str"):
        return "Alternate accessors and preformatted strings are redundant with the canonical fields used for rendering."
    if "forward_args" in name or "forward_kwargs" in name or "captured_" in name:
        return "Rich replay data is intentionally excluded from the static graph view to avoid clutter."
    return "Currently not used by the visualization; retained for future overlays, interactivity, or debugging without being silently discarded."


def build_render_audit(trace: Any) -> TorchLensRenderAudit:
    """Return an explicit used/unused field audit for the bridge."""
    model_fields = _list_public_fields(trace)
    entry_sample = trace.layer_list[0] if getattr(trace, "layer_list", None) else None
    entry_fields = _list_public_fields(entry_sample) if entry_sample is not None else []
    module_values = list(trace.modules) if hasattr(trace, "modules") else []
    module_fields = _list_public_fields(module_values[0]) if module_values else []

    return TorchLensRenderAudit(
        trace_used=dict(MODELLOG_FIELD_USAGE),
        trace_unused={
            name: _unused_justification(name)
            for name in model_fields
            if name not in MODELLOG_FIELD_USAGE
        },
        entry_used=dict(ENTRY_FIELD_USAGE),
        entry_unused={
            name: _unused_justification(name)
            for name in entry_fields
            if name not in ENTRY_FIELD_USAGE
        },
        module_used=dict(MODULE_FIELD_USAGE),
        module_unused={
            name: _unused_justification(name)
            for name in module_fields
            if name not in MODULE_FIELD_USAGE
        },
    )


def build_torchlens_caption(trace: Any) -> str:
    """Build the user-facing graph caption."""
    num_params = int(getattr(trace, "num_params", 0) or 0)
    trainable = int(getattr(trace, "num_params_trainable", 0) or 0)
    frozen = int(getattr(trace, "num_params_frozen", 0) or 0)
    if num_params == 0:
        params_line = "0 params"
    elif frozen == 0:
        params_line = f"{num_params:,} params (all trainable, {human_readable_size(getattr(trace, 'param_memory', 0) or 0)})"
    elif trainable == 0:
        params_line = f"{num_params:,} params (all frozen, {human_readable_size(getattr(trace, 'param_memory', 0) or 0)})"
    else:
        params_line = (
            f"{num_params:,} params "
            f"({trainable:,}/{num_params:,} trainable, {human_readable_size(getattr(trace, 'param_memory', 0) or 0)})"
        )
    return (
        f"{getattr(trace, 'model_name', 'TorchLens model')}\n"
        f"{getattr(trace, 'num_tensors', 0)} tensors total "
        f"({human_readable_size(getattr(trace, 'total_out_memory', 0) or 0)})\n"
        f"{params_line}"
    )


def _to_dagua_direction(direction: str) -> str:
    mapping = {"bottomup": "BT", "topdown": "TB", "leftright": "LR"}
    return mapping.get(direction, direction)


def _entries_for_mode(trace: Any, vis_mode: str) -> list[Any]:
    if vis_mode == "unrolled":
        return list(trace.layer_dict_main_keys.values())
    if vis_mode == "rolled":
        return list(trace.layer_logs.values())
    raise ValueError("vis_mode must be 'unrolled' or 'rolled'")


def _display_label(entry: Any, vis_mode: str) -> str:
    if vis_mode == "rolled":
        base = getattr(entry, "layer_label_no_pass", None) or getattr(entry, "layer_label", "")
        ops = int(getattr(entry, "num_calls", 1) or 1)
        base_str = str(base)
        return f"{base_str} (x{ops})" if ops > 1 else base_str
    return str(getattr(entry, "layer_label_w_pass", None) or getattr(entry, "layer_label", ""))


def _shape_memory_line(entry: Any) -> str | None:
    shape = getattr(entry, "shape", None)
    if not shape:
        return None
    shape_str = "x".join(str(v) for v in shape)
    memory = getattr(entry, "memory", None)
    if memory:
        return f"{shape_str}  {human_readable_size(memory)}"
    return shape_str


def _param_summary_line(entry: Any) -> str | None:
    total = int(getattr(entry, "num_params", 0) or 0)
    if total <= 0:
        return None
    trainable = int(getattr(entry, "num_params_trainable", 0) or 0)
    frozen = int(getattr(entry, "num_params_frozen", 0) or 0)
    shapes = getattr(entry, "param_shapes", None) or []
    shape_frag = ""
    if shapes:
        shape_frag = " · " + ", ".join("x".join(str(v) for v in shp) for shp in shapes[:2])
        if len(shapes) > 2:
            shape_frag += ", ..."
    if frozen == 0:
        return f"{total:,} trainable{shape_frag}"
    if trainable == 0:
        return f"{total:,} frozen{shape_frag}"
    return f"{trainable:,}T/{frozen:,}F{shape_frag}"


def _module_line(entry: Any) -> str | None:
    if getattr(entry, "is_buffer", False):
        addr = getattr(entry, "buffer_address", None)
        return f"@{addr}" if addr else "@buffer"
    addr = getattr(entry, "module", None)
    if addr:
        return f"@{str(addr).split(':')[0]}"
    return None


def _build_node_label(entry: Any, vis_mode: str) -> str:
    lines: list[str] = []
    if getattr(entry, "is_terminal_bool", False):
        val = getattr(entry, "bool_value", None)
        if val is not None:
            lines.append("TRUE" if bool(val) else "FALSE")
    lines.append(_display_label(entry, vis_mode))
    for extra in (_shape_memory_line(entry), _param_summary_line(entry), _module_line(entry)):
        if extra:
            lines.append(extra)
    return "\n".join(lines)


def _interventions_summary(entry: Any) -> str:
    """Return a compact summary of intervention fire records for an entry.

    Parameters
    ----------
    entry:
        Layer-pass or layer log entry.

    Returns
    -------
    str
        Human-readable fire-record summary, or an empty string when no records
        are attached.
    """

    records = list(getattr(entry, "interventions", ()) or ())
    if not records:
        return ""
    engines = sorted({str(getattr(record, "engine", "") or "unknown") for record in records})
    return f"{len(records)} fire record(s): {', '.join(engines)}"


def _base_node_type(entry: Any) -> str:
    if getattr(entry, "is_input", False):
        return "input"
    if getattr(entry, "is_output", False):
        return "output"
    if getattr(entry, "is_buffer", False):
        return "buffer"
    if getattr(entry, "is_terminal_bool", False):
        return "bool"
    trainable = int(getattr(entry, "num_params_trainable", 0) or 0)
    frozen = int(getattr(entry, "num_params_frozen", 0) or 0)
    if trainable and frozen:
        return "mixed_params"
    if trainable:
        return "trainable_params"
    if frozen:
        return "frozen_params"
    return "default"


def _module_chain(entry: Any, vis_mode: str, vis_call_depth: int) -> list[str]:
    modules = list(getattr(entry, "modules", None) or [])
    if vis_mode == "rolled":
        modules = [str(m).split(":")[0] for m in modules]
    if vis_call_depth > 0:
        modules = modules[:vis_call_depth]
    return modules


def _labels_match(label: str, candidate: str) -> bool:
    return label == candidate or str(label).split(":")[0] == str(candidate).split(":")[0]


def _argument_edge_label(child: Any, parent_label: str) -> str | None:
    arg_locs = getattr(child, "parent_arg_positions", None) or {}
    func_name = (
        getattr(child, "func_name", None) or getattr(child, "layer_type", "") or ""
    ).lower()
    if len(getattr(child, "parents", None) or []) <= 1:
        return None
    if func_name in COMMUTATIVE_FUNCS:
        return None
    labels: list[str] = []
    for pos, source in (arg_locs.get("args", {}) or {}).items():
        if _labels_match(str(source), parent_label):
            labels.append(f"arg {pos}")
    for key, source in (arg_locs.get("kwargs", {}) or {}).items():
        if _labels_match(str(source), parent_label):
            labels.append(f"kwarg {key}")
    if labels:
        return ", ".join(labels)
    return None


def _is_skip_edge(parent: Any, child: Any) -> bool:
    p = getattr(parent, "min_distance_from_input", None)
    c = getattr(child, "min_distance_from_input", None)
    if p is None or c is None:
        return False
    try:
        return int(c) - int(p) > 1
    except Exception:
        return False


def _is_recurrent_edge(parent: Any, child: Any) -> bool:
    p_group = getattr(parent, "recurrent_ops", None)
    c_group = getattr(child, "recurrent_ops", None)
    if p_group and c_group and p_group == c_group:
        return True
    return (
        int(getattr(parent, "num_calls", 1) or 1) > 1
        or int(getattr(child, "num_calls", 1) or 1) > 1
    )


def _classify_forward_edge(parent: Any, child: Any) -> str:
    child_label = getattr(child, "layer_label", "")
    if child_label in (getattr(parent, "cond_branch_then_children", None) or []):
        return "then"
    if child_label in (getattr(parent, "cond_branch_start_children", None) or []):
        return "if"
    if getattr(parent, "is_buffer", False):
        return "buffer"
    if _is_recurrent_edge(parent, child):
        return "recurrent"
    if _is_skip_edge(parent, child):
        return "skip"
    return "default"


def _resolve_container_path(vis_outpath: str, vis_fileformat: str) -> str:
    p = Path(vis_outpath)
    if p.suffix.lower() in {
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".svg",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }:
        return str(p)
    if p.suffix.lower() == ".gv":
        return str(p.with_suffix(f".{vis_fileformat}"))
    return str(p.with_suffix(f".{vis_fileformat}"))


def _torchlens_layout_config(num_nodes: int, direction: str) -> Any:
    import dagua

    if torch.cuda.is_available() and num_nodes <= 25_000:
        device = "cuda"
    else:
        device = "cpu"
    if num_nodes <= 150:
        return dagua.LayoutConfig(
            device=device,
            direction=direction,
            steps=120,
            edge_opt_steps=20,
            node_sep=28,
            rank_sep=52,
            seed=42,
        )
    if num_nodes <= 800:
        return dagua.LayoutConfig(
            device=device,
            direction=direction,
            steps=90,
            edge_opt_steps=12,
            node_sep=30,
            rank_sep=56,
            seed=42,
        )
    if num_nodes <= 5_000:
        return dagua.LayoutConfig(
            device=device,
            direction=direction,
            steps=60,
            edge_opt_steps=6,
            node_sep=26,
            rank_sep=50,
            seed=42,
        )
    return dagua.LayoutConfig(
        device=device,
        direction=direction,
        steps=40,
        edge_opt_steps=-1,
        node_sep=22,
        rank_sep=42,
        seed=42,
    )


def trace_to_dagua_graph(
    trace: Any,
    vis_mode: str = "unrolled",
    vis_call_depth: int = 1000,
    show_buffer_layers: bool = False,
    direction: str = "bottomup",
    include_grad_edges: bool | None = None,
) -> Any:
    """Translate a TorchLens Trace into a DaguaGraph."""
    import dagua

    g = dagua.DaguaGraph(direction=_to_dagua_direction(direction))
    entries = _entries_for_mode(trace, vis_mode)
    if not show_buffer_layers:
        entries = [e for e in entries if not getattr(e, "is_buffer", False)]
    entries_by_label = {getattr(e, "layer_label"): e for e in entries}
    site_labels, cone_labels = intervention_site_and_cone_labels(trace, show_cone=True)
    g.is_intervention_site = []
    g.is_in_cone = []
    g.interventions_summary = []
    g.has_direct_writes = bool(getattr(trace, "_has_direct_writes", False))

    module_shape = dagua.NodeStyle(shape="roundrect", corner_radius=8.0)
    op_shape = dagua.NodeStyle(shape="ellipse")
    buffer_shape = dagua.NodeStyle(shape="rect", corner_radius=3.0)

    for entry in entries:
        node_id = getattr(entry, "layer_label")
        g.add_node(node_id, label=_build_node_label(entry, vis_mode), type=_base_node_type(entry))
        node_idx = g._id_to_index[node_id]
        is_site = node_id in site_labels
        is_cone = node_id in cone_labels
        g.is_intervention_site.append(is_site)
        g.is_in_cone.append(is_cone)
        g.interventions_summary.append(_interventions_summary(entry))

        shape_override = op_shape
        if getattr(entry, "is_buffer", False):
            shape_override = buffer_shape
        elif getattr(entry, "is_submodule_output", False) or getattr(
            entry, "is_atomic_module_op", False
        ):
            shape_override = module_shape

        if not getattr(entry, "has_input_ancestor", True):
            shape_override = dagua.NodeStyle(
                shape=shape_override.shape,
                corner_radius=shape_override.corner_radius,
                stroke_dash="dashed",
            )
        if is_site or is_cone:
            shape_override = dagua.NodeStyle(
                shape=shape_override.shape,
                corner_radius=shape_override.corner_radius,
                stroke_dash=shape_override.stroke_dash,
                stroke=INTERVENTION_SITE_COLOR if is_site else INTERVENTION_CONE_COLOR,
            )
        g.node_styles[node_idx] = shape_override

    edges_seen = set()
    for child in entries:
        child_id = getattr(child, "layer_label", "")
        for parent_label in getattr(child, "parents", None) or []:
            if parent_label not in entries_by_label:
                continue
            parent = entries_by_label[parent_label]
            edge_key = (parent_label, child_id, "forward")
            if edge_key in edges_seen:
                continue
            edges_seen.add(edge_key)

            edge_type = _classify_forward_edge(parent, child)
            edge_label = (
                "IF"
                if edge_type == "if"
                else "THEN"
                if edge_type == "then"
                else _argument_edge_label(child, parent_label)
            )

            style = None
            if not getattr(parent, "has_input_ancestor", True):
                style = dagua.EdgeStyle(style="dashed")
            g.add_edge(parent_label, child_id, label=edge_label, type=edge_type, style=style)

    if include_grad_edges is None:
        include_grad_edges = bool(getattr(trace, "has_grads", False))

    if include_grad_edges and vis_mode == "unrolled":
        for child in entries:
            if not getattr(child, "has_grad", False):
                continue
            child_id = getattr(child, "layer_label", "")
            for parent_label in getattr(child, "parents", None) or []:
                parent = entries_by_label.get(parent_label)
                if parent is None or not getattr(parent, "has_grad", False):
                    continue
                edge_key = (child_id, parent_label, "back")
                if edge_key in edges_seen:
                    continue
                edges_seen.add(edge_key)
                g.add_edge(child_id, parent_label, type="back")

    kept_modules: dict[str, list[str]] = {}
    for entry in entries:
        mods = _module_chain(entry, vis_mode, vis_call_depth)
        if not mods:
            continue
        node_id = getattr(entry, "layer_label")
        for mod in mods:
            kept_modules.setdefault(mod, []).append(node_id)

    module_logs = {}
    if hasattr(trace, "modules"):
        try:
            module_logs = {
                m.address: m for m in trace.modules if getattr(m, "address", "self") != "self"
            }
        except Exception:
            module_logs = {}

    for mod, members in kept_modules.items():
        if not members:
            continue
        parent = None
        parts = mod.split(".")
        for depth in range(len(parts) - 1, 0, -1):
            candidate = ".".join(parts[:depth])
            if candidate in kept_modules:
                parent = candidate
                break
        module_log = module_logs.get(mod)
        class_name = getattr(module_log, "class_name", "") if module_log is not None else ""
        label = f"@{mod} ({class_name})" if class_name else f"@{mod}"
        g.add_cluster(mod, members, label=label, parent=parent, strict=False)

    return g


def render_trace_with_dagua(
    trace: Any,
    vis_mode: str = "unrolled",
    vis_call_depth: int = 1000,
    vis_outpath: str = "graph.gv",
    vis_save_only: bool = False,
    vis_fileformat: str = "pdf",
    vis_buffers: bool = False,
    vis_direction: str = "bottomup",
    vis_theme: str = "torchlens",
) -> Any:
    """Render a TorchLens Trace with Dagua."""
    import dagua

    graph = trace_to_dagua_graph(
        trace,
        vis_mode=vis_mode,
        vis_call_depth=vis_call_depth,
        show_buffer_layers=vis_buffers,
        direction=vis_direction,
    )
    graph._theme = dagua.get_theme(vis_theme)
    output = _resolve_container_path(vis_outpath, vis_fileformat)
    config = _torchlens_layout_config(graph.num_nodes, graph.direction)
    dagua.draw(
        graph,
        config=config,
        output=output,
        title=build_torchlens_caption(trace),
    )
    return graph.to_json() if hasattr(graph, "to_json") else output
