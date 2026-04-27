"""Live visualization helpers for fastlog dry-run traces."""

from __future__ import annotations

from collections import Counter, defaultdict
from html import escape
from typing import Any

import graphviz

from ..fastlog.types import RecordContext, RecordingTrace

_RAIL_COLORS = (
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
)


def _shape_text(ctx: RecordContext) -> str:
    """Return a compact tensor shape string."""

    if ctx.tensor_shape is None:
        return ""
    return "x".join(str(dim) for dim in ctx.tensor_shape)


def _dtype_text(ctx: RecordContext) -> str:
    """Return a compact dtype string."""

    if ctx.tensor_dtype is None:
        return ""
    return str(ctx.tensor_dtype).replace("torch.", "")


def _event_kept(trace: RecordingTrace, index: int) -> bool:
    """Return whether a trace event was selected by the predicate."""

    if index >= len(trace.decisions):
        return False
    return trace.decisions[index]


def _module_key(ctx: RecordContext) -> str:
    """Return the best module grouping key for an event."""

    if ctx.module_address:
        return ctx.module_address
    if ctx.module_stack:
        return ctx.module_stack[-1].module_address
    return "self"


def _module_color(module_address: str, color_map: dict[str, str]) -> str:
    """Return a stable rail color for a module address."""

    if module_address not in color_map:
        color_map[module_address] = _RAIL_COLORS[len(color_map) % len(_RAIL_COLORS)]
    return color_map[module_address]


def print_tree(trace: RecordingTrace) -> str:
    """Return a unicode-indented tree of dry-run events.

    Parameters
    ----------
    trace:
        Dry-run trace to render.

    Returns
    -------
    str
        One row per event, indented by module stack depth.
    """

    rows: list[str] = []
    for index, ctx in enumerate(trace.events):
        depth = len(ctx.module_stack)
        prefix = "  " * max(depth - 1, 0) + ("└─ " if depth else "")
        kept = "kept" if _event_kept(trace, index) else "skip"
        module = _module_key(ctx)
        op_type = ctx.layer_type or ctx.func_name or ctx.kind
        rows.append(f"{prefix}{ctx.event_index:04d} {ctx.kind} {op_type} [{module}] {kept}")
    return "\n".join(rows)


def to_pandas(trace: RecordingTrace) -> Any:
    """Return a pandas DataFrame with one row per trace event."""

    import pandas as pd

    rows = [
        {
            "pass_num": ctx.pass_index,
            "step_num": ctx.event_index,
            "kind": ctx.kind,
            "op_type": ctx.layer_type or ctx.func_name,
            "module_address": _module_key(ctx),
            "shape": ctx.tensor_shape,
            "dtype": ctx.tensor_dtype,
        }
        for ctx in trace.events
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "pass_num",
            "step_num",
            "kind",
            "op_type",
            "module_address",
            "shape",
            "dtype",
        ],
    )


def _graph_label(ctx: RecordContext, kept: bool, rail_color: str) -> str:
    """Return a Graphviz HTML-like label with a module rail cell."""

    op_type = escape(str(ctx.layer_type or ctx.func_name or ctx.kind))
    shape = escape(_shape_text(ctx))
    dtype = escape(_dtype_text(ctx))
    status = "kept" if kept else "skip"
    return (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">'
        f'<TR><TD BGCOLOR="{rail_color}" WIDTH="8"> </TD>'
        f'<TD ALIGN="LEFT">{op_type}</TD></TR>'
        f'<TR><TD BGCOLOR="{rail_color}" WIDTH="8"> </TD>'
        f'<TD ALIGN="LEFT">{shape} {dtype}</TD></TR>'
        f'<TR><TD BGCOLOR="{rail_color}" WIDTH="8"> </TD>'
        f'<TD ALIGN="LEFT">{status}</TD></TR>'
        "</TABLE>>"
    )


def show_graph(
    trace: RecordingTrace,
    *,
    vis_outpath: str | None = None,
    vis_save_only: bool = True,
    vis_fileformat: str = "pdf",
) -> str:
    """Render a flat Graphviz graph of op events with module-color rails.

    Parameters
    ----------
    trace:
        Dry-run trace to render.
    vis_outpath:
        Optional output path stem.
    vis_save_only:
        Whether to render without opening a viewer when ``vis_outpath`` is set.
    vis_fileformat:
        Graphviz output format.

    Returns
    -------
    str
        Graphviz DOT source.
    """

    dot = graphviz.Digraph("fastlog_dry_run")
    dot.attr(rankdir="LR")
    color_map: dict[str, str] = {}
    op_contexts = [ctx for ctx in trace.events if ctx.kind in {"op", "input", "buffer"}]
    event_to_index = {ctx.event_index: index for index, ctx in enumerate(trace.events)}
    for op_number, ctx in enumerate(op_contexts):
        trace_index = event_to_index[ctx.event_index]
        module_address = _module_key(ctx)
        rail_color = _module_color(module_address, color_map)
        fillcolor = "#98FB98" if _event_kept(trace, trace_index) else "#E6E6E6"
        dot.node(
            f"event_{ctx.event_index}",
            label=_graph_label(ctx, _event_kept(trace, trace_index), rail_color),
            shape="box",
            style="filled,rounded",
            fillcolor=fillcolor,
            tooltip=module_address,
        )
        if op_number > 0:
            dot.edge(f"event_{op_contexts[op_number - 1].event_index}", f"event_{ctx.event_index}")
    if vis_outpath is not None:
        dot.render(vis_outpath, format=vis_fileformat, cleanup=True, view=not vis_save_only)
    return dot.source


def summary(trace: RecordingTrace) -> str:
    """Return counts for a dry-run trace."""

    by_kind = Counter(ctx.kind for ctx in trace.events)
    by_op_type = Counter(str(ctx.layer_type or ctx.func_name or ctx.kind) for ctx in trace.events)
    kept = sum(1 for index, _ in enumerate(trace.events) if _event_kept(trace, index))
    lines = [
        f"RecordingTrace(total_events={len(trace.events)}, kept={kept}, rejected={len(trace.events) - kept})",
        "by_kind: " + ", ".join(f"{name}={count}" for name, count in sorted(by_kind.items())),
        "by_op_type: " + ", ".join(f"{name}={count}" for name, count in sorted(by_op_type.items())),
    ]
    if trace.predicate_failures:
        lines.append(f"predicate_failures={len(trace.predicate_failures)}")
    return "\n".join(lines)


def timeline_html(trace: RecordingTrace) -> Any:
    """Return a horizontal module-row timeline as ``IPython.display.HTML``."""

    from IPython.display import HTML

    module_rows: dict[str, list[RecordContext]] = defaultdict(list)
    for ctx in trace.events:
        module_rows[_module_key(ctx)].append(ctx)
    row_html: list[str] = []
    max_events = max((len(events) for events in module_rows.values()), default=1)
    for module_address, events in module_rows.items():
        tags = []
        for ctx in events:
            label = escape(str(ctx.layer_type or ctx.func_name or ctx.kind))
            tags.append(f'<span class="tl-fastlog-tag">{label}</span>')
        row_html.append(
            '<div class="tl-fastlog-row">'
            f'<div class="tl-fastlog-module">{escape(module_address)}</div>'
            f'<div class="tl-fastlog-events" style="grid-template-columns: repeat({max_events}, max-content);">'
            + "".join(tags)
            + "</div></div>"
        )
    html = (
        "<style>"
        ".tl-fastlog-row{display:grid;grid-template-columns:180px 1fr;gap:8px;margin:4px 0;"
        "font-family:system-ui,sans-serif;font-size:12px}"
        ".tl-fastlog-module{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}"
        ".tl-fastlog-events{display:grid;gap:4px;overflow-x:auto}"
        ".tl-fastlog-tag{display:inline-block;padding:2px 6px;border:1px solid #bbb;"
        "background:#f2f2f2;border-radius:4px;white-space:nowrap}"
        "</style><div>" + "".join(row_html) + "</div>"
    )
    return HTML(html)
