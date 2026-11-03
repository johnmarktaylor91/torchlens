"""Trace drawing, overlay, animation and preview functions (record rebuild destination).

Bodies of the ``Trace`` visualization verbs live here; ``Trace`` keeps one thin
method per verb that forwards to these functions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from html import escape
from typing import TYPE_CHECKING, Any, Literal, cast

from .._data_classes.module import Module
from .._deprecations import MISSING, MissingType
from .._errors import InvalidArgumentError
from .._literals import (
    BufferVisibilityLiteral,
    CollapseLiteral,
    FoldRepeatsLiteral,
    VisDirectionLiteral,
    VisInterventionModeLiteral,
    VisModeLiteral,
    VisNodeModeLiteral,
    VisNodePlacementLiteral,
    VisRendererLiteral,
)
from ..intervention.types import FireRecord
from ._render_dot import draw as _render_draw
from ._render_entrypoints import render_backward_graph, render_combined_graph
from .fastlog_preview import preview_fastlog as _preview_fastlog

if TYPE_CHECKING:
    from .._data_classes.trace import Trace
    from ..experimental.dagua._bridge import TorchLensRenderAudit
    from .code_panel import CodePanelOption


def _flatten_backward_fire_ref(value: Any) -> tuple[FireRecord, ...]:
    """Return the fire records stored on a backward call reference."""

    if isinstance(value, FireRecord):
        return (value,)
    if isinstance(value, tuple):
        return tuple(item for item in value if isinstance(item, FireRecord))
    return ()


def show(trace: Trace, kind: str | None = None, **kwargs: Any) -> str | None:
    """Body of ``Trace.show``; the method docstring is the contract."""

    if kwargs.get("vis_mode") == "none":
        return None
    if kind == "repr":
        return repr(trace)
    if kind == "html":
        return trace._repr_html_()
    return cast("str | None", trace.draw(**kwargs))


def draw(  # noqa: PLR0913 - mirrors the ratified public grammar of Trace.draw
    trace: Trace,
    view: VisModeLiteral | MissingType = MISSING,
    depth: int | MissingType = MISSING,
    renderer: VisRendererLiteral | MissingType = MISSING,
    layout: VisNodePlacementLiteral | MissingType = MISSING,
    node_style: VisNodeModeLiteral | MissingType = MISSING,
    vis_mode: VisModeLiteral = "unrolled",
    vis_call_depth: int = 1000,
    path: str | None = None,
    vis_graph_overrides: dict[str, Any] | None = None,
    module: Module | str | None = None,
    node_mode: VisNodeModeLiteral = "default",
    node_spec_fn: Callable[..., Any] | None = None,
    collapsed_node_spec_fn: Callable[..., Any] | None = None,
    collapse_fn: Callable[..., Any] | None = None,
    collapse: CollapseLiteral = "none",
    fold_repeats: FoldRepeatsLiteral = None,
    fold_patterns: Any = None,
    skip_fn: Callable[..., Any] | None = None,
    vis_edge_overrides: dict[str, Any] | None = None,
    vis_grad_edge_overrides: dict[str, Any] | None = None,
    vis_module_overrides: dict[str, Any] | None = None,
    vis_save_only: bool = False,
    file_format: str = "pdf",
    show_buffers: BufferVisibilityLiteral = "meaningful",
    direction: VisDirectionLiteral = "bottomup",
    vis_node_placement: VisNodePlacementLiteral = "auto",
    vis_renderer: VisRendererLiteral = "graphviz",
    theme: str = "torchlens",
    vis_intervention_mode: VisInterventionModeLiteral = "node_mark",
    vis_show_cone: bool = False,
    code_panel: CodePanelOption = False,
    node_overlay: str | Mapping[str, Any] | Callable[[Any], Any] | None = None,
    node_label_fields: list[str] | None = None,
    show_legend: bool | None = None,
    font_size: int | None = None,
    dpi: int | None = None,
    for_paper: bool = False,
    return_graph: bool = False,
    order_siblings: bool = True,
    show_containers: Literal[False, "labels", "collapsed", "nodes"] = False,
    container_max_inline: int = 12,
    show_input_transform_summary: bool = False,
    show_orphans: bool = False,  # Invariants support flipping this; owner visual review pending.
    *,
    color_by: str | Callable[[Any], Any] | None = None,
    size_by: str | Callable[[Any], Any] | None = None,
    scale: str | None = None,
    stack_by: str | bool | Callable[[Any], Any] | None = None,
    show_redundant_args: bool = False,
    show_saved_for_backward: bool = False,
) -> Any:
    """Body of ``Trace.draw``; the method docstring is the contract."""

    if view is not MISSING:
        vis_mode = cast(VisModeLiteral, view)
    if depth is not MISSING:
        vis_call_depth = cast(int, depth)
    if renderer is not MISSING:
        vis_renderer = cast(VisRendererLiteral, renderer)
    if layout is not MISSING:
        vis_node_placement = cast(VisNodePlacementLiteral, layout)
    if node_style is not MISSING:
        node_mode = cast(VisNodeModeLiteral, node_style)
    if vis_mode not in ("none", "rolled", "unrolled"):
        # Name the spelling the caller used and every accepted value; the
        # renderer-level check downstream never sees "none" or "view".
        mode_argument = "view" if view is not MISSING else "vis_mode"
        raise InvalidArgumentError(
            f"{mode_argument} must be 'none', 'rolled', or 'unrolled'; received {vis_mode!r}",
            code="visualization_mode_invalid",
            remedy=f"pass {mode_argument}='none', 'rolled', or 'unrolled'",
            argument=mode_argument,
        )
    if vis_mode == "none":
        return None

    return _render_draw(
        trace,
        vis_mode=vis_mode,
        vis_call_depth=vis_call_depth,
        vis_outpath=path,
        vis_graph_overrides=vis_graph_overrides,
        module=module,
        node_mode=node_mode,
        node_spec_fn=node_spec_fn,
        collapsed_node_spec_fn=collapsed_node_spec_fn,
        collapse_fn=collapse_fn,
        collapse=collapse,
        fold_repeats=fold_repeats,
        fold_patterns=fold_patterns,
        skip_fn=skip_fn,
        vis_edge_overrides=vis_edge_overrides,
        vis_grad_edge_overrides=vis_grad_edge_overrides,
        vis_module_overrides=vis_module_overrides,
        vis_save_only=vis_save_only,
        vis_fileformat=file_format,
        show_buffer_layers=show_buffers,
        direction=direction,
        vis_node_placement=vis_node_placement,
        vis_renderer=vis_renderer,
        vis_theme=theme,
        vis_intervention_mode=vis_intervention_mode,
        vis_show_cone=vis_show_cone,
        code_panel=code_panel,
        node_overlay=node_overlay,
        node_label_fields=node_label_fields,
        show_legend=show_legend,
        font_size=font_size,
        dpi=dpi,
        for_paper=for_paper,
        return_graph=return_graph,
        order_siblings=order_siblings,
        show_containers=show_containers,
        container_max_inline=container_max_inline,
        show_input_transform_summary=show_input_transform_summary,
        show_orphans=show_orphans,
        color_by=color_by,
        size_by=size_by,
        scale=scale,
        stack_by=stack_by,
        show_redundant_args=show_redundant_args,
        show_saved_for_backward=show_saved_for_backward,
    )


def animate(trace: Trace, site: Any) -> str:
    """Body of ``Trace.animate``; the method docstring is the contract."""

    label = str(getattr(site, "layer_label", site))
    base_label = label.split(":", 1)[0]
    if base_label not in trace._layer_logs:
        raise KeyError(f"Unknown layer site {label!r}.")
    layer = trace._layer_logs[base_label]
    pass_entries = list(getattr(layer, "ops", ()) or [])
    if not pass_entries:
        pass_entries = [trace[label]]
    frames = [
        {
            "pass": int(getattr(entry, "pass_number", index + 1) or index + 1),
            "label": str(getattr(entry, "layer_label", base_label)),
            "shape": "x".join(str(dim) for dim in getattr(entry, "shape", ()) or ()),
            "memory": str(getattr(entry, "activation_memory", "")),
        }
        for index, entry in enumerate(pass_entries)
    ]
    frame_markup = "".join(
        "<li data-frame='{idx}'>{label} pass {call_index}: {shape} {memory}</li>".format(
            idx=index,
            label=escape(str(frame["label"])),
            call_index=frame["pass"],
            shape=escape(str(frame["shape"] or "scalar")),
            memory=escape(str(frame["memory"])),
        )
        for index, frame in enumerate(frames)
    )
    return (
        "<div class='tl-pass-animation' data-site='"
        + escape(base_label)
        + "'><button type='button' data-action='play'>Play</button>"
        + "<button type='button' data-action='pause'>Pause</button><ol>"
        + frame_markup
        + "</ol><script>(function(){var root=document.currentScript.parentElement;"
        + "var items=root.querySelectorAll('li');var i=0,t=null;"
        + "function show(){items.forEach(function(x,j){x.style.display=j===i?'':'none';});}"
        + "show();root.querySelector('[data-action=play]').onclick=function(){"
        + "if(t)return;t=setInterval(function(){i=(i+1)%items.length;show();},500);};"
        + "root.querySelector('[data-action=pause]').onclick=function(){clearInterval(t);t=null;};"
        + "})();</script></div>"
    )


def draw_backward(  # noqa: PLR0913 - mirrors the ratified public grammar of Trace.draw_backward
    trace: Trace,
    path: str = "backward_modelgraph",
    vis_graph_overrides: dict[str, Any] | None = None,
    node_spec_fn: Callable[..., Any] | None = None,
    collapsed_node_spec_fn: Callable[..., Any] | None = None,
    vis_node_mode: VisNodeModeLiteral = "default",
    vis_edge_overrides: dict[str, Any] | None = None,
    vis_save_only: bool = False,
    file_format: str = "pdf",
    vis_direction: VisDirectionLiteral = "topdown",
    code_panel: CodePanelOption = False,
    vis_mode: VisModeLiteral = "rolled",
    bwd: int | Iterable[int] | None = None,
    show_legend: bool | None = None,
    *,
    theme: str = "torchlens",
    show_module_paths: bool = False,
    show_grad_values: bool = False,
) -> str:
    """Body of ``Trace.draw_backward``; the method docstring is the contract."""

    return render_backward_graph(
        trace,
        vis_outpath=path,
        vis_graph_overrides=vis_graph_overrides,
        node_spec_fn=node_spec_fn,
        collapsed_node_spec_fn=collapsed_node_spec_fn,
        vis_node_mode=vis_node_mode,
        vis_edge_overrides=vis_edge_overrides,
        vis_save_only=vis_save_only,
        vis_fileformat=file_format,
        direction=vis_direction,
        code_panel=code_panel,
        vis_mode=vis_mode,
        bwd=bwd,
        show_legend=show_legend,
        theme=theme,
        show_module_paths=show_module_paths,
        show_grad_values=show_grad_values,
    )


def draw_combined(  # noqa: PLR0913 - mirrors the ratified public grammar of Trace.draw_combined
    trace: Trace,
    path: str = "combined_modelgraph",
    vis_graph_overrides: dict[str, Any] | None = None,
    node_spec_fn: Callable[..., Any] | None = None,
    backward_node_spec_fn: Callable[..., Any] | None = None,
    vis_edge_overrides: dict[str, Any] | None = None,
    vis_save_only: bool = False,
    file_format: str = "pdf",
    vis_direction: VisDirectionLiteral = "leftright",
    vis_mode: VisModeLiteral = "unrolled",
    intervening_cluster: Literal["upstream", "outside", "downstream", "own"] = "upstream",
    show_buffers: BufferVisibilityLiteral = "meaningful",
    bwd: int | Iterable[int] | None = None,
    show_legend: bool | None = None,
) -> str:
    """Body of ``Trace.draw_combined``; the method docstring is the contract."""

    return render_combined_graph(
        trace,
        vis_outpath=path,
        vis_graph_overrides=vis_graph_overrides,
        node_spec_fn=node_spec_fn,
        backward_node_spec_fn=backward_node_spec_fn,
        vis_edge_overrides=vis_edge_overrides,
        vis_save_only=vis_save_only,
        vis_fileformat=file_format,
        direction=vis_direction,
        vis_mode=vis_mode,
        intervening_cluster=intervening_cluster,
        show_buffer_layers=show_buffers,
        bwd=bwd,
        show_legend=show_legend,
    )


def preview_fastlog(
    trace: Trace,
    predicate: Callable[..., Any] | None = None,
    keep_op: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> str:
    """Body of ``Trace.preview_fastlog``; the method docstring is the contract."""

    return _preview_fastlog(
        trace,
        predicate=predicate,
        keep_op=keep_op,
        **kwargs,
    )


def last_run_records(trace: Trace) -> tuple[FireRecord, ...]:
    """Body of ``Trace.last_run_records``; the method docstring is the contract."""

    ctx = getattr(trace, "last_run", None)
    if not isinstance(ctx, dict):
        return ()
    timestamp = ctx.get("timestamp")
    if not isinstance(timestamp, (int, float)):
        return ()
    records = []
    for layer in ([] if trace.__dict__.get("_tl_cleaned_up", False) else trace._layer_list) or []:
        for record in getattr(layer, "edits", []) or []:
            record_timestamp = getattr(record, "timestamp", None)
            if isinstance(record_timestamp, (int, float)) and record_timestamp >= timestamp:
                records.append(record)
    for grad_fn in (
        {} if trace.__dict__.get("_tl_cleaned_up", False) else trace._grad_fn_logs
    ).values():
        calls = getattr(getattr(grad_fn, "calls", None), "_list", [])
        for call in calls:
            for record in _flatten_backward_fire_ref(getattr(call, "fire_record", None)):
                record_timestamp = getattr(record, "timestamp", None)
                if isinstance(record_timestamp, (int, float)) and record_timestamp >= timestamp:
                    records.append(record)
    return tuple(records)


def render_dagua_graph(  # noqa: PLR0913 - mirrors the ratified public grammar of Trace.render_dagua_graph
    trace: Trace,
    vis_mode: str = "unrolled",
    vis_call_depth: int = 1000,
    path: str = "graph.gv",
    vis_save_only: bool = False,
    file_format: str = "pdf",
    show_buffers: bool = False,
    vis_direction: str = "bottomup",
    theme: str = "torchlens",
) -> str:
    """Body of ``Trace.render_dagua_graph``; the method docstring is the contract."""
    from ..experimental.dagua._bridge import render_trace_with_dagua as _impl

    return cast(
        str,
        _impl(
            trace,
            vis_mode=vis_mode,
            vis_call_depth=vis_call_depth,
            path=path,
            vis_save_only=vis_save_only,
            file_format=file_format,
            show_buffers=show_buffers,
            vis_direction=vis_direction,
            theme=theme,
        ),
    )


def to_dagua_graph(  # noqa: PLR0913 - mirrors the ratified public grammar of Trace.to_dagua_graph
    trace: Trace,
    vis_mode: str = "unrolled",
    vis_call_depth: int = 1000,
    show_buffers: bool = False,
    direction: str = "bottomup",
    include_grad_edges: bool | None = None,
) -> Any:
    """Body of ``Trace.to_dagua_graph``; the method docstring is the contract."""
    from ..experimental.dagua import trace_to_dagua_graph as _impl

    return _impl(
        trace,
        vis_mode=vis_mode,
        vis_call_depth=vis_call_depth,
        show_buffers=show_buffers,
        direction=direction,
        include_grad_edges=include_grad_edges,
    )


def visualization_field_audit(trace: Trace) -> TorchLensRenderAudit:
    """Body of ``Trace.visualization_field_audit``; the method docstring is the contract."""
    from ..experimental.dagua._bridge import build_render_audit as _impl

    return _impl(trace)


__all__ = [
    "animate",
    "draw",
    "draw_backward",
    "draw_combined",
    "last_run_records",
    "preview_fastlog",
    "render_dagua_graph",
    "show",
    "to_dagua_graph",
    "visualization_field_audit",
]
