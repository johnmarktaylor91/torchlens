"""Trace drawing, overlay, animation and preview functions (record rebuild destination).

Bodies of the ``Trace`` visualization verbs live here; ``Trace`` keeps one thin
method per verb that forwards to these functions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .._data_classes.module import Module
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
from ..viz._drawing import Drawing, resolve_draw_target
from ._color_sources import resolve_color_source
from ._pass_animation import pass_animation_html
from ._render_dot import draw as _render_draw
from ._render_entrypoints import render_backward_graph, render_combined_graph
from ._render_utils import _KNOWN_EXTS, DEFAULT_DRAW_STEM, collision_safe_outpath
from ._trace_draw_arguments import _forward_render_arguments
from .fastlog_preview import preview_fastlog as _preview_fastlog
from .request import ShowContainersLiteral

if TYPE_CHECKING:
    from .._data_classes.trace import Trace
    from .code_panel import CodePanelOption

GraphLiteral = Literal["forward", "backward", "combined"]


def _flatten_backward_fire_ref(value: Any) -> tuple[FireRecord, ...]:
    """Return the fire records stored on a backward call reference."""

    if isinstance(value, FireRecord):
        return (value,)
    if isinstance(value, tuple):
        return tuple(item for item in value if isinstance(item, FireRecord))
    return ()


def show(trace: Trace, kind: str | None = None, **kwargs: Any) -> Any:
    """Body of ``Trace.show``; the method docstring is the contract."""

    if kind == "repr":
        return repr(trace)
    if kind == "html":
        return trace._repr_html_()
    return trace.draw(**kwargs)


def _render_backward(  # noqa: PLR0913 - the backward branch of the one draw door
    trace: Trace,
    *,
    outpath: str,
    vis_mode: VisModeLiteral,
    direction: VisDirectionLiteral | None,
    graph_overrides: dict[str, Any] | None,
    node_spec_fn: Callable[..., Any] | None,
    collapsed_node_spec_fn: Callable[..., Any] | None,
    node_style: VisNodeModeLiteral,
    edge_overrides: dict[str, Any] | None,
    open_viewer: bool,
    file_format: str,
    code_panel: CodePanelOption,
    bwd: int | Iterable[int] | None,
    show_legend: bool | None,
    theme: str,
    show_module_paths: bool,
    show_grad_values: bool,
) -> Any:
    """Render ``graph="backward"`` for the draw door in the backward renderer's spellings."""
    return render_backward_graph(
        trace,
        vis_outpath=outpath,
        vis_graph_overrides=graph_overrides,
        node_spec_fn=node_spec_fn,
        collapsed_node_spec_fn=collapsed_node_spec_fn,
        vis_node_mode=node_style,
        vis_edge_overrides=edge_overrides,
        vis_save_only=not open_viewer,
        vis_fileformat=file_format,
        direction="topdown" if direction is None else direction,
        code_panel=code_panel,
        vis_mode=vis_mode,
        bwd=bwd,
        show_legend=show_legend,
        theme=theme,
        show_module_paths=show_module_paths,
        show_grad_values=show_grad_values,
    )


def draw(  # noqa: PLR0913 - the one draw door: every VisualizationOptions field is a keyword
    trace: Trace,
    path: str | None = None,
    *,
    graph: GraphLiteral = "forward",
    rolled: bool = False,
    depth: int = 1000,
    module: Module | str | None = None,
    collapse: CollapseLiteral = "none",
    fold_repeats: FoldRepeatsLiteral = None,
    fold_patterns: Any = None,
    theme: str = "torchlens",
    layout: VisNodePlacementLiteral = "auto",
    renderer: VisRendererLiteral = "graphviz",
    direction: VisDirectionLiteral | None = None,
    node_style: VisNodeModeLiteral = "default",
    show_buffers: BufferVisibilityLiteral = "meaningful",
    show_legend: bool | None = None,
    color_by: Any = None,
    size_by: str | Callable[[Any], Any] | None = None,
    size_scale: str | None = None,
    stack_by: str | bool | Callable[[Any], Any] | None = None,
    node_overlay: str | Mapping[str, Any] | Callable[[Any], Any] | None = None,
    node_label_fields: list[str] | None = None,
    graph_overrides: dict[str, Any] | None = None,
    edge_overrides: dict[str, Any] | None = None,
    grad_edge_overrides: dict[str, Any] | None = None,
    module_overrides: dict[str, Any] | None = None,
    node_spec_fn: Callable[..., Any] | None = None,
    collapsed_node_spec_fn: Callable[..., Any] | None = None,
    collapse_fn: Callable[..., Any] | None = None,
    skip_fn: Callable[..., Any] | None = None,
    intervention_mode: VisInterventionModeLiteral = "node_mark",
    show_cone: bool = False,
    code_panel: CodePanelOption = False,
    font_size: int | None = None,
    dpi: int | None = None,
    order_siblings: bool = True,
    show_containers: ShowContainersLiteral = False,
    container_max_inline: int = 12,
    show_input_transform_summary: bool = False,
    show_orphans: bool = False,
    show_redundant_args: bool = False,
    show_saved_for_backward: bool = False,
    bwd: int | Iterable[int] | None = None,
    show_module_paths: bool = False,
    show_grad_values: bool = False,
    backward_node_spec_fn: Callable[..., Any] | None = None,
    intervening_cluster: Literal["upstream", "outside", "downstream", "own"] = "upstream",
    file_format: str | None = None,
    open_viewer: bool = False,
) -> Drawing:
    """Body of ``Trace.draw``; the method docstring is the contract."""

    if graph not in ("forward", "backward", "combined"):
        raise InvalidArgumentError(
            f"graph must be 'forward', 'backward', or 'combined'; received {graph!r}",
            code="visualization_mode_invalid",
            remedy="pass graph='forward', 'backward', or 'combined'",
            argument="graph",
        )
    num_ops = len(trace) if hasattr(trace, "__len__") else None
    policy: dict[str, Any] = {
        "graph": graph,
        "rolled": rolled,
        "theme": theme,
        "collapse": collapse,
        "module": module,
        "depth": depth,
    }
    target = resolve_draw_target(path, file_format, _KNOWN_EXTS)
    path, file_format = target.path, target.file_format
    vis_mode: VisModeLiteral = "rolled" if rolled else "unrolled"
    if graph == "backward":
        outpath = "backward_modelgraph" if path is None else path
        rendered: Any = _render_backward(
            trace,
            outpath=outpath,
            vis_mode=vis_mode,
            direction=direction,
            graph_overrides=graph_overrides,
            node_spec_fn=node_spec_fn,
            collapsed_node_spec_fn=collapsed_node_spec_fn,
            node_style=node_style,
            edge_overrides=edge_overrides,
            open_viewer=open_viewer,
            file_format=file_format,
            code_panel=code_panel,
            bwd=bwd,
            show_legend=show_legend,
            theme=theme,
            show_module_paths=show_module_paths,
            show_grad_values=show_grad_values,
        )
    elif graph == "combined":
        outpath = "combined_modelgraph" if path is None else path
        rendered = render_combined_graph(
            trace,
            vis_outpath=outpath,
            vis_graph_overrides=graph_overrides,
            node_spec_fn=node_spec_fn,
            backward_node_spec_fn=backward_node_spec_fn,
            vis_edge_overrides=edge_overrides,
            vis_save_only=not open_viewer,
            vis_fileformat=file_format,
            direction="leftright" if direction is None else direction,
            vis_mode=vis_mode,
            intervening_cluster=intervening_cluster,
            show_buffer_layers=show_buffers,
            bwd=bwd,
            show_legend=show_legend,
        )
    else:
        outpath = collision_safe_outpath(DEFAULT_DRAW_STEM, file_format) if path is None else path
        rendered = _render_draw(trace, **_forward_render_arguments(locals(), resolve_color_source))
    return target.finish(str(rendered), str(outpath), num_ops=num_ops, policy=policy)


def animate(
    trace: Trace,
    site: Any,
    path: str | None = None,
    *,
    open_viewer: bool = False,
) -> Drawing:
    """Body of ``Trace.animate``; the method docstring is the contract."""

    html = pass_animation_html(trace, site)
    num_ops = len(trace) if hasattr(trace, "__len__") else None
    drawing = Drawing(
        dot="",
        file_format="html",
        data=html.encode("utf-8"),
        num_ops=num_ops,
        policy={"animate": str(getattr(site, "layer_label", site))},
    )
    if path is not None:
        target = Path(path)
        if target.suffix.lower() != ".html":
            target = target.with_name(target.name + ".html")
        drawing.path = drawing.save(target)
    if open_viewer:
        drawing.show()
    return drawing


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


__all__ = [
    "animate",
    "draw",
    "last_run_records",
    "preview_fastlog",
    "show",
    "to_dagua_graph",
]
