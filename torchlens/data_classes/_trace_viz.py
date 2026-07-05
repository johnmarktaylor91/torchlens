# mypy: ignore-errors
"""Trace visualization mixin."""

from collections.abc import Iterable, Mapping
from html import escape
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, cast

import torch

if TYPE_CHECKING:
    from ..experimental.dagua._bridge import TorchLensRenderAudit
    from ..intervention.types import FireRecord
    from ..visualization.code_panel import CodePanelOption
    from .trace import Trace
from .._deprecations import MISSING, MissingType
from .._literals import (
    BufferVisibilityLiteral,
    CollapseLiteral,
    FoldRunsLiteral,
    VisDirectionLiteral,
    VisInterventionModeLiteral,
    VisModeLiteral,
    VisNodeModeLiteral,
    VisNodePlacementLiteral,
    VisRendererLiteral,
)
from .._source_links import file_line_text, terminal_file_line_link, vscode_file_line_link
from ..intervention.types import FireRecord
from .module import Module


def _flatten_backward_fire_ref(value: Any) -> tuple["FireRecord", ...]:
    """Return fire records stored on a backward call reference.

    Parameters
    ----------
    value:
        FireRecord, tuple of records, or another value.

    Returns
    -------
    tuple[FireRecord, ...]
        Fire records contained in ``value``.
    """

    if isinstance(value, FireRecord):
        return (value,)
    if isinstance(value, tuple):
        return tuple(item for item in value if isinstance(item, FireRecord))
    return ()


class TraceVisualizationMixin:
    def show(self, method: str = "graph", **kwargs: Any) -> str | None:
        """Render this trace using a lightweight notebook-friendly dispatcher.

        Parameters
        ----------
        method:
            Display method. ``"graph"`` delegates to :meth:`draw`, ``"repr"``
            returns ``repr(self)``, and ``"html"`` returns ``_repr_html_()``.
        **kwargs:
            Visualization keyword arguments forwarded to :meth:`draw`. The
            legacy ``vis_opt`` spelling is accepted as an alias for ``vis_mode``.

        Returns
        -------
        str | None
            Rendered representation, Graphviz DOT source, or ``None`` when
            rendering is explicitly disabled with ``vis_opt="none"`` or
            ``vis_mode="none"``.
        """

        vis_opt = kwargs.pop("vis_opt", None)
        if vis_opt is not None and "vis_mode" not in kwargs:
            kwargs["vis_mode"] = vis_opt
        if kwargs.get("vis_mode") == "none":
            return None
        if method == "repr":
            return repr(self)
        if method == "html":
            return self._repr_html_()
        return cast("str | None", self.draw(**kwargs))

    def draw(
        self,
        vis_opt: VisModeLiteral | MissingType = MISSING,
        view: VisModeLiteral | MissingType = MISSING,
        depth: int | MissingType = MISSING,
        renderer: VisRendererLiteral | MissingType = MISSING,
        layout: VisNodePlacementLiteral | MissingType = MISSING,
        node_style: VisNodeModeLiteral | MissingType = MISSING,
        vis_mode: VisModeLiteral = "unrolled",
        vis_call_depth: int = 1000,
        vis_outpath: str = "modelgraph",
        vis_graph_overrides: Optional[Dict[str, Any]] = None,
        module: "Module | str | None" = None,
        node_mode: VisNodeModeLiteral = "default",
        vis_node_mode: VisNodeModeLiteral | MissingType = MISSING,
        node_spec_fn: Optional[Callable[..., Any]] = None,
        collapsed_node_spec_fn: Optional[Callable[..., Any]] = None,
        collapse_fn: Optional[Callable[..., Any]] = None,
        collapse: CollapseLiteral = "none",
        fold_runs: FoldRunsLiteral = None,
        skip_fn: Optional[Callable[..., Any]] = None,
        vis_edge_overrides: Optional[Dict[str, Any]] = None,
        vis_grad_edge_overrides: Optional[Dict[str, Any]] = None,
        vis_module_overrides: Optional[Dict[str, Any]] = None,
        vis_save_only: bool = False,
        vis_fileformat: str = "pdf",
        vis_buffers: BufferVisibilityLiteral | bool | MissingType = MISSING,
        show_buffer_layers: BufferVisibilityLiteral | bool = "meaningful",
        vis_direction: VisDirectionLiteral | MissingType = MISSING,
        direction: VisDirectionLiteral = "bottomup",
        vis_node_placement: VisNodePlacementLiteral = "auto",
        vis_renderer: VisRendererLiteral = "graphviz",
        vis_theme: str = "torchlens",
        vis_intervention_mode: VisInterventionModeLiteral = "node_mark",
        vis_show_cone: bool = True,
        code_panel: "CodePanelOption" = False,
        node_overlay: str | Mapping[str, Any] | None = None,
        node_label_fields: list[str] | None = None,
        show_legend: bool = False,
        font_size: int | None = None,
        dpi: int | None = None,
        for_paper: bool = False,
        return_graph: bool = False,
        order_siblings: bool = True,
        show_containers: Literal[False, "labels", "cluster", "collapsed", "auto"] = False,
        container_max_inline: int = 12,
        show_input_transform_summary: bool = False,
    ) -> Any:
        """Render the computational graph for this model log.

        Parameters
        ----------
        vis_mode, vis_call_depth, vis_outpath, vis_graph_overrides, module, node_mode, \
        node_spec_fn, collapsed_node_spec_fn, collapse_fn, skip_fn, vis_edge_overrides, \
        vis_grad_edge_overrides, vis_module_overrides, vis_save_only, vis_fileformat, \
        show_buffer_layers, direction, vis_node_placement, vis_renderer, vis_theme, \
        vis_intervention_mode, vis_show_cone, code_panel, order_siblings, show_containers,
        container_max_inline, show_input_transform_summary:
            Forwarded unchanged to :func:`torchlens.visualization.rendering.draw`.
            ``show_buffer_layers`` accepts ``"never"``, ``"meaningful"``, or
            ``"always"``. Legacy bools are deprecated but supported by the
            Graphviz renderer.
        collapse:
            Smart module-collapse mode. ``"none"`` preserves the full graph,
            ``"auto"`` uses the v2 readability-targeted engine, and ``"max"``
            aggressively condenses eligible modules and segment boxes. A float
            in ``[0.0, 1.0]`` selects a deterministic monotone schedule:
            ``0.0`` is equivalent to ``"none"``, ``1.0`` is equivalent to
            ``"max"``, and larger values never increase the visible node count
            or uncollapse a collapsed unit. ``"auto"`` is the first schedule
            point whose visible count enters the readable band, while its
            current implementation remains unchanged for compatibility.
        fold_runs:
            Run-fold policy. ``None`` preserves the default policy: off for
            ``collapse="none"`` and band-pressure two-pass folding for
            ``"auto"``/``"max"``. ``True`` folds every eligible repeated run,
            including standalone folding with ``collapse="none"``. ``False``
            disables run folding.

        Returns
        -------
        Any
            Graphviz DOT source, renderer-specific output, or renderer object
            when ``return_graph=True``.
        """
        from ..visualization.rendering import draw as _impl

        if vis_opt is not MISSING:
            vis_mode = cast(VisModeLiteral, vis_opt)
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
        if vis_node_mode is not MISSING:
            node_mode = cast(VisNodeModeLiteral, vis_node_mode)
        if vis_buffers is not MISSING:
            show_buffer_layers = cast(BufferVisibilityLiteral | bool, vis_buffers)
        if vis_direction is not MISSING:
            direction = cast(VisDirectionLiteral, vis_direction)

        return _impl(
            self,
            vis_mode=vis_mode,
            vis_call_depth=vis_call_depth,
            vis_outpath=vis_outpath,
            vis_graph_overrides=vis_graph_overrides,
            module=module,
            node_mode=node_mode,
            node_spec_fn=node_spec_fn,
            collapsed_node_spec_fn=collapsed_node_spec_fn,
            collapse_fn=collapse_fn,
            collapse=collapse,
            fold_runs=fold_runs,
            skip_fn=skip_fn,
            vis_edge_overrides=vis_edge_overrides,
            vis_grad_edge_overrides=vis_grad_edge_overrides,
            vis_module_overrides=vis_module_overrides,
            vis_save_only=vis_save_only,
            vis_fileformat=vis_fileformat,
            show_buffer_layers=show_buffer_layers,
            direction=direction,
            vis_node_placement=vis_node_placement,
            vis_renderer=vis_renderer,
            vis_theme=vis_theme,
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
        )

    def add_node_overlay(
        self,
        scores: Mapping[str, Any],
        *,
        name: str = "overlay",
    ) -> "Trace":
        """Register external per-node overlay scores for later rendering.

        Parameters
        ----------
        scores:
            Mapping from layer labels to scalar or displayable values.
        name:
            Overlay name stored on the log for discoverability.

        Returns
        -------
        Trace
            This log, allowing chained calls before ``draw``.
        """

        self._node_overlay_scores = dict(scores)
        self._node_overlay_name = name
        return self

    def animate_ops(self, site: Any) -> str:
        """Return a minimal HTML animation for repeated ops at ``site``.

        Parameters
        ----------
        site:
            Layer label, pass-qualified layer label, or object with a
            ``layer_label`` attribute.

        Returns
        -------
        str
            Self-contained HTML fragment with play/pause controls.

        Raises
        ------
        KeyError
            If the requested site cannot be resolved.
        """

        label = str(getattr(site, "layer_label", site))
        base_label = label.split(":", 1)[0]
        if base_label not in self.layer_logs:
            raise KeyError(f"Unknown layer site {label!r}.")
        layer = self.layer_logs[base_label]
        pass_entries = list(getattr(layer, "ops", ()) or [])
        if not pass_entries:
            pass_entries = [self[label]]
        frames = [
            {
                "pass": int(getattr(entry, "pass_index", index + 1) or index + 1),
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

    def first_nonfinite(self, link_format: Literal["terminal", "html", "text"] = "terminal") -> str:
        """Return a text answer describing the first saved non-finite out.

        Parameters
        ----------
        link_format:
            Source-location link style. ``"terminal"`` emits OSC 8 hyperlinks,
            ``"html"`` emits VS Code URI anchors, and ``"text"`` emits plain
            ``path:line`` text.

        Returns
        -------
        str
            Human-readable single-paragraph answer naming the layer, operation,
            module, shape, dtype, parents, and source location.
        """

        for layer in getattr(self, "layer_list", []) or []:
            out = getattr(layer, "out", None)
            if not isinstance(out, torch.Tensor) or out.numel() == 0:
                continue
            try:
                has_nonfinite = bool((~torch.isfinite(out.detach())).any().item())
            except (RuntimeError, TypeError):
                continue
            if not has_nonfinite:
                continue
            stack = getattr(layer, "code_context", None) or []
            location = "source unavailable"
            if stack:
                frame = stack[0]
                file_path = str(getattr(frame, "file", "unknown"))
                line_number = getattr(frame, "line_number", "unknown")
                if link_format == "terminal":
                    location = terminal_file_line_link(file_path, line_number)
                elif link_format == "html":
                    location = vscode_file_line_link(file_path, line_number)
                elif link_format == "text":
                    location = file_line_text(file_path, line_number)
                else:
                    raise ValueError("link_format must be 'terminal', 'html', or 'text'.")
            parents = ", ".join(getattr(layer, "parents", None) or []) or "none"
            module = getattr(layer, "module", None) or "no module"
            return (
                f"First non-finite saved out is in layer {layer.layer_label} "
                f"(op {getattr(layer, 'func_name', 'unknown')}, module {module}), "
                f"shape={getattr(layer, 'shape', None)}, "
                f"dtype={getattr(layer, 'dtype', None)}, parents={parents}, "
                f"source={location}."
            )
        return "No non-finite tensor values found in saved outs."

    def draw_backward(
        self,
        vis_outpath: str = "backward_modelgraph",
        vis_graph_overrides: Optional[Dict[str, Any]] = None,
        node_spec_fn: Optional[Callable[..., Any]] = None,
        collapsed_node_spec_fn: Optional[Callable[..., Any]] = None,
        vis_node_mode: VisNodeModeLiteral = "default",
        vis_edge_overrides: Optional[Dict[str, Any]] = None,
        vis_save_only: bool = False,
        vis_fileformat: str = "pdf",
        vis_direction: VisDirectionLiteral = "topdown",
        code_panel: "CodePanelOption" = False,
        vis_mode: VisModeLiteral = "rolled",
        bwd: int | Iterable[int] | None = None,
    ) -> str:
        """Render the captured backward grad_fn_handle graph.

        Parameters
        ----------
        vis_outpath, vis_graph_overrides, node_spec_fn, collapsed_node_spec_fn, \
        vis_node_mode, vis_edge_overrides, vis_save_only, vis_fileformat, \
        vis_direction, code_panel, vis_mode, bwd:
            Forwarded unchanged to
            :func:`torchlens.visualization.rendering.render_backward_graph`.
            ``collapsed_node_spec_fn`` and ``vis_node_mode`` are accepted for
            forward-visualization API symmetry but are not applied because
            backward graphs do not render collapsed module nodes.

        Returns
        -------
        str
            Graphviz DOT source.
        """
        from ..visualization.rendering import render_backward_graph as _impl

        return _impl(
            self,
            vis_outpath=vis_outpath,
            vis_graph_overrides=vis_graph_overrides,
            node_spec_fn=node_spec_fn,
            collapsed_node_spec_fn=collapsed_node_spec_fn,
            vis_node_mode=vis_node_mode,
            vis_edge_overrides=vis_edge_overrides,
            vis_save_only=vis_save_only,
            vis_fileformat=vis_fileformat,
            direction=vis_direction,
            code_panel=code_panel,
            vis_mode=vis_mode,
            bwd=bwd,
        )

    def draw_combined(
        self,
        vis_outpath: str = "combined_modelgraph",
        vis_graph_overrides: Optional[Dict[str, Any]] = None,
        node_spec_fn: Optional[Callable[..., Any]] = None,
        backward_node_spec_fn: Optional[Callable[..., Any]] = None,
        vis_edge_overrides: Optional[Dict[str, Any]] = None,
        vis_save_only: bool = False,
        vis_fileformat: str = "pdf",
        vis_direction: VisDirectionLiteral = "leftright",
        vis_mode: VisModeLiteral = "unrolled",
        intervening_cluster: Literal["upstream", "outside", "downstream", "own"] = "upstream",
        show_buffer_layers: BufferVisibilityLiteral | bool = "meaningful",
        bwd: int | Iterable[int] | None = None,
    ) -> str:
        """Render forward ops and backward grad_fns in one graph.

        Parameters
        ----------
        vis_outpath, vis_graph_overrides, node_spec_fn, backward_node_spec_fn, \
        vis_edge_overrides, vis_save_only, vis_fileformat, vis_direction, \
        vis_mode, intervening_cluster, show_buffer_layers, bwd:
            Forwarded unchanged to
            :func:`torchlens.visualization.rendering.render_combined_graph`.

        Returns
        -------
        str
            Graphviz DOT source.
        """
        from ..visualization.rendering import render_combined_graph as _impl

        return _impl(
            self,
            vis_outpath=vis_outpath,
            vis_graph_overrides=vis_graph_overrides,
            node_spec_fn=node_spec_fn,
            backward_node_spec_fn=backward_node_spec_fn,
            vis_edge_overrides=vis_edge_overrides,
            vis_save_only=vis_save_only,
            vis_fileformat=vis_fileformat,
            direction=vis_direction,
            vis_mode=vis_mode,
            intervening_cluster=intervening_cluster,
            show_buffer_layers=show_buffer_layers,
            bwd=bwd,
        )

    def preview_fastlog(
        self,
        predicate: Optional[Callable[..., Any]] = None,
        keep_op: Optional[Callable[..., Any]] = None,
        keep_module: Optional[Callable[..., Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Render a fastlog predicate preview for this model graph.

        Parameters
        ----------
        predicate, keep_op, keep_module:
            Predicate callables that receive synthesized fastlog ``RecordContext``
            objects.
        **kwargs:
            Forwarded to :func:`torchlens.visualization.fastlog_preview.preview_fastlog`.

        Returns
        -------
        str
            Graphviz DOT source.
        """

        from ..visualization.fastlog_preview import preview_fastlog as _impl

        return _impl(
            self,
            predicate=predicate,
            keep_op=keep_op,
            keep_module=keep_module,
            **kwargs,
        )

    def last_run_records(self) -> tuple["FireRecord", ...]:
        """Return fire records from the most recent replay, rerun, or live capture.

        Returns
        -------
        tuple[FireRecord, ...]
            Immutable snapshot of matching intervention fire records.
        """

        ctx = getattr(self, "last_run", None)
        if not isinstance(ctx, dict):
            return ()
        timestamp = ctx.get("timestamp")
        if not isinstance(timestamp, (int, float)):
            return ()
        records = []
        for layer in getattr(self, "layer_list", []) or []:
            for record in getattr(layer, "interventions", []) or []:
                record_timestamp = getattr(record, "timestamp", None)
                if isinstance(record_timestamp, (int, float)) and record_timestamp >= timestamp:
                    records.append(record)
        for grad_fn in getattr(self, "grad_fn_logs", {}).values():
            calls = getattr(getattr(grad_fn, "calls", None), "_list", [])
            for call in calls:
                for record in _flatten_backward_fire_ref(
                    getattr(call, "intervention_fire_ref", None)
                ):
                    record_timestamp = getattr(record, "timestamp", None)
                    if isinstance(record_timestamp, (int, float)) and record_timestamp >= timestamp:
                        records.append(record)
        return tuple(records)

    def summary(
        self,
        level: Literal[
            "overview", "graph", "memory", "control_flow", "compute", "cost", "waterfall", "output"
        ] = "overview",
        *,
        fields: Optional[List[str]] = None,
        mode: Literal["auto", "rolled", "unrolled"] = "auto",
        show_ops: bool = False,
        preset: Optional[
            Literal[
                "overview",
                "graph",
                "memory",
                "control_flow",
                "compute",
                "cost",
                "waterfall",
                "output",
            ]
        ] = None,
        columns: Optional[List[str]] = None,
        include_ops: Optional[bool] = None,
        max_rows: Optional[int] = 200,
        print_to: Optional[Callable[[str], None]] = None,
        count_fma_as_two: bool = False,
        show_input_preprocessing_details: bool = False,
    ) -> str:
        """Render a concise text summary of the logged model.

        Parameters
        ----------
        level:
            Summary level to render.
        fields:
            Explicit column selection for the primary table.
        mode:
            Operation aggregation mode. ``"rolled"`` uses aggregate layer rows,
            while ``"unrolled"`` uses per-pass operation rows.
        show_ops:
            Whether to append an operation table.
        preset:
            Alias for ``level`` retained for compatibility with the design spec.
        columns:
            Alias for ``fields``.
        include_ops:
            Alias for ``show_ops`` retained for compatibility with the design spec.
        max_rows:
            Maximum number of rows to render per table. ``None`` disables truncation.
        print_to:
            Optional callable that receives the rendered summary text.
        count_fma_as_two:
            FLOP/MAC convention marker for summary consumers. Current captured
            counts are displayed as stored; this flag reserves the public
            convention toggle without changing saved metadata.
        show_input_preprocessing_details:
            Whether to include verification/source detail for input
            preprocessing records.

        Returns
        -------
        str
            Rendered summary string.
        """
        from ..visualization._summary_internal import render_model_summary

        del count_fma_as_two
        return render_model_summary(
            self,
            level=level,
            preset=preset,
            fields=fields,
            columns=columns,
            mode=mode,
            show_ops=show_ops,
            include_ops=include_ops,
            max_rows=max_rows,
            print_to=print_to,
            show_input_preprocessing_details=show_input_preprocessing_details,
        )

    def render_dagua_graph(
        self,
        vis_mode: str = "unrolled",
        vis_call_depth: int = 1000,
        vis_outpath: str = "graph.gv",
        vis_save_only: bool = False,
        vis_fileformat: str = "pdf",
        vis_buffers: bool = False,
        vis_direction: str = "bottomup",
        vis_theme: str = "torchlens",
    ) -> str:
        """Render this model log with the experimental Dagua backend.

        Parameters
        ----------
        vis_mode, vis_call_depth, vis_outpath, vis_save_only, vis_fileformat, \
        vis_buffers, vis_direction, vis_theme:
            Forwarded unchanged to
            :func:`torchlens.experimental.dagua.render_trace_with_dagua`.

        Returns
        -------
        str
            Serialized Dagua graph output or the rendered artifact path.
        """
        from ..experimental.dagua import render_trace_with_dagua as _impl

        return cast(
            str,
            _impl(
                self,
                vis_mode=vis_mode,
                vis_call_depth=vis_call_depth,
                vis_outpath=vis_outpath,
                vis_save_only=vis_save_only,
                vis_fileformat=vis_fileformat,
                vis_buffers=vis_buffers,
                vis_direction=vis_direction,
                vis_theme=vis_theme,
            ),
        )

    def to_dagua_graph(
        self,
        vis_mode: str = "unrolled",
        vis_call_depth: int = 1000,
        show_buffer_layers: bool = False,
        direction: str = "bottomup",
        include_grad_edges: Optional[bool] = None,
    ) -> Any:
        """Translate this model log into an experimental Dagua graph.

        Parameters
        ----------
        vis_mode, vis_call_depth, show_buffer_layers, direction, include_grad_edges:
            Forwarded unchanged to
            :func:`torchlens.experimental.dagua.trace_to_dagua_graph`.

        Returns
        -------
        Any
            Dagua graph object.
        """
        from ..experimental.dagua import trace_to_dagua_graph as _impl

        return _impl(
            self,
            vis_mode=vis_mode,
            vis_call_depth=vis_call_depth,
            show_buffer_layers=show_buffer_layers,
            direction=direction,
            include_grad_edges=include_grad_edges,
        )

    def visualization_field_audit(self) -> "TorchLensRenderAudit":
        """Return the visualization field-usage audit for this model log.

        Returns
        -------
        TorchLensRenderAudit
            Audit of used and unused fields in the visualization bridge.
        """
        from ..experimental.dagua import build_render_audit as _impl

        return _impl(self)
