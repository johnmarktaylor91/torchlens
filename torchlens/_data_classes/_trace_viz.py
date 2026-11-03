"""Trace visualization mixin."""

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .._visualization.code_panel import CodePanelOption
    from ..intervention.types import FireRecord
    from ..viz._drawing import Drawing
    from .trace import Trace

    _TraceMixinBase = Trace
else:
    _TraceMixinBase = object
from .._literals import (
    BufferVisibilityLiteral,
    CollapseLiteral,
    FoldRepeatsLiteral,
    VisDirectionLiteral,
    VisInterventionModeLiteral,
    VisNodeModeLiteral,
    VisNodePlacementLiteral,
    VisRendererLiteral,
)
from . import _doors
from ._nonfinite import nonfinite_coverage, nonfinite_op_labels
from ._trace_viz_arguments import _forward_draw, _forward_summary
from .module import Module

GraphLiteral = Literal["forward", "backward", "combined"]


class TraceVisualizationMixin(_TraceMixinBase):
    """``Trace`` visualization surface: ``draw``, ``show``, and collapse diagnostics."""

    def show(self: "Trace", kind: str | None = None, **kwargs: Any) -> Any:
        """Render this trace using a lightweight notebook-friendly dispatcher.

        Parameters
        ----------
        kind:
            Picture kind. ``None`` or ``"graph"`` delegates to :meth:`draw`, ``"repr"``
            returns ``repr(self)``, and ``"html"`` returns ``_repr_html_()``.
        **kwargs:
            Visualization keyword arguments forwarded to :meth:`draw`.

        Returns
        -------
        Any
            The :class:`~torchlens.viz.Drawing` from :meth:`draw`, or the
            requested text representation.
        """

        return _doors.visualization()._trace_draw.show(self, kind=kind, **kwargs)

    def draw(  # noqa: PLR0913 - the one draw door: every VisualizationOptions field is a keyword
        self: "Trace",
        path: str | None = None,
        *,
        graph: GraphLiteral = "forward",
        rolled: bool = False,
        depth: int = 1000,
        module: "Module | str | None" = None,
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
        code_panel: "CodePanelOption" = False,
        font_size: int | None = None,
        dpi: int | None = None,
        order_siblings: bool = True,
        show_containers: Literal[False, "labels", "collapsed", "nodes"] = False,
        container_max_inline: int = 12,
        show_input_transform_summary: bool = False,
        show_orphans: bool = False,  # Invariants support flipping this; owner visual review pending.
        show_redundant_args: bool = False,
        show_saved_for_backward: bool = False,
        bwd: int | Iterable[int] | None = None,
        show_module_paths: bool = False,
        show_grad_values: bool = False,
        backward_node_spec_fn: Callable[..., Any] | None = None,
        intervening_cluster: Literal["upstream", "outside", "downstream", "own"] = "upstream",
        file_format: str = "pdf",
        open_viewer: bool = False,
    ) -> "Drawing":
        """Draw this trace and return the picture as a :class:`~torchlens.viz.Drawing`.

        The keywords are the :class:`~torchlens.options.VisualizationOptions`
        field names. ``str(drawing)`` is the Graphviz DOT source, ``drawing.save(path)``
        writes it and ``drawing.show()`` displays it.

        Parameters
        ----------
        path:
            Output path stem. ``None`` (the default) writes a collision-safe
            file in the current directory (``modelgraph.<format>``, then
            ``modelgraph-2.<format>``, ...) and prints its path, so a bare
            ``draw()`` never overwrites an earlier picture. An explicit path is
            written exactly (overwriting) and printed only when the trace is
            verbose. Backward and combined graphs default to
            ``backward_modelgraph`` and ``combined_modelgraph``.
        graph:
            Which graph: ``"forward"`` (the ops), ``"backward"`` (the captured
            grad_fn graph) or ``"combined"`` (both in one picture).
        rolled:
            ``True`` draws each repeated module once (the rolled graph);
            ``False`` (default) draws every pass (the unrolled graph).
        depth:
            Maximum module nesting depth shown.
        module:
            Focus the picture on one module (a ``Module`` or its address).
        collapse:
            Smart module-collapse mode. ``"none"`` preserves the full graph,
            ``"auto"`` uses the readability-targeted engine, ``"max"``
            condenses every eligible module, and a float in ``[0.0, 1.0]``
            selects a deterministic monotone schedule (``0.0`` is ``"none"``,
            ``1.0`` is ``"max"``; larger values never increase the visible
            node count or uncollapse a collapsed unit).
        fold_repeats:
            Repeat-fold policy. ``None`` keeps the default (off for
            ``collapse="none"``, band-pressure folding otherwise), ``True``
            folds every eligible repeated run, ``False`` disables folding.
        fold_patterns:
            Declarative user-named pattern folding (DOCUMENTED-UNSTABLE):
            ``None`` = off; ``"idiomatic"`` selects the curated conv-bn-relu
            preset; a mapping declares named linear paths such as
            ``{"ConvBnRelu": "conv2d > batch_norm > relu"}``. Combining with
            automatic collapse refuses typed
            (``pattern_collapse_combination_unsupported``).
        theme:
            Skin preset: ``"torchlens"`` (default), ``"paper"``, ``"dark"``,
            ``"colorblind"`` or ``"high_contrast"``.
        layout:
            Node placement: ``"auto"``, ``"dot"`` or ``"rank"``.
        renderer:
            ``"graphviz"`` (default) or the experimental ``"dagua"`` after opt-in.
        direction:
            Edge flow: ``"bottomup"``, ``"topdown"``, ``"leftright"`` or
            ``"rightleft"``. ``None`` takes the graph's default (forward
            ``"bottomup"``, backward ``"topdown"``, combined ``"leftright"``).
        node_style:
            Node label style: ``"default"`` or ``"profiling"``.
        show_buffers:
            Buffer visibility: ``"never"``, ``"meaningful"`` or ``"always"``.
        show_legend:
            Tri-state legend visibility. ``None`` (default) draws a legend only
            when an encoding channel is active, ``True`` draws the full theme
            legend, ``False`` disables the legend even with channels active.
        color_by:
            Fill encoding channel (keyword-only; unstable). A Layer/Op field
            name (``"flops_forward"``), a scalar builtin (``"time"``,
            ``"flops"``, ``"bytes"``, ``"magnitude"``, ``"grad_norm"``), a
            callable ``node -> value``, a mapping of layer label or module
            address to value (an overlay; the legend names it ``overlay``), or
            an attribution result or an iterable of them (each layer-scoped
            result paints its layer with the ``abs_sum`` of its values; another
            reduction is a mapping you compute). Encoded nodes are filled from
            a colorblind-safe sequential ramp; missing or non-finite values
            leave nodes unencoded (disclosed in the legend). Requires the dot
            layout: ``layout="rank"`` refuses.
        size_by:
            Size encoding channel (keyword-only; unstable): a Layer/Op field
            name, the ``"dims"`` shape token, or a callable ``node -> scalar``.
            Encoded nodes get width/height minimums (labels are never
            truncated), clamped to 4x the default area. A rolled multi-pass
            layer whose source cannot be certified single-valued refuses typed
            (``size_by_rolled_varying``).
        size_scale:
            Size-channel transform: ``"sqrt"`` (default) or ``"linear"``.
            Supplied without ``size_by`` it refuses (``scale_requires_size_by``).
        stack_by:
            Rank encoding channel (keyword-only; unstable): nodes sharing an
            annotation value pin to one Graphviz rank, the unrolled-RNN
            timestep diagram. ``True`` derives the pass index (unrolled
            multi-pass ops only); rolled graphs refuse
            (``stack_by_requires_unrolled``).
        node_overlay:
            Per-node label row: a stock overlay name (``"flops"``, ``"time"``,
            ``"bytes"``, ``"magnitude"``, ``"nan"``, ``"grad_norm"``,
            ``"intervention"``, ``"bundle_delta"``), a mapping of layer label
            to value, or a callable ``node -> value``.
        node_label_fields:
            Explicit list of label rows to show on every node.
        graph_overrides, edge_overrides, grad_edge_overrides, module_overrides:
            Graphviz attribute overrides for the graph, forward edges, gradient
            edges and module boxes.
        node_spec_fn, collapsed_node_spec_fn, collapse_fn, skip_fn:
            Per-node styling and selection callbacks (see ``tl.viz.nodes``).
        intervention_mode:
            How intervened sites are marked: ``"node_mark"`` or ``"cone"``.
        show_cone:
            Shade the downstream cone of every intervention.
        code_panel:
            Source-location panel option (``False``, ``True`` or a panel spec).
        font_size, dpi:
            Typography and raster resolution overrides.
        order_siblings:
            Keep sibling ops in execution order (a post-pass; off while
            ``stack_by`` is active).
        show_containers:
            Container rendering: ``False`` (default: the plain graph),
            ``"labels"`` (the key or index on each edge into a container
            leaf), ``"collapsed"`` (``"labels"`` plus the fold of an unrolled
            same-shape container over ``container_max_inline`` leaves into one
            dashed box) or ``"nodes"`` (``"collapsed"`` plus one dashed record
            node per container with keyed source and sink edges). A string
            setting forces the ``dot`` engine; the removed ``"auto"`` and
            ``"cluster"`` refuse naming the replacement (``docs/containers.md``).
        container_max_inline:
            Inline container size ceiling for ``show_containers="collapsed"``
            and ``"nodes"``.
        show_input_transform_summary:
            Add the input-transform summary row.
        show_orphans:
            Render orphan (island) ops, captured but unreachable from inputs
            and outputs, as a dashed greyed cluster (capture with
            ``keep_orphans=True``).
        show_redundant_args:
            Show constructor-arg label rows that provably duplicate a captured
            shape dimension (suppressed by default; the rule only ever reveals
            more).
        show_saved_for_backward:
            Add a ``saved for backward: N tensors, X MB`` row on every op whose
            grad_fn measurably retained tensors.
        bwd:
            Backward and combined graphs: which backward pass or passes to draw.
        show_module_paths, show_grad_values:
            Backward graph: opt-in rows naming the owning module of each
            backward node and summarizing its captured output gradient.
        backward_node_spec_fn:
            Combined graph: per-node styling callback for the backward nodes.
        intervening_cluster:
            Combined graph: placement of backward-only nodes (``"upstream"``,
            ``"outside"``, ``"downstream"`` or ``"own"``).
        file_format:
            Output file format (``"pdf"`` default, ``"svg"``, ``"png"``, ...).
        open_viewer:
            ``True`` opens the written picture in the system viewer; the
            default never opens anything.

        Returns
        -------
        Drawing
            The picture: DOT source (or animation HTML), its format, the written
            path when a file was written, and the resolved policy.
        """

        return _forward_draw(_doors.visualization()._trace_draw.draw, locals())

    def animate(
        self: "Trace",
        site: Any,
        path: str | None = None,
        *,
        open_viewer: bool = False,
    ) -> "Drawing":
        """Animate the passes at ``site`` and return the picture as a :class:`~torchlens.viz.Drawing`.

        The animation is its own door: ``draw`` draws graphs, ``animate`` plays
        the repeated ops at one site.

        Parameters
        ----------
        site:
            A site label, pass-qualified label or record (anything with a
            ``layer_label``). The picture is the HTML animation of the repeated
            ops at that site: ``str(drawing)`` is the HTML, ``drawing.show()``
            plays it and ``drawing.save(path)`` writes it.
        path:
            Output path; ``.html`` is appended when the suffix differs.
            ``None`` (the default) keeps the animation in memory.
        open_viewer:
            ``True`` opens the animation in the system viewer.

        Returns
        -------
        Drawing
            The HTML animation (``file_format == "html"``).

        Raises
        ------
        KeyError
            ``site`` names no layer of this trace.
        """

        return _doors.visualization()._trace_draw.animate(
            self, site, path=path, open_viewer=open_viewer
        )

    @property
    def nonfinite_ops(self: "Trace") -> tuple[str, ...]:
        """Return pass-qualified labels of ops whose output held NaN or Inf.

        DOCUMENTED-UNSTABLE spelling (pending naming-session ratification; no
        deprecation shim owed on rename). This is the queryable per-op record:
        when this capture ran with ``CaptureOptions(on_nonfinite='track')`` it
        serves the capture-time verdicts (covering ops that retained no
        payload); otherwise it derives the answer from the memoized
        saved-payload scan already backing ``print(trace)``, at zero
        capture-time cost. An empty tuple is only as strong as its coverage --
        read :attr:`nonfinite_coverage` before trusting a clean answer from a
        capture that retained few payloads.

        Returns
        -------
        tuple[str, ...]
            Pass-qualified op labels (``Op.label``) in scan order; each is a
            valid ``trace[label]`` key.
        """

        return nonfinite_op_labels(self)

    @property
    def nonfinite_coverage(self: "Trace") -> Any:
        """Return the evidence basis and coverage behind :attr:`nonfinite_ops`.

        DOCUMENTED-UNSTABLE spelling (pending naming-session ratification; no
        deprecation shim owed on rename). A clean :attr:`nonfinite_ops` answer
        must not read as a whole-capture verdict when the scan could not
        examine everything; this discloses the basis (``"capture"`` vs
        ``"saved_payloads"``) and the checked / unchecked / unexamined counts.

        Returns
        -------
        NonfiniteCoverage
            Frozen coverage record (see
            :class:`torchlens._data_classes._nonfinite.NonfiniteCoverage`).
        """

        return nonfinite_coverage(self)

    def first_nonfinite(
        self: "Trace", link_format: Literal["terminal", "html", "text"] = "text"
    ) -> str:
        """Return a text answer describing the first saved non-finite out.

        Parameters
        ----------
        link_format:
            Source-location link style. ``"text"`` (the default) emits plain
            ``path:line`` text; ``"terminal"`` emits OSC 8 hyperlinks and
            ``"html"`` emits VS Code URI anchors. The styled registers are
            OPT-IN because both embed the capture machine's RESOLVED ABSOLUTE
            path inside the link target: returned strings flow into logs,
            saved reports, JSON, and MCP responses, where an escape byte
            breaks diffs/caches and an absolute path is a leak.

        Returns
        -------
        str
            Human-readable single-paragraph answer naming the layer, operation,
            module, shape, dtype, parents, and source location.
        """

        return _doors.report()._trace_summary.first_nonfinite(self, link_format=link_format)

    def preview_fastlog(
        self: "Trace",
        predicate: Callable[..., Any] | None = None,
        keep_op: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Render a fastlog predicate preview for this model graph.

        Parameters
        ----------
        predicate, keep_op:
            Predicate callables that receive synthesized fastlog ``RecordContext``
            objects.
        **kwargs:
            Forwarded to :func:`torchlens._visualization.fastlog_preview.preview_fastlog`.

        Returns
        -------
        str
            Graphviz DOT source.
        """

        return _doors.visualization()._trace_draw.preview_fastlog(
            self, predicate=predicate, keep_op=keep_op, **kwargs
        )

    def last_run_records(self: "Trace") -> tuple["FireRecord", ...]:
        """Return fire records from the most recent replay, rerun, or live capture.

        Returns
        -------
        tuple[FireRecord, ...]
            Immutable snapshot of matching intervention fire records.
        """

        return _doors.visualization()._trace_draw.last_run_records(self)

    def _repr_html_(self: "Trace") -> str:
        """Return the notebook HTML representation for this model log.

        Returns
        -------
        str
            HTML fragment for IPython/Jupyter display.

        Document composition (F08 x C05 x F16): the Trace card (treescope
        memo B1/B2 -- identity, honesty badge, NaN/Inf line, and the
        budgeted lookup-key index, assembled through
        ``torchlens._notebook.cards``) renders first, and on finished
        traces the rebuilt summary table (F08; summary memo 3.11 -- the
        bare ``trace`` cell is the most likely first touch in Jupyter)
        follows as a sibling fragment. A summary failure degrades to the
        card alone; any failure THERE degrades inside ``safe_card_html``
        to a one-line ``card unavailable`` fragment (never-raise
        boundary). The treescope-bridge suppression sentinel replaces the
        WHOLE document exactly when the bridge just rendered this object
        (memo 3.6) -- appending the summary there would duplicate the
        render treescope owns.
        """

        return _doors.report()._trace_summary.repr_html(self)

    def summary(  # noqa: PLR0913 - the ratified public grammar: every axis is a named keyword
        self: "Trace",
        level: str | None = None,
        *,
        # Rebuilt grammar (F08; summary memo 3.10). Spellings
        # DOCUMENTED-UNSTABLE pending naming-session ratification.
        view: str | None = None,
        depth: Any = None,
        columns: list[str] | None = None,
        where: Any = None,
        buffers: str | None = None,
        fold_repeats: Any = None,
        max_rows: int | None = None,
        flop_convention: str | None = None,
        units: str | None = None,
        style: str | None = None,
        max_scan_elements: int | None = None,
        # One-call-only spellings (typed errors here, never no-ops).
        input_shape: Any = None,
        execution_mode: str | None = None,
        grad_mode: str | None = None,
        # Internal plumbing from the one-call door (not public grammar).
        _execution_note: str | None = None,
        _input_synthesis: str | None = None,
        # Anything else refuses typed (removed legacy spellings name their
        # successor; unknown names get the nearest grammar option).
        **unsupported: Any,
    ) -> str:
        """Render a summary of this capture.

        A bare ``trace.summary()`` resolves the automatic view ladder
        (hybrid / folded module tree / elision) under a 48-body-row budget
        and returns a detached typed report whose ``str`` payload is
        canonical byte-stable ASCII. The legacy spellings (``level=
        "graph"``, ``preset=``, ``fields=``, ``show_ops=``, ...) are
        removed: each refuses typed and names its replacement.

        Parameters
        ----------
        level:
            Rebuilt row grain: ``"auto"`` (default) | ``"module"`` |
            ``"op"``. With ``view="stats"``, ``"op"`` (and ``"auto"``) is one
            row per executed op and ``"module"`` one row per module-output
            payload.
        view:
            ``"overview"`` (default) | ``"compute"`` | ``"stats"`` (per-site
            payload statistics; see docs/reference/summary.md, "The stats view").
        depth:
            ``"auto"`` | ``"all"`` | int module-tree depth.
        columns:
            Bundle name, exact ordered list, or ``+name``/``-name`` deltas.
        where:
            Regex string or ``row -> bool`` callable; presentation-only
            (whole-model totals preserved and coverage disclosed).
        buffers:
            ``"summary"`` (footer totals; default) | ``"hide"``.
        fold_repeats:
            ``"auto"``/``True`` fold repeated sibling runs; ``False``
            restores every member.
        max_rows:
            Body-row budget for the auto ladder (default 48).
        flop_convention:
            ``"fma2"`` (stored convention; default) | ``"fma1"``.
        units:
            ``"human"`` (K/M/G) | ``"raw"`` (full digits) in rendered text.
        style:
            ``"auto"`` | ``"ascii"`` | ``"unicode"`` for display helpers;
            ``str(result)`` is ALWAYS the canonical ASCII payload.
        max_scan_elements:
            The payload-scan element budget of ``view="stats"``; later retained
            rows become typed ``unscanned`` rows. Refused typed
            (``summary_option_invalid``) with any other view.
        **unsupported:
            Refused typed (``summary_option_invalid``): a removed legacy
            spelling (``preset``, ``fields``, ``show_ops``, ``include_ops``,
            ``mode``, ``print_to``, ``count_fma_as_two``,
            ``show_input_preprocessing_details``) names its replacement;
            any other name gets the nearest grammar option.
        input_size, execution_mode, grad_mode:
            ONE-CALL-ONLY spellings: valid on ``tl.summary(model, ...)``,
            typed errors here (this method reports an existing capture).

        Returns
        -------
        str
            A ``SummaryReport`` (``str`` subclass) carrying typed rows,
            totals, capture facts, and the
            render/print/to_pandas/to_markdown/to_html result API.
        """

        return _forward_summary(_doors.report()._trace_summary.summary, locals())

    def to_dagua_graph(
        self: "Trace",
        rolled: bool = False,
        depth: int = 1000,
        show_buffers: bool = False,
        direction: str = "bottomup",
        include_grad_edges: bool | None = None,
    ) -> Any:
        """Translate this model log into an experimental Dagua graph.

        Parameters
        ----------
        rolled, depth, show_buffers, direction, include_grad_edges:
            Forwarded unchanged to
            :func:`torchlens.experimental.dagua.trace_to_dagua_graph`.

        Returns
        -------
        Any
            Dagua graph object.
        """

        return _doors.visualization()._trace_draw.to_dagua_graph(
            self,
            vis_mode="rolled" if rolled else "unrolled",
            vis_call_depth=depth,
            show_buffers=show_buffers,
            direction=direction,
            include_grad_edges=include_grad_edges,
        )
