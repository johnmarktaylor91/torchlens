"""Trace visualization mixin."""

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .._visualization.code_panel import CodePanelOption
    from ..experimental.dagua._bridge import TorchLensRenderAudit
    from ..intervention.types import FireRecord
    from .trace import Trace

    _TraceMixinBase = Trace
else:
    _TraceMixinBase = object
from .._deprecations import MISSING, MissingType
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
from . import _doors
from ._nonfinite import nonfinite_coverage, nonfinite_op_labels
from .module import Module


class TraceVisualizationMixin(_TraceMixinBase):
    """``Trace`` visualization surface: ``draw``, ``show``, and collapse diagnostics."""

    def show(self: "Trace", kind: str | None = None, **kwargs: Any) -> str | None:
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
        str | None
            Rendered representation, Graphviz DOT source, or ``None`` when
            rendering is explicitly disabled with ``vis_mode="none"``.
        """

        return _doors.visualization()._trace_draw.show(self, kind=kind, **kwargs)

    def draw(
        self: "Trace",
        view: VisModeLiteral | MissingType = MISSING,
        depth: int | MissingType = MISSING,
        renderer: VisRendererLiteral | MissingType = MISSING,
        layout: VisNodePlacementLiteral | MissingType = MISSING,
        node_style: VisNodeModeLiteral | MissingType = MISSING,
        vis_mode: VisModeLiteral = "unrolled",
        vis_call_depth: int = 1000,
        path: str | None = None,
        vis_graph_overrides: dict[str, Any] | None = None,
        module: "Module | str | None" = None,
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
        code_panel: "CodePanelOption" = False,
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
        """Render the computational graph for this model log.

        Parameters
        ----------
        vis_mode, vis_call_depth, path, vis_graph_overrides, module, node_mode, \
        node_spec_fn, collapsed_node_spec_fn, collapse_fn, skip_fn, vis_edge_overrides, \
        vis_grad_edge_overrides, vis_module_overrides, vis_save_only, vis_fileformat, \
        show_buffers, direction, vis_node_placement, vis_renderer, theme, \
        vis_intervention_mode, vis_show_cone, code_panel, order_siblings, show_containers,
        container_max_inline, show_input_transform_summary, show_orphans:
            Forwarded unchanged to :func:`torchlens._visualization._render_dot.draw`.
            ``show_orphans=True`` renders orphan (island) ops -- captured but unreachable from both
            inputs and outputs -- as a dashed, greyed cluster of edgeless nodes instead of omitting
            them; orphans must have been retained at capture time (``keep_orphans=True``).
            ``show_buffers`` accepts ``"never"``, ``"meaningful"``, or ``"always"``. ``direction``
            accepts ``"bottomup"``, ``"topdown"``, ``"leftright"``, or ``"rightleft"``.
            ``show_containers`` accepts ``False`` (default: the plain graph), ``"labels"`` (the key
            or index on each edge into a container leaf), ``"collapsed"`` (``"labels"`` plus the
            fold of an unrolled same-shape container over ``container_max_inline`` leaves into one
            dashed box) or ``"nodes"`` (``"collapsed"`` plus one dashed record node per container
            with keyed source and sink edges, member-of ties, and the dotted box around a
            single-module tuple); a string setting forces the ``dot`` engine, and the removed
            ``"auto"`` and ``"cluster"`` refuse naming the replacement (``docs/containers.md``).
            ``vis_outpath=None`` (the default) writes a collision-safe file in the current
            directory -- ``modelgraph.<format>``, then ``modelgraph-2.<format>``,
            ``modelgraph-3.<format>`` -- and always prints its path, so a bare ``draw()`` never
            overwrites an earlier picture. An explicit path is written exactly (overwriting) and
            is printed only when the trace is verbose.
        collapse:
            Smart module-collapse mode. ``"none"`` preserves the full graph, ``"auto"`` uses the
            v2 readability-targeted engine, and ``"max"`` aggressively condenses eligible modules
            and segment boxes. A float in ``[0.0, 1.0]`` selects a deterministic monotone
            schedule: ``0.0`` is equivalent to ``"none"``, ``1.0`` is equivalent to ``"max"``, and
            larger values never increase the visible node count or uncollapse a collapsed unit.
            ``"auto"`` is the first schedule point whose visible count enters the readable band,
            while its current implementation remains unchanged for compatibility.
        fold_repeats:
            Repeat-fold policy. ``None`` preserves the default policy: off for ``collapse="none"``
            and band-pressure two-pass folding for ``"auto"``/``"max"``. ``True`` folds every
            eligible repeated run, including standalone folding with ``collapse="none"``.
            ``False`` disables run folding.
        fold_patterns:
            Declarative user-named pattern folding (DOCUMENTED-UNSTABLE spelling; collapse memo
            D11). ``None`` = off; ``"idiomatic"`` selects the curated conv-bn-relu preset; a mapping
            declares named linear paths (for example ``{"ConvBnRelu": "conv2d > batch_norm >
            relu"}``; exact repetition ``atom{k}``; an atom naming an earlier pattern inlines it).
            Matched instances render as dashed ``PATTERN '<name>' -- N ops`` chips; honesty-refused
            sites are counted and disclosed. v1 supports the pattern-only view
            (``collapse="none"``); combining with automatic collapse refuses typed
            (``pattern_collapse_combination_unsupported``).
        show_legend:
            Tri-state legend visibility. ``None`` (default) is AUTO: no
            legend unless an encoding channel is active, in which case a
            channel-only disclosure legend is emitted. ``True`` renders the
            full theme legend (plus channel rows when active); ``False``
            disables the legend even with channels active — a deliberate
            act that leaves the encoding undisclosed.
        color_by:
            UNSTABLE (keyword-only; no deprecation shim owed). Encoding
            channel value source: a Layer/Op field name (``"flops_forward"``),
            a scalar builtin (``"time"``, ``"flops"``, ``"bytes"``,
            ``"magnitude"``, ``"grad_norm"``), or a callable ``node ->
            value``. Encoded nodes are filled from a colorblind-safe
            sequential ramp normalized linear min-max over visible nodes;
            missing/non-finite values leave nodes unencoded (disclosed in
            the legend). Requires the Graphviz dot layout: under
            ``layout="auto"`` an active channel forces dot; explicit
            ``layout="rank"`` refuses. On rolled multi-pass layers, field
            sources resolve through the rolled-aggregate allowlist —
            per-pass-varying and first-pass-only sources stay unencoded with
            a legend note rather than painting an unprovable uniform value.
        size_by:
            UNSTABLE (keyword-only; no deprecation shim owed). Size encoding
            channel source: a Layer/Op field name, the closed ``"dims"``
            shape token (numel of the non-batch output shape — the D4
            default mapping, applied as default because D4 is unruled), or
            a callable ``node -> scalar``. Encoded nodes get width/height
            MINIMUMS (``fixedsize=false``: labels are never truncated,
            fonts never scale) with encoded area clamped to 4x the default
            node area. STRICTLY OPT-IN: plain ``draw()`` keeps uniform
            boxes. On a rolled multi-pass layer a size source that cannot
            be certified single-valued refuses typed
            (``size_by_rolled_varying``): size has no honest "n/a"
            rendering, so it refuses where color degrades. Callables bypass
            the rolled table (disclosed in the legend).
        scale:
            UNSTABLE (keyword-only). Size-channel scale transform:
            ``"sqrt"`` (default) or ``"linear"`` (the literal area motif).
            Supplied without ``size_by`` it refuses
            (``scale_requires_size_by``). Every legend drawn states the
            active scale.
        stack_by:
            UNSTABLE (keyword-only). Rank encoding channel: nodes sharing an
            annotation value pin to one Graphviz rank (column/row), the
            classic unrolled-RNN timestep diagram. STRICTLY OPT-IN.
            ``True``/``"auto"`` derives the annotation (``pass_index`` on
            multi-pass ops only) under the lockstep license — the
            multi-pass execution order must be globally monotone, else it
            refuses (``stack_by_auto_underivable``); an explicit field name
            or callable bypasses the license (the caption disclosed what
            was used). Rolled graphs refuse (``stack_by_requires_unrolled``).
            While stacking is active the sibling-ordering post-pass no-ops
            (two rank-constraint systems would fight), and collapsed boxes/
            fold reps stay un-annotated.
        show_redundant_args:
            UNSTABLE (keyword-only). Checked suppression of redundant
            constructor-arg label rows is DEFAULT-ON: an arg such as
            ``in_features=4`` is omitted exactly when its value provably
            equals the captured shape dimension it duplicates on THIS
            trace (a closed torch-module candidate table; the check is
            data equality on records). A mismatch or unavailable shape
            keeps the arg VISIBLE — the rule can only reveal more, never
            hide a discrepancy. Pass ``True`` to show every captured arg.
        show_saved_for_backward:
            UNSTABLE (keyword-only). Saved-for-backward annotation: adds a
            label row (``saved for backward: N tensors, X MB``) on every op
            whose grad_fn measurably retained tensors for the backward pass,
            from the capture-time ``num_autograd_tensors`` /
            ``autograd_memory`` measurements — the memory autograd is
            actually holding, made visible per node. Ops that saved nothing
            (or whose backward graph was never built, e.g. under
            ``torch.no_grad``) get no row: an absent row makes no claim. On
            rolled multi-pass layers the stored measurements are cross-pass
            sums and the row discloses ``(total across passes)``. Composes
            with ``color_by="autograd_memory"`` for a ramp over the same
            quantity.

        Returns
        -------
        Any
            Graphviz DOT source, renderer-specific output, or renderer object
            when ``return_graph=True``.
        """

        return _doors.visualization()._trace_draw.draw(
            self,
            view=view,
            depth=depth,
            renderer=renderer,
            layout=layout,
            node_style=node_style,
            vis_mode=vis_mode,
            vis_call_depth=vis_call_depth,
            path=path,
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
            file_format=file_format,
            show_buffers=show_buffers,
            direction=direction,
            vis_node_placement=vis_node_placement,
            vis_renderer=vis_renderer,
            theme=theme,
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

    def add_node_overlay(
        self: "Trace",
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

    def animate_ops(self: "Trace", site: Any) -> str:
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

        return _doors.visualization()._trace_draw.animate_ops(self, site=site)

    @property
    def nonfinite_ops(self: "Trace") -> tuple[str, ...]:
        """Return pass-qualified labels of ops whose output held NaN or Inf.

        DOCUMENTED-UNSTABLE spelling (pending naming-session ratification; no
        deprecation shim owed on rename). This is the queryable per-op record:
        when this capture ran with ``CaptureOptions(track_nonfinite=True)`` it
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

    def draw_backward(
        self: "Trace",
        path: str = "backward_modelgraph",
        vis_graph_overrides: dict[str, Any] | None = None,
        node_spec_fn: Callable[..., Any] | None = None,
        collapsed_node_spec_fn: Callable[..., Any] | None = None,
        vis_node_mode: VisNodeModeLiteral = "default",
        vis_edge_overrides: dict[str, Any] | None = None,
        vis_save_only: bool = False,
        file_format: str = "pdf",
        vis_direction: VisDirectionLiteral = "topdown",
        code_panel: "CodePanelOption" = False,
        vis_mode: VisModeLiteral = "rolled",
        bwd: int | Iterable[int] | None = None,
        show_legend: bool | None = None,
        *,
        theme: str = "torchlens",
        show_module_paths: bool = False,
        show_grad_values: bool = False,
    ) -> str:
        """Render the captured backward grad_fn_handle graph.

        Parameters
        ----------
        path, vis_graph_overrides, node_spec_fn, collapsed_node_spec_fn, \
        vis_node_mode, vis_edge_overrides, vis_save_only, file_format, \
        vis_direction, code_panel, vis_mode, bwd, show_legend:
            Forwarded unchanged to
            :func:`torchlens._visualization._render_entrypoints.render_backward_graph`.
            ``collapsed_node_spec_fn`` and ``vis_node_mode`` are accepted for
            forward-visualization API symmetry but are not applied because
            backward graphs do not render collapsed module nodes.
        theme:
            Keyword-only. Skin preset, the names :meth:`draw` takes: ``"torchlens"``
            (default), ``"paper"``, ``"dark"``, ``"colorblind"`` or
            ``"high_contrast"``. It sets the page, text, typography and key
            ink; the backward node and arrow colours stay the same.
        show_module_paths:
            UNSTABLE (keyword-only). Opt-in row naming the module that owns
            each backward node (``module <address>``; AccumulateGrad nodes
            read ``param <address>``). Nodes no module owns get no row.
        show_grad_values:
            UNSTABLE (keyword-only). Opt-in row summarizing each node's
            captured output gradient (``grad norm <x>, absmax <y>``; ``grad
            values not captured`` when the capture kept no tensor).

        Returns
        -------
        str
            Graphviz DOT source.
        """

        return _doors.visualization()._trace_draw.draw_backward(
            self,
            path=path,
            vis_graph_overrides=vis_graph_overrides,
            node_spec_fn=node_spec_fn,
            collapsed_node_spec_fn=collapsed_node_spec_fn,
            vis_node_mode=vis_node_mode,
            vis_edge_overrides=vis_edge_overrides,
            vis_save_only=vis_save_only,
            file_format=file_format,
            vis_direction=vis_direction,
            code_panel=code_panel,
            vis_mode=vis_mode,
            bwd=bwd,
            show_legend=show_legend,
            theme=theme,
            show_module_paths=show_module_paths,
            show_grad_values=show_grad_values,
        )

    def draw_combined(
        self: "Trace",
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
        """Render forward ops and backward grad_fns in one graph.

        Parameters
        ----------
        path, vis_graph_overrides, node_spec_fn, backward_node_spec_fn, \
        vis_edge_overrides, vis_save_only, file_format, vis_direction, \
        vis_mode, intervening_cluster, show_buffers, bwd, show_legend:
            Forwarded unchanged to
            :func:`torchlens._visualization._render_entrypoints.render_combined_graph`.

        Returns
        -------
        str
            Graphviz DOT source.
        """

        return _doors.visualization()._trace_draw.draw_combined(
            self,
            path=path,
            vis_graph_overrides=vis_graph_overrides,
            node_spec_fn=node_spec_fn,
            backward_node_spec_fn=backward_node_spec_fn,
            vis_edge_overrides=vis_edge_overrides,
            vis_save_only=vis_save_only,
            file_format=file_format,
            vis_direction=vis_direction,
            vis_mode=vis_mode,
            intervening_cluster=intervening_cluster,
            show_buffers=show_buffers,
            bwd=bwd,
            show_legend=show_legend,
        )

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
        filter: Any = None,  # noqa: A002 - the memo-ratified axis name
        buffers: str | None = None,
        fold_repeats: Any = None,
        max_rows: int | None = None,
        flop_convention: str | None = None,
        units: str | None = None,
        style: str | None = None,
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
            ``"op"``.
        view:
            Column/footer preset: ``"overview"`` (default) | ``"compute"``.
        depth:
            ``"auto"`` | ``"all"`` | int module-tree depth.
        columns:
            Bundle name, exact ordered list, or ``+name``/``-name`` deltas.
        filter:
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

        return _doors.report()._trace_summary.summary(
            self,
            level=level,
            view=view,
            depth=depth,
            columns=columns,
            filter=filter,
            buffers=buffers,
            fold_repeats=fold_repeats,
            max_rows=max_rows,
            flop_convention=flop_convention,
            units=units,
            style=style,
            input_shape=input_shape,
            execution_mode=execution_mode,
            grad_mode=grad_mode,
            _execution_note=_execution_note,
            _input_synthesis=_input_synthesis,
            **unsupported,
        )

    def render_dagua_graph(
        self: "Trace",
        vis_mode: str = "unrolled",
        vis_call_depth: int = 1000,
        path: str = "graph.gv",
        vis_save_only: bool = False,
        file_format: str = "pdf",
        show_buffers: bool = False,
        vis_direction: str = "bottomup",
        theme: str = "torchlens",
    ) -> str:
        """Render this model log with the experimental Dagua backend.

        Parameters
        ----------
        vis_mode, vis_call_depth, path, vis_save_only, file_format, \
        show_buffers, vis_direction, theme:
            Forwarded unchanged to
            ``torchlens.experimental.dagua._bridge.render_trace_with_dagua``.

        Returns
        -------
        str
            Serialized Dagua graph output or the rendered artifact path.
        """

        return _doors.visualization()._trace_draw.render_dagua_graph(
            self,
            vis_mode=vis_mode,
            vis_call_depth=vis_call_depth,
            path=path,
            vis_save_only=vis_save_only,
            file_format=file_format,
            show_buffers=show_buffers,
            vis_direction=vis_direction,
            theme=theme,
        )

    def to_dagua_graph(
        self: "Trace",
        vis_mode: str = "unrolled",
        vis_call_depth: int = 1000,
        show_buffers: bool = False,
        direction: str = "bottomup",
        include_grad_edges: bool | None = None,
    ) -> Any:
        """Translate this model log into an experimental Dagua graph.

        Parameters
        ----------
        vis_mode, vis_call_depth, show_buffers, direction, include_grad_edges:
            Forwarded unchanged to
            :func:`torchlens.experimental.dagua.trace_to_dagua_graph`.

        Returns
        -------
        Any
            Dagua graph object.
        """

        return _doors.visualization()._trace_draw.to_dagua_graph(
            self,
            vis_mode=vis_mode,
            vis_call_depth=vis_call_depth,
            show_buffers=show_buffers,
            direction=direction,
            include_grad_edges=include_grad_edges,
        )

    def visualization_field_audit(self: "Trace") -> "TorchLensRenderAudit":
        """Return the visualization field-usage audit for this model log.

        Returns
        -------
        TorchLensRenderAudit
            Audit of used and unused fields in the visualization bridge.
        """

        return _doors.visualization()._trace_draw.visualization_field_audit(self)
