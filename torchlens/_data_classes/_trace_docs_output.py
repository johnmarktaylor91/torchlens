"""Full docstrings of the ``Trace`` drawing, summaries, export, stacking and narration methods.

The methods on ``Trace`` carry a one-line summary; ``_trace_docs.attach_trace_docs`` binds the
text here as their ``__doc__`` when ``trace.py`` finishes defining the class, so ``help()``
and notebooks show the full reference while ``trace.py`` stays inside its file cap.
"""

from __future__ import annotations

__all__ = ["DOCS"]

DOCS: dict[str, str] = {
    "stack": (
        """Stack selector-matched saved outputs in recorded execution order.
        
        Selector evaluation reuses :meth:`Trace.find_sites`. Matching operations
        are then sorted explicitly by their recorded ``ordinal_index``; selector
        iteration order is never treated as authoritative. The resulting rows can
        feed ``torchlens.stats.cka``, across-layer PCA, or ``torchlens.repgeom.rdm``.
        
        Parameters
        ----------
        selector:
            TorchLens selector accepted by :meth:`Trace.find_sites`.
        
        Returns
        -------
        StackedActivations
            Tensor rows and their aligned layer labels.
        
        Raises
        ------
        ValueError
            If no operations match, execution ordinals are invalid, a payload was
            not saved, a primary output is not one tensor, or shapes differ.
        """
    ),
    "logged_values": (
        """Return values recorded via ``log_value`` during this capture.
        
        DOCUMENTED-UNSTABLE spelling (naming ratification pending). This is
        the read-back for ``torchlens.observers.log_value``, which was a
        writer nothing read back: values land in
        ``annotations["logged_values"]`` and persist with the trace.
        Returns a copy -- write through ``log_value`` during capture, never by
        mutating this view.
        
        Returns
        -------
        dict[str, Any]
            Name-to-value mapping of capture-time logged values (empty when
            none were recorded).
        """
    ),
    "to_agent_json": (
        """Return a self-describing machine-readable dump of this trace.
        
        DOCUMENTED-UNSTABLE spelling (naming ratification pending). The dump
        is a navigation aid over the SAME public surface a human drives: it
        embeds a ``guide`` block mapping every record back to the live
        spelling to call next (``trace[label]``, ``trace.explain``, ...),
        and carries the capture outcome/verification honesty facts verbatim.
        Tensor payloads are never inlined.
        
        Parameters
        ----------
        max_ops:
            Optional cap on emitted op rows (execution order, first ``max_ops``
            kept). Any omission is disclosed in the ``truncation`` block.
        
        Returns
        -------
        dict[str, Any]
            JSON-serializable dump under the ``torchlens.agent_trace.v1``
            schema.
        """
    ),
    "_attach_dataframe_honesty": (
        """Stamp ``frame.attrs`` with this trace's capture-honesty facts.
        
        The one place the records reach ``_capture_honesty`` for tabular exports:
        ``to_pandas`` here and ``_accessor_base.attach_source_honesty`` both call it.
        """
    ),
    "to_pandas": (
        """Return a dataframe containing one row per layer pass.
        
        Parameters
        ----------
        include_decoded_output_summary:
            If ``True``, add a gated ``decoded_output_summary`` column populated
            only on output-node rows. The default remains schema-stable.
        
        Returns
        -------
        pd.DataFrame
            Layer-pass table for this model log.
        
        Raises
        ------
        RuntimeError
            If called before the forward pass has completed.
        ValueError
            If an exported field such as ``grad`` is ambiguous across multiple
            backward passes and must be accessed through a pass-specific record.
        """
    ),
    "decode_output": (
        """Return decoded output rows: the captured rows, or a decode on demand.
        
        With no keywords this returns the rows the automatic capture-time
        decode produced. Naming a ``style``, a ``head`` or a ``tokenizer``
        decodes again from the retained output tensors, the persisted
        ``output_id2label`` and the ImageNet bank, never ``decoded_output``.
        A classification decode returns ``{"kind": "batch_topk", "rows": ...}``
        with ``batch_item``, ``rank``, ``class_index``, ``label`` and ``prob``
        rows; a text decode (``"hf_text"``) returns the top next tokens at the
        last position as a list of ``batch_item``, ``rank``, ``token_id``,
        ``text`` and ``prob`` rows, ``prob`` being the softmax over the
        vocabulary at that position.
        
        Parameters
        ----------
        k:
            Optional maximum rank to return per batch item.
        style:
            Decode style: ``"classification"``, ``"imagenet"``,
            ``"imagenet1k"``, ``"hf_text"`` or a registered custom style.
            ``None`` lets the registered detectors pick one.
        head:
            Output head to decode, as an output container path such as
            ``"logits"``, ``"0"`` or ``"0.logits"``.
        tokenizer:
            Tokenizer with a ``decode`` method, used by ``"hf_text"``; without
            one the rows carry each ``token_id`` and ``text`` is ``None``.
        
        Returns
        -------
        Any
            JSON-primitive decoded output.
        
        Raises
        ------
        ValueError
            If decoded output was not captured or the requested ``k`` exceeds
            the capture-time bound (``decoded_output_unavailable``); a style
            no detector resolves (``decode_style_unknown``); a head that is
            not an output head (``decode_head_unknown``) or an ambiguous head
            (``decode_head_ambiguous``).
        """
    ),
    "output_table": (
        """Return decoded output as a batch top-k dataframe.
        
        A classifier gives its top ``k`` classes per batch item. A language
        model (a text decode) gives its top ``k`` next tokens at the last
        position per batch item, with each token's probability under the
        softmax over the vocabulary at that position.
        
        Parameters
        ----------
        k:
            Maximum number of rows (classes, or next tokens) per batch item.
        batch_items:
            ``None`` returns all captured batch items, an integer returns the
            first ``N`` batch items, and a sequence selects explicit batch item
            indices.
        style:
            Optional decode style for a decode on demand (see ``decode_output``).
        head:
            Optional output head for a decode on demand (see ``decode_output``).
        
        Returns
        -------
        pd.DataFrame
            Table with columns ``batch_item``, ``rank``, ``label``, and ``prob``
            for a classification decode, and ``batch_item``, ``rank``,
            ``token_id``, ``text`` and ``prob`` for a text decode (``text`` is
            ``None`` when no tokenizer was available).
        
        Raises
        ------
        ValueError
            If the output logits were not decoded or cannot be recomputed.
        """
    ),
    "show": (
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
    ),
    "draw": (
        """Draw this trace and return the picture as a :class:`~torchlens.viz.Drawing`.
        
        The keywords are the :class:`~torchlens.options.VisualizationOptions`
        field names. ``str(drawing)`` is the Graphviz DOT source, ``drawing.save(path)``
        writes it and ``drawing.show()`` displays it.
        
        Parameters
        ----------
        path:
            Output path; its extension picks the format (``"g.png"`` writes PNG).
            An explicit path is always written (overwriting), printed only when
            the trace is verbose. ``None`` (the default): inside a Jupyter kernel
            the picture displays inline and no file is written; elsewhere a
            collision-safe file is written in the current directory
            (``modelgraph.<format>``, then ``modelgraph-2.<format>``, ...) and
            its path printed. Backward and combined graphs write
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
            ``"graphviz"`` (default) or a renderer registered out of tree.
        direction:
            Edge flow: ``"bottomup"``, ``"topdown"``, ``"leftright"`` or
            ``"rightleft"``. ``None`` takes the graph's default (forward
            ``"bottomup"``, backward ``"topdown"``, combined ``"leftright"``).
        node_style:
            Node label style: ``"default"`` or ``"profiling"``.
        show_buffers:
            Buffer visibility: ``"never"``, ``"meaningful"`` or ``"always"``.
        show_edge_shapes:
            Label forward dataflow edges with producer output shapes (default False);
            multi-output calls label each tensor slot separately.
        show_arg_labels:
            Label the edges into a multi-input operation with the argument position each
            parent fills (``arg 0``, ``arg (0, 1)``); ``False`` draws no argument labels
            (default True).
        show_legend:
            Tri-state legend visibility. ``None`` (default) draws a legend only
            when an encoding channel is active, and the key on a ``module=``
            focused draw; ``True`` draws the full theme legend (on a focused
            draw, with a reading key explaining the picture's notation),
            ``False`` disables the legend even with channels active.
        color_by:
            Fill encoding channel (keyword-only; unstable). A Layer/Op field
            name (``"flops_forward"``), a scalar builtin (``"time"``,
            ``"flops"``, ``"bytes"``, ``"magnitude"``, ``"grad_norm"``), a
            callable ``node -> value``, a mapping of layer label or module
            address to value (an overlay; the legend names it ``overlay``; a
            collapsed module box takes the value at its own address), or
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
            Explicit list of label rows to show on every node. ``"label"`` keeps
            the default title's pass marks (``:2``, ``(x3)``); ``"pass"`` adds
            no row on a single-pass node.
        graph_overrides, edge_overrides, grad_edge_overrides, module_overrides:
            Graphviz attribute overrides for the graph, forward edges, gradient
            edges and module boxes. A static forward-edge ``fontsize`` also sizes
            the edge annotations (pass ranges, IF/THEN, argument labels).
        node_spec_fn, collapsed_node_spec_fn, collapse_fn, skip_fn:
            Per-node styling and selection callbacks (see ``tl.viz.nodes``).
        intervention_mode:
            How intervened sites are marked: ``"node_mark"`` or ``"as_node"``.
        show_cone:
            Shade the downstream cone of every intervention.
        code_panel:
            Source-location panel option (``False``, ``True`` or a panel spec).
        font_size, dpi:
            Typography and raster resolution overrides; ``font_size`` sizes the
            edge annotations when ``edge_overrides`` sets no ``fontsize``.
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
            Format for a path with no extension, or a bare call (``"pdf"``, or
            ``"svg"`` inline); a different format than the path's extension refuses.
        open_viewer:
            ``True`` opens the written picture in the system viewer; the
            default never opens anything.
        view:
            Keyword-only. A premade view: a bundle of draw settings plus
            evidence checks and a caption under the header, each answering one
            question. ``None`` (the default) draws the plain graph.
            ``"overview"`` (what is this model?), ``"debug"`` (what is
            broken?), ``"speed"`` (where does time go?), ``"memory"`` (what is
            big in bytes?), ``"compute"`` (where are the FLOPs?), ``"dims"``
            (what shape is the data?), ``"sequence"`` (how do the passes
            unfold?), ``"transformer"`` (how is attention wired?) and
            ``"interventions"`` (what did the edits change?). A view whose
            evidence or subject is missing refuses typed with a remedy (for
            example ``view_attention_subject_absent``, or
            ``surgery_evidence_missing`` for ``"interventions"`` on a trace
            with no recorded edit, whose full edit record lands beside a
            written figure as ``<path>.census.txt``); any other value refuses
            ``draw_view_invalid`` naming the allowed values. The other keywords
            you pass win over the view's own settings; ``theme=`` stays the
            colour skin and composes with any view.
        
        Returns
        -------
        Drawing
            The picture: DOT source (or animation HTML), its format, the written
            path when a file was written, and the resolved policy.
        """
    ),
    "animate": (
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
    ),
    "nonfinite_ops": (
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
    ),
    "nonfinite_coverage": (
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
    ),
    "first_nonfinite": (
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
    ),
    "preview_fastlog": (
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
    ),
    "last_run_records": (
        """Return fire records from the most recent replay, rerun, or live capture.
        
        Returns
        -------
        tuple[FireRecord, ...]
            Immutable snapshot of matching intervention fire records.
        """
    ),
    "_repr_html_": (
        """Return the notebook HTML representation for this model log.
        
        Returns
        -------
        str
            HTML fragment for IPython/Jupyter display.
        
        Document composition: the Trace card (identity, honesty badge, NaN/Inf line, and the
        budgeted lookup-key index, assembled through
        ``torchlens.report._notebook.cards``) renders first, and on finished
        traces the rebuilt summary table (the
        bare ``trace`` cell is the most likely first touch in Jupyter)
        follows as a sibling fragment. A summary failure degrades to the
        card alone; any failure THERE degrades inside ``safe_card_html``
        to a one-line ``card unavailable`` fragment (never-raise
        boundary). The treescope-bridge suppression sentinel replaces the
        WHOLE document exactly when the bridge just rendered this object
        -- appending the summary there would duplicate the
        render treescope owns.
        """
    ),
    "summary": (
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
    ),
    "narrate": (
        """Render this trace's events in the live-narration grammar.
        
        Post-hoc narration works when ``narrate=`` was never enabled: the same
        renderer serves ``Trace``, ``PartialTrace``, and ``Recording``. Final
        labels render natively here; ``trace.nonfinite_ops`` evidence (when
        available) flags the first nonfinite op in the footer.
        
        Parameters
        ----------
        last:
            Keep only the last ``last`` rows (the crash-tail idiom).
        select:
            Optional filter: a substring, or a callable over
            :class:`torchlens.types.NarrationEvent` rows.
        
        Returns
        -------
        str
            Rendered narration block (no trailing newline).
        """
    ),
}
