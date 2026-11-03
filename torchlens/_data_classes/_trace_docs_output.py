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
        decodes again from the retained output tensors (the output ops'
        values), the persisted ``output_id2label`` and the shipped ImageNet
        bank, without touching ``decoded_output``.
        
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
            Tokenizer with a ``decode`` method, required by ``"hf_text"``.
        
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
        
        Parameters
        ----------
        k:
            Maximum number of class rows per batch item.
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
            Table with columns ``batch_item``, ``rank``, ``label``, and ``prob``.
        
        Raises
        ------
        ValueError
            If classification logits were not decoded or cannot be recomputed.
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
        str | None
            Rendered representation, Graphviz DOT source, or ``None`` when
            rendering is explicitly disabled with ``vis_mode="none"``.
        """
    ),
    "draw": (
        """Render the computational graph for this model log.
        
        Parameters
        ----------
        vis_mode, vis_call_depth, path, vis_graph_overrides, module, node_mode,         node_spec_fn, collapsed_node_spec_fn, collapse_fn, skip_fn, vis_edge_overrides,         vis_grad_edge_overrides, vis_module_overrides, vis_save_only, vis_fileformat,         show_buffers, direction, vis_node_placement, vis_renderer, theme,         vis_intervention_mode, vis_show_cone, code_panel, order_siblings, show_containers,
        container_max_inline, show_input_transform_summary, show_orphans:
            Forwarded unchanged to :func:`torchlens._visualization._render_dot.draw`.
            ``show_orphans=True`` renders orphan (island) ops -- captured but unreachable
            from both inputs and outputs -- as a dashed, greyed cluster of edgeless nodes,
            instead of omitting them. Orphans must have been retained at capture time
            (capture with ``keep_orphans=True``).
            ``show_buffers`` accepts ``"never"``, ``"meaningful"``, or
            ``"always"``. ``direction`` accepts ``"bottomup"``, ``"topdown"``,
            ``"leftright"``, or ``"rightleft"``.
            ``vis_outpath=None`` (the default) writes a collision-safe file in
            the current directory -- ``modelgraph.<format>``, then
            ``modelgraph-2.<format>``, ``modelgraph-3.<format>`` -- and always
            prints its path, so a bare ``draw()`` never overwrites an earlier
            picture. An explicit path is written exactly (overwriting) and is
            printed only when the trace is verbose.
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
        fold_repeats:
            Repeat-fold policy. ``None`` preserves the default policy: off for
            ``collapse="none"`` and band-pressure two-pass folding for
            ``"auto"``/``"max"``. ``True`` folds every eligible repeated run,
            including standalone folding with ``collapse="none"``. ``False``
            disables run folding.
        fold_patterns:
            Declarative user-named pattern folding (DOCUMENTED-UNSTABLE
            spelling; collapse memo D11). ``None`` = off; ``"idiomatic"``
            selects the curated conv-bn-relu preset; a mapping declares named
            linear paths (for example ``{"ConvBnRelu": "conv2d > batch_norm
            > relu"}``; exact repetition ``atom{k}``; an atom naming an
            earlier pattern inlines it). Matched instances render as dashed
            ``PATTERN '<name>' -- N ops`` chips; honesty-refused sites are
            counted and disclosed. v1 supports the pattern-only view
            (``collapse="none"``); combining with automatic collapse refuses
            typed (``pattern_collapse_combination_unsupported``).
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
    ),
    "add_node_overlay": (
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
    ),
    "animate_ops": (
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
    ),
    "nonfinite_ops": (
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
    "draw_backward": (
        """Render the captured backward grad_fn_handle graph.
        
        Parameters
        ----------
        path, vis_graph_overrides, node_spec_fn, collapsed_node_spec_fn,         vis_node_mode, vis_edge_overrides, vis_save_only, file_format,         vis_direction, code_panel, vis_mode, bwd, show_legend:
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
    ),
    "draw_combined": (
        """Render forward ops and backward grad_fns in one graph.
        
        Parameters
        ----------
        path, vis_graph_overrides, node_spec_fn, backward_node_spec_fn,         vis_edge_overrides, vis_save_only, file_format, vis_direction,         vis_mode, intervening_cluster, show_buffers, bwd, show_legend:
            Forwarded unchanged to
            :func:`torchlens._visualization._render_entrypoints.render_combined_graph`.
        
        Returns
        -------
        str
            Graphviz DOT source.
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
    ),
    "render_dagua_graph": (
        """Render this model log with the experimental Dagua backend.
        
        Parameters
        ----------
        vis_mode, vis_call_depth, path, vis_save_only, file_format,         show_buffers, vis_direction, theme:
            Forwarded unchanged to
            ``torchlens.experimental.dagua._bridge.render_trace_with_dagua``.
        
        Returns
        -------
        str
            Serialized Dagua graph output or the rendered artifact path.
        """
    ),
    "to_dagua_graph": (
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
    ),
    "visualization_field_audit": (
        """Return the visualization field-usage audit for this model log.
        
        Returns
        -------
        TorchLensRenderAudit
            Audit of used and unused fields in the visualization bridge.
        """
    ),
    "narrate": (
        """Render this trace's events in the live-narration grammar (snoop D5).
        
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
