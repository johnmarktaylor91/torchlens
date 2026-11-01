"""Full docstrings of the ``Trace`` statistics, views and reports methods.

The methods on ``Trace`` carry a one-line summary; ``_trace_docs.attach_trace_docs`` binds the
text here as their ``__doc__`` when ``trace.py`` finishes defining the class, so ``help()``
and notebooks show the full reference while ``trace.py`` stays inside its file cap.
"""

from __future__ import annotations

__all__ = ["DOCS"]

DOCS: dict[str, str] = {
    "_cost_forward_peak_memory_pair": (
        """Session-time (live, resident) forward peak pair with backend named.
        
        Full contract: :func:`torchlens._capture.peak_memory.read_peak_pair`
        (F20 brainpipe D-7; spelling DOCUMENTED-UNSTABLE). ``None`` on loaded
        artifacts, which never re-measure.
        """
    ),
    "_layers_recurrent": (
        """Access Layers with more than one captured pass.
        
        Returns
        -------
        LayerAccessor
            Accessor containing aggregate Layer records whose ``num_passes`` is
            greater than 1.
        """
    ),
    "_provenance_kind": (
        """Return this product's capture kind: ``"episode"`` or ``"plain"``.
        
        Discoverability seam only (lane F40a; spelling DOCUMENTED-UNSTABLE
        pending the naming session): delegates to
        ``torchlens._capture._episode_ledger.capture_kind_for``. A fresh
        re-execution product whose episode evidence the annotations travel
        policy dropped reads ``"plain"``.
        """
    ),
    "episode": (
        """Return the parsed episode step ledger, or ``None``.
        
        Public step access (lane F40a; spelling DOCUMENTED-UNSTABLE pending
        the naming session): the returned ``EpisodeLedger`` exposes
        ``header``, ordered per-step ``rows``, ``steps_completed``, and
        ``truncated_at_step``. ``None`` for plain captures, pre-finalize
        declarations, and quarantined/dropped payloads -- delegates to
        ``torchlens._capture._episode_ledger.episode_ledger_for``.
        """
    ),
    "episode_coupling": (
        """Return the attested coupling verdict, or ``None`` (lane F42).
        
        The coupling read door (spelling DOCUMENTED-UNSTABLE pending the
        naming session): recomputes the capture digest from this product's
        own persisted facts and compares it to the ledger's binding -- the
        positive "this ledger belongs to this product" claim -- and serves
        the intervention digest, per-step fire counts, fidelity basis, and
        per-segment facts. ``None`` for plain captures and quarantined
        payloads; a digest mismatch refuses typed
        (``episode_coupling_unbound``) -- delegates to
        ``torchlens._capture._episode_coupling.attest_coupling``.
        """
    ),
    "_cost_total_flops_forward": (
        """Total forward FLOPs (the canonical identity-partition aggregation).
        
        Reads the ONE aggregation service (costreport D1) -- never a
        private re-sum over raw per-op fields.
        """
    ),
    "_cost_flops_by_op_type": (
        """Group FLOPs by layer type.
        
        Returns:
            Callable dict mapping layer_type to forward/backward/count totals.
        """
    ),
    "_cost_total_macs_forward": (
        """Total TRUE forward MACs over ops with a known MAC split.
        
        Ops whose split is unknown are EXCLUDED and named by
        ``macs_unknown_split_ops``; when that set is nonempty the figure is a
        lower bound (the summary footer says so).
        """
    ),
    "factcore": (
        """The ONE numbers substrate for this trace (C02; sumfam D1/D6).
        
        DOCUMENTED-UNSTABLE spelling. Counts (with the explicit grain
        menu), the identity spine with refusing joins, parameter truth,
        the canonical compute aggregation, and the D8 payload-scope
        memory figures -- every reporting surface reads THIS, never raw
        trace attributes.
        """
    ),
    "health_facts": (
        """Normalized, serializable health observations (C02; sumfam D9).
        
        DOCUMENTED-UNSTABLE spelling. Basis + pass-qualified nonfinite
        identities (alias rows excluded per D19, disclosed separately) +
        coverage counts + capture revision; persists through save/load on
        the annotations channel; an artifact without a re-derivable basis
        renders NOT-CHECKED, never clean.
        """
    ),
    "nonfinite_verdict": (
        """Three-state health verdict: found / checked_and_clean / not_checked.
        
        DOCUMENTED-UNSTABLE spelling (C02; sumfam D5). The ONLY health
        fact a builder may branch on -- ``nonfinite_ops`` truthiness is a
        false-negative trap on payload-stripped artifacts and is linted
        against in builder code.
        """
    ),
    "unknown_flop_ops": (
        """Named ledger of ops with unknown FLOPs, grouped by op name.
        
        DOCUMENTED-UNSTABLE spelling (A07; costreport D3). Each group carries
        count, example labels/shapes, and the exact ``register_op_rule``
        remedy invocation -- a work queue, never an anonymous exclusion.
        """
    ),
    "compute_coverage": (
        """Four-way compute coverage counts over every layer-list row.
        
        DOCUMENTED-UNSTABLE spelling (A07; costreport D2/D3). Keys:
        ``known`` / ``zero_by_rule`` / ``not_applicable`` / ``unknown`` --
        every row lands in exactly one class (totality is pinned).
        """
    ),
    "_cost_macs_by_kind": (
        """Group TRUE forward MACs by layer type.
        
        Returns:
            Callable dict mapping layer_type to forward/backward/count
            entries. ``backward`` is always ``None`` (not derivable).
        """
    ),
    "transforms": (
        """Return transform-boundary operation records.
        
        Returns
        -------
        tuple[Op, ...]
            Ops whose ``is_transform`` role flag is true.
        """
    ),
    "model_profile": (
        """Return a computed semantic I/O profile descriptor.
        
        Returns
        -------
        ModelProfile
            Runtime-only descriptor derived from preprocessing, raw-input, and
            output-label metadata. This property is not persisted.
        """
    ),
    "profile": (
        """Return a unified resource profile assembled from this trace.
        
        Parameters
        ----------
        level:
            Profile granularity: operation, module address, or module call.
        sort_by:
            Resource column used for sorting. Time is descending by default.
        ascending:
            Whether to sort the selected metric ascending.
        k:
            Number of rows to keep after sorting, or ``None`` for all rows.
        group_by:
            ``"source_line"`` folds op rows into one row per user source line
            (activation bytes, FLOPs and time summed; the groups sum to the
            ungrouped view). Only with ``level="op"``. DOCUMENTED-UNSTABLE.
        
        Returns
        -------
        torchlens.report.ProfileReport
            Printable table object that also exposes ``to_pandas()``.
        
        Notes
        -----
        Per-operation wall-times are captured under instrumentation. Treat them
        as relative hotspot guidance, not clean benchmark timings.
        """
    ),
    "sites": (
        """Return the tabular view of this trace's structural sites.
        
        DOCUMENTED-UNSTABLE spelling (pending naming-session ratification).
        One row per distinct L1 ``site_key`` in first-occurrence execution
        order, aggregating the ops that share the site: ``site_key``,
        ``module_site``, ``layer_type``, ``output_slot``, ``call_ordinal``,
        ``n_ops``, ``labels``, ``layer_labels``, ``passes``, ``shapes``.
        Requires the ``tabular`` extra (pandas).
        
        Returns
        -------
        pandas.DataFrame
            The sites table.
        
        Raises
        ------
        InvalidArgumentError
            ``site_key_unavailable`` for legacy artifacts whose ops carry no
            site keys — consistent with the L1 site accessors, never a
            silently empty table.
        """
    ),
    "bill_of_materials": (
        """Return the inventory of what this trace actually contains.
        
        DOCUMENTED-UNSTABLE spelling (pending naming-session ratification).
        A nested, JSON-friendly dict of sections — ``capture``, ``graph``,
        ``parameters``, ``buffers``, ``activations``, ``backward``, and
        ``annotations`` — every figure read from fields the trace already
        carries (this rollup mints no new claims).
        
        Returns
        -------
        dict[str, Any]
            The inventory sections.
        """
    ),
    "explain": (
        """Explain this trace in plain language (DOCUMENTED-UNSTABLE).
        
        Parameters
        ----------
        audience:
            Report style: ``"researcher"`` adds graph-pattern detail,
            ``"practitioner"`` emphasizes operational status, ``"auto"``
            balances the two.
        max_tokens:
            Optional token budget. Whole sections drop low-value-first until
            the report fits, every drop is disclosed, and the capture-status
            section never drops.
        
        Returns
        -------
        Any
            ``torchlens.report.ExplainReport``, a ``str`` of multi-section prose.
            The machine-readable projection of the same facts is
            ``trace.to_agent_json()``.
        """
    ),
    "capabilities": (
        """Return the report-family capability card for THIS trace (F09/D24).
        
        DOCUMENTED-UNSTABLE spelling (pending naming-session ratification).
        The SurfaceRegistry filtered by the object in hand: which family
        members work, which refuse and why, and what each costs here.
        Metadata-only -- building the card never scans payloads, captures,
        or runs a forward.
        
        Returns
        -------
        Any
            ``torchlens.report._registry.CapabilityCard``.
        """
    ),
    "receptive_fields": (
        """Return the trace-wide geometric receptive-field table.
        
        Parameters
        ----------
        level:
            Table granularity: operation, layer, module invocation, or module.
        input_op:
            Optional model-input ``Op`` handle. String layer names are rejected.
        statuses:
            Optional typed receptive-field-status filter.
        sort_by:
            Optional table column used for stable sorting.
        ascending:
            Whether an explicit ``sort_by`` sorts ascending.
        
        Returns
        -------
        torchlens.receptive_field.ReceptiveFieldProfile
            Frozen table wrapper exposing ``to_pandas()``.
        """
    ),
    "module_collapse_order": (
        """Every module's smart-collapse priority (``Module.collapse_score``), highest first.
        
        The priority is the module's hidden-op count (its subtree's distinct
        ops minus one) over the largest hidden-op count of any module in the
        trace, rounded to 6 places; a module the renderer never collapses (the
        root, one with at most one Layer, or the only child of a non-root module
        that hides the same ops, whose parent ranks instead) reads 0.0. It ranks
        candidates and does not depend on the plan the optimizer selects
        (``collapse_plan()``), which may skip a high-priority module. Past the pathological-input pre-gate (20x
        ``COLLAPSE_OPTIMIZER_MAX_OPS`` ops) the ranking is empty.
        
        Returns
        -------
        list[tuple[str, float]]
            ``(module_address, priority)`` sorted by ``(-priority, address)``.
        """
    ),
    "collapse_order": (
        """Return smart-collapse ranking for a policy.
        
        The documented-inert ``weights=`` parameter is REMOVED (collapse
        memo D9; clean-v2 hard-rename posture, no warn shim; MIGRATIONS.md).
        
        Parameters
        ----------
        mode:
            ``"auto"`` or ``"max"`` landmark policy.
        
        Returns
        -------
        list[tuple[str, float]]
            ``(module_address, rounded_score)`` sorted by ``(-score, address)``.
        """
    ),
    "collapse_plan": (
        """Return the v2 diagnostic collapse plan for this trace.
        
        Parameters
        ----------
        mode:
            Collapse policy to plan, either ``"auto"`` for a readable overview
            or ``"max"`` for aggressive condensation. A float in ``[0.0, 1.0]``
            selects the public monotone collapse schedule, where ``0.0`` is
            the full graph and ``1.0`` is byte-identical to ``"max"``.
        context:
            Optional :class:`torchlens._visualization.collapse_plan.RenderContext`.
            When omitted, the default unrolled Graphviz context is used.
        
        Returns
        -------
        CollapsePlan
            Renderer-faithful diagnostic plan. Node kinds are ``ModuleBox`` for
            collapsed module calls, ``RawOp`` for exposed operations,
            ``RepeatFold`` for representative-plus-ellipsis repeated runs,
            ``OpSegment`` and ``ChildSegment`` for condensed segment boxes, and
            ``Boundary`` for input/output or renderer boundary nodes.
        
        Raises
        ------
        ValueError
            If ``mode`` is not ``"auto"``, ``"max"``, or a float in
            ``[0.0, 1.0]``, or the v2 optimizer declines to produce a plan for
            the provided context.
        """
    ),
    "collapse_schedule": (
        """Return the public monotone float collapse schedule.
        
        Parameters
        ----------
        context:
            Optional :class:`torchlens._visualization.collapse_plan.RenderContext`.
            When omitted, the default unrolled Graphviz context is used.
        
        Returns
        -------
        CollapseSchedule
            Ordered schedule from ``t=0.0`` full graph to ``t=1.0`` max
            collapse. Each step includes the selected plan, visible count, and
            collapsed module-address set.
        """
    ),
    "modules_with_facet": (
        """Yield Modules whose semantic facet view has a facet available now.
        
        Parameters
        ----------
        name:
            Facet name to look up using the current capture's available values.
        """
    ),
    "_root_call": (
        """Return the top-level ModuleCall for internal call-tree traversal.
        
        When multiple top-level calls exist, the first one in trace insertion
        order is returned.
        """
    ),
    "call_tree": (
        """Return the root ModuleCall for the nested dynamic call tree.
        
        Returns
        -------
        ModuleCall
            Top-level call whose ``call_children`` labels form the nested
            ModuleCall tree displayed by ``show_call_tree``.
        """
    ),
    "walk_calls": (
        """Yield every ModuleCall in call-tree depth-first order.
        
        Yields
        ------
        ModuleCall
            ModuleCall records ordered by dynamic call-tree traversal.
        """
    ),
    "show_call_tree": (
        """Print this Trace's full ModuleCall tree as an ASCII tree.
        
        Parameters
        ----------
        max_depth:
            Maximum descendant depth to print, or ``None`` for no limit.
        include_atomic:
            Whether to include atomic module leaves.
        show_call_index:
            Whether to include the ``:N`` call suffix in labels.
        file:
            Optional output stream. ``None`` prints to stdout.
        """
    ),
    "_graph_branching_factor": (
        """Mean fan-out: children (consumers) per compute Op.
        
        Computed over a single, consistent node set (compute Ops) so the ratio is
        a coherent mean out-degree: ~1.0 for a plain chain, >1.0 when ops are
        reused (residual streams, shared embeddings, dense connectivity). Counts
        every child of each compute Op (including boundary/buffer consumers).
        """
    ),
    "_autograd_intermediate_derived_grads": (
        """Store exact op-level derived gradient records.
        
        Parameters
        ----------
        value
            Accessor to expose through ``trace.autograd.intermediate_derived_grads``.
        """
    ),
    "_autograd_grad_fn_fire_timings": (
        """Return live per-fire backward timing spans keyed like ``grad_fn_calls``.
        
        DOCUMENTED-UNSTABLE spelling (L9 memo 1.3; pending naming-session
        ratification). Serves the paired ``time.perf_counter()`` stamps
        carried by the runtime ``GradFnFired`` events: a timed fire yields
        its span as a :class:`~torchlens._quantities.Duration`, an untimed
        fire (empty keyed LIFO, stale-key discard, or timing-registration
        failure) yields ``None`` -- never a false zero. Keys are
        ``"<grad_fn_label>:<call_index>"`` in fold order.
        
        Raises
        ------
        InvalidArgumentError
            ``grad_fn_fire_timing_unavailable`` on a trace without its
            runtime capture event stream (loaded artifacts, cleaned traces):
            events never persist, so such a read has no timing evidence
            until the coordinated tlspec bump persists the pairs.
        """
    ),
    "grad_fn_site_summary": (
        """Return the read-only per-site rollup of backward grad-fn facts.
        
        DOCUMENTED-UNSTABLE spelling (L9 memo 1.1 grouped-backward floor;
        pending naming-session ratification). Aggregates GradFn/GradFnCall
        facts per L1 ``site_key`` (read-only L1 consumption -- reused-module
        grad-fns share one entry): per entry ``grad_fn_labels``,
        ``fire_count``, ``pass_coverage``, and -- when live per-fire timing
        evidence exists -- ``timed_fire_count`` plus ``total_fire_duration``
        (``None`` when no fire carries timing evidence, never a false zero).
        Grad-fns without an op FK (AccumulateGrad and other unattributed
        nodes) aggregate under the ``None`` key. Accessor-level only: no
        persisted fields.
        
        Raises
        ------
        InvalidArgumentError
            ``site_key_unavailable`` when op-backed grad-fns exist but no op
            carries a site key (legacy pre-site-key artifact) -- consistent
            with the L1 site accessors, never a silently keyless rollup.
        """
    ),
    "_module_calls_with_output_in": (
        """Access ModuleCalls with at least one output Op among ``op_labels``.
        
        The root call's layer-space output labels resolve to their Op labels
        first (``op_label_aliases``), so the root call counts like any other.
        
        Parameters
        ----------
        op_labels:
            Pass-qualified Op labels to match against each call's outputs.
        
        Returns
        -------
        Accessor[Any]
            Matching ModuleCalls keyed by call label, in call order.
        """
    ),
    "num_buffer_layers": (
        """Number of distinct buffer-boundary Layers.
        
        ``buffer_layers`` lists a buffer Layer once per pass (a buffer read in
        a loop repeats), so the count dedupes it.
        """
    ),
    "buffer_write_ops": (
        """Labels for buffer Ops that record registered-buffer write events.
        
        Note: buffer writes are only tracked in exhaustive capture. A Trace
        cooked from a ``record(...)`` Recording (predicate mode) does not track
        buffer writes, so this list is empty even for a model that mutated its
        buffers (e.g. training-mode BatchNorm); ``buffer_write_kind is None``
        there means "not tracked", not "read-only". See
        ``Recording.to_trace``'s Notes.
        """
    ),
    "num_uncalled_modules": (
        """Number of registered source-model modules not called in this Trace.
        
        Recorded when the capture finished, so it survives the source model
        being garbage-collected. ``None`` when it cannot be known: a loaded or
        unpickled trace (the record is session-time) whose model is gone.
        """
    ),
    "between": (
        """Return the sub-DAG view carrying influence from sources to sinks.
        
        The graph-VIEW binding of the one influence-region idea: the same
        member set the ``tl.between(sources, sinks)`` producer selects,
        presented as a :class:`torchlens.types.TraceSlice` — member
        ops, internal dataflow edges, and an EXPLICIT boundary (every edge
        crossing in or out is declared, so external dependencies are
        visible rather than silently dropped). A slice is a presenter,
        never a ``Trace``: it offers no save/replay/validate. No directed
        path yields the EMPTY slice (emptiness is disclosure).
        DOCUMENTED-UNSTABLE spelling.
        
        Parameters
        ----------
        sources:
            One region or a list of regions: site label strings,
            ``Op``/``Layer`` handles, or any ACT selection.
        sinks:
            Same forms as ``sources``.
        
        Returns
        -------
        torchlens.types.TraceSlice
            Frozen sub-DAG view (session-time only; never persisted).
        """
    ),
    "subgraph": (
        """Return the sub-DAG view of any ACT region of this trace.
        
        The general slice door: whatever produced the region —
        ``tl.neighborhood(...)``, ``tl.between(...)``, an explicit
        ``tl.units(...)``, an ``Op``/``Layer``, or a future graph-motif
        producer's hits — its touched-site FAMILY becomes the member set
        (element masks never shrink a graph region), presented with the
        same explicit-boundary :class:`~torchlens.types.TraceSlice`
        contract as :meth:`between`. DOCUMENTED-UNSTABLE spelling.
        
        Parameters
        ----------
        selection:
            A selection-shaped ACT region (Selection, ResolvedSelection,
            Op/Layer, receptive-field region, or site label string).
            PARAM/EDGE selections refuse ``selection_kind_incompatible``.
        
        Returns
        -------
        torchlens.types.TraceSlice
            Frozen sub-DAG view (session-time only; never persisted).
        """
    ),
}
