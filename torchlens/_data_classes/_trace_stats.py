"""Trace computed stats mixin."""

from collections import OrderedDict
from collections.abc import Collection, Iterator
from typing import TYPE_CHECKING, Any, Literal, TextIO, cast

if TYPE_CHECKING:
    from .._trace_slice import TraceSlice
    from .._visualization.collapse_plan import CollapsePlan, CollapseSchedule, RenderContext
    from ..receptive_field._types import (
        ReceptiveFieldDirection,
        ReceptiveFieldProfile,
        ReceptiveFieldStatus,
    )
    from ..report._explain import ExplainReport
    from ..report._profile import ProfileReport
    from ..report._trace_facts import _CallableDict
    from .buffer import BufferAccessor
    from .layer import LayerAccessor
    from .module import Module, ModuleCall
    from .trace import Trace

    _TraceMixinBase = Trace
else:
    _TraceMixinBase = object

from .._errors import InvalidArgumentError
from .._quantities import Duration, Flops, Macs, as_duration
from . import (
    _call_tree,
    _doors,
    _trace_inventory,
    _trace_profile,
    _trace_stats_grad_fn_site,
    _trace_views,
)
from ._accessor_base import Accessor
from ._trace_accessors import (
    OrphanAccessor,
    TraceGradFnCallAccessor,
    TraceModuleCallAccessor,
    TraceOpAccessor,
)
from ._trace_groups import TraceAutograd, TraceCost, TraceGraph, TraceProvenance
from ._trace_groups_capture import TraceHealth, TraceOptions, TraceSource
from ._trace_profile import ModelProfile
from ._trace_stats_grad_fn_site import _fire_timings_by_key
from .backward_pass import BackwardPass, BackwardPassAccessor
from .derived_grad import IntermediateDerivedGradAccessor
from .grad_fn import GradFnAccessor
from .layer import Layer
from .module import ModuleAccessor
from .op import Op
from .param import ParamAccessor


class TraceStatsMixin(_TraceMixinBase):
    """``Trace`` computed-statistics surface: derived counts, edges, and summaries."""

    # ********************************************
    # ********** Computed Properties *************
    # ********************************************

    @property
    def cost(self: "Trace") -> "TraceCost":
        """Compute, memory and timing totals of a Trace."""
        return TraceCost(self)

    @property
    def graph(self: "Trace") -> "TraceGraph":
        """Graph-shape facts of a Trace: edges, degrees, branching and conditionals."""

        return TraceGraph(self)

    @property
    def autograd(self: "Trace") -> "TraceAutograd":
        """Backward-pass records and grad_fn facts of a Trace."""

        return TraceAutograd(self)

    @property
    def source(self: "Trace") -> "TraceSource":
        """Source facts of the traced model class: locations, signatures, docstrings."""

        return TraceSource(self)

    @property
    def provenance(self: "Trace") -> "TraceProvenance":
        """How, when and from what this Trace was captured."""

        return TraceProvenance(self)

    @property
    def options(self: "Trace") -> "TraceOptions":
        """The capture options this Trace ran with (the stored snapshot)."""

        return TraceOptions(self)

    @property
    def health(self: "Trace") -> "TraceHealth":
        """Honesty diagnostics of the capture: witness and escape-detector facts."""

        return TraceHealth(self)

    @property
    def _graph_is_recurrent(self: "Trace") -> bool:
        """Whether any layer has more than one pass."""
        return any(v > 1 for v in self._layer_num_calls.values())

    @property
    def _cost_forward_peak_memory_pair(self: "Trace") -> "dict[str, Any] | None":
        """Session-time (live, resident) forward peak pair with backend named.

        Full contract: :func:`torchlens._capture.peak_memory.read_peak_pair`
        (F20 brainpipe D-7; spelling DOCUMENTED-UNSTABLE). ``None`` on loaded
        artifacts, which never re-measure.
        """
        from .._capture.peak_memory import read_peak_pair

        return read_peak_pair(self)

    @property
    def _layers_recurrent(self: "Trace") -> "LayerAccessor":
        """Access Layers with more than one captured pass.

        Returns
        -------
        LayerAccessor
            Accessor containing aggregate Layer records whose ``num_passes`` is
            greater than 1.
        """

        return _trace_views.layers_recurrent(self)

    @property
    def _graph_max_ops_per_layer(self: "Trace") -> int:
        """Maximum number of ops for any layer."""
        return max(self._layer_num_calls.values(), default=1)

    @property
    def _graph_is_branching(self: "Trace") -> bool:
        """Whether any layer has more than one child."""
        return any(len(entry.children) > 1 for entry in self._layer_list)

    @property
    def _graph_has_conditional_branching(self: "Trace") -> bool:
        """Whether any layer is in a conditional branch."""
        return any(entry.is_in_conditional_body for entry in self._layer_list)

    @property
    def _graph_has_conditionals(self: "Trace") -> bool:
        """Whether this Trace contains conditional-flow records."""

        return len(self._graph_conditionals) > 0

    @property
    def _graph_is_dynamic(self: "Trace") -> bool:
        """Whether execution depends on runtime tensor values."""

        return self.graph.has_conditionals

    @property
    def _source_forward_location(self: "Trace") -> str | None:
        """Combined forward source location."""

        if self._source_forward_file is None or self._source_forward_line is None:
            return None
        return f"{self._source_forward_file}:{self._source_forward_line}"

    @property
    def _source_class_location(self: "Trace") -> str | None:
        """Combined model class source location."""

        if self._source_class_file is None or self._source_class_line is None:
            return None
        return f"{self._source_class_file}:{self._source_class_line}"

    @property
    def _source_init_location(self: "Trace") -> str | None:
        """Combined model ``__init__`` source location."""

        if self._source_init_file is None or self._source_init_line is None:
            return None
        return f"{self._source_init_file}:{self._source_init_line}"

    @property
    def _cost_last_backward_duration(self: "Trace") -> Duration | None:
        """Most recent backward-pass duration, if any."""

        if not self._cost_backward_durations:
            return None
        return as_duration(self._cost_backward_durations[-1])

    @property
    def _cost_total_backward_duration(self: "Trace") -> Duration:
        """Sum of all captured backward-pass durations."""

        return Duration(sum(self._cost_backward_durations))

    @property
    def _autograd_last_backward_root_grad_fn_object_id(self: "Trace") -> int | None:
        """Most recent backward root grad_fn_handle object id, if any."""

        if not self._autograd_backward_root_grad_fn_object_ids:
            return None
        return self._autograd_backward_root_grad_fn_object_ids[-1]

    @property
    def _cost_overhead_duration(self: "Trace") -> Duration:
        """Time spent on TorchLens overhead (total minus function calls)."""
        return self._provenance_duration - self._cost_func_calls_duration

    @property
    def _provenance_duration(self: "Trace") -> Duration:
        """Total capture-phase duration in seconds."""

        if not self._provenance_start_time or not self._provenance_end_time:
            return Duration(0)
        return Duration(self._provenance_end_time - self._provenance_start_time)

    @property
    def _provenance_kind(self: "Trace") -> str:
        """Return this product's capture kind: ``"episode"`` or ``"plain"``.

        Discoverability seam only (lane F40a; spelling DOCUMENTED-UNSTABLE
        pending the naming session): delegates to
        ``torchlens._capture._episode_ledger.capture_kind_for``. A fresh
        re-execution product whose episode evidence the annotations travel
        policy dropped reads ``"plain"``.
        """

        return _doors.report()._trace_facts.provenance_kind(self)

    @property
    def episode(self: "Trace") -> Any | None:
        """Return the parsed episode step ledger, or ``None``.

        Public step access (lane F40a; spelling DOCUMENTED-UNSTABLE pending
        the naming session): the returned ``EpisodeLedger`` exposes
        ``header``, ordered per-step ``rows``, ``steps_completed``, and
        ``truncated_at_step``. ``None`` for plain captures, pre-finalize
        declarations, and quarantined/dropped payloads -- delegates to
        ``torchlens._capture._episode_ledger.episode_ledger_for``.
        """

        return _doors.report()._trace_facts.episode(self)

    @property
    def episode_coupling(self: "Trace") -> Any | None:
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

        return _doors.report()._trace_facts.episode_coupling(self)

    # ********************************************
    # ************* FLOPs Properties *************
    # ********************************************
    # FLOPs are estimated per-operation during logging (flops_forward,
    # flops_backward on each Op).  These properties aggregate
    # across the entire model.  Layers with None FLOPs (unknown ops) are
    # skipped, so the totals may undercount.

    @property
    def _cost_total_flops_forward(self: "Trace") -> Flops:
        """Total forward FLOPs (the canonical identity-partition aggregation).

        Reads the ONE aggregation service (costreport D1) -- never a
        private re-sum over raw per-op fields.
        """

        return _doors.report()._trace_facts.cost_total_flops_forward(self)

    @property
    def _cost_total_flops_backward(self: "Trace") -> Flops:
        """Total backward FLOPs across all layers (skipping None/unknown)."""
        return Flops(
            sum(
                entry.flops_backward
                for entry in self._layer_list
                if entry.flops_backward is not None
            )
        )

    @property
    def _cost_total_flops(self: "Trace") -> Flops:
        """Total FLOPs (forward + backward)."""
        return Flops(self.cost.total_flops_forward + self.cost.total_flops_backward)

    @property
    def _cost_flops_by_op_type(self: "Trace") -> "_CallableDict":
        """Group FLOPs by layer type.

        Returns:
            Callable dict mapping layer_type to forward/backward/count totals.
        """

        return _doors.report()._trace_facts.cost_flops_by_op_type(self)

    # ********************************************
    # ************** MACs Properties *************
    # ********************************************
    # TRUE multiply-accumulate counts, derived from each op's two-term
    # compute record (fma_macs / other_flops) -- NEVER flops // 2. A ReLU has
    # zero MACs; a biased Linear's bias adds are not MACs. Backward MACs are
    # not derivable (backward FLOPs are multiplier estimates) and read None.

    @property
    def _cost_total_macs_forward(self: "Trace") -> Macs:
        """Total TRUE forward MACs over ops with a known MAC split.

        Ops whose split is unknown are EXCLUDED and named by
        ``macs_unknown_split_ops``; when that set is nonempty the figure is a
        lower bound (the summary footer says so).
        """

        return _doors.report()._trace_facts.cost_total_macs_forward(self)

    @property
    def factcore(self: "Trace") -> Any:
        """The ONE numbers substrate for this trace (C02; sumfam D1/D6).

        DOCUMENTED-UNSTABLE spelling. Counts (with the explicit grain
        menu), the identity spine with refusing joins, parameter truth,
        the canonical compute aggregation, and the D8 payload-scope
        memory figures -- every reporting surface reads THIS, never raw
        trace attributes.
        """

        return _doors.report()._trace_facts.factcore(self)

    @property
    def health_facts(self: "Trace") -> Any:
        """Normalized, serializable health observations (C02; sumfam D9).

        DOCUMENTED-UNSTABLE spelling. Basis + pass-qualified nonfinite
        identities (alias rows excluded per D19, disclosed separately) +
        coverage counts + capture revision; persists through save/load on
        the annotations channel; an artifact without a re-derivable basis
        renders NOT-CHECKED, never clean.
        """

        return _doors.report()._trace_facts.health_facts(self)

    @property
    def nonfinite_verdict(self: "Trace") -> str:
        """Three-state health verdict: found / checked_and_clean / not_checked.

        DOCUMENTED-UNSTABLE spelling (C02; sumfam D5). The ONLY health
        fact a builder may branch on -- ``nonfinite_ops`` truthiness is a
        false-negative trap on payload-stripped artifacts and is linted
        against in builder code.
        """

        return _doors.report()._trace_facts.nonfinite_verdict(self)

    @property
    def unknown_flop_ops(self: "Trace") -> tuple[Any, ...]:
        """Named ledger of ops with unknown FLOPs, grouped by op name.

        DOCUMENTED-UNSTABLE spelling (A07; costreport D3). Each group carries
        count, example labels/shapes, and the exact ``register_op_rule``
        remedy invocation -- a work queue, never an anonymous exclusion.
        """

        return _doors.report()._trace_facts.unknown_flop_ops(self)

    @property
    def compute_coverage(self: "Trace") -> dict[str, int]:
        """Four-way compute coverage counts over every layer-list row.

        DOCUMENTED-UNSTABLE spelling (A07; costreport D2/D3). Keys:
        ``known`` / ``zero_by_rule`` / ``not_applicable`` / ``unknown`` --
        every row lands in exactly one class (totality is pinned).
        """

        return _doors.report()._trace_facts.compute_coverage(self)

    @property
    def macs_unknown_split_ops(self: "Trace") -> tuple[str, ...]:
        """Labels of compute ops whose exact MAC split cannot be derived."""

        return _doors.report()._trace_facts.macs_unknown_split_ops(self)

    @property
    def _cost_total_macs_backward(self: "Trace") -> Macs | None:
        """Backward MACs are not derivable and always ``None`` (estimates only)."""

        return None

    @property
    def _cost_total_macs(self: "Trace") -> Macs | None:
        """Total MACs are not derivable and always ``None`` (see total_macs_backward)."""

        return None

    @property
    def _cost_macs_by_kind(self: "Trace") -> "_CallableDict":
        """Group TRUE forward MACs by layer type.

        Returns:
            Callable dict mapping layer_type to forward/backward/count
            entries. ``backward`` is always ``None`` (not derivable).
        """

        return _doors.report()._trace_facts.cost_macs_by_kind(self)

    # ********************************************
    # ************* Params Accessor **************
    # ********************************************

    @property
    def params(self: "Trace") -> ParamAccessor:
        """Access parameter metadata by address, short name, or index."""
        return self._param_logs

    # Parameter-truth read surface (A2/A3; DOCUMENTED-UNSTABLE spellings
    # pending naming-session ratification). The accounting basis everywhere is
    # the Python Parameter OBJECT: ``param_logs`` is object-deduplicated at the
    # pre-forward scan, so ties are counted once and disclosed, never merged.

    @property
    def _graph_tied_param_groups(self: "Trace") -> tuple[tuple[str, ...], ...]:
        """Alias-address groups of parameters registered at more than one path."""

        return _trace_views.graph_tied_param_groups(self)

    @property
    def ops(self: "Trace") -> TraceOpAccessor:
        """Access per-invocation Op records by label or index."""

        return _trace_views.ops(self)

    @property
    def transforms(self: "Trace") -> tuple[Op, ...]:
        """Return transform-boundary operation records.

        Returns
        -------
        tuple[Op, ...]
            Ops whose ``is_transform`` role flag is true.
        """

        return _trace_views.transforms(self)

    @property
    def model_profile(self: "Trace") -> ModelProfile:
        """Return a computed semantic I/O profile descriptor.

        Returns
        -------
        ModelProfile
            Runtime-only descriptor derived from preprocessing, raw-input, and
            output-label metadata. This property is not persisted.
        """

        return _trace_profile.model_profile(self)

    def profile(
        self: "Trace",
        level: Literal["op", "module", "call"] = "op",
        *,
        sort_by: Literal["time", "flops", "activation_memory", "param_count"] = "time",
        ascending: bool = False,
        k: int | None = None,
        group_by: Literal["source_line"] | None = None,
    ) -> "ProfileReport":
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

        return _doors.report()._trace_facts.profile(
            self, level, sort_by=sort_by, ascending=ascending, k=k, group_by=group_by
        )

    def sites(self: "Trace") -> Any:
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

        return _trace_inventory.build_sites_table(self)

    def bill_of_materials(self: "Trace") -> dict[str, Any]:
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

        return _doors.report()._trace_facts.bill_of_materials(self)

    def explain(
        self: "Trace",
        *,
        audience: Literal["researcher", "practitioner", "auto"] = "auto",
        max_tokens: int | None = None,
    ) -> "ExplainReport":
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

        return _doors.report()._trace_facts.explain(self, audience=audience, max_tokens=max_tokens)

    def capabilities(self: "Trace") -> Any:
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

        return _doors.report()._trace_facts.capabilities(self)

    def receptive_fields(
        self: "Trace",
        level: Literal["op", "layer", "call", "module"] = "op",
        *,
        input_op: "Op | None" = None,
        statuses: "Collection[ReceptiveFieldStatus] | None" = None,
        sort_by: str | None = None,
        ascending: bool = True,
        direction: "ReceptiveFieldDirection | str" = "receptive",
    ) -> "ReceptiveFieldProfile":
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

        return _doors.receptive_field()._trace_tables.receptive_fields(
            self,
            level,
            input_op=input_op,
            statuses=statuses,
            sort_by=sort_by,
            ascending=ascending,
            direction=direction,
        )

    def projective_fields(
        self: "Trace",
        level: Literal["op", "layer", "call", "module"] = "op",
        *,
        input_op: "Op | None" = None,
        statuses: "Collection[ReceptiveFieldStatus] | None" = None,
        sort_by: str | None = None,
        ascending: bool = True,
    ) -> "ReceptiveFieldProfile":
        """Return the trace-wide source-anchored projective-field table."""

        return _doors.receptive_field()._trace_tables.projective_fields(
            self, level, input_op=input_op, statuses=statuses, sort_by=sort_by, ascending=ascending
        )

    @property
    def layers(self: "Trace") -> "LayerAccessor":
        """Access aggregate per-layer metadata by label, index, or pass notation."""

        return _trace_views.layers(self)

    @property
    def modules(self: "Trace") -> "ModuleAccessor":
        """Access structured per-module metadata by address, index, or pass notation."""
        return self._module_logs

    @property
    def module_collapse_order(self: "Trace") -> list[tuple[str, float]]:
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

        return _doors.visualization()._trace_collapse.module_collapse_order(self)

    def collapse_order(
        self: "Trace",
        mode: Literal["auto", "max"] = "auto",
    ) -> list[tuple[str, float]]:
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

        return _doors.visualization()._trace_collapse.collapse_order(self, mode)

    def collapse_plan(
        self: "Trace",
        mode: Literal["auto", "max"] | float = "auto",
        context: "RenderContext | None" = None,
    ) -> "CollapsePlan":
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

        return _doors.visualization()._trace_collapse.collapse_plan(self, mode, context)

    def collapse_schedule(
        self: "Trace",
        context: "RenderContext | None" = None,
    ) -> "CollapseSchedule":
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

        return _doors.visualization()._trace_collapse.collapse_schedule(self, context)

    def modules_with_facet(self: "Trace", name: str) -> Iterator[Any]:
        """Yield Modules whose semantic facet view has a facet available now.

        Parameters
        ----------
        name:
            Facet name to look up using the current capture's available values.
        """

        return (module for module in self.modules if module.facets.has(name))

    def attention_blocks(self: "Trace") -> Iterator[Any]:
        """Yield Modules with an available query-projection facet."""

        return self.modules_with_facet("q")

    @property
    def module_calls(self: "Trace") -> TraceModuleCallAccessor:
        """Access per-invocation ModuleCall records by call label or index."""

        return _trace_views.module_calls(self)

    def _root_call(self: "Trace") -> "ModuleCall":
        """Return the top-level ModuleCall for internal call-tree traversal.

        When multiple top-level calls exist, the first one in trace insertion
        order is returned.
        """

        return _call_tree.root_call(self)

    @property
    def call_tree(self: "Trace") -> "ModuleCall":
        """Return the root ModuleCall for the nested dynamic call tree.

        Returns
        -------
        ModuleCall
            Top-level call whose ``call_children`` labels form the nested
            ModuleCall tree displayed by ``show_call_tree``.
        """

        return self._root_call()

    def walk_calls(self: "Trace") -> Iterator["ModuleCall"]:
        """Yield every ModuleCall in call-tree depth-first order.

        Yields
        ------
        ModuleCall
            ModuleCall records ordered by dynamic call-tree traversal.
        """

        return _call_tree.walk_calls(self)

    def show_call_tree(
        self: "Trace",
        max_depth: int | None = None,
        include_atomic: bool = True,
        show_call_index: bool = True,
        file: TextIO | None = None,
    ) -> None:
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

        return _call_tree.show_call_tree(self, max_depth, include_atomic, show_call_index, file)

    @property
    def root_module(self: "Trace") -> "Module":
        """The root module (the model itself)."""
        return cast("Module", self._module_logs["self"])

    @property
    def num_edges(self: "Trace") -> int:
        """Distinct edges in the per-pass Op graph, including boundary edges."""

        return _trace_views.num_edges(self)

    @property
    def _graph_num_compute_edges(self: "Trace") -> int:
        """Distinct Op graph edges whose endpoints are both compute Ops."""

        return _trace_views.graph_num_compute_edges(self)

    @property
    def _graph_num_buffer_edges(self: "Trace") -> int:
        """Distinct Op graph edges with at least one buffer endpoint."""

        return _trace_views.graph_num_buffer_edges(self)

    @property
    def _graph_num_layer_edges(self: "Trace") -> int:
        """Distinct edges in the aggregate Layer graph."""

        return _trace_views.graph_num_layer_edges(self)

    @property
    def _autograd_num_backward_edges(self: "Trace") -> int | None:
        """Distinct edges in the backward GradFn graph, or ``None`` if ungated."""

        return _trace_views.autograd_num_backward_edges(self)

    @property
    def _graph_branching_factor(self: "Trace") -> float:
        """Mean fan-out: children (consumers) per compute Op.

        Computed over a single, consistent node set (compute Ops) so the ratio is
        a coherent mean out-degree: ~1.0 for a plain chain, >1.0 when ops are
        reused (residual streams, shared embeddings, dense connectivity). Counts
        every child of each compute Op (including boundary/buffer consumers).
        """

        return _trace_views.graph_branching_factor(self)

    @property
    def _graph_max_in_degree(self: "Trace") -> int:
        """Maximum number of parents over compute Ops."""

        return _trace_views.graph_max_in_degree(self)

    @property
    def _graph_max_out_degree(self: "Trace") -> int:
        """Maximum number of children over compute Ops."""

        return _trace_views.graph_max_out_degree(self)

    @property
    def _autograd_intermediate_derived_grads(self: "Trace") -> IntermediateDerivedGradAccessor:
        """Access exact op-level derived gradient records, when a backend provides them.

        Returns
        -------
        IntermediateDerivedGradAccessor
            Records keyed by pass-qualified op label. Backends that did not run
            an intermediate-derived-gradient pass return an empty accessor.
        """

        return _trace_views.autograd_intermediate_derived_grads(self)

    @_autograd_intermediate_derived_grads.setter
    def _autograd_intermediate_derived_grads(
        self: "Trace", value: IntermediateDerivedGradAccessor
    ) -> None:
        """Store exact op-level derived gradient records.

        Parameters
        ----------
        value
            Accessor to expose through ``trace.autograd.intermediate_derived_grads``.
        """

        self.__dict__["_intermediate_derived_grads"] = value

    @property
    def has_trainable_params(self: "Trace") -> bool:
        """Whether this Trace contains at least one trainable parameter."""

        return self.num_params_trainable > 0

    @property
    def has_frozen_params(self: "Trace") -> bool:
        """Whether this Trace contains at least one frozen parameter."""

        return self.num_params_frozen > 0

    @property
    def buffers(self: "Trace") -> "BufferAccessor":
        """Access buffer metadata by address, short name, or index."""
        return self._buffer_accessor  # type: ignore[return-value]

    @property
    def _graph_orphans(self: "Trace") -> OrphanAccessor:
        """Access retained orphan island operations by raw or final label."""

        return _trace_views.graph_orphans(self)

    @property
    def _autograd_grad_fns(self: "Trace") -> GradFnAccessor:
        """Access backward grad_fn_handle metadata by label, index, pass label, or substring."""

        return _trace_views.autograd_grad_fns(self)

    @property
    def _autograd_grad_fn_calls(self: "Trace") -> TraceGradFnCallAccessor:
        """Access per-invocation GradFnCall records by qualified label or index."""

        return _trace_views.autograd_grad_fn_calls(self)

    @property
    def _autograd_grad_fn_fire_timings(self: "Trace") -> "OrderedDict[str, Duration | None]":
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

        from .._ir.events import GradFnFired

        stream = self.__dict__.get("capture_events") or self.__dict__.get("_capture_events")
        fired = [
            event
            for event in getattr(stream, "backward_events", ())
            if isinstance(event, GradFnFired)
        ]
        # Evidence test, not a load flag: a rehydrated trace owns a fresh
        # EMPTY stream, so backward records without any fire event mean the
        # runtime evidence did not travel (loaded artifact or cleaned trace).
        if stream is None or (
            not fired and ({} if self.__dict__.get("_tl_cleaned_up", False) else self._grad_fn_logs)
        ):
            raise InvalidArgumentError(
                "Per-fire backward timing is served from the runtime capture "
                "event stream, which never persists: this trace (loaded from "
                "an artifact, or already cleaned up) carries no per-fire "
                "timing evidence for its backward records.",
                code="grad_fn_fire_timing_unavailable",
                remedy=(
                    "read grad_fn_fire_timings on the live capturing trace; "
                    "persisted per-fire timing activates at the coordinated "
                    "tlspec version bump"
                ),
            )
        grad_fn_logs = {} if self.__dict__.get("_tl_cleaned_up", False) else self._grad_fn_logs
        return _fire_timings_by_key(fired, grad_fn_logs)

    @property
    def grad_fn_site_summary(self: "Trace") -> "OrderedDict[str | None, dict[str, Any]]":
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

        return _trace_stats_grad_fn_site.grad_fn_site_summary(self)

    @property
    def _autograd_backward_passes(self: "Trace") -> BackwardPassAccessor:
        """Access backward pass records by 0-based position or named pass number."""

        return _trace_views.autograd_backward_passes(self)

    @property
    def _autograd_last_backward_pass(self: "Trace") -> BackwardPass | None:
        """Return the most recent backward pass record, if any."""

        backward_passes = self.autograd.backward_passes
        if not backward_passes:
            return None
        return backward_passes[-1]

    def _sync_backward_projection_if_needed(self: "Trace") -> None:
        """Synchronize lazy backward projections from runtime sidecar events."""

        if getattr(self, "_tl_active_backward_bracket", False):
            return
        if not getattr(self, "backward_events", ()):
            return
        from ..backends.torch.backward import (
            _backward_finalize_pending,
            _close_implicit_backward_pass_if_open,
            _materialize_backward_projections,
        )

        _close_implicit_backward_pass_if_open(self)
        # A read from INSIDE an engine invocation journals but must not
        # materialize while the close's FINALIZE step is still pending:
        # materializing there would publish records ahead of the R36-1 D2H
        # fence (L9 memo 1.2). The first post-pass read finalizes fully.
        if not _backward_finalize_pending(self):
            _materialize_backward_projections(self)

    @property
    def saved_ops(self: "Trace") -> Accessor[Op]:
        """Access Ops with saved activations."""

        return _trace_views.saved_ops(self)

    @property
    def saved_grad_ops(self: "Trace") -> Accessor[Op]:
        """Access Ops with saved gradients."""

        return _trace_views.saved_grad_ops(self)

    @property
    def saved_layers(self: "Trace") -> Accessor[Layer]:
        """Access Layers containing at least one Op with a saved activation."""

        return _trace_views.saved_layers(self)

    @property
    def _layers_saved_grad(self: "Trace") -> Accessor[Layer]:
        """Access Layers containing at least one Op with a saved gradient."""

        return _trace_views.layers_saved_grad(self)

    @property
    def saved_module_calls(self: "Trace") -> Accessor[Any]:
        """Access ModuleCalls whose outputs include saved activations."""

        return self._module_calls_with_output_in(set(self.saved_ops.keys()))

    def _module_calls_with_output_in(self: "Trace", op_labels: set[str]) -> Accessor[Any]:
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

        from .._postprocess.saved_summary import op_label_aliases

        aliases = op_label_aliases(self)
        calls: OrderedDict[str, Any] = OrderedDict()
        for call in self.module_calls:
            if any(
                aliases.get(label, label) in op_labels for label in getattr(call, "output_ops", [])
            ):
                calls[call.label] = call
        return TraceModuleCallAccessor(calls)

    @property
    def saved_modules(self: "Trace") -> Accessor[Any]:
        """Access Modules with at least one saved-activation ModuleCall."""

        return _trace_views.saved_modules(self)

    @property
    def saved_grad_module_calls(self: "Trace") -> Accessor[Any]:
        """Access ModuleCalls whose outputs include saved gradients."""

        return self._module_calls_with_output_in(set(self.saved_grad_ops.keys()))

    @property
    def saved_grad_modules(self: "Trace") -> Accessor[Any]:
        """Access Modules with at least one saved-gradient ModuleCall."""

        return _trace_views.saved_grad_modules(self)

    @property
    def _autograd_saved_grad_fn_calls(self: "Trace") -> Accessor[Any]:
        """Access GradFnCall records with saved gradient inputs or outputs."""

        return _trace_views.autograd_saved_grad_fn_calls(self)

    @property
    def _autograd_saved_grad_fns(self: "Trace") -> GradFnAccessor:
        """Access GradFns containing at least one saved GradFnCall."""

        return _trace_views.autograd_saved_grad_fns(self)

    @property
    def compute_ops(self: "Trace") -> Accessor[Op]:
        """Access Ops that are not graph-boundary sentinels."""

        return _trace_views.compute_ops(self)

    @property
    def _layers_compute(self: "Trace") -> Accessor[Layer]:
        """Access Layers whose representative Op is not a boundary sentinel."""

        return _trace_views.layers_compute(self)

    @property
    def input_ops(self: "Trace") -> Accessor[Op]:
        """Access flat input-boundary Ops."""

        return _trace_views.input_ops(self)

    @property
    def num_input_layers(self: "Trace") -> int:
        """Number of distinct input-boundary Layers (a multi-pass Layer counts once)."""

        return len(dict.fromkeys(self.input_layers))

    @property
    def output_ops(self: "Trace") -> Accessor[Op]:
        """Access flat output-boundary Ops."""

        return _trace_views.output_ops(self)

    @property
    def num_output_layers(self: "Trace") -> int:
        """Number of distinct output-boundary Layers (a multi-pass Layer counts once)."""

        return len(dict.fromkeys(self.output_layers))

    @property
    def num_buffer_layers(self: "Trace") -> int:
        """Number of distinct buffer-boundary Layers.

        ``buffer_layers`` lists a buffer Layer once per pass (a buffer read in
        a loop repeats), so the count dedupes it.
        """

        return len(dict.fromkeys(self.buffer_layers))

    @property
    def buffer_read_ops(self: "Trace") -> list[str]:
        """Labels for buffer Ops that read registered-buffer values into the graph."""

        return _trace_views.buffer_read_ops(self)

    @property
    def buffer_write_ops(self: "Trace") -> list[str]:
        """Labels for buffer Ops that record registered-buffer write events.

        Note: buffer writes are only tracked in exhaustive capture. A Trace
        cooked from a ``record(...)`` Recording (predicate mode) does not track
        buffer writes, so this list is empty even for a model that mutated its
        buffers (e.g. training-mode BatchNorm); ``buffer_write_kind is None``
        there means "not tracked", not "read-only". See
        ``Recording.to_trace``'s Notes.
        """

        return _trace_views.buffer_write_ops(self)

    @property
    def _layers_internal_source(self: "Trace") -> Accessor[Layer]:
        """Access Layers representing internal-source positions."""

        return _trace_views.layers_internal_source(self)

    @property
    def _layers_internal_sink(self: "Trace") -> Accessor[Layer]:
        """Access Layers representing internal-sink positions."""

        return _trace_views.layers_internal_sink(self)

    @property
    def num_uncalled_modules(self: "Trace") -> int | None:
        """Number of registered source-model modules not called in this Trace.

        Recorded when the capture finished, so it survives the source model
        being garbage-collected. ``None`` when it cannot be known: a loaded or
        unpickled trace (the record is session-time) whose model is gone.
        """

        return _trace_views.num_uncalled_modules(self)

    @property
    def ops_with_params(self: "Trace") -> Accessor[Op]:
        """Access Ops that use at least one parameter tensor."""

        return _trace_views.ops_with_params(self)

    def between(self: "Trace", sources: Any, sinks: Any) -> "TraceSlice":
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

        return _doors.trace_slice().build_slice_between(self, sources, sinks)

    def subgraph(self: "Trace", selection: Any) -> "TraceSlice":
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

        return _doors.trace_slice().build_slice_from_selection(self, selection)
