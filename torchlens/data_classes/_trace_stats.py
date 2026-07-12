"""Trace computed stats mixin."""

from collections import OrderedDict
from collections.abc import Collection, Iterator, Mapping
from typing import TYPE_CHECKING, Any, Dict, List, Literal, TextIO, Tuple, cast


if TYPE_CHECKING:
    from ..receptive_field._types import ReceptiveFieldProfile, ReceptiveFieldStatus
    from ..report._profile import TraceProfile
    from ..visualization.collapse_plan import CollapsePlan, CollapseSchedule, RenderContext
    from .buffer import BufferAccessor
    from .layer import LayerAccessor
    from .module import Module, ModuleCall
    from .trace import Trace

    _TraceMixinBase = Trace
else:
    _TraceMixinBase = object

from .._deprecations import warn_deprecated_alias
from ..quantities import Duration, Flops, Macs, as_duration
from ._accessor_base import Accessor
from ._trace_profile import (
    ModelProfile,
    _infer_input_modality,
    _raw_input_contains_images,
    _raw_input_num_stimuli,
)
from ._trace_accessors import (
    OrphanAccessor,
    TraceGradFnCallAccessor,
    TraceModuleCallAccessor,
    TraceOpAccessor,
    _TRACE_LAYER_ACCESSOR_CACHE,
    _TRACE_OP_ACCESSOR_CACHE,
)
from .backward_pass import BackwardPass, BackwardPassAccessor
from .derived_grad import IntermediateDerivedGradAccessor
from .grad_fn import GradFnAccessor
from .layer import Layer
from .module import ModuleAccessor
from .op import Op
from .param import ParamAccessor


class _CallableDict(dict[Any, Any]):
    """Dict that returns a plain dict when called.

    This preserves legacy ``log.report_by_type()`` ergonomics for budgeted
    report properties that should not remain inspectable custom_methods.
    """

    def __call__(self) -> dict[Any, Any]:
        """Return a plain-dict copy of this report.

        Returns
        -------
        dict[Any, Any]
            Plain dict containing this report's items.
        """

        return dict(self)


def _legacy_conditional_then_entry_edges(
    conditional_arm_entry_edges: Mapping[Tuple[int, str], List[Tuple[str, str]]],
) -> List[Tuple[str, str]]:
    """Return the legacy THEN-edge view from canonical conditional arm edges.

    Parameters
    ----------
    conditional_arm_entry_edges:
        Canonical ``(cond_id, branch_kind) -> edge list`` mapping.

    Returns
    -------
    List[Tuple[str, str]]
        Legacy ``(parent, child)`` THEN-edge view.
    """

    return [
        edge
        for (_conditional_id, branch_kind), edges in conditional_arm_entry_edges.items()
        if branch_kind == "then"
        for edge in edges
    ]


def _legacy_conditional_elif_entry_edges(
    conditional_arm_entry_edges: Mapping[Tuple[int, str], List[Tuple[str, str]]],
) -> List[Tuple[int, int, str, str]]:
    """Return the legacy ELIF-edge view from canonical conditional arm edges.

    Parameters
    ----------
    conditional_arm_entry_edges:
        Canonical ``(cond_id, branch_kind) -> edge list`` mapping.

    Returns
    -------
    List[Tuple[int, int, str, str]]
        Legacy ``(cond_id, elif_index, parent, child)`` ELIF-edge view.
    """

    return [
        (conditional_id, int(branch_kind.split("_", 1)[1]), parent, child)
        for (conditional_id, branch_kind), edges in conditional_arm_entry_edges.items()
        if branch_kind.startswith("elif_")
        for parent, child in edges
    ]


def _legacy_conditional_else_entry_edges(
    conditional_arm_entry_edges: Mapping[Tuple[int, str], List[Tuple[str, str]]],
) -> List[Tuple[int, str, str]]:
    """Return the legacy ELSE-edge view from canonical conditional arm edges.

    Parameters
    ----------
    conditional_arm_entry_edges:
        Canonical ``(cond_id, branch_kind) -> edge list`` mapping.

    Returns
    -------
    List[Tuple[int, str, str]]
        Legacy ``(cond_id, parent, child)`` ELSE-edge view.
    """

    return [
        (conditional_id, parent, child)
        for (conditional_id, branch_kind), edges in conditional_arm_entry_edges.items()
        if branch_kind == "else"
        for parent, child in edges
    ]


class TraceStatsMixin(_TraceMixinBase):
    # ********************************************
    # ********** Computed Properties *************
    # ********************************************

    @property
    def conditional_then_entry_edges(self: "Trace") -> List[Tuple[str, str]]:
        """Deprecated THEN-edge view derived from ``conditional_arm_entry_edges``.

        Returns
        -------
        List[Tuple[str, str]]
            Legacy ``(parent, child)`` edge view.
        """

        warn_deprecated_alias("conditional_then_entry_edges", "conditional_arm_entry_edges")
        return _legacy_conditional_then_entry_edges(self.conditional_arm_entry_edges)

    @conditional_then_entry_edges.setter
    def conditional_then_entry_edges(self: "Trace", value: List[Tuple[str, str]]) -> None:
        """Set the deprecated THEN-edge view by updating canonical arm edges.

        Parameters
        ----------
        value:
            Legacy ``(parent, child)`` edge list. Edges are assigned to
            conditional id 0 because the legacy view did not carry ids.
        """

        warn_deprecated_alias("conditional_then_entry_edges", "conditional_arm_entry_edges")
        self.conditional_arm_entry_edges = {
            key: edges
            for key, edges in self.conditional_arm_entry_edges.items()
            if key[1] != "then"
        }
        if value:
            self.conditional_arm_entry_edges[(0, "then")] = list(value)

    @property
    def conditional_elif_entry_edges(self: "Trace") -> List[Tuple[int, int, str, str]]:
        """Deprecated ELIF-edge view derived from ``conditional_arm_entry_edges``.

        Returns
        -------
        List[Tuple[int, int, str, str]]
            Legacy ``(cond_id, elif_index, parent, child)`` edge view.
        """

        warn_deprecated_alias("conditional_elif_entry_edges", "conditional_arm_entry_edges")
        return _legacy_conditional_elif_entry_edges(self.conditional_arm_entry_edges)

    @conditional_elif_entry_edges.setter
    def conditional_elif_entry_edges(self: "Trace", value: List[Tuple[int, int, str, str]]) -> None:
        """Set the deprecated ELIF-edge view by updating canonical arm edges.

        Parameters
        ----------
        value:
            Legacy ``(cond_id, elif_index, parent, child)`` edge list.
        """

        warn_deprecated_alias("conditional_elif_entry_edges", "conditional_arm_entry_edges")
        self.conditional_arm_entry_edges = {
            key: edges
            for key, edges in self.conditional_arm_entry_edges.items()
            if not key[1].startswith("elif_")
        }
        for conditional_id, elif_index, parent, child in value:
            self.conditional_arm_entry_edges.setdefault(
                (conditional_id, f"elif_{elif_index}"), []
            ).append((parent, child))

    @property
    def conditional_else_entry_edges(self: "Trace") -> List[Tuple[int, str, str]]:
        """Deprecated ELSE-edge view derived from ``conditional_arm_entry_edges``.

        Returns
        -------
        List[Tuple[int, str, str]]
            Legacy ``(cond_id, parent, child)`` edge view.
        """

        warn_deprecated_alias("conditional_else_entry_edges", "conditional_arm_entry_edges")
        return _legacy_conditional_else_entry_edges(self.conditional_arm_entry_edges)

    @conditional_else_entry_edges.setter
    def conditional_else_entry_edges(self: "Trace", value: List[Tuple[int, str, str]]) -> None:
        """Set the deprecated ELSE-edge view by updating canonical arm edges.

        Parameters
        ----------
        value:
            Legacy ``(cond_id, parent, child)`` edge list.
        """

        warn_deprecated_alias("conditional_else_entry_edges", "conditional_arm_entry_edges")
        self.conditional_arm_entry_edges = {
            key: edges
            for key, edges in self.conditional_arm_entry_edges.items()
            if key[1] != "else"
        }
        for conditional_id, parent, child in value:
            self.conditional_arm_entry_edges.setdefault((conditional_id, "else"), []).append(
                (parent, child)
            )

    @property
    def is_recurrent(self: "Trace") -> bool:
        """Whether any layer has more than one pass."""
        return any(v > 1 for v in self.layer_num_calls.values())

    @property
    def recurrent_layers(self: "Trace") -> "LayerAccessor":
        """Access Layers with more than one captured pass.

        Returns
        -------
        LayerAccessor
            Accessor containing aggregate Layer records whose ``num_passes`` is
            greater than 1.
        """
        from .layer import LayerAccessor

        return LayerAccessor(
            OrderedDict(
                (label, layer) for label, layer in self.layer_logs.items() if layer.num_passes > 1
            ),
            source_trace=self,
        )

    @property
    def max_layer_op_count(self: "Trace") -> int:
        """Maximum number of ops for any layer."""
        return max(self.layer_num_calls.values(), default=1)

    @property
    def is_branching(self: "Trace") -> bool:
        """Whether any layer has more than one child."""
        return any(len(entry.children) > 1 for entry in self.layer_list)

    @property
    def has_conditional_branching(self: "Trace") -> bool:
        """Whether any layer is in a conditional branch."""
        return any(entry.is_in_conditional_body for entry in self.layer_list)

    @property
    def has_conditionals(self: "Trace") -> bool:
        """Whether this Trace contains conditional-flow records."""

        return len(self.conditionals) > 0

    @property
    def num_conditionals(self: "Trace") -> int:
        """Number of conditional-flow records."""

        return len(self.conditionals)

    @property
    def is_dynamic_graph(self: "Trace") -> bool:
        """Whether execution depends on runtime tensor values."""

        return self.has_conditionals

    @property
    def forward_source_location(self: "Trace") -> str | None:
        """Combined forward source location."""

        if self.forward_source_file is None or self.forward_source_line is None:
            return None
        return f"{self.forward_source_file}:{self.forward_source_line}"

    @property
    def class_source_location(self: "Trace") -> str | None:
        """Combined model class source location."""

        if self.class_source_file is None or self.class_source_line is None:
            return None
        return f"{self.class_source_file}:{self.class_source_line}"

    @property
    def init_source_location(self: "Trace") -> str | None:
        """Combined model ``__init__`` source location."""

        if self.init_source_file is None or self.init_source_line is None:
            return None
        return f"{self.init_source_file}:{self.init_source_line}"

    @property
    def num_tensors(self: "Trace") -> int:
        """Total number of tensor operations."""
        return len(self)

    @property
    def last_backward_duration(self: "Trace") -> Duration | None:
        """Most recent backward-pass duration, if any."""

        if not self.backward_durations:
            return None
        return as_duration(self.backward_durations[-1])

    @property
    def total_backward_duration(self: "Trace") -> Duration:
        """Sum of all captured backward-pass durations."""

        return Duration(sum(self.backward_durations))

    @property
    def last_backward_root_grad_fn_object_id(self: "Trace") -> int | None:
        """Most recent backward root grad_fn_handle object id, if any."""

        if not self.backward_root_grad_fn_object_ids:
            return None
        return self.backward_root_grad_fn_object_ids[-1]

    @property
    def overhead_duration(self: "Trace") -> Duration:
        """Time spent on TorchLens overhead (total minus function calls)."""
        return self.capture_duration - self.func_calls_duration

    @property
    def capture_duration(self: "Trace") -> Duration:
        """Total capture-phase duration in seconds."""

        if not self.capture_start_time or not self.capture_end_time:
            return Duration(0)
        return Duration(self.capture_end_time - self.capture_start_time)

    # ********************************************
    # ************* FLOPs Properties *************
    # ********************************************
    # FLOPs are estimated per-operation during logging (flops_forward,
    # flops_backward on each Op).  These properties aggregate
    # across the entire model.  Layers with None FLOPs (unknown ops) are
    # skipped, so the totals may undercount.

    @property
    def total_flops_forward(self: "Trace") -> Flops:
        """Total forward FLOPs across all layers (skipping None/unknown)."""
        return Flops(
            sum(entry.flops_forward for entry in self.layer_list if entry.flops_forward is not None)
        )

    @property
    def total_flops_backward(self: "Trace") -> Flops:
        """Total backward FLOPs across all layers (skipping None/unknown)."""
        return Flops(
            sum(
                entry.flops_backward
                for entry in self.layer_list
                if entry.flops_backward is not None
            )
        )

    @property
    def total_flops(self: "Trace") -> Flops:
        """Total FLOPs (forward + backward)."""
        return Flops(self.total_flops_forward + self.total_flops_backward)

    @property
    def flops_by_op_type(self: "Trace") -> _CallableDict:
        """Group FLOPs by layer type.

        Returns:
            Callable dict mapping layer_type to forward/backward/count totals.
        """
        result: Dict[str, Dict[str, int | Flops]] = {}
        for entry in self.layer_list:
            lt = entry.layer_type
            if lt not in result:
                result[lt] = {"forward": Flops(0), "backward": Flops(0), "count": 0}
            result[lt]["count"] += 1
            if entry.flops_forward is not None:
                result[lt]["forward"] += entry.flops_forward
            if entry.flops_backward is not None:
                result[lt]["backward"] += entry.flops_backward
        return _CallableDict(result)

    # ********************************************
    # ************** MACs Properties *************
    # ********************************************
    # MACs (multiply-accumulate operations) = FLOPs / 2.

    @property
    def total_macs_forward(self: "Trace") -> Macs:
        """Total forward MACs across all layers (skipping None/unknown)."""
        return Macs(self.total_flops_forward // 2)

    @property
    def total_macs_backward(self: "Trace") -> Macs:
        """Total backward MACs across all layers (skipping None/unknown)."""
        return Macs(self.total_flops_backward // 2)

    @property
    def total_macs(self: "Trace") -> Macs:
        """Total MACs (forward + backward)."""
        return Macs(self.total_flops // 2)

    @property
    def macs_by_op_type(self: "Trace") -> _CallableDict:
        """Group MACs by layer type.

        Returns:
            Callable dict mapping layer_type to forward/backward/count totals.
        """
        result: Dict[str, Dict[str, int | Macs]] = {}
        for entry in self.layer_list:
            lt = entry.layer_type
            if lt not in result:
                result[lt] = {"forward": Macs(0), "backward": Macs(0), "count": 0}
            result[lt]["count"] += 1
            if entry.flops_forward is not None:
                result[lt]["forward"] += Macs(entry.flops_forward // 2)
            if entry.flops_backward is not None:
                result[lt]["backward"] += Macs(entry.flops_backward // 2)
        return _CallableDict(result)

    # ********************************************
    # ************* Params Accessor **************
    # ********************************************

    @property
    def params(self: "Trace") -> ParamAccessor:
        """Access parameter metadata by address, short name, or index."""
        return self.param_logs

    @property
    def ops(self: "Trace") -> TraceOpAccessor:
        """Access per-invocation Op records by label or index."""

        cache_key = len(self.layer_list)
        cache_entry = _TRACE_OP_ACCESSOR_CACHE.get(self)
        if cache_entry is None or cache_entry[0] != cache_key:
            accessor = TraceOpAccessor(self.layer_list, self.layer_num_calls)
            _TRACE_OP_ACCESSOR_CACHE[self] = (cache_key, accessor)
            return accessor
        return cache_entry[1]

    @property
    def transforms(self: "Trace") -> tuple[Op, ...]:
        """Return transform-boundary operation records.

        Returns
        -------
        tuple[Op, ...]
            Ops whose ``is_transform`` role flag is true.
        """

        return tuple(op for op in self.ops if getattr(op, "is_transform", False))

    @property
    def model_profile(self: "Trace") -> ModelProfile:
        """Return a computed semantic I/O profile descriptor.

        Returns
        -------
        ModelProfile
            Runtime-only descriptor derived from preprocessing, raw-input, and
            output-label metadata. This property is not persisted.
        """

        input_preprocessor = self.input_preprocessor
        output_postprocessor = self.output_postprocessor
        input_source = input_preprocessor.source if input_preprocessor is not None else None
        output_source = output_postprocessor.source if output_postprocessor is not None else None
        has_output_labels = bool(self.output_id2label)
        output_label_count = (
            len(self.output_id2label)
            if self.output_id2label is not None
            else self.output_num_classes
        )
        has_raw_images = _raw_input_contains_images(self.raw_input)
        input_modality = _infer_input_modality(self.raw_input, input_source)
        return ModelProfile(
            input_modality=input_modality,
            input_preprocessing_source=input_source,
            output_postprocessing_source=output_source,
            output_label_count=output_label_count,
            has_output_labels=has_output_labels,
            num_stimuli=_raw_input_num_stimuli(self.raw_input),
            has_raw_images=has_raw_images,
            keystone_applicable=bool(
                input_modality == "image"
                and has_raw_images
                and has_output_labels
                and output_label_count is not None
            ),
        )

    def profile(
        self: "Trace",
        level: Literal["op", "module", "call"] = "op",
        *,
        sort_by: Literal["time", "flops", "activation_memory", "param_count"] = "time",
        ascending: bool = False,
    ) -> "TraceProfile":
        """Return a unified resource profile assembled from this trace.

        Parameters
        ----------
        level:
            Profile granularity: operation, module address, or module call.
        sort_by:
            Resource column used for sorting. Time is descending by default.
        ascending:
            Whether to sort the selected metric ascending.

        Returns
        -------
        torchlens.report.TraceProfile
            Printable table object that also exposes ``to_pandas()``.

        Notes
        -----
        Per-operation wall-times are captured under instrumentation. Treat them
        as relative hotspot guidance, not clean benchmark timings.
        """

        from ..report._profile import build_profile

        return build_profile(self, level=level, sort_by=sort_by, ascending=ascending)

    def receptive_fields(
        self: "Trace",
        level: Literal["op", "layer", "call", "module"] = "op",
        *,
        input: "Op | None" = None,
        statuses: "Collection[ReceptiveFieldStatus] | None" = None,
        sort_by: str | None = None,
        ascending: bool = True,
    ) -> "ReceptiveFieldProfile":
        """Return the trace-wide geometric receptive-field table.

        Parameters
        ----------
        level:
            Table granularity: operation, layer, module invocation, or module.
        input:
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

        from ..receptive_field._table import build_rf_profile

        return build_rf_profile(
            self,
            level=level,
            input=input,
            statuses=statuses,
            sort_by=sort_by,
            ascending=ascending,
        )

    @property
    def layers(self: "Trace") -> "LayerAccessor":
        """Access aggregate per-layer metadata by label, index, or pass notation."""
        from .layer import LayerAccessor

        cache_key = len(self.layer_logs)
        cache_entry = _TRACE_LAYER_ACCESSOR_CACHE.get(self)
        if cache_entry is None or cache_entry[0] != cache_key:
            accessor = LayerAccessor(self.layer_logs, source_trace=self)
            _TRACE_LAYER_ACCESSOR_CACHE[self] = (cache_key, accessor)
            return accessor
        return cast("LayerAccessor", cache_entry[1])

    @property
    def modules(self: "Trace") -> "ModuleAccessor":
        """Access structured per-module metadata by address, index, or pass notation."""
        return self._module_logs

    @property
    def module_collapse_order(self: "Trace") -> list[tuple[str, float]]:
        """Canonical smart-collapse module ranking.

        Returns
        -------
        list[tuple[str, float]]
            ``(module_address, rounded_score)`` sorted by ``(-score, address)``.
        """

        from ..visualization.auto_collapse import collapse_order

        return collapse_order(self)

    def collapse_order(
        self: "Trace",
        weights: Any | None = None,
        mode: Literal["auto", "max"] = "auto",
    ) -> list[tuple[str, float]]:
        """Return smart-collapse ranking for a custom policy.

        Parameters
        ----------
        weights:
            Ignored legacy mapping retained for call compatibility.
        mode:
            ``"auto"`` or ``"max"`` landmark policy.

        Returns
        -------
        list[tuple[str, float]]
            ``(module_address, rounded_score)`` sorted by ``(-score, address)``.
        """

        from ..visualization.auto_collapse import collapse_order

        return collapse_order(self, weights=weights, mode=mode)

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
            Optional :class:`torchlens.visualization.collapse_plan.RenderContext`.
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

        if isinstance(mode, float):
            if not 0.0 <= mode <= 1.0:
                raise ValueError("collapse float level must be in [0.0, 1.0].")
        elif mode not in {"auto", "max"}:
            raise ValueError("mode must be one of 'auto', 'max', or a float in [0.0, 1.0].")

        from ..visualization.collapse_optimizer import select_collapse_level, select_collapse_plan
        from ..visualization.collapse_plan import RenderContext

        resolved_context = RenderContext() if context is None else context
        result = (
            select_collapse_level(self, resolved_context, mode)
            if isinstance(mode, float)
            else select_collapse_plan(self, resolved_context, mode=mode)
        )
        if result.declined:
            reason = result.reason or "unsupported render context"
            raise ValueError(f"collapse plan unavailable: {reason}")
        return result.plan

    def collapse_schedule(
        self: "Trace",
        context: "RenderContext | None" = None,
    ) -> "CollapseSchedule":
        """Return the public monotone float collapse schedule.

        Parameters
        ----------
        context:
            Optional :class:`torchlens.visualization.collapse_plan.RenderContext`.
            When omitted, the default unrolled Graphviz context is used.

        Returns
        -------
        CollapseSchedule
            Ordered schedule from ``t=0.0`` full graph to ``t=1.0`` max
            collapse. Each step includes the selected plan, visible count, and
            collapsed module-address set.
        """

        from ..visualization.collapse_optimizer import collapse_schedule
        from ..visualization.collapse_plan import RenderContext

        resolved_context = RenderContext() if context is None else context
        return collapse_schedule(self, resolved_context)

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

        calls: OrderedDict[str, Any] = OrderedDict()
        for module in self._module_logs:
            for call in module.calls.values():
                calls[call.call_label] = call
        return TraceModuleCallAccessor(calls)

    @property
    def num_module_calls(self: "Trace") -> int:
        """Total number of module invocations recorded in this Trace."""

        return len(self.module_calls)

    def _root_call(self: "Trace") -> "ModuleCall":
        """Return the top-level ModuleCall for internal call-tree traversal.

        When multiple top-level calls exist, the first one in trace insertion
        order is returned.
        """

        for call in self.module_calls:
            if call.call_parent is None:
                return cast("ModuleCall", call)
        raise RuntimeError("Trace has no root ModuleCall")

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

        if not self.module_calls:
            return
        yield from self._root_call().walk_descendants(include_self=True)

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

        if not self.module_calls:
            return
        self._root_call().show_call_tree(
            max_depth=max_depth,
            include_atomic=include_atomic,
            show_call_index=show_call_index,
            file=file,
        )

    @property
    def root_module(self: "Trace") -> "Module":
        """The root module (the model itself)."""
        return cast("Module", self._module_logs["self"])

    @property
    def num_layers(self: "Trace") -> int:
        """Number of distinct Layer records in this Trace."""

        return len(self.layers)

    @property
    def num_compute_layers(self: "Trace") -> int:
        """Number of compute Layers in this Trace."""

        return len(self.compute_layers)

    @property
    def num_compute_ops(self: "Trace") -> int:
        """Number of compute Ops in this Trace."""

        return len(self.compute_ops)

    @property
    def num_edges(self: "Trace") -> int:
        """Distinct edges in the per-pass Op graph, including boundary edges."""

        return len(
            {
                (op.label, self.ops[child_label].label)
                for op in self.ops
                for child_label in op.children
            }
        )

    @property
    def num_compute_edges(self: "Trace") -> int:
        """Distinct Op graph edges whose endpoints are both compute Ops."""

        compute_labels = {op.label for op in self.compute_ops}
        return len(
            {
                (op.label, self.ops[child_label].label)
                for op in self.compute_ops
                for child_label in op.children
                if self.ops[child_label].label in compute_labels
            }
        )

    @property
    def num_buffer_edges(self: "Trace") -> int:
        """Distinct Op graph edges with at least one buffer endpoint."""

        return len(
            {
                (op.label, self.ops[child_label].label)
                for op in self.ops
                for child_label in op.children
                if op.is_buffer or self.ops[child_label].is_buffer
            }
        )

    @property
    def num_layer_edges(self: "Trace") -> int:
        """Distinct edges in the aggregate Layer graph."""

        return len(
            {
                (layer.layer_label, child_label)
                for layer in self.layers
                for child_label in layer.children
            }
        )

    @property
    def num_backward_edges(self: "Trace") -> int | None:
        """Distinct edges in the backward GradFn graph, or ``None`` if ungated."""

        if not self.has_backward_pass:
            return None
        return len(
            {
                (grad_fn.label, child_label)
                for grad_fn in self.grad_fns
                for child_label in grad_fn.children
            }
        )

    @property
    def branching_factor(self: "Trace") -> float:
        """Mean fan-out: children (consumers) per compute Op.

        Computed over a single, consistent node set (compute Ops) so the ratio is
        a coherent mean out-degree: ~1.0 for a plain chain, >1.0 when ops are
        reused (residual streams, shared embeddings, dense connectivity). Counts
        every child of each compute Op (including boundary/buffer consumers).
        """

        compute_ops = list(self.compute_ops)
        if not compute_ops:
            return 0.0
        return sum(op.num_children for op in compute_ops) / len(compute_ops)

    @property
    def max_in_degree(self: "Trace") -> int:
        """Maximum number of parents over compute Ops."""

        return max((op.num_parents for op in self.compute_ops), default=0)

    @property
    def max_out_degree(self: "Trace") -> int:
        """Maximum number of children over compute Ops."""

        return max((op.num_children for op in self.compute_ops), default=0)

    @property
    def num_saved_grad_ops(self: "Trace") -> int:
        """Number of Ops with saved gradients."""

        return len(self.saved_grad_ops)

    @property
    def intermediate_derived_grads(self: "Trace") -> IntermediateDerivedGradAccessor:
        """Access exact op-level derived gradient records, when a backend provides them.

        Returns
        -------
        IntermediateDerivedGradAccessor
            Records keyed by pass-qualified op label. Backends that did not run
            an intermediate-derived-gradient pass return an empty accessor.
        """

        records = self.__dict__.get("_intermediate_derived_grads")
        if isinstance(records, IntermediateDerivedGradAccessor):
            return records
        return IntermediateDerivedGradAccessor()

    @intermediate_derived_grads.setter
    def intermediate_derived_grads(self: "Trace", value: IntermediateDerivedGradAccessor) -> None:
        """Store exact op-level derived gradient records.

        Parameters
        ----------
        value
            Accessor to expose through ``trace.intermediate_derived_grads``.
        """

        self.__dict__["_intermediate_derived_grads"] = value

    @property
    def num_saved_grad_layers(self: "Trace") -> int:
        """Number of Layers containing at least one saved-gradient Op."""

        return len(self.saved_grad_layers)

    @property
    def num_param_tensors_trainable(self: "Trace") -> int:
        """Number of trainable parameter tensors in this Trace."""

        return sum(1 for param in self.params if param.is_trainable)

    @property
    def num_param_tensors_frozen(self: "Trace") -> int:
        """Number of frozen parameter tensors in this Trace."""

        return sum(1 for param in self.params if not param.is_trainable)

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
    def num_modules(self: "Trace") -> int:
        """Total number of registered source-model submodules."""

        return len(self._module_logs)

    @num_modules.deleter
    def num_modules(self: "Trace") -> None:
        """Ignore cleanup deletion for derived module count."""

    @property
    def orphans(self: "Trace") -> OrphanAccessor:
        """Access retained orphan island operations by raw or final label."""

        orphan_dict = OrderedDict(
            (log.layer_label, log) for log in self._orphan_logs if getattr(log, "is_orphan", False)
        )
        return OrphanAccessor(orphan_dict)

    @property
    def grad_fns(self: "Trace") -> GradFnAccessor:
        """Access backward grad_fn_handle metadata by label, index, pass label, or substring."""
        self._sync_backward_projection_if_needed()
        return GradFnAccessor(self.grad_fn_logs, self.grad_fn_order)

    @property
    def grad_fn_calls(self: "Trace") -> TraceGradFnCallAccessor:
        """Access per-invocation GradFnCall records by qualified label or index."""

        self._sync_backward_projection_if_needed()
        calls: OrderedDict[str, Any] = OrderedDict()
        for grad_fn_handle in self.grad_fns:
            for call_index, call in grad_fn_handle.calls.items():
                call.source_trace = self
                calls[f"{grad_fn_handle.label}:{call_index}"] = call
        return TraceGradFnCallAccessor(calls)

    @property
    def backward_passes(self: "Trace") -> BackwardPassAccessor:
        """Access backward pass records by 0-based position or named pass number."""

        if getattr(self, "backend", "torch") in {"jax", "mlx", "tinygrad"}:
            raise ValueError(
                f"{getattr(self, 'backend', 'backend')} traces do not support true backward "
                "capture. Use trace.derived_grads for leaf-level derived gradients computed "
                "by the backend preview."
            )
        self._sync_backward_projection_if_needed()
        return BackwardPassAccessor(self.backward_pass_logs)

    @property
    def last_backward_pass(self: "Trace") -> BackwardPass | None:
        """Return the most recent backward pass record, if any."""

        backward_passes = self.backward_passes
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
            _close_implicit_backward_pass_if_open,
            _materialize_backward_projections,
        )

        _close_implicit_backward_pass_if_open(self)
        _materialize_backward_projections(self)

    @property
    def num_grad_fn_calls(self: "Trace") -> int:
        """Total number of GradFnCall records in this Trace."""

        return len(self.grad_fn_calls)

    @property
    def saved_ops(self: "Trace") -> Accessor[Op]:
        """Access Ops with saved activations."""

        return TraceOpAccessor(
            [op for op in self.layer_list if op.has_saved_activation],
            self.layer_num_calls,
        )

    @property
    def saved_grad_ops(self: "Trace") -> Accessor[Op]:
        """Access Ops with saved gradients."""

        if getattr(self, "backend", "torch") in {"jax", "mlx", "tinygrad"}:
            raise ValueError(
                f"{getattr(self, 'backend', 'backend')} traces do not expose op-level saved "
                "gradients. Use trace.derived_grads for leaf-level derived gradients computed "
                "by the backend preview."
            )
        return TraceOpAccessor(
            [op for op in self.layer_list if op.has_grad],
            self.layer_num_calls,
        )

    @property
    def saved_layers(self: "Trace") -> Accessor[Layer]:
        """Access Layers containing at least one Op with a saved activation."""

        from .layer import LayerAccessor

        return LayerAccessor(
            OrderedDict(
                (label, layer)
                for label, layer in self.layer_logs.items()
                if any(op.has_saved_activation for op in layer.ops.values())
            ),
            source_trace=self,
        )

    @property
    def saved_grad_layers(self: "Trace") -> Accessor[Layer]:
        """Access Layers containing at least one Op with a saved gradient."""

        from .layer import LayerAccessor

        return LayerAccessor(
            OrderedDict(
                (label, layer)
                for label, layer in self.layer_logs.items()
                if any(op.has_grad for op in layer.ops.values())
            ),
            source_trace=self,
        )

    @property
    def saved_module_calls(self: "Trace") -> Accessor[Any]:
        """Access ModuleCalls whose outputs include saved activations."""

        saved_labels = set(self.saved_ops.keys())
        calls: OrderedDict[str, Any] = OrderedDict()
        for call in self.module_calls:
            if any(label in saved_labels for label in getattr(call, "output_ops", [])):
                calls[call.call_label] = call
        return TraceModuleCallAccessor(calls)

    @property
    def saved_modules(self: "Trace") -> Accessor[Any]:
        """Access Modules with at least one saved-activation ModuleCall."""

        saved_addresses = {call.address for call in self.saved_module_calls}
        return ModuleAccessor(
            OrderedDict(
                (module.address, module)
                for module in self.modules
                if module.address in saved_addresses
            )
        )

    @property
    def num_saved_modules(self: "Trace") -> int:
        """Number of Modules with at least one saved-activation ModuleCall."""

        return len(self.saved_modules)

    @property
    def saved_grad_module_calls(self: "Trace") -> Accessor[Any]:
        """Access ModuleCalls whose outputs include saved gradients."""

        saved_labels = set(self.saved_grad_ops.keys())
        calls: OrderedDict[str, Any] = OrderedDict()
        for call in self.module_calls:
            if any(label in saved_labels for label in getattr(call, "output_ops", [])):
                calls[call.call_label] = call
        return TraceModuleCallAccessor(calls)

    @property
    def saved_grad_modules(self: "Trace") -> Accessor[Any]:
        """Access Modules with at least one saved-gradient ModuleCall."""

        saved_addresses = {call.address for call in self.saved_grad_module_calls}
        return ModuleAccessor(
            OrderedDict(
                (module.address, module)
                for module in self.modules
                if module.address in saved_addresses
            )
        )

    @property
    def num_saved_grad_module_calls(self: "Trace") -> int:
        """Number of ModuleCalls whose outputs include saved gradients."""

        return len(self.saved_grad_module_calls)

    @property
    def num_saved_grad_modules(self: "Trace") -> int:
        """Number of Modules with at least one saved-gradient ModuleCall."""

        return len(self.saved_grad_modules)

    @property
    def saved_grad_fn_calls(self: "Trace") -> Accessor[Any]:
        """Access GradFnCall records with saved gradient inputs or outputs."""

        calls: OrderedDict[str, Any] = OrderedDict()
        for label, call in self.grad_fn_calls.items():
            if (
                getattr(call, "grad_inputs", None) is not None
                or getattr(call, "grad_outputs", None) is not None
            ):
                calls[label] = call
        return TraceGradFnCallAccessor(calls)

    @property
    def saved_grad_fns(self: "Trace") -> GradFnAccessor:
        """Access GradFns containing at least one saved GradFnCall."""

        items = OrderedDict(
            (grad_fn_object_id, grad_fn_handle)
            for grad_fn_object_id, grad_fn_handle in self.grad_fn_logs.items()
            if any(
                getattr(call, "grad_inputs", None) is not None
                or getattr(call, "grad_outputs", None) is not None
                for call in grad_fn_handle.calls.values()
            )
        )
        return GradFnAccessor(items, list(items))

    @property
    def compute_ops(self: "Trace") -> Accessor[Op]:
        """Access Ops that are not graph-boundary sentinels."""

        return TraceOpAccessor(
            [op for op in self.layer_list if not (op.is_input or op.is_output or op.is_buffer)],
            self.layer_num_calls,
        )

    @property
    def compute_layers(self: "Trace") -> Accessor[Layer]:
        """Access Layers whose representative Op is not a boundary sentinel."""

        from .layer import LayerAccessor

        return LayerAccessor(
            OrderedDict(
                (label, layer)
                for label, layer in self.layer_logs.items()
                if label in {op.layer_label for op in self.compute_ops}
            ),
            source_trace=self,
        )

    @property
    def input_ops(self: "Trace") -> Accessor[Op]:
        """Access flat input-boundary Ops."""

        return TraceOpAccessor([self[label] for label in self.input_layers], self.layer_num_calls)

    @property
    def num_input_layers(self: "Trace") -> int:
        """Number of input-boundary Layers."""

        return len(self.input_layers)

    @property
    def num_input_ops(self: "Trace") -> int:
        """Number of flat input-boundary Ops."""

        return len(self.input_ops)

    @property
    def output_ops(self: "Trace") -> Accessor[Op]:
        """Access flat output-boundary Ops."""

        return TraceOpAccessor([self[label] for label in self.output_layers], self.layer_num_calls)

    @property
    def num_output_layers(self: "Trace") -> int:
        """Number of output-boundary Layers."""

        return len(self.output_layers)

    @property
    def num_output_ops(self: "Trace") -> int:
        """Number of flat output-boundary Ops."""

        return len(self.output_ops)

    @property
    def num_buffer_layers(self: "Trace") -> int:
        """Number of buffer-boundary Layers."""

        return len(self.buffer_layers)

    @property
    def buffer_read_ops(self: "Trace") -> list[str]:
        """Labels for buffer Ops that read registered-buffer values into the graph."""

        return [op.label for op in self.layer_list if op.is_buffer and op.buffer_write_kind is None]

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

        return [
            op.label for op in self.layer_list if op.is_buffer and op.buffer_write_kind is not None
        ]

    @property
    def num_buffer_read_ops(self: "Trace") -> int:
        """Number of buffer Ops that read registered-buffer values into the graph."""

        return len(self.buffer_read_ops)

    @property
    def num_buffer_write_ops(self: "Trace") -> int:
        """Number of buffer Ops that record registered-buffer write events."""

        return len(self.buffer_write_ops)

    @property
    def num_buffer_source_ops(self: "Trace") -> int:
        """Number of buffer source Ops that read registered buffers."""

        return len(self.buffer_read_ops)

    @property
    def num_buffer_sink_ops(self: "Trace") -> int:
        """Number of buffer sink Ops that record registered-buffer writes."""

        return len(self.buffer_write_ops)

    @property
    def internal_source_layers(self: "Trace") -> Accessor[Layer]:
        """Access Layers representing internal-source positions."""

        from .layer import LayerAccessor

        return LayerAccessor(
            OrderedDict(
                (self[label].layer_label, self.layers[self[label].layer_label])
                for label in self.internal_source_ops
            ),
            source_trace=self,
        )

    @property
    def num_internal_source_layers(self: "Trace") -> int:
        """Number of internal-source Layers."""

        return len(self.internal_source_layers)

    @property
    def num_internal_source_ops(self: "Trace") -> int:
        """Number of internal-source Ops."""

        return len(self.internal_source_ops)

    @property
    def internal_sink_layers(self: "Trace") -> Accessor[Layer]:
        """Access Layers representing internal-sink positions."""

        from .layer import LayerAccessor

        return LayerAccessor(
            OrderedDict(
                (self[label].layer_label, self.layers[self[label].layer_label])
                for label in self.internal_sink_ops
            ),
            source_trace=self,
        )

    @property
    def num_internal_sink_layers(self: "Trace") -> int:
        """Number of internal-sink Layers."""

        return len(self.internal_sink_layers)

    @property
    def num_internal_sink_ops(self: "Trace") -> int:
        """Number of internal-sink Ops."""

        return len(self.internal_sink_ops)

    @property
    def num_uncalled_modules(self: "Trace") -> int:
        """Number of registered source-model modules not called in this Trace."""

        return len(self.uncalled_modules)

    @property
    def ops_with_params(self: "Trace") -> Accessor[Op]:
        """Access Ops that use at least one parameter tensor."""

        return TraceOpAccessor(
            [op for op in self.layer_list if op.num_params > 0],
            self.layer_num_calls,
        )

    @property
    def num_ops_with_params(self: "Trace") -> int:
        """Number of Ops that use at least one parameter tensor."""

        return len(self.ops_with_params)

    @property
    def num_grad_fns(self: "Trace") -> int:
        """Number of unique autograd grad_fn_handle nodes discovered."""
        return len(self.grad_fn_logs)

    @property
    def num_grad_fns_with_op(self: "Trace") -> int:
        """Number of GradFn records paired with a forward Op."""

        return sum(1 for grad_fn_handle in self.grad_fn_logs.values() if grad_fn_handle.has_op)

    @property
    def num_grad_fns_without_op(self: "Trace") -> int:
        """Number of grad_fn_handle nodes without a corresponding forward Layer."""
        return sum(1 for grad_fn_handle in self.grad_fn_logs.values() if not grad_fn_handle.has_op)
